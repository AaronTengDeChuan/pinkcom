# coding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import math

from layers import layers
import layers.operations as op
from layers.bert_modeling import BertConfig, PreTrainedBertModel, BertModel, BertLayerNorm, gelu
from nets.BertDownstream import BertModelWrapper

from collections import OrderedDict, Iterable
from functools import reduce
import itertools
import os
import sys
import pickle

from losses.loss import CrossEntropyLoss
from optimizers.optimizer import AdamOptimizer

# base_work_dir = os.path.dirname(os.getcwd())
# sys.path.append(base_work_dir)


from utils import utils
from utils.utils import varname

logger = utils.get_logger()

def optimizer_grouped_parameters(model, optimizer_params, model_params):
    bert_trainable = model_params["bert_trainable"]

    grouped_parameters = []
    all_modules = dict(model.named_children())
    logger.info("\n" + str(list(all_modules.keys())))
    bert_parameters = list(all_modules.pop("bert").named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    if bert_trainable:
        grouped_parameters.append(
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)], **optimizer_params["bert"],
             "weight_decay_rate": 0.01})
        grouped_parameters.append(
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], **optimizer_params["bert"],
             "weight_decay_rate": 0.0})

    other_parameters = reduce(lambda x, y: x + list(y[1].named_parameters()), all_modules.items(), [])
    grouped_parameters.append({"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)]})
    grouped_parameters.append(
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0})

    return grouped_parameters


class BERTSMNModel(nn.Module):
    """SMN Module contains ."""

    def __init__(self, config):
        super(BERTSMNModel, self).__init__()
        # hyperparameters
        self.bert_hidden_size = config["bert_hidden_size"] if "bert_hidden_size" in config else 768
        self.hidden_size = config["hidden_size"] if "hidden_size" in config else 200
        self.rnn_units = config["rnn_units"] if "rnn_units" in config else 200
        self.bert_layers = config["bert_layers"] if "bert_layers" in config else [11]
        self.feature_maps = config["feature_maps"] if "feature_maps" in config else 8
        self.dense_out_dim = config["dense_out_dim"] if "dense_out_dim" in config else 50
        self.drop_prob = config["drop_prob"] if "drop_prob" in config else 0.0
        self.max_num_utterance = config["max_num_utterance"] if "max_num_utterance" in config else 10
        self.max_sentence_len = config["max_sentence_len"] if "max_sentence_len" in config else 50
        self.final_out_features = config["final_out_features"] if "final_out_features" in config else 2
        self.device = config["device"]
        assert "bert_model_dir" in config
        self.bert_model_dir = config["bert_model_dir"]
        self.bert_trainable = config["bert_trainable"]

        # build model
        # network
        self.bert_config = BertConfig.from_json_file(os.path.join(self.bert_model_dir, 'bert_config.json'))
        # self.output_layernorm = BertLayerNorm(self.bert_config)
        self.activation = gelu

        self.dropout = nn.Dropout(self.drop_prob)
        ## Sentence GRU: default batch_first is False
        self.sentence_gru = nn.GRU(self.bert_hidden_size, self.hidden_size, batch_first=True)
        ## Linear Transformation
        self.a_matrix = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.a_matrixs = utils.clones(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False), len(self.bert_layers))

        ## Convolution Layer
        ## valid cross-correlation padding, 2 in_channels, 8 out_channels, kernel_size 3*3
        ## relu activation function and 2d valid max_pooling
        in_channels = 1 + len(self.bert_layers)
        self.conv1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=in_channels, out_channels=self.feature_maps, kernel_size=(3, 3))),
            ("batchnorm", nn.BatchNorm2d(self.feature_maps)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)))
        ]))

        ## Dense: fully connected layer
        in_features = op.calculate_dim_with_initialDim_conv((self.max_sentence_len, self.max_sentence_len),
                                                               self.conv1, in_channels=in_channels)
        self.dense = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features=in_features, out_features=self.dense_out_dim)),
            ("tanh1", nn.Tanh())
        ]))

        ## Final GRU: time major
        self.final_gru = nn.GRU(self.dense_out_dim, self.rnn_units)
        ## SMN Last: Linear Transformation
        self.smn_last_linear = nn.Linear(self.rnn_units, self.final_out_features)

        self.apply(self.init_weights)

        ## Bert pretrained model
        self.bert = BertModelWrapper.from_pretrained(self.bert_model_dir, cache_dir=None)
        # self.emb_linear = nn.Linear(self.bert_hidden_size, self.hidden_size)
        # self.ctxemb_linear = nn.Linear(self.bert_hidden_size, self.hidden_size)
        # self.dense_linear = nn.Linear(self.max_sentence_len * self.max_sentence_len, self.final_out_features)

        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_hidden(self, batch_shape, hidden_size, hidden=None):
        weight = next(self.parameters()).data
        if not isinstance(batch_shape, Iterable):
            batch_shape = [batch_shape]
        bsz = reduce(lambda x, y: x * y, batch_shape, 1)
        if hidden is not None and hidden.shape[1] == bsz:
            return utils.repackage_hidden(hidden)
        else:
            return weight.new(1, bsz, hidden_size).zero_()

    def forward(self, inputs):
        # First Layer --- utterances-Response Matching
        # context_id: torch.Size([None, 10, 50]); response_id: torch.Size([None, 50])
        context_id, context_len, context_mask, context_segment = \
            inputs["context_id"], inputs["context_len"], inputs["context_mask"], inputs["context_segment"]
        response_id, response_len, response_mask, response_segment = \
            inputs["response_id"], inputs["response_len"], inputs["response_mask"], inputs["response_segment"]

        with torch.set_grad_enabled(self.bert_trainable):
            if not self.bert_trainable: self.bert.eval()
            ## push context into bert
            context_encoded_layers, context_pooled_output, context_embeds = self.bert(
                context_id.view(-1, context_id.shape[-1]),
                context_segment.view(-1, context_segment.shape[-1]),
                context_mask.view(-1, context_mask.shape[-1]),
                output_all_encoded_layers=False,
                output_embeddings=True, only_embeddings=True) # torch.Size([None * 10, 50, hidden_size]), torch.Size([None * 10, hidden_size]), torch.Size([None * 10, 50, hidden_size])

            ## push response into bertresponse_output
            response_encoded_layers, response_pooled_output, response_embeds = self.bert(response_id, response_segment, response_mask, output_all_encoded_layers=False,
                                           output_embeddings=True, only_embeddings=True) # torch.Size([None, 50, hidden_size]), torch.Size([None, hidden_size]), torch.Size([None, 50, hidden_size])

        # context_embeds = context_encoded_layers
        # response_embeds = response_encoded_layers
        ## push responses into sentence gru
        self.hidden1 = self.init_hidden(response_embeds.shape[:-2], self.hidden_size, self.hidden1)
        self.sentence_gru.flatten_parameters()
        response_output, response_ht = op.pack_and_pad_sequences_for_rnn(response_embeds, response_len, self.sentence_gru,
                                                                 self.hidden1)  # torch.Size([None, 50, 200] and torch.Size([1, None, 200])

        ## push utterances into sentence gru
        self.hidden2 = self.init_hidden(context_embeds.shape[:-2], self.hidden_size, self.hidden2)
        self.sentence_gru.flatten_parameters()
        context_output, _ = op.pack_and_pad_sequences_for_rnn(context_embeds, context_len, self.sentence_gru,
                                                          self.hidden2)  # torch.Size([None * 10, 50, 200]) and torch.Size([1, None * 10, 200])

        matrix1 = torch.matmul(
            torch.transpose(context_embeds.view(*context_id.shape[:-1], *context_embeds.shape[1:]), 0, 1),
            torch.transpose(response_embeds, 1, 2))
        matrixs = [matrix1]

        matrix2 = self.a_matrix(
            context_output.view(*context_id.shape[:-1], *context_output.shape[1:]))  # torch.Size([None, 10, 50, 200])
        matrix2 = torch.matmul(torch.transpose(matrix2, 0, 1),
                               torch.transpose(response_output, 1, 2))  # torch.Size([10, None, 50, 50])
        matrixs.append(matrix2)

        '''
        matrix = torch.matmul(
            context_embeds.view(*context_id.shape[:-1], *context_embeds.shape[1:])[:, -1],
            torch.transpose(response_embeds, 1, 2))  # [None, 50, 50]
        logits = self.dense_linear(matrix.view(matrix.shape[0], -1))

        context_pooled_output = self.emb_linear(context_pooled_output)
        response_pooled_output = self.emb_linear(response_pooled_output)
        logits = torch.matmul(context_pooled_output.view(*context_id.shape[:-1], -1)[:, -1].unsqueeze(1),
                              response_pooled_output.unsqueeze(1).transpose(1, 2))
        
        
        # context_embeds = self.embeds_layernorm(context_embeds)
        # response_embeds = self.embeds_layernorm(response_embeds)
        # context_output = self.output_layernorm(context_output)
        # response_output = self.output_layernorm(response_output)
        context_embeds = self.dropout(self.emb_linear(context_embeds))
        response_embeds = self.dropout(self.emb_linear(response_embeds))
        context_output = self.dropout(self.ctxemb_linear(context_output))
        response_output = self.dropout(self.ctxemb_linear(response_output))
        '''

        '''
        ## calculate matrix1
        ## [10, None, 50, hidden_size] * [None, hidden_size, 50] -> [10, None, 50, 50]
        matrix1 = torch.matmul(
            torch.transpose(context_embeds.view(*context_id.shape[:-1], *context_embeds.shape[1:]), 0, 1),
            torch.transpose(response_embeds, 1, 2))
        matrixs = [matrix1]

        ## TODO: calculate matrix2: check this
        for i, layer_number in enumerate(self.bert_layers):
            ## [None, 10, 50, hidden_size] * [hidden_size, hidden_size] -> [None, 10, 50, hidden_size]
            ## [10, None, 50, hidden_size] * [None, hidden_size, 50] -> [10, None, 50, 50]
            context_output = context_encoded_layers[layer_number]
            response_output = response_encoded_layers[layer_number]
            matrix2 = self.a_matrixs[i](context_output.view(*context_id.shape[:-1], *context_output.shape[1:])) # torch.Size([None, 10, 50, 200])
            matrix2 = torch.matmul(torch.transpose(matrix2, 0, 1), torch.transpose(response_output, 1, 2))  # torch.Size([10, None, 50, 50])
            matrixs.append(matrix2)
        '''

        ## convolute two matrixes
        ## in_channels: [10, None, 2, 50, 50], out_channels: [10, None, 8, 16, 16]
        conv_output = op.stack_channels_for_conv2d(matrixs, self.conv1)

        conv_output = conv_output.transpose(2,4).transpose(2,3).contiguous()    # torch.Size([10, None, 8, 16, 16])

        # Second Layer --- Matching Accumulation
        ## flat conv_output
        flatten_input = conv_output.view(conv_output.shape[:-3] + tuple([-1]))  # torch.Size([10, None, 2048])
        ## dense layer
        dense_output = self.dense(flatten_input)    # torch.Size([10, None, 50])
        ## push dense_output into final gru
        ## TODO: Why not mask the empty utterances ?
        self.hidden3 = self.init_hidden(dense_output.shape[-2], self.rnn_units, self.hidden3)
        self.final_gru.flatten_parameters()
        final_output, last_hidden = self.final_gru(dense_output, self.hidden3)  # torch.Size([10, None, 50]) and torch.Size([1, None, 50])
        logits = self.smn_last_linear(last_hidden).squeeze(0)  # torch.Size([None, 2])

        return logits.squeeze(-1)


if __name__ == "__main__":
    smn = SMNModel({"device": torch.device("cpu")})
    utt_inputs = torch.randint(0, 434511, (1, 10, 50), dtype=torch.int64)
    utt_inputs = torch.cat([utt_inputs] * 2, dim=0)
    utt_len_inputs = torch.sum(utt_inputs != 0, dim=-1)
    resp_inputs = torch.randint(0, 434511, (2, 50), dtype=torch.int64)
    resp_len_inputs = torch.sum(resp_inputs != 0, dim=-1)
    targets = torch.tensor([1, 0], dtype=torch.int64)
    inputs = {
        "utt": utt_inputs,
        "utt_len": utt_len_inputs,
        "resp": resp_inputs,
        "resp_len": resp_len_inputs,
        "target": targets
    }
    loss_fn = CrossEntropyLoss({})
    optimizer = AdamOptimizer({"lr": 0.01}).ops(smn.parameters())
    for i in range(100):
        smn.train()
        optimizer.zero_grad()
        logits = smn(inputs)
        loss, num_labels, batch_total_loss = loss_fn(logits, inputs["target"])
        loss.backward()
        optimizer.step()

    # print(logits)
    # print(torch.nn.functional.softmax(logits, dim=-1))
    # print(loss.item())
