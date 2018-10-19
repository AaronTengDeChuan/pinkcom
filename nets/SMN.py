# coding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np

from layers import layers
import layers.operations as op

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

class SMNModel(nn.Module):
    """SMN Module contains ."""

    def __init__(self, config):
        super(SMNModel, self).__init__()
        # hyperparameters
        embeddings = config["embeddings"] if "embeddings" in config else None
        self.vocabulary_size = config["vocabulary_size"] if "vocabulary_size" in config else 434511
        self.rnn_units = config["hidden_size"] if "hidden_size" in config else 200
        self.feature_maps = config["feature_maps"] if "feature_maps" in config else 8
        self.dense_out_dim = config["dense_out_dim"] if "dense_out_dim" in config else 50
        self.word_embedding_size = config["embedding_dim"] if "embedding_dim" in config else 200
        self.drop_prob = config["drop_prob"] if "drop_prob" in config else 0.0
        self.max_num_utterance = config["max_num_utterance"] if "max_num_utterance" in config else 10
        self.max_sentence_len = config["max_sentence_len"] if "max_sentence_len" in config else 50
        self.embeddings_trainable = \
            config["emb_trainable"] if "emb_trainable" in config else True
        self.device = config["device"]

        # build model
        ## Embedding
        self.embeddings = nn.Embedding(self.vocabulary_size, self.word_embedding_size)
        if isinstance(embeddings, np.ndarray):
            # TODO: check whether share the storage
            self.embeddings.from_pretrained(torch.tensor(embeddings), freeze=not self.embeddings_trainable)
        elif isinstance(embeddings, torch.Tensor):
            # TODO: check whether share the storage
            self.embeddings.from_pretrained(embeddings, freeze=not self.embeddings_trainable)
        else:
            self.embeddings.weight.requires_grad = self.embeddings_trainable

        # network
        ## Sentence GRU: default batch_first is False
        self.sentence_gru = nn.GRU(self.word_embedding_size, self.rnn_units, batch_first=True)
        ## Linear Transformation
        self.a_matrix = nn.Linear(in_features=self.rnn_units, out_features=self.rnn_units, bias=False)
        # self.a_matrix = torch.nn.Parameter(nn.Linear(in_features=self.rnn_units, out_features=self.rnn_units, bias=False).weight.transpose(0,1))

        ## Convolution Layer
        ## valid cross-correlation padding, 2 in_channels, 8 out_channels, kernel_size 3*3
        ## tanh activation function
        ## 2d valid max_pooling
        self.conv1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=2, out_channels=self.feature_maps, kernel_size=(3, 3))),
            ("batch_norm", nn.BatchNorm2d(self.feature_maps)),
            ("relu1", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)))
        ]))

        ## Dense: fully connected layer
        in_features = op.calculate_dim_with_initialDim_conv((self.max_sentence_len, self.max_sentence_len),
                                                               self.conv1)
        self.dense = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features=in_features, out_features=self.dense_out_dim)),
            # ("batch_norm", nn.BatchNorm1d(50)),
            ("tanh1", nn.Tanh())
        ]))

        ## Final GRU: time major
        self.final_gru = nn.GRU(self.dense_out_dim, self.rnn_units)
        ## SMN Last: Linear Transformation
        self.smn_last_linear = nn.Linear(self.rnn_units, 2)

        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

        # TODO: check the initialization
        self.orthogonal = nn.init.orthogonal_
        self.xavier_normal = nn.init.xavier_normal_
        ## by kaiming he for relu
        self.kaiming_normal = nn.init.kaiming_normal_
        # self.init_parameters()
        # self.init_parameters_from_tf()

    def init_parameters(self):
        # TODO: test the performance with initialization or without initialization
        ## use orthogonal initializer to init kernel of sentence gru and final gru
        ## TODO: not exact initialization
        for parameter in self.sentence_gru.parameters():
            if len(parameter.shape) == 2:
                self.orthogonal(parameter)

        for parameter in self.final_gru.parameters():
            if len(parameter.shape) == 2:
                self.orthogonal(parameter)

        ## use xavier_normal initializer to init a_matrix, fully connected layer, and smn last linear layer
        self.xavier_normal(self.a_matrix.weight)
        self.xavier_normal(self.dense[0].weight)
        self.xavier_normal(self.smn_last_linear.weight)

        ## use kaiming_normal initializer to init kernel of convolution layer
        for i, j in itertools.product(range(self.conv1[0].out_channels), range(self.conv1[0].in_channels)):
            self.kaiming_normal(self.conv1[0].weight[i][j], nonlinearity="relu")

    def init_parameters_from_tf(self):
        weights_path = "/users3/dcteng/work/Dialogue/MultiTurnResponseSelection-master/tensorflow_src/model/tf_model-1.pkl"
        with open(weights_path, 'rb') as f:
            name2value = pickle.load(f)
        name2value["A_matrix_v"] = torch.tensor(name2value["A_matrix_v"].transpose(1, 0))
        name2value["conv/kernel"] = torch.tensor(name2value["conv/kernel"].transpose(3, 2, 0, 1))
        name2value["conv/bias"] = torch.tensor(name2value["conv/bias"])
        name2value["matching_v/kernel"] = torch.tensor(name2value["matching_v/kernel"].transpose(1, 0))
        name2value["matching_v/bias"] = torch.tensor(name2value["matching_v/bias"])
        name2value["final_v/kernel"] = torch.tensor(name2value["final_v/kernel"].transpose(1, 0))
        name2value["final_v/bias"] = torch.tensor(name2value["final_v/bias"])

        assert self.a_matrix.weight.shape == name2value["A_matrix_v"].shape, "{}\t{}".format(self.a_matrix.weight.shape, name2value["A_matrix_v"].shape)
        assert self.conv1[0].weight.shape == name2value["conv/kernel"].shape, "{}\t{}".format(self.conv1[0].weight.shape, name2value["conv/kernel"].shape)
        assert self.conv1[0].bias.shape == name2value["conv/bias"].shape, "{}\t{}".format(self.conv1[0].bias.shape, name2value["conv/bias"].shape)
        assert self.dense[0].weight.shape == name2value["matching_v/kernel"].shape, "{}\t{}".format(self.dense[0].weight.shape, name2value["matching_v/kernel"].shape)
        assert self.dense[0].bias.shape == name2value["matching_v/bias"].shape, "{}\t{}".format(self.dense[0].bias.shape, name2value["matching_v/bias"].shape)
        assert self.smn_last_linear.weight.shape == name2value["final_v/kernel"].shape, "{}\t{}".format(self.smn_last_linear.weight.shape, name2value["final_v/kernel"].shape)
        assert self.smn_last_linear.bias.shape == name2value["final_v/bias"].shape, "{}\t{}".format(self.smn_last_linear.bias.shape, name2value["final_v/bias"].shape)

        self.a_matrix.weight.data = name2value["A_matrix_v"]
        self.conv1[0].weight.data = name2value["conv/kernel"]
        self.conv1[0].bias.data = name2value["conv/bias"]
        self.dense[0].weight.data = name2value["matching_v/kernel"]
        self.dense[0].bias.data = name2value["matching_v/bias"]
        self.smn_last_linear.weight.data = name2value["final_v/kernel"]
        self.smn_last_linear.bias.data = name2value["final_v/bias"]

    def init_hidden(self, batch_shape, hidden=None):
        weight = next(self.parameters()).data
        if not isinstance(batch_shape, Iterable):
            batch_shape = [batch_shape]
        bsz = reduce(lambda x, y: x * y, batch_shape, 1)
        if hidden is not None and hidden.shape[1] == bsz:
            return utils.repackage_hidden(hidden)
        else:
            return weight.new(1, bsz, self.rnn_units).zero_()

    def forward(self, inputs):
        # First Layer --- utterances-Response Matching

        ## embeddings
        all_utt_embeds = self.embeddings(inputs["utt"])
        # varname(all_utt_embeds)  # torch.Size([None, 10, 50, 200])
        response_embeds = self.embeddings(inputs["resp"])
        # varname(response_embeds)  # torch.Size([None, 50, 200])
        ## len
        all_utt_lens = inputs["utt_len"]
        # varname(all_utt_lens)  # torch.Size([None, 10])
        resp_lens = inputs["resp_len"]
        # varname(resp_lens)  # torch.Size([None])

        ## push responses into sentence gru
        self.hidden1 = self.init_hidden(response_embeds.shape[:-2], self.hidden1)
        resp_output, resp_ht = op.pack_and_pad_sequences_for_rnn(response_embeds, resp_lens, self.sentence_gru, self.hidden1)
        # resp_output, resp_ht = self.sentence_gru(response_embeds)
        # varname(resp_output)  # torch.Size([None, 50, 200])
        # varname(resp_ht)  # torch.Size([1, None, 200])

        ## calculate matrix1
        ## [10, None, 50, 200] * [None, 200, 50] -> [10, None, 50, 50]
        matrix1 = torch.matmul(torch.transpose(all_utt_embeds, 0, 1), torch.transpose(response_embeds, 1, 2))
        # varname(matrix1)  # torch.Size([10, None, 50, 50])

        ## push utterances into sentence gru
        self.hidden2 = self.init_hidden(all_utt_embeds.shape[:-2], self.hidden2)
        utt_output, _ = op.pack_and_pad_sequences_for_rnn(all_utt_embeds, all_utt_lens, self.sentence_gru, self.hidden2)
        #utt_output, _ = self.sentence_gru(all_utt_embeds.view(tuple([-1]) +  all_utt_embeds.shape[-2:]))
        # utt_output = utt_output.view(all_utt_embeds.shape[:-2] + utt_output.shape[-2:])
        # varname(utt_output)  # torch.Size([None, 10, 50, 200])
        # varname(utt_ht)  # torch.Size([1, None, 10, 200])

        ## TODO: calculate matrix2: check this
        ## [None, 10, 50, 200] * [200, 200] -> [None, 10, 50, 200]
        ## [10, None, 50, 200] * [None, 200, 50] -> [10, None, 50, 50]
        matrix2 = self.a_matrix(utt_output)
        # matrix2 = torch.einsum('abij,jk->abik', (utt_output, self.a_matrix))
        # varname(matrix2)  # torch.Size([None, 10, 50, 200])
        matrix2 = torch.matmul(torch.transpose(matrix2, 0, 1), torch.transpose(resp_output, 1, 2))
        # varname(matrix2)  # torch.Size([10, None, 50, 50])

        ## convolute two matrixes
        ## in_channels: [10, None, 2, 50, 50]
        ## out_channels: [10, None, 8, 16, 16]
        conv_output = op.stack_channels_for_conv2d((matrix1, matrix2), self.conv1)
        # conv_outputs = []
        # for mat1, mat2 in zip(matrix1, matrix2):
        #     conv_outputs.append(op.stack_channels_for_conv2d((mat1, mat2), self.conv1))
        # conv_output = torch.stack(conv_outputs, dim=0)
        conv_output = conv_output.transpose(2,4).transpose(2,3).contiguous()
        # varname(conv_output)  # torch.Size([10, None, 8, 16, 16])

        # Second Layer --- Matching Accumulation

        ## flat conv_output
        flatten_input = conv_output.view(conv_output.shape[:-3] + tuple([-1]))
        # varname(flatten_input)  # torch.Size([10, None, 2048])
        ## dense layer
        dense_output = self.dense(flatten_input)
        # dense_outputs = []
        # for fi in flatten_input:
        #     dense_outputs.append(self.dense(fi))
        # dense_output = torch.stack(dense_outputs, dim=0)
        # varname(dense_output)  # torch.Size([10, None, 50])
        ## push dense_output into final gru
        ## TODO: Why not mask the empty utterances ?
        self.hidden3 = self.init_hidden(dense_output.shape[-2], self.hidden3)
        final_output, last_hidden = self.final_gru(dense_output, self.hidden3)
        # varname(final_output)  # torch.Size([10, None, 50])
        # varname(last_hidden)  # torch.Size([1, None, 50])
        logits = self.smn_last_linear(last_hidden)
        # varname(logits)  # torch.Size([1, None, 2])
        return torch.squeeze(logits, dim=0)


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
