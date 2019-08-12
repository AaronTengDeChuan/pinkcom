# coding: utf-8
from utils import utils
from utils.utils import varname

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import layers
import layers.operations as op

from collections import OrderedDict
import itertools
import os
import sys
import pickle

from losses.loss import CrossEntropyLoss
from optimizers.optimizer import AdamOptimizer

logger = utils.get_logger()


def tensor_hook(grad):
    print('grad:', grad)
    input("\nnext hook:")

def tensor_info(tensor):
    print (tensor)
    tensor.register_hook(tensor_hook)
    input("\nnext tensor:")


def output_result(predictions, params):
    scores = None
    labels = predictions[2]["target"]
    if isinstance(predictions[0][0], list):
        scores = [s[1] for s in predictions[0]]
    else:
        scores = predictions[0]

    with open(params["file_name"], 'w', encoding="utf-8") as f:
        for score, label in zip(scores, labels):
            f.write("{}\t{}\n".format(score, label))


class DAMModel(nn.Module):
    """DAM Module contains ."""

    def __init__(self, config):
        super(DAMModel, self).__init__()
        # hyperparameters
        embeddings = config["embeddings"] if "embeddings" in config else None
        self.vocabulary_size = config["vocabulary_size"] if "vocabulary_size" in config else 434511
        self.word_embedding_size = config["embedding_dim"] if "embedding_dim" in config else 200
        self.max_num_utterance = config["max_num_utterance"] if "max_num_utterance" in config else 10
        self.max_sentence_len = config["max_sentence_len"] if "max_sentence_len" in config else 50

        self.is_positional = config["is_positional"] if "is_positional" in config else False
        self.stack_num = config["stack_num"] if "stack_num" in config else 5

        self.head_num = config["head_num"] if "head_num" in config else 0

        self.is_layer_norm = config["is_layer_norm"] if "is_layer_norm" in config else True
        self.drop_prob = config["drop_prob"] if "drop_prob" in config else None

        self.attention_type = config["attention_type"] if "attention_type" in config else "dot"
        self.is_mask = config["is_mask"] if "is_mask" in config else True

        self.final_out_features = config["final_out_features"] if "final_out_features" in config else 2

        self.embeddings_trainable = config["emb_trainable"] if "emb_trainable" in config else True
        self.device = config["device"]

        # build model
        ## Embedding
        self.embeddings = op.init_embedding(self.vocabulary_size, self.word_embedding_size, embeddings=embeddings,
                                            embeddings_trainable=self.embeddings_trainable)

        self.position_encoder = layers.PositionEncoder({"lambda_size": self.max_sentence_len, "max_timescale": 10})

        self.self_blocks = nn.ModuleList()
        for index in range(self.stack_num):
            if self.head_num <= 0:
                self.self_blocks.append(layers.AttentiveModule(
                    {"name": "self_block_{}".format(index), "x_dim": self.word_embedding_size,
                     "y_dim": self.word_embedding_size, "is_layer_norm": self.is_layer_norm,
                     "drop_prob": self.drop_prob}))
            else:
                self.self_blocks.append(layers.MultiHeadedAttentiveModule(
                    {"name": "self_block_{}".format(index), "x_dim": self.word_embedding_size,
                     "y_dim": self.word_embedding_size, "head_num": self.head_num, "is_layer_norm": self.is_layer_norm,
                     "drop_prob": self.drop_prob}))

        self.t_a_r_blocks = nn.ModuleList()
        for index in range(self.stack_num + 1):
            if self.head_num <= 0:
                self.t_a_r_blocks.append(layers.AttentiveModule(
                    {"name": "t_a_r_block_{}".format(index), "x_dim": self.word_embedding_size,
                     "y_dim": self.word_embedding_size, "is_layer_norm": self.is_layer_norm,
                     "drop_prob": self.drop_prob}))
            else:
                self.t_a_r_blocks.append(layers.MultiHeadedAttentiveModule(
                    {"name": "t_a_r_block_{}".format(index), "x_dim": self.word_embedding_size,
                     "y_dim": self.word_embedding_size, "head_num": self.head_num, "is_layer_norm": self.is_layer_norm,
                     "drop_prob": self.drop_prob}))

        self.r_a_t_blocks = nn.ModuleList()
        for index in range(self.stack_num + 1):
            if self.head_num <= 0:
                self.r_a_t_blocks.append(layers.AttentiveModule(
                    {"name": "r_a_t_block_{}".format(index), "x_dim": self.word_embedding_size,
                     "y_dim": self.word_embedding_size, "is_layer_norm": self.is_layer_norm,
                     "drop_prob": self.drop_prob}))
            else:
                self.r_a_t_blocks.append(layers.MultiHeadedAttentiveModule(
                    {"name": "r_a_t_block_{}".format(index), "x_dim": self.word_embedding_size,
                     "y_dim": self.word_embedding_size, "head_num": self.head_num, "is_layer_norm": self.is_layer_norm,
                     "drop_prob": self.drop_prob}))

        self.creat_conv()

        in_features = op.calculate_dim_with_initialDim_conv(
            (self.max_num_utterance, self.max_sentence_len, self.max_sentence_len), self.conv)

        self.final = nn.Linear(in_features=in_features, out_features=self.final_out_features)

        # self._reset_parameters()

    def creat_conv(self):
        # calculate padding
        input_shape = (self.max_num_utterance, self.max_sentence_len, self.max_sentence_len)
        conv1_padding, output_shape = op.calculate_padding_for_cnn(input_shape, (3, 3, 3), (1, 1, 1))
        maxpool1_padding, output_shape = op.calculate_padding_for_cnn(output_shape, (3, 3, 3), (3, 3, 3))
        conv2_padding, output_shape = op.calculate_padding_for_cnn(output_shape, (3, 3, 3), (1, 1, 1))
        maxpool2_padding, output_shape = op.calculate_padding_for_cnn(output_shape, (3, 3, 3), (3, 3, 3))
        # creat conv
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * (self.stack_num + 1), out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=conv1_padding),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=maxpool1_padding),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=conv2_padding),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=maxpool2_padding),
        )

    def _reset_parameters(self):
        # CNN
        stdv = 0.01
        self.conv[0].weight.data.uniform_(-stdv, stdv)
        self.conv[0].bias.data.fill_(0)
        self.conv[3].weight.data.uniform_(-stdv, stdv)
        self.conv[3].bias.data.fill_(0)

    def forward(self, inputs):
        device = inputs["target"].device
        dtype = torch.get_default_dtype()
        # response part
        Hr = self.embeddings(inputs["resp"]) # [batch, max_sentence_len, emb_size]
        response_len = inputs["resp_len"]

        if self.is_positional and self.stack_num > 0:
            Hr = self.position_encoder(Hr)  # [batch, max_sentence_len, emb_size]
        Hr_stack = [Hr]

        for index in range(self.stack_num):
            Hr = self.self_blocks[index](Hr, Hr, Hr, Q_lengths=response_len, K_lengths=response_len,
                                         attention_type=self.attention_type, is_mask=self.is_mask)
            Hr_stack.append(Hr)

        # context part
        bHu = self.embeddings(inputs["utt"])    # [batch, max_num_utterance, max_sentence_len, emb_size]
        utterance_len = inputs["utt_len"]   # [batch, max_num_utterance]

        shape_save = bHu.shape[:-2]

        bHu = bHu.view(-1, *bHu.shape[-2:]) # [batch * max_num_utterance, max_sentence_len, emb_size]
        b_utterance_len = utterance_len.view(-1)   # [batch * max_num_utterance]

        if self.is_positional and self.stack_num > 0:
            bHu = self.position_encoder(bHu)    # [batch * max_num_utterance, max_sentence_len, emb_size]
        bHu_stack = [bHu.view(*shape_save, *bHu.shape[-2:])]    # [batch, max_num_utterance, max_sentence_len, emb_size]

        for index in range(self.stack_num):
            bHu = self.self_blocks[index](bHu, bHu, bHu, Q_lengths=b_utterance_len, K_lengths=b_utterance_len,
                                         attention_type=self.attention_type, is_mask=self.is_mask)  # [batch * max_num_utterance, max_sentence_len, emb_size]
            bHu_stack.append(bHu.view(*shape_save, *bHu.shape[-2:]))    # [batch, max_num_utterance, max_sentence_len, emb_size]

        b_t_a_r_stack = []
        b_r_a_t_stack = []

        for index in range(self.stack_num + 1):
            b_t_a_r = []
            b_r_a_t = []
            for Hu, utt_len in zip(bHu_stack[index].transpose(0, 1), utterance_len.transpose(0, 1)):
                t_a_r = self.t_a_r_blocks[index](Hu, Hr_stack[index], Hr_stack[index], Q_lengths=utt_len,
                                                 K_lengths=response_len, attention_type=self.attention_type,
                                                 is_mask=self.is_mask)  # [batch, max_sentence_len, emb_size]
                r_a_t = self.r_a_t_blocks[index](Hr_stack[index], Hu, Hu, Q_lengths=response_len,
                                                 K_lengths=utt_len, attention_type=self.attention_type,
                                                 is_mask=self.is_mask)  # [batch, max_sentence_len, emb_size]
                b_t_a_r.append(t_a_r)
                b_r_a_t.append(r_a_t)
            b_t_a_r = torch.stack(b_t_a_r, dim=1)   # [batch, max_num_utterance, max_sentence_len, emb_size]
            b_r_a_t = torch.stack(b_r_a_t, dim=1)   # [batch, max_num_utterance, max_sentence_len, emb_size]

            b_t_a_r_stack.append(b_t_a_r)
            b_r_a_t_stack.append(b_r_a_t)

        bHu = torch.stack(bHu_stack, dim=1) # [batch, stack_num+1, max_num_utterance, max_sentence_len, emb_size]
        Hr = torch.stack(Hr_stack, dim=1)    # [batch, stack_num+1, max_sentence_len, emb_size]
        sim_1 = torch.einsum("bsaik,bsjk->bsaij", (bHu, Hr))    # [batch, stack_num+1, max_num_utterance, max_sentence_len, max_sentence_len]

        b_t_a_r = torch.stack(b_t_a_r_stack, dim=1) # [batch, stack_num+1, max_num_utterance, max_sentence_len, emb_size]
        b_r_a_t = torch.stack(b_r_a_t_stack, dim=1) # [batch, stack_num+1, max_num_utterance, max_sentence_len, emb_size]
        sim_2 = torch.einsum("bsaik,bsajk->bsaij", (b_t_a_r, b_r_a_t))  # [batch, stack_num+1, max_num_utterance, max_sentence_len, max_sentence_len]
        # sim shape [batch, 2*(stack_num+1), max_num_utterance, max_sentence_len, max_sentence_len]
        sim = torch.cat((sim_2, sim_1), dim=1) / torch.sqrt(torch.tensor([200], dtype=dtype, device=device))

        final_info = self.conv(sim) # final_info shape [batch, 16, 4, 4, 4]

        logits = self.final(final_info.view(final_info.shape[0], -1)) # logits shape [batch, 1]

        # utils.varname(logits, fn=tensor_info)

        return logits.squeeze(-1)

if __name__ == "__main__":
    dam = DAMModel({"device": torch.device("cpu")})

    # for k, v in dam.named_parameters():
    #     logger.info("{}".format(k))

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
    optimizer = AdamOptimizer(dam.parameters(), lr=0.001)

    for i in range(100):
        dam.train()
        optimizer.zero_grad()
        logits = dam(inputs)
        loss, num_labels, batch_total_loss = loss_fn(logits, inputs["target"])
        loss.backward()
        optimizer.step()

        print ("epoch: {}".format(i + 1))
        print(logits)
        print(torch.nn.functional.softmax(logits, dim=-1))
        print(loss.item(), end="\n\n")
