# coding: utf-8
from utils import utils
from utils.utils import varname

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import layers
import layers.operations as op

from functools import reduce
from collections import OrderedDict
import itertools
import os
import sys
import pickle

from losses.loss import CrossEntropyLoss
from optimizers.optimizer import AdamOptimizer

logger = utils.get_logger()

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
            self.self_blocks.append(layers.AttentiveModule(
                {"name": "self_block_{}".format(index), "x_dim": self.word_embedding_size,
                 "y_dim": self.word_embedding_size, "is_layer_norm": self.is_layer_norm, "drop_prob": self.drop_prob}))

        self.t_a_r_blocks = nn.ModuleList()
        for index in range(self.stack_num + 1):
            self.t_a_r_blocks.append(layers.AttentiveModule(
                {"name": "t_a_r_block_{}".format(index), "x_dim": self.word_embedding_size,
                 "y_dim": self.word_embedding_size, "is_layer_norm": self.is_layer_norm, "drop_prob": self.drop_prob}))

        self.r_a_t_blocks = nn.ModuleList()
        for index in range(self.stack_num + 1):
            self.r_a_t_blocks.append(layers.AttentiveModule(
                {"name": "r_a_t_block_{}".format(index), "x_dim": self.word_embedding_size,
                 "y_dim": self.word_embedding_size, "is_layer_norm": self.is_layer_norm, "drop_prob": self.drop_prob}))

        self.creat_conv()

        in_features = op.calculate_dim_with_initialDim_conv(
            (self.max_num_utterance, self.max_sentence_len, self.max_sentence_len), self.conv)

        self.final = nn.Linear(in_features=in_features, out_features=self.final_out_features)

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
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=maxpool1_padding),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=conv2_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=maxpool2_padding),
        )

    def forward(self, inputs):
        device = inputs["target"].device
        dtype = torch.get_default_dtype()
        # response part
        Hr = self.embeddings(inputs["resp"]) # [batch, time, emb_size]
        response_len = inputs["resp_len"]

        if self.is_positional and self.stack_num > 0:
            Hr = self.position_encoder(Hr)
        Hr_stack = [Hr]

        for index in range(self.stack_num):
            Hr = self.self_blocks[index](Hr, Hr, Hr, Q_lengths=response_len, K_lengths=response_len,
                                         attention_type=self.attention_type, is_mask=self.is_mask)
            Hr_stack.append(Hr)

        # context part
        # a list of length max_num_utterance, every element is a tensor with shape [batch, max_sentence_len]
        list_utt = torch.unbind(inputs["utt"], dim=1)
        list_utt_len = torch.unbind(inputs["utt_len"], dim=1)

        matching_vectors = []
        # for every utt calculate matching vector
        for utt, utt_len in zip(list_utt, list_utt_len):
            Hu = self.embeddings(utt) #[batch, max_sentence_len, emb_size]

            if self.is_positional and self.stack_num > 0:
                Hu = self.position_encoder(Hu)
            Hu_stack = [Hu]

            for index in range(self.stack_num):
                Hu = self.self_blocks[index](Hu, Hu, Hu, Q_lengths=utt_len, K_lengths=utt_len,
                                         attention_type=self.attention_type, is_mask=self.is_mask)
                Hu_stack.append(Hu)

            t_a_r_stack = []
            r_a_t_stack = []

            for index in range(self.stack_num + 1):
                t_a_r = self.t_a_r_blocks[index](Hu_stack[index], Hr_stack[index], Hr_stack[index], Q_lengths=utt_len,
                                                 K_lengths=response_len, attention_type=self.attention_type,
                                                 is_mask=self.is_mask)
                r_a_t = self.r_a_t_blocks[index](Hr_stack[index], Hu_stack[index], Hu_stack[index], Q_lengths=response_len,
                                                 K_lengths=utt_len, attention_type=self.attention_type,
                                                 is_mask=self.is_mask)
                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            # t_a_r and r_a_t: shape [batch, 2*(stack_num+1), max_sentence_len, emb_size]
            t_a_r = torch.stack(t_a_r_stack, dim=1)
            r_a_t = torch.stack(r_a_t_stack, dim=1)

            # calculate similarity matrix
            # sim shape [batch, 2*stack_num+1, max_sentence_len, max_sentence_len]
            # divide sqrt(200) to prevent gradient explosion
            sim = torch.einsum('bsik,bsjk->bsij', (t_a_r, r_a_t)) / torch.sqrt(
                torch.tensor([200], dtype=dtype, device=device))
            matching_vectors.append(sim)

        # cnn and aggregation
        # sim shape [batch, 2*stack_num+1, max_num_utterance, max_sentence_len, max_sentence_len]
        sim = torch.stack(matching_vectors, dim=2)

        final_info = self.conv(sim) # final_info shape [batch, 16, 4, 4, 4]

        logits = self.final(final_info.view(final_info.shape[0], -1)) # logits shape [batch, 1]

        return logits.squeeze()


if __name__ == "__main__":
    dam = DAMModel({"device": torch.device("cpu")})

    # for k, v in dam.named_parameters():
    #     logger.info("{}".format(k))
    batch_size = 10
    utt_inputs = torch.randint(0, 434511, (batch_size, 10, 50), dtype=torch.int64)
    # utt_inputs = torch.cat([utt_inputs] * 2, dim=0)
    utt_len_inputs = torch.sum(utt_inputs != 0, dim=-1)
    resp_inputs = torch.randint(0, 434511, (batch_size, 50), dtype=torch.int64)
    resp_len_inputs = torch.sum(resp_inputs != 0, dim=-1)
    targets = torch.randint(0, 2, (batch_size,), dtype=torch.int64)
    inputs = {
        "utt": utt_inputs,
        "utt_len": utt_len_inputs,
        "resp": resp_inputs,
        "resp_len": resp_len_inputs,
        "target": targets
    }
    loss_fn = CrossEntropyLoss({})
    optimizer = AdamOptimizer(dam.parameters(), lr=0.001)

    # for i in range(100):
    #     dam.train()
    #     optimizer.zero_grad()
    #     logits = dam(inputs)
    #     loss, num_labels, batch_total_loss = loss_fn(logits, inputs["target"])
    #     loss.backward()
    #     optimizer.step()
    #
    #     print ("epoch: {}".format(i + 1))
    #     print(logits)
    #     print(torch.nn.functional.softmax(logits, dim=-1))
    #     print(loss.item(), end="\n\n")
