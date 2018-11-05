# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np

from utils import utils

logger = utils.get_logger()


# TODO: Related to Embedding

def init_embedding(vocabulary_size, word_embedding_size, embeddings=None, embeddings_trainable=True):
    dtype = torch.get_default_dtype()
    if isinstance(embeddings, np.ndarray):
        # TODO: check whether share the storage
        assert list(embeddings.shape) == [vocabulary_size, word_embedding_size]
        embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=dtype), freeze=not embeddings_trainable)
        logger.info("The embeddings are initialized by pretrained embeddings" + (
            "." if embeddings_trainable else " and frozen."))
    elif isinstance(embeddings, torch.Tensor):
        # TODO: check whether share the storage
        assert dtype == embeddings.dtype \
               and list(embeddings.shape) == [vocabulary_size, word_embedding_size]
        embeddings = nn.Embedding.from_pretrained(embeddings, freeze=not embeddings_trainable)
        logger.info("The embeddings are initialized by pretrained embeddings" + (
            "." if embeddings_trainable else " and frozen."))
    else:
        embeddings = nn.Embedding(vocabulary_size, word_embedding_size)
        embeddings.weight.requires_grad = embeddings_trainable
        logger.info("The embeddings are randomly initialized" + ("." if embeddings_trainable else " and frozen."))
    return embeddings


# TODO: Related to Linear Layer

def dense(x, linear_module, bias=None):
    '''
        Add dense connected layer, Wx + b.
            Args:
                x: a tensor with shape [batch, time, dimension]
                weight: W
                bias: b

            Return:
                a tensor with shape [batch, time, out_dimension]

            Raises:
    '''
    assert x.shape[-1] == linear_module.in_features

    # TODO: check this
    if bias is not None:
        return linear_module(x) + bias
    else:
        return linear_module(x)


# TODO: Related to RNN

def pack_and_pad_sequences_for_rnn(seq_embeds, seq_lens, rnn_module, hidden):
    '''
    similar to dynamic rnn
    supported rnn including GRU, LSTM
    batch_first of rnn_module must be True
    :param seq_embeds:
    :param seq_lens:
    :param rnn_module: rnn module in Pytorch
    :return: rnn_output, rnn_ht
    '''
    sorted_seq_lens, seq_indices = torch.sort(seq_lens.view(-1), dim=0, descending=True)
    sorted_seq_embeds = torch.index_select(seq_embeds.view(-1, seq_embeds.shape[-2], seq_embeds.shape[-1]), dim=0,
                                           index=seq_indices)
    # varname(sorted_seq_embeds) # torch.Size([None, 50, 200])
    sorted_seq_lens = sorted_seq_lens + torch.tensor(sorted_seq_lens == 0, device=sorted_seq_lens.device,
                                                     dtype=torch.int64)

    # Packs a Tensor containing padded sequences of variable length in order to obtain a PackedSequence object
    packed_seq_input = pack_padded_sequence(sorted_seq_embeds, sorted_seq_lens, batch_first=True)
    # varname(packed_seq_input) # torch.Size([478, 200]), torch.Size([50])
    packed_seq_output, seq_ht = rnn_module(packed_seq_input, hidden)
    # varname(packed_seq_output) # torch.Size([478, 200]), torch.Size([50])
    # varname(seq_ht) # torch.Size([1, None, 200])

    # Pads a packed batch of variable length sequences
    # Ensure that the shape of output is not changed: check this
    seq_output, _ = pad_packed_sequence(packed_seq_output, batch_first=True, total_length=seq_embeds.shape[-2])
    # varname(seq_output) # torch.Size([None, 50, 200])

    # restore original order
    _, original_indices = torch.sort(seq_indices, dim=0, descending=False)
    seq_output = torch.index_select(seq_output, dim=0, index=original_indices)
    # varname(seq_output) # torch.Size([None, 50, 200])

    # restore original shape
    seq_output = seq_output.view(seq_embeds.shape[:-2] + seq_output.shape[-2:])
    # varname(seq_output)

    assert seq_output.shape[-2] == seq_embeds.shape[-2]

    if isinstance(seq_ht, torch.Tensor):
        seq_ht = seq_ht.view(seq_ht.shape[:-2] + seq_embeds.shape[:-2] + seq_ht.shape[-1:])
    else:
        seq_ht = tuple([ht.view(ht.shape[:-2] + seq_embeds.shape[:-2] + ht.shape[-1:]) for ht in seq_ht])
    # varname(seq_ht)
    return seq_output, seq_ht


# TODO: Related to CNN

def calculate_padding_for_cnn(input_shape, kernal_size, stride):
    '''
    This method is designed for "SAME" mode in convolution or pooling layer
    Calculate 'padding' which controls the amount of implicit zero-paddings on both sides for 'padding' number of points for each dimension
    :param input_shape:   the shape of n dimensions tensor
    :param kernal_size: the kernal size of convolution or pooling layer
    :param stride:      the stride of convolution or pooling layer
    :return:            padding, output_shape
    '''
    assert len(input_shape) == len(kernal_size) and len(input_shape) == len(stride)
    output_shape = []
    padding = []
    for i in range(len(input_shape)):
        output_shape.append(math.ceil(float(input_shape[i]) / float(stride[i])))
        _padding = max(kernal_size[i] - (input_shape[i] - 1)%stride[i] - 1, 0)
        pad_head = _padding // 2
        pad_tail = _padding - pad_head
        # padding.append((pad_head, pad_tail))
        # padding.append(pad_head)
        padding.append(pad_tail)
    return tuple(padding), tuple(output_shape)


def calculate_dim_with_initialDim_conv(initial_dim, conv):
    '''
        Calculate input dim of first fully connected layer
        :param initial_dim: a list of integers respresenting the shape of last n-2 dimensions of a tensor
        :param conv:        nn.Sequential containing conv, activation function, and pooling
        :return:            input dim of first fully connected layer
    '''
    batch_size = 2
    dtype = torch.get_default_dtype()
    inputs = torch.randn((batch_size, conv[0].in_channels, *initial_dim), dtype=dtype)
    outputs = conv(inputs).view(batch_size, -1)
    return outputs.shape[1]


def stack_channels_for_conv2d(channels, conv2d_module):
    """
    :param channels: tuple or tensor
    :param conv2d_module:
    :return: conv output
    """
    stack_dim = -3
    if isinstance(channels, tuple):
        channels = torch.stack(channels, dim=stack_dim)
    # varname(channels)
    conv_input = channels.view(tuple([-1]) + channels.shape[-3:])
    # varname(conv_input)
    conv_output = conv2d_module(conv_input)
    # varname(conv_output)

    # restore original shape of conv_output
    conv_output = conv_output.view(channels.shape[:-3] + conv_output.shape[-3:])
    # varname(conv_output)

    return conv_output


# TODO: Related to Mask

def sequence_mask(lengths, max_length):
    '''
    Returns a mask tensor representing the first N positions of each cell
    :param lengths:     a tensor with shape [batch]
    :param max_length:  an integer
    :return:            a tensor with shape [batch, max_length] and dtype torch.uint8
    '''
    device = lengths.device
    left = torch.ones(lengths.shape[0], max_length, dtype=torch.int64, device=device) * torch.arange(1, max_length + 1,
                                                                                                     device=device)
    right = lengths.unsqueeze(dim=-1).expand(-1, max_length)
    return left <= right


def mask(row_lengths, col_lengths, max_row_length, max_col_length):
    '''
        Return a mask tensor representing the first N positions of each row and each column.
            Args:
                row_lengths: a tensor with shape [batch]
                col_lengths: a tensor with shape [batch]

            Returns:
                a mask tensor with shape [batch, max_row_length, max_col_length]

            Raises:
    '''
    row_mask = sequence_mask(row_lengths, max_row_length) #bool, [batch, max_row_len]
    col_mask = sequence_mask(col_lengths, max_col_length) #bool, [batch, max_col_len]
    dtype = torch.get_default_dtype()
    row_mask = row_mask.unsqueeze(dim=-1).to(dtype=dtype)
    col_mask = col_mask.unsqueeze(dim=-1).to(dtype=dtype)

    # TODO: check this
    return torch.einsum('bik,bjk->bij', (row_mask, col_mask))


# TODO: Related to Math Operations

def weighted_sum(weight, values):
    '''
        Calcualte the weighted sum.
            Args:
                weight: a tensor with shape [batch, time, dimension]
                values: a tensor with shape [batch, dimension, values_dimension]

            Return:
                a tensor with shape [batch, time, values_dimension]

            Raises:
    '''
    assert weight.shape[-1] == values.shape[-2]

    # TODO: check this
    return torch.einsum('bij,bjk->bik', (weight, values))


def reduce_mean(x, axis=None, keepdim=False):
    '''

    :param x:       a tensor with shape [d1, d2, ...]
    :param axis:    a list of integers representing dims normalized, [index1, index2, ...],
                    required: reverse order [..., index2, index1]
    :param keepdim:
    :return:        a tensor
    '''
    if axis is None:
        axis = [-1]

    num_elements = 1
    for i in axis:
        num_elements *= x.shape[i]

    return torch.sum(x, dim=axis, keepdim=keepdim) / num_elements


def bilinear_sim(x, y, linear_module, is_nor=True):
    '''
        Calculate bilinear similarity with two tensor.
            Args:
                x: a tensor with shape [batch, time_x, dimension_x]
                y: a tensor with shape [batch, time_y, dimension_y]

            Returns:
                a tensor with shape [batch, time_x, time_y]

            Raises:
                ValueError: if
                    the shapes of x and y are not match;
                    bilinear matrix reuse error.
    '''
    assert x.shape[-1] == linear_module.in_features and linear_module.out_features == y.shape[-1]

    # TODO: check this
    sim = torch.einsum('bik,bjk->bij', (linear_module(x), y))

    device = sim.device
    dtype = torch.get_default_dtype()

    if is_nor:
        scale = torch.sqrt(torch.tensor([ x.shape[-1] * y.shape[-1] ], dtype=dtype, device=device))
        scale = torch.max(scale, torch.tensor([1], dtype=dtype, device=device))
        return sim / scale
    else:
        return sim


def dot_sim(x, y, is_nor=True):
    '''
        Calculate dot similarity with two tensor.

            Args:
                x: a tensor with shape [batch, time_x, dimension]
                y: a tensor with shape [batch, time_y, dimension]

            Returns:
                a tensor with shape [batch, time_x, time_y]

            Raises:
                AssertionError: if
                    the shapes of x and y are not match.
    '''
    assert x.shape[-1] == y.shape[-1]

    # TODO: check this
    sim = torch.einsum('bik,bjk->bij', (x, y))

    device = sim.device
    dtype = torch.get_default_dtype()

    if is_nor:
        scale = torch.sqrt(torch.tensor([x.shape[-1]], dtype=dtype, device=device))
        scale = torch.max(scale, torch.tensor([1], dtype=dtype, device=device))
        return sim / scale
    else:
        return sim


if __name__ == "__main__":
    input_shape = (9, 50, 50)
    conv1_padding, output_shape = calculate_padding_for_cnn(input_shape, (3, 3, 3), (1, 1, 1))
    print (conv1_padding, output_shape)
    maxpool1_padding, output_shape = calculate_padding_for_cnn(output_shape, (3, 3, 3), (3, 3, 3))
    print (maxpool1_padding, output_shape)
    conv2_padding, output_shape = calculate_padding_for_cnn(output_shape, (3, 3, 3), (1, 1, 1))
    print(conv2_padding, output_shape)
    maxpool2_padding, output_shape = calculate_padding_for_cnn(output_shape, (3, 3, 3), (3, 3, 3))
    print(maxpool2_padding, output_shape)

    conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * (5 + 1), out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=conv1_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=maxpool1_padding),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=conv2_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=maxpool2_padding),
        )
    print (calculate_dim_with_initialDim_conv(input_shape, conv))