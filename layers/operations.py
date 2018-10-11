# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def bilinear_sim(x, y, weight, is_nor=True):
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
    assert x.shape[-1] == weight.shape[0] and weight.shape[1] == y.shape[-1]

    # TODO: check this
    sim = torch.einsum('bik,kl,bjl->bij', (x, weight, y))

    device = sim.device

    if is_nor:
        scale = torch.sqrt(torch.Tensor([ x.shape[-1] * y.shape[-1] ], device=device))
        scale = torch.max(scale, torch.Tensor([1], device=device))
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

    if is_nor:
        scale = torch.sqrt(torch.Tensor([x.shape[-1]], device=device))
        scale = torch.max(scale, torch.Tensor([1], device=device))
        return sim / scale
    else:
        return sim


def dense(x, weight, bias=None):
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
    assert x.shape[-1] == weight.shape[0]

    # TODO: check this
    if bias is not None:
        return torch.einsum('bik,kj->bij', (x, weight)) + bias
    else:
        return torch.einsum('bik,kj->bij', (x, weight))


def calculate_dim_with_initialDim_conv(initial_dim, conv):
    '''
    calculate input dim of first fully connected layer
    :param initial_dim:
    :param conv: nn.Sequential containing conv, activation function, and pooling
    :return: input dim of first fully connected layer
    '''
    batch_size = 2
    inputs = torch.randn((batch_size, conv[0].in_channels, *initial_dim), dtype=torch.float32)
    outputs = conv(inputs).view(batch_size, -1)
    return outputs.shape[1]


def sequence_mask(lengths, max_length):
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

    row_mask = row_mask.unsqueeze(dim=-1).to(dtype=torch.float32)
    col_mask = col_mask.unsqueeze(dim=-1).to(dtype=torch.float32)

    # TODO: check this
    return torch.einsum('bik,bjk->bij', (row_mask, col_mask))


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
            Args:
                x: a tensor
                axis: the dims to normalize, [d1_index, d2_index, ...]
                    required: reverse order [..., d2_index, d1_index]
    '''
    if axis is None:
        axis = [-1]

    num_elements = 1
    for i in axis:
        num_elements *= x.shape[i]

    return torch.sum(x, dim=axis, keepdim=keepdim) / num_elements