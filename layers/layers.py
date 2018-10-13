# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers.operations as op

import math
from copy import deepcopy
from utils import utils

logger = utils.get_logger()

class PositionEncoder(nn.Module):

    '''
        Adds a bunch of sinusoids of different frequencies to a tensor.
            Args:
                x: a tensor with shape [batch, length, channels]
                min_timescale: a float
                max_timescale: a float

            Returns:
                a tensor the same shape as x.

            Raises:
    '''

    def __init__(self, config):
        super(PositionEncoder, self).__init__()
        assert "lambda_size" in config
        self.lambda_size = config["lambda_size"]
        self.min_timescale = config["min_timescale"] if "min_timescale" in config else 1.0
        self.max_timescale = config["max_timescale"] if "max_timescale" in config else 1.0e4
        self.fill_value = config["fill_value"] if "fill_value" in config else 0

        self._lambda = nn.Parameter(torch.full((self.lambda_size, 1), self.fill_value))

        self.name = config["name"] if "name" in config else "Position Encoder"
        logger.info(
            utils.generate_module_info(self.name, "lambda_size", self.lambda_size, "min_timescale", self.min_timescale,
                                       "max_timescale", self.max_timescale, "fill_value", self.fill_value))

    def forward(self, x):
        assert x.shape[1] == self.lambda_size
        device = x.device
        dtype = torch.get_default_dtype()
        channels = x.shape[2]

        position = torch.arange(0., self.lambda_size, device=device)

        num_timescales = channels // 2
        log_timescale_increment = math.log(float(self.max_timescale) / float(self.min_timescale)) / (
                    torch.tensor([num_timescales], dtype=dtype, device=device) - 1)
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(0., num_timescales, device=device) * -log_timescale_increment)

        scaled_time = torch.unsqueeze(position, dim=1) * torch.unsqueeze(inv_timescales, dim=0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        signal = F.pad(signal, (0, channels % 2), mode='constant', value=0)
        signal = torch.unsqueeze(signal, dim=0)

        return x + self._lambda * signal


class Attention(nn.Module):

    '''
        Add attention layer.
            Args:
                Q: a tensor with shape [batch, Q_time, Q_dimension]
                K: a tensor with shape [batch, time, K_dimension]
                V: a tensor with shape [batch, time, V_dimension]

                Q_length: a tensor with shape [batch]
                K_length: a tensor with shape [batch]

            Returns:
                a tensor with shape [batch, Q_time, V_dimension]

            Raises:
                AssertionError: if
                    Q_dimension not equal to K_dimension when attention type is dot.
    '''

    def __init__(self, config):
        super(Attention, self).__init__()
        assert "x_dim" in config and "y_dim" in config
        self.x_dim = config["x_dim"]
        self.y_dim = config["y_dim"]
        self.drop_prob = config["drop_prob"] if "drop_prob" in config else None
        self.bilinear_matrix = nn.Linear(in_features=self.x_dim, out_features=self.y_dim, bias=False)
        if self.drop_prob is not None:
            self.dropout = nn.Dropout(p=self.drop_prob)

        self.name = config["name"] if "name" in config else "Attention"
        logger.info(
            utils.generate_module_info(self.name, "x_dim", self.x_dim, "y_dim", self.y_dim, "drop_prob",
                                       self.drop_prob))

    def forward(self, Q, K, V, Q_lengths, K_lengths, attention_type="dot", is_mask=True, mask_value=-2 ** 32 + 1):
        assert attention_type in ('dot', 'bilinear')
        if attention_type == 'dot':
            assert Q.shape[-1] == K.shape[-1]
        else:
            assert Q.shape[-1] == self.x_dim and K.shape[-1] == self.y_dim

        Q_time = Q.shape[1]
        K_time = K.shape[1]

        if attention_type == 'dot':
            logits = op.dot_sim(Q, K)  # [batch, Q_time, K_time]
        if attention_type == 'bilinear':
            logits = op.bilinear_sim(Q, K, linear_module=self.bilinear_matrix)

        if is_mask:
            mask = op.mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
            logits = mask * logits + (1 - mask) * mask_value

        attention = F.softmax(logits, dim=-1)

        if self.drop_prob is not None:
            attention = self.dropout(attention)

        return op.weighted_sum(attention, V)

# NOTE: In PyTorch, torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True)
#       has the same behaviour as class LayerNorm(nn.Module)
class LayerNorm(nn.Module):

    '''
    Add layer normalization.
        Args:
            x: a tensor
            parameter_shape: [d1, d2, ...]
            axis: the dimensions to normalize, [d1_index, d2_index, ...]
                required: reverse order [..., d2_index, d1_index]

        Returns:
            a tensor the same shape as x.

        Raises:
    '''

    def __init__(self, config):
        super(LayerNorm, self).__init__()
        assert "parameter_shape" in config and isinstance(config["parameter_shape"], list or tuple)
        self.parameter_shape = config["parameter_shape"]
        assert "axis" in config and isinstance(config["axis"], list or tuple)
        self.axis = config["axis"]
        self.scale = nn.Parameter(torch.ones(self.parameter_shape))
        self.bias = nn.Parameter(torch.zeros(self.parameter_shape))

        self.name = config["name"] if "name" in config else "Layer Norm"
        logger.info(
            utils.generate_module_info(self.name, "parameter_shape", self.parameter_shape, "axis", self.axis))

    def forward(self, x, epsilon=1e-6):
        mean = op.reduce_mean(x, axis=self.axis[::-1], keepdim=True)
        variance = op.reduce_mean(torch.pow(x - mean, 2), axis=self.axis[::-1], keepdim=True)
        norm = (x - mean) * torch.rsqrt(variance + epsilon)
        return self.scale * norm + self.bias


class FFN(nn.Module):
    '''
        A feed-forward network
        Add two dense connected layer, max(0, x*W0+b0)*W1+b1.
            Args:
                x: a tensor with shape [batch, time, dimension]
                out_dimension: a number which is the output dimension

            Returns:
                a tensor with shape [batch, time, out_dimension]

            Raises:
    '''

    def __init__(self, config):
        super(FFN, self).__init__()
        assert "input_dim" in config and "out_dim_0" in config and "out_dim_1" in config
        self.input_dim = config["input_dim"]
        self.out_dim_0 = config["out_dim_0"]
        self.out_dim_1 = config["out_dim_1"]

        self.linear_1 = nn.Linear(in_features=self.input_dim, out_features=self.out_dim_0, bias=False)
        self.bias_1 = nn.Parameter(torch.zeros(1))

        self.relu = nn.ReLU(inplace=True)

        self.linear_2 = nn.Linear(in_features=self.out_dim_0, out_features=self.out_dim_1, bias=False)
        self.bias_2 = nn.Parameter(torch.zeros(1))

        self.name = config["name"] if "name" in config else "FFN"
        logger.info(
            utils.generate_module_info(self.name, "input_dim", self.input_dim, "out_dim_0", self.out_dim_0, "out_dim_1",
                                       self.out_dim_1))

    def forward(self, x):
        y = op.dense(x, self.linear_1, bias=self.bias_1)
        self.relu(y)
        z = op.dense(y, self.linear_2, bias=self.bias_2)
        return z


class AttentiveModule(nn.Module):

    '''
        Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
            Args:
                Q: a tensor with shape [batch, Q_time, Q_dimension]
                K: a tensor with shape [batch, time, K_dimension]
                V: a tensor with shape [batch, time, V_dimension]

                Q_length: a tensor with shape [batch]
                K_length: a tensor with shape [batch]

            Returns:
                a tensor with shape [batch, time, dimension]

            Raises:
    '''

    def __init__(self, config):
        super(AttentiveModule, self).__init__()
        assert "x_dim" in config and "y_dim" in config
        # Attention layer
        attention_config = deepcopy(config)
        attention_config["name"] = "Attention"
        self.attention = Attention(attention_config)

        self.is_layer_norm = config["is_layer_norm"] if "is_layer_norm" in config else True
        if self.is_layer_norm:
            # Attention layer norm
            self.attention_layer_norm = nn.LayerNorm([config["y_dim"]])
            # self.attention_layer_norm = LayerNorm(
            #     {"name": "Attention_layer_norm", "parameter_shape": [config["x_dim"]], "axis": [-1]})

            # FFN layer norm
            self.ffn_layer_norm = nn.LayerNorm([config["y_dim"]])
            # self.ffn_layer_norm = LayerNorm(
            #     {"name": "FFN_layer_norm", "parameter_shape": [config["x_dim"]], "axis": [-1]})

        self.ffn = FFN({"name": "FFN", "input_dim": config["y_dim"], "out_dim_0": config["y_dim"],
                        "out_dim_1": config["y_dim"]})

        self.name = config["name"] if "name" in config else "AttentiveModule"
        logger.info(
            utils.generate_module_info(self.name, "is_layer_norm", self.is_layer_norm))

    def forward(self, Q, K, V, Q_lengths, K_lengths, attention_type="dot", is_mask=True, mask_value=-2 ** 32 + 1):
        att = self.attention(Q, K, V, Q_lengths, K_lengths, attention_type=attention_type, is_mask=is_mask,
                             mask_value=mask_value) # [batch, Q_time, V_dimension]
        if self.is_layer_norm:
            y = self.attention_layer_norm(Q + att)
        else:
            y = Q + att

        z = self.ffn(y)

        if self.is_layer_norm:
            w = self.ffn_layer_norm(y + z)
        else:
            w = y + z

        return w
