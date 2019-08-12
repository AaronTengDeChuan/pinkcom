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


class Attention(nn.Module):

    '''
        Add attention layer.
            Args:
                Q: a tensor with shape [batch, *, Q_time, Q_dimension]
                K: a tensor with shape [batch, *, time, K_dimension]
                V: a tensor with shape [batch, *, time, V_dimension]

                Q_length: a tensor with shape [batch]
                K_length: a tensor with shape [batch]

            Returns:
                a tensor with shape [batch, *, Q_time, V_dimension]

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

        Q_time = Q.shape[-2]
        K_time = K.shape[-2]

        if attention_type == 'dot':
            logits = op.dot_sim(Q, K)  # [batch, *, Q_time, K_time]
        if attention_type == 'bilinear':
            logits = op.bilinear_sim(Q, K, linear_module=self.bilinear_matrix)

        if is_mask:
            mask = op.mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
            if len(logits.shape) == 4:
                mask = mask.unsqueeze(1)
            logits = mask * logits + (1 - mask) * mask_value

        attention = F.softmax(logits, dim=-1)

        if self.drop_prob is not None:
            attention = self.dropout(attention)

        return op.weighted_sum(attention, V)


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
            self.attention_layer_norm = nn.LayerNorm([config["y_dim"]], eps=1e-6)
            # self.attention_layer_norm = LayerNorm(
            #     {"name": "Attention_layer_norm", "parameter_shape": [config["x_dim"]], "axis": [-1]})

            # FFN layer norm
            self.ffn_layer_norm = nn.LayerNorm([config["y_dim"]], eps=1e-6)
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


class MultiHeadedAttentiveModule(nn.Module):

    '''
        This method has the same behavior as pure AttentiveModule.
    '''

    def __init__(self, config):
        super(MultiHeadedAttentiveModule, self).__init__()
        assert "x_dim" in config and "y_dim" in config and "head_num" in config
        # Attention layer
        attention_config = deepcopy(config)
        attention_config["name"] = "Attention"
        self.attention = Attention(attention_config)

        self.input_dim = config["x_dim"]
        self.output_dim = config["y_dim"]
        self.head_num = config["head_num"]
        assert self.input_dim % self.head_num == 0
        self.sub_input_dim = self.input_dim // self.head_num

        self.input_linears = utils.clones(nn.Linear(self.input_dim, self.output_dim), 3)
        self.output_linear = nn.Linear(self.output_dim, self.output_dim)

        self.is_layer_norm = config["is_layer_norm"] if "is_layer_norm" in config else True
        if self.is_layer_norm:
            # Attention layer norm
            self.attention_layer_norm = nn.LayerNorm([self.output_dim], eps=1e-6)
            # FFN layer norm
            self.ffn_layer_norm = nn.LayerNorm([self.output_dim], eps=1e-6)

        self.ffn = FFN({"name": "FFN", "input_dim": self.output_dim, "out_dim_0": self.output_dim,
                        "out_dim_1": self.output_dim})

        self.name = config["name"] if "name" in config else "MultiHeadAttentiveModule"
        logger.info(
            utils.generate_module_info(self.name, "head_num", self.head_num, "input_dim", self.input_dim, "output_dim",
                                       self.output_dim, "is_layer_norm", self.is_layer_norm))

    def forward(self, Q, K, V, Q_lengths, K_lengths, attention_type="dot", is_mask=True, mask_value=-2 ** 32 + 1):
        nbatches = Q.size(0)
        # 1) Do all the linear projections in batch from input_dim => head_num * sub_input_dim
        q, k, v = [l(x).view(nbatches, -1, self.head_num, self.sub_input_dim).transpose(1, 2) for l, x in
                   zip(self.input_linears, (Q, K, V))]
        # 2) Apply attention on all the projected vectors in batch
        att = self.attention(q, k, v, Q_lengths, K_lengths, attention_type=attention_type, is_mask=is_mask,
                             mask_value=mask_value)  # [batch, *, Q_time, V_dimension]
        # 3) "Concat" using a view and apply a final linear
        att = self.output_linear(
            att.transpose(1, 2).contiguous().view(nbatches, -1, self.head_num * self.sub_input_dim))
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


# TODO: Below are layers for FlowQA Model

def set_seq_dropout(option): # option = True or False
    global do_seq_dropout
    do_seq_dropout = option


def set_my_dropout_prob(p): # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p


def seq_dropout(x, p=0.0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = 1.0 / (1-p) * torch.bernoulli((1-p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0.0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3: # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    elif p == 0:
        return x
    else:
        return F.dropout(x, p=p, training=training, inplace=True)


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, concat_layers=False, do_residual=False,
                 add_feat=0, dialog_flow=False, bidir=True):
        super(StackedBRNN, self).__init__()
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.do_residual = do_residual
        self.dialog_flow = dialog_flow
        self.hidden_size = hidden_size

        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else (2 * hidden_size + add_feat if i == 1 else 2 * hidden_size)
            if self.dialog_flow == True:
                input_size += 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidir))

    def forward(self, x, x_len=None, x_mask=None, return_list=False, additional_x=None, previous_hiddens=None):
        '''
        :param x: a tensor with shape of [batch_size, seq_len, input_size]
        :param x_len: a tensor with shape of [batch_size]
        :param x_mask: a tensor with shape of [batch_size, seq_len]
        :param return_list: return a list for layers of hidden vectors
        :param additional_x: corresponds to the parameter "add_feat"
        :param previous_hiddens: corresponds to the parameter "dialog_flow"
        :return:
        '''
        if x_len is not None and x_mask is not None:
            assert torch.equal(x_len, x.size(1) - torch.sum(x_mask, dim=1))

        if additional_x is not None:
            additional_x = additional_x.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            if i == 1 and additional_x is not None:
                rnn_input = torch.cat((rnn_input, additional_x), 2)
            # Apply dropout to input
            if my_dropout_p > 0:
                rnn_input = dropout(rnn_input, p=my_dropout_p, training=self.training)
            if self.dialog_flow == True:
                if previous_hiddens is not None:
                    dialog_memory = previous_hiddens[i-1]
                else:
                    dialog_memory = rnn_input.new_zeros((rnn_input.size(0), rnn_input.size(1), self.hidden_size * 2))
                rnn_input = torch.cat((rnn_input, dialog_memory), 2)
            # Forward
            if x_len is not None:
                rnn_output = op.pack_and_pad_sequences_for_rnn(rnn_input, x_len, self.rnns[i])[0]
            else:
                rnn_output = self.rnns[i](rnn_input)[0]
            if self.do_residual and i > 0:
                rnn_output = rnn_output + hiddens[-1]
            hiddens.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output


# Attention layers
class AttentionScore(nn.Module):
    """
    Use S(x, y) = Relu(Ux) D Relu(Uy) to compute the attention score between x, y,
        where U, D are trainable parameters and D is a diagonal matrix.
    Especially, when elements in D's diagonal are same, the resulting attention score is similarity score.
    """
    def __init__(self, input_size, attention_hidden_size, similarity_score = False):
        super(AttentionScore, self).__init__()
        self.linear = nn.Linear(input_size, attention_hidden_size, bias=False)

        if similarity_score:
            self.linear_final = nn.Parameter(torch.ones(1, 1, 1) / (attention_hidden_size ** 0.5), requires_grad = False)
        else:
            self.linear_final = nn.Parameter(torch.ones(1, 1, attention_hidden_size), requires_grad = True)

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        scores: batch * len1 * len2 <the scores are not masked>
        """
        x1 = dropout(x1, p=my_dropout_p, training=self.training)
        x2 = dropout(x2, p=my_dropout_p, training=self.training)

        x1_rep = self.linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), x1.size(1), -1)
        x2_rep = self.linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), x2.size(1), -1)

        x1_rep = F.relu(x1_rep)
        x2_rep = F.relu(x2_rep)
        final_v = self.linear_final.expand_as(x2_rep)

        x2_rep_v = final_v * x2_rep
        scores = x1_rep.bmm(x2_rep_v.transpose(1, 2))
        return scores


class GetAttentionHiddens(nn.Module):
    def __init__(self, input_size, attention_hidden_size, similarity_attention = False):
        super(GetAttentionHiddens, self).__init__()
        self.scoring = AttentionScore(input_size, attention_hidden_size, similarity_score=similarity_attention)

    def forward(self, x1, x2, x2_mask, x3=None, scores=None, return_scores=False, drop_diagonal=False):
        """
        Using x1, x2 to calculate attention score, but x1 will take back info from x3.
        If x3 is not specified, x1 will attend on x2.

        x1: batch * len1 * x1_input_size
        x2: batch * len2 * x2_input_size
        x2_mask: batch * len2

        x3: batch * len2 * x3_input_size (or None)
        """
        if x3 is None:
            x3 = x2

        if scores is None:
            scores = self.scoring(x1, x2)

        # Mask padding
        x2_mask = x2_mask.unsqueeze(1).expand_as(scores)
        negative_inf = torch.tensor(-float('inf'), device=scores.device).max()
        # negative_inf = torch.tensor(-2 ** 32 + 1.0, device=scores.device)
        scores.data.masked_fill_(x2_mask.data, negative_inf)
        if drop_diagonal:
            assert(scores.size(1) == scores.size(2))
            # +0 for avoiding shared storage
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores) + 0
            scores.data.masked_fill_(diag_mask, negative_inf)

            # address error: full '-inf' elements in dim=2
            inf_bool = 1 - (torch.max(scores, dim=2)[0] == negative_inf)
            # inf_bool = 1 - utils.float_is_equal(torch.max(scores, dim=2)[0], negative_inf)
            # diag_mask.data.masked_fill_(inf_bool.unsqueeze(-1).expand_as(scores), 0)
            # scores.data.masked_fill_(diag_mask, 0)
            scores.data.masked_fill_((1 - inf_bool).unsqueeze(-1).expand_as(scores), 0)

        # Normalize with softmax
        num_negative_inf = torch.sum(torch.max(scores, dim=2)[0] == negative_inf).item()
        # num_negative_inf = torch.sum(utils.float_is_equal(torch.max(scores, dim=2)[0], negative_inf)).item()
        if num_negative_inf > 0:
            # print (drop_diagonal, end=' ', flush=True)
            # print (x1.shape)
            # print (x2.shape)
            # print (len(x3) if x3 is not None else None)
            print (num_negative_inf, end=' ', flush=True)
        alpha = F.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(x3)
        if return_scores:
            return matched_seq, scores
        else:
            return matched_seq


class DeepAttention(nn.Module):
    def __init__(self, opt, abstr_list_cnt, deep_att_hidden_size_per_abstr, do_similarity=False, word_hidden_size=None,
                 do_self_attn=False, dialog_flow=False, no_rnn=False):
        super(DeepAttention, self).__init__()

        self.no_rnn = no_rnn

        word_hidden_size = opt['embedding_dim'] if word_hidden_size is None else word_hidden_size
        abstr_hidden_size = opt['hidden_size'] * 2

        att_size = abstr_hidden_size * abstr_list_cnt + word_hidden_size

        self.int_attn_list = nn.ModuleList()
        for i in range(abstr_list_cnt+1):
            self.int_attn_list.append(
                GetAttentionHiddens(att_size, deep_att_hidden_size_per_abstr, similarity_attention=do_similarity))

        rnn_input_size = abstr_hidden_size * abstr_list_cnt * 2 + (opt['hidden_size'] * 2)

        self.att_final_size = rnn_input_size
        if not self.no_rnn:
            self.rnn = StackedBRNN(rnn_input_size, opt['hidden_size'], num_layers=1, dialog_flow=dialog_flow)
            self.output_size = opt['hidden_size'] * 2
        #print('Deep attention x {}: Each with {} rays in {}-dim space'.format(abstr_list_cnt, deep_att_hidden_size_per_abstr, att_size))
        #print('Deep attention RNN input {} -> output {}'.format(self.rnn_input_size, self.output_size))

        self.opt = opt
        self.do_self_attn = do_self_attn

    def forward(self, x1_word, x1_abstr, x2_word, x2_abstr, x1_mask, x2_mask, return_bef_rnn=False, previous_hiddens=None):
        """
        x1_word, x2_word, x1_abstr, x2_abstr are list of 3D tensors.
        3D tensor: batch_size * length * hidden_size
        """
        # the last tensor of x2_abstr is an addtional tensor
        x1_att = torch.cat(x1_word + x1_abstr, 2)
        x2_att = torch.cat(x2_word + x2_abstr[:-1], 2)
        x1 = torch.cat(x1_abstr, 2)

        x2_list = x2_abstr
        for i in range(len(x2_list)):
            attn_hiddens = self.int_attn_list[i](x1_att, x2_att, x2_mask, x3=x2_list[i], drop_diagonal=self.do_self_attn)
            x1 = torch.cat((x1, attn_hiddens), 2)

        if not self.no_rnn:
            x1_hiddens = self.rnn(x1, x1_mask, previous_hiddens=previous_hiddens)
            if return_bef_rnn:
                return x1_hiddens, x1
            else:
                return x1_hiddens
        else:
            return x1


# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class BilinearLayer(nn.Module):
    def __init__(self, x_size, y_size, class_num):
        super(BilinearLayer, self).__init__()
        self.linear = nn.Linear(y_size, x_size * class_num)
        self.class_num = class_num

    def forward(self, x, y):
        """
        x = batch * h1
        y = batch * h2
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        y = dropout(y, p=my_dropout_p, training=self.training)

        Wy = self.linear(y)
        Wy = Wy.view(Wy.size(0), self.class_num, x.size(1))
        xWy = torch.sum(x.unsqueeze(1).expand_as(Wy) * Wy, dim=2)
        return xWy.squeeze(-1) # size = batch * class_num