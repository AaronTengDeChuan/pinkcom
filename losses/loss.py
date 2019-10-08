# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import logging
from utils import utils

logger = utils.get_logger()


def gather_statistics(loss, target, reduction):
    # count the number of targets
    num_labels = reduce(lambda x, y: x * y, target.shape, 1)

    # calculate total_loss
    if reduction != "none":
        total_loss = loss.item() * (num_labels if reduction == "elementwise_mean" else 1)
    else:
        total_loss = torch.sum(loss).item()

    return num_labels, total_loss


def smooth_label(labels, ratio, epsilon=0.1):
    assert len(labels.shape) == 1
    assert labels.dtype == torch.int64
    pos_score = labels.new_full(labels.shape, ((1 - epsilon) * 1) + (epsilon / ratio), dtype=torch.get_default_dtype())
    neg_score = labels.new_full(labels.shape, ((1 - epsilon) * 0) + (epsilon / ratio), dtype=torch.get_default_dtype())
    pos_prob = torch.where(labels > 0, pos_score, neg_score)
    return torch.stack((1 - pos_prob, pos_prob), dim=-1)


class MarginRankingLoss(nn.Module):
    '''
    This loss combines a Sigmoid layer and the MarginRankingLoss which measures the loss given inputs x1, x2,
        two 1D mini-batch Tensors, and a label 1D mini-batch tensor (containing 1 or -1).
    Computes sigmoid cross entropy given logits.
    '''
    def __init__(self, config):
        super(MarginRankingLoss, self).__init__()
        self.margin = config["margin"] if "margin" in config else 0.0
        self.reduction = config["reduction"] if "reduction" in config else "elementwise_mean"

        self.sigmoid = torch.nn.Sigmoid()
        self.loss_module = torch.nn.MarginRankingLoss(margin=self.margin, reduction=self.reduction)
        self.name = config["name"] if "name" in config else "Margin Ranking Loss"
        logger.info(utils.generate_module_info(self.name, "margin", self.margin, "reduction", self.reduction))


    def forward(self, input, target):
        """
        :param input: (N, 1)
        :param target: (N)
        :return: scalar . If reduction is "none", then the same size as the target: (N).
        """
        assert input.shape == target.shape
        if input.shape[0] % 2 != 0:
            if input.shape[0] == 1:
                return 0, 0, 0
            input = input[:-1]
            target = target[:-1]

        input = self.sigmoid(input)

        loss = self.loss_module(input[::2], input[1::2],
                                target.new_ones((target.shape[0] // 2,), dtype=torch.get_default_dtype()))
        num_labels, total_loss = gather_statistics(loss, target, self.reduction)

        return loss, num_labels, total_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        if "weight" in config and isinstance(config["weight"], (list, tuple, np.ndarray, torch.Tensor)):
            self.weight = torch.Tensor(config["weight"])
        else:
            self.weight = None
        self.ignore_index = config["ignore_index"] if "ignore_index" in config else -100
        self.reduction = config["reduction"] if "reduction" in config else "elementwise_mean"

        self.loss_module = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                                     reduction=self.reduction)
        self.name = config["name"] if "name" in config else "Cross Entropy Loss"
        logger.info(
            utils.generate_module_info(self.name, "weight", self.weight, "ignore_index", self.ignore_index, "reduction",
                                       self.reduction))


    def forward(self, input, target):
        """
        :param input: (N, C, d1, d2, ..., dK) with K >= 2 in the case of K-dimensional loss.
        :param target: (N, d1, d2, ..., dK) with K >= 2 in the case of K-dimensional loss.
        :return: scalar . If reduction is "none", then the same size as the target:
            (N), or (N, d1, d2, ..., dK) with K >= 2 in the case of K-dimensional loss.
        """
        assert input.shape[0] == target.shape[0]
        if self.weight is not None:
            assert self.weight.shape[0] == input.shape[1]

        loss = self.loss_module(input, target)
        num_labels, total_loss = gather_statistics(loss, target, self.reduction)

        return loss, num_labels, total_loss


class KLDivLoss(nn.Module):
    def __init__(self, config):
        super(KLDivLoss, self).__init__()
        self.size_average = config["size_average"] if "size_average" in config else False
        self.reduce = config["reduce"] if "reduce" in config else True
        self.reduction = config["reduction"] if "reduction" in config else "elementwise_mean"

        # parameters for smoothing label
        self.do_label_smoothing = config["do_label_smoothing"] if "do_label_smoothing" in config else False
        self.ratio = config["ratio"] if "ratio" in config else 2
        self.epsilon = config["epsilon"] if "epsilon" in config else 0.1

        self.loss_module = torch.nn.KLDivLoss(size_average=self.size_average, reduce=self.reduce,
                                              reduction=self.reduction)

        self.name = config["name"] if "name" in config else " KL Divergence Loss"
        logger.info(
            utils.generate_module_info(self.name, "size_average", self.size_average, "reduce", self.reduce, "reduction",
                                       self.reduction))

    def forward(self, input, target):
        if self.do_label_smoothing:
            assert input.shape[0] == target.shape[0]
            target = smooth_label(target, self.ratio, self.epsilon)
        else:
            assert input.shape == target.shape

        input = F.log_softmax(input, dim=-1)

        loss = self.loss_module(input, target)
        num_labels = input.shape[0]
        total_loss = loss.item()
        if self.size_average == False:
            loss /= input.shape[0]
        else:
            total_loss *= num_labels
        return loss, num_labels, total_loss


class BCEWithLogitsLoss(nn.Module):
    '''
    This loss combines a Sigmoid layer and the BCELoss which measures the Binary Cross Entropy between the target and the output in one single class.
        Computes sigmoid cross entropy given logits.
        Measures the probability error in discrete classification tasks in which each class is independent and not mutually exclusive.
        For instance, one could perform multilabel classification where a picture can contain both an elephant and a dog at the same time.
    '''
    def __init__(self, config):
        super(BCEWithLogitsLoss, self).__init__()
        if "weight" in config and isinstance(config["weight"], (list, tuple, np.ndarray, torch.Tensor)):
            self.weight = torch.Tensor(config["weight"])
        else:
            self.weight = None
        if "pos_weight" in config and isinstance(config["pos_weight"], (list, tuple, np.ndarray, torch.Tensor)):
            self.pos_weight = torch.Tensor(config["pos_weight"])
        else:
            self.pos_weight = None
        self.reduction = config["reduction"] if "reduction" in config else "elementwise_mean"

        self.loss_module = nn.BCEWithLogitsLoss(weight=self.weight, reduction=self.reduction, pos_weight=self.pos_weight)
        self.name = config["name"] if "name" in config else "Binary Cross Entropy"
        logger.info(
            utils.generate_module_info(self.name, "weight", self.weight, "pos_weight", self.pos_weight, "reduction",
                                       self.reduction))

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0]
        if self.weight is not None:
            assert self.weight.shape[0] == input.shape[0]
        if self.pos_weight is not None:
            assert self.pos_weight.shape[0] == input.shape[-1]

        if input.shape != target.shape:
            target = target.unsqueeze(-1).expand(input.shape)

        loss = self.loss_module(input, target.to(dtype=torch.get_default_dtype()))
        num_labels, total_loss = gather_statistics(loss, target, self.reduction)

        return loss, num_labels, total_loss


if __name__ == "__main__":
    input = torch.randn(2,1)
    target = torch.randint(0, 2, (2,1), dtype=torch.int64)
    # loss_fn1 = CrossEntropyLoss({})
    loss_fn2 = BCEWithLogitsLoss({"weight": [1., 2.]})
    print (input)
    print (target)
    # print (loss_fn1(input, target))
    print (loss_fn2(input, target))
    # print (nn.LogSoftmax(dim=-1)(input))
    s = nn.Sigmoid()(input)
    print (torch.log(s))
    print (torch.log(1 - s))