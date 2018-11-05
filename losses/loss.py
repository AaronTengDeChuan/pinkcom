# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
import logging
from utils import utils

logger = utils.get_logger()

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

        # count the number of targets
        num_labels = reduce(lambda x, y: x * y, target.shape, 1)

        # calculate total_loss
        if self.reduction != "none":
            total_loss = loss.item() * (num_labels if self.reduction == "elementwise_mean" else 1)
        else:
            total_loss = torch.sum(loss).item()

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
        assert input.shape == target.shape
        if self.weight is not None:
            assert self.weight.shape[0] == input.shape[0]
        if self.pos_weight is not None:
            assert self.pos_weight.shape[0] == input.shape[-1]

        loss = self.loss_module(input, target.to(dtype=torch.get_default_dtype()))

        # count the number of targets
        num_labels = reduce(lambda x, y: x * y, target.shape, 1)

        # calculate total_loss
        if self.reduction != "none":
            total_loss = loss.item() * (num_labels if self.reduction == "elementwise_mean" else 1)
        else:
            total_loss = torch.sum(loss).item()

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