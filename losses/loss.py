# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
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
        self.name = "Cross Entropy Loss"
        logger.info("| {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "weight", self.weight,
            "ignore_index", self.ignore_index,
            "reduction", self.reduction)
        )


    def forward(self, input, target):
        """
        :param input: (N, C, d1, d2, ..., dK) with K >= 2 in the case of K-dimensional loss.
        :param target: (N, d1, d2, ..., dK) with K >= 2 in the case of K-dimensional loss.
        :return: scalar . If reduction is "none", then the same size as the target:
            (N), or (N, d1, d2, ..., dK) with K >= 2 in the case of K-dimensional loss.
        """
        if self.weight is not None:
            assert self.weight.shape[0] == input.shape[1]
        assert input.shape[0] == target.shape[0]

        loss = self.loss_module(input, target)

        # count the number of targets
        num_labels = 1
        for shape in target.shape:
            num_labels *= shape

        # calculate total_loss
        if self.reduction != "none":
            total_loss = loss.item() * (num_labels if self.reduction == "elementwise_mean" else 1)
        else:
            total_loss = torch.sum(loss).item()

        return loss, num_labels, total_loss
