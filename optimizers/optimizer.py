# coding=utf-8

import torch
import logging
from utils import utils

logger = utils.get_logger()

class AdamOptimizer(object):
    def __init__(self, config):
        self.lr = config["lr"] if "lr" in config else 0.001
        self.betas = config["betas"] if "betas" in config else (0.9, 0.999)
        self.eps = config["eps"] if "eps" in config else 1e-08
        self.weight_decay = config["weight_decay"] if "weight_decay" in config else 0
        self.amsgrad = config["amsgrad"] if "amsgrad" in config else False
        self.name = "Adam Optimizer"
        logger.info("| {} | {}: {} | {}: {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "lr", self.lr,
            "betas", self.betas,
            "eps", self.eps,
            "weight_decay", self.weight_decay,
            "amsgrad", self.amsgrad)
        )

    def ops(self, params):
        return torch.optim.Adam(params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay,
                                amsgrad=self.amsgrad)