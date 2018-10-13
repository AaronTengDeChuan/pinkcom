# coding=utf-8

import torch
import logging
from utils import utils
from copy import deepcopy
from inspect import isclass

logger = utils.get_logger()


class LrScheduler(object):
    def __init__(self, config=None):
        config = deepcopy(config)
        if config == None or not isinstance(config, dict):
            config = {
                "function": "torch.optim.lr_scheduler.StepLR",
                "params": {
                    "step_size": 1000000,
                    "gamma": 1
                }
            }
        self.lr_scheduler = utils.name2function(config["function"])
        self.lr_scheduler_params = config["params"]
        if isclass(self.lr_scheduler) and issubclass(self.lr_scheduler, (
        torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
            self.name = config["function"].strip().rsplit('.', maxsplit=1)[-1] + " Scheduler"
            logger.info(
                utils.generate_module_info(self.name, **self.lr_scheduler_params)
            )

    def ops(self, optimizer):
        return self.lr_scheduler(optimizer, **self.lr_scheduler_params)


class Optimizer(object):
    def __init__(self, config):
        # lr_scheduler
        if "lr_scheduler" in config["params"]:
            self.lr_scheduler = LrScheduler(config["params"]["lr_scheduler"])
            config["params"].pop("lr_scheduler")
        else:
            self.lr_scheduler = LrScheduler()
        # optimizer
        self.optimizer = utils.name2function(config["function"])
        self.optimizer_params = config["params"]
        if isclass(self.optimizer) and issubclass(self.optimizer, torch.optim.Optimizer):
            self.name = config["function"].strip().rsplit('.', maxsplit=1)[-1]
            logger.info(
                utils.generate_module_info(self.name, **self.optimizer_params)
            )

    def ops(self, params):
        optimizer = self.optimizer(params, **self.optimizer_params)
        return self.lr_scheduler.ops(optimizer), optimizer


def AdamOptimizer(params, **config):
    lr = config["lr"] if "lr" in config else 0.001
    betas = config["betas"] if "betas" in config else (0.9, 0.999)
    eps = config["eps"] if "eps" in config else 1e-08
    weight_decay = config["weight_decay"] if "weight_decay" in config else 0
    amsgrad = config["amsgrad"] if "amsgrad" in config else False

    name = "Adam Optimizer"
    logger.info(
        utils.generate_module_info(name, "lr", lr, "betas", betas, "eps", eps, "weight_decay", weight_decay, "amsgrad",
                                   amsgrad))

    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)


def SGDOptimizer(params, **config):
    lr = config["lr"] if "lr" in config else 0.001
    momentum = config["momentum"] if "momentum" in config else 0
    weight_decay = config["weight_decay"] if "weight_decay" in config else 0
    dampening = config["dampening"] if "dampening" in config else 0
    nesterov = config["nesterov"] if "nesterov" in config else False

    name = "SGD Optimizer"
    logger.info(
        utils.generate_module_info(name, "lr", lr, "momentum", momentum, "weight_decay", weight_decay, "dampening",
                                   dampening, "nesterov", nesterov))

    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening,
                           nesterov=nesterov)