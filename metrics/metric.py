# coding=utf-8

import torch
from utils import utils
from utils.utils import varname
import logging

logger = utils.get_logger()

class Recall_N_at_K(object):
    '''
    Designed for 2 classes
    y_pred: [x*N, 2]
    y_true: [x*N]
    '''
    def __init__(self, config):
        self.N = config["N"] if "N" in config else 10
        self.K = config["K"] if "K" in config else 1
        self.skip = config["skip"] if "skip" in config else True
        self.name = "R{}@{}".format(self.N, self.K)
        logger.info("| {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "N", self.N,
            "K", self.K,
            "skip", self.skip)
        )

    def ops(self, y_pred, y_true):
        # TODO: check this
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        # varname(y_pred)
        # varname(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:,1].tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        total_recall = 0.
        for i in range(x):
            if self.skip:
                y_p = y_pred[i::x]
                y_t = y_true[i::x]
            else:
                y_p = y_pred[i*self.N: (i+1)*self.N]
                y_t = y_true[i*self.N: (i+1)*self.N]
            total = sum(y_t)
            recall = 0.
            c = list(zip(y_p, y_t))
            c = sorted(c, key=lambda x: x[0], reverse=True)
            for j, (p, t) in enumerate(c):
                if j >= self.K:
                    break
                if t == 1:
                    recall += 1
            total_recall += recall / total

        return total_recall, x


class Accurary(object):
    '''
    y_pred: [N, d1, ..., dk, C]
    y_true: [N, d1, ..., dk]
    '''
    def __init__(self, config):
        self.name = "Accuracy"
        logger.info("| {} |".format(self.name))

    def ops(self, y_pred, y_true):
        # TODO: check this
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[:-1] == y_true.shape
        correct = torch.sum(torch.eq(torch.argmax(y_pred, dim=-1),y_true)).item()
        total = 1
        for shape in y_true.shape:
            total *= shape
        return correct, total


if __name__ == "__main__":
    # test Rn@k
    R10_at_1 = Recall_N_at_K({"N":5, "K":5, "skip":True})
    y_pred = torch.randn(10,2)
    y_true = torch.tensor([1,1,0,0,0,0,0,0,0,0])
    print (y_pred[:,1])
    print (y_true)
    print (R10_at_1.ops(y_pred,y_true))

    # test Accurary
    acc = Accurary({})
    y_pred = torch.randn(2,3,4)
    y_true = torch.randint(0, 4, (2,3), dtype=torch.int64)
    print (y_pred)
    print (y_true)
    print (acc.ops(y_pred, y_true))
    print (acc.name)
