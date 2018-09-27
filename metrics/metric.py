# coding=utf-8

import torch
from utils import utils
from utils.utils import varname
import logging

logger = utils.get_logger()

class Recall_N_at_K(object):
    '''
    Input:
        Designed for 2 classes
        y_pred: [x*N, 2]
        y_true: [x*N]
    Aim:
        Calculate the recall of the true positive replies among the k selected ones as the main evaluation metric,
        denoted as
            Rn@k = Σi=1~k(yi) / Σi=1~n(yi),
        where yi is the binary label for each candidate.
    Params:
        N: the number of candidates for every sample [Default: 10]
        AN: the number of available candidates for every sample,
            namely 'n' in the above equation 'Rn@k = Σi=1~k(yi) / Σi=1~n(yi)' [Default: N]
        K: select K best-matched result from AN available candidates,
            namely 'k' in the above equation 'Rn@k = Σi=1~k(yi) / Σi=1~n(yi)' [Default: 1]
        skip: if skip is true, candidates of a sample have saltatory indexes,
            i.e. i i+x i+2x ... i+(N-1)x [Default: False]
    '''
    def __init__(self, config):
        config = utils.lower_dict(config, recursive=True)
        self.N = config["n"] if "n" in config else 10
        self.AN = config["an"] if 'an' in config else self.N
        self.K = config["k"] if "k" in config else 1
        assert self.AN >= self.K
        self.skip = config["skip"] if "skip" in config else False
        self.name = "R{}@{}".format(self.AN, self.K)
        logger.info("| {} | {}: {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "N", self.N,
            "AN", self.AN,
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
            y_p = (y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N])[:self.AN]
            y_t = (y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N])[:self.AN]
            total = sum(y_t)
            c = list(zip(y_p, y_t))
            c = sorted(c, key=lambda x: x[0], reverse=True)
            recall = 0. + [pt[1] for pt in c][:self.K].count(1)
            total_recall += recall / total
        return total_recall, x


class MAP_in_N(object):
    '''
    Mean Average Precision
    P -> AP -> MAP
    '''
    def __init__(self, config):
        config = utils.lower_dict(config, recursive=True)
        self.N = config["n"] if "n" in config else 10
        self.AN = config["an"] if 'an' in config else self.N
        assert self.AN <= self.N
        self.skip = config["skip"] if "skip" in config else False
        self.name = "MAP_in_{}".format(self.AN)
        logger.info("| {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "N", self.N,
            "AN", self.AN,
            "skip", self.skip)
        )

    def ops(self, y_pred, y_true):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:, 1].tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        APs = 0
        for i in range(x):
            y_p = (y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N])[:self.AN]
            y_t = (y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N])[:self.AN]
            c = list(zip(y_p, y_t))
            c = sorted(c, key=lambda x: x[0], reverse=True)
            num_refs = 0
            Ps = 0
            for index in range(self.AN):
                if c[index][1] == 1:
                    num_refs += 1
                    Ps += 1.0 * num_refs / (index + 1)
            APs += Ps / num_refs
        return APs, x


class MRR_in_N(object):
    '''
    Mean Reciprocal Rank
    '''
    def __init__(self, config):
        config = utils.lower_dict(config, recursive=True)
        self.N = config["n"] if "n" in config else 10
        self.AN = config["an"] if 'an' in config else self.N
        assert self.AN <= self.N
        self.skip = config["skip"] if "skip" in config else False
        self.name = "MRR_in_{}".format(self.AN)
        logger.info("| {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "N", self.N,
            "AN", self.AN,
            "skip", self.skip)
        )

    def ops(self, y_pred, y_true):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:, 1].tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        RRs = 0
        for i in range(x):
            y_p = (y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N])[:self.AN]
            y_t = (y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N])[:self.AN]
            c = list(zip(y_p, y_t))
            c = [pt[1] for pt in sorted(c, key=lambda x: x[0], reverse=True)]
            assert 1 in c
            RRs += 1.0 / (1 + c.index(1))
        return RRs, x


class Precision_N_at_K(object):
    '''
    Precision among top k
    '''
    def __init__(self, config):
        config = utils.lower_dict(config, recursive=True)
        self.N = config["n"] if "n" in config else 10
        self.AN = config["an"] if 'an' in config else self.N
        self.K = config["k"] if "k" in config else 1
        assert self.AN >= self.K
        self.skip = config["skip"] if "skip" in config else False
        self.name = "Precision_{}@{}".format(self.AN, self.K)
        logger.info("| {} | {}: {} | {}: {} | {}: {} | {}: {}".format(
            self.name,
            "N", self.N,
            "AN", self.AN,
            "K", self.K,
            "skip", self.skip)
        )

    def ops(self, y_pred, y_true):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:, 1].tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        Ps = 0
        for i in range(x):
            y_p = (y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N])[:self.AN]
            y_t = (y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N])[:self.AN]
            c = list(zip(y_p, y_t))
            c = [pt[1] for pt in sorted(c, key=lambda x: x[0], reverse=True)]
            Ps += 1.0 if 1 in c[:self.K] else 0
        return Ps, x


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
        correct = torch.sum(torch.eq(torch.argmax(y_pred, dim=-1), y_true)).item()
        total = 1
        for shape in y_true.shape:
            total *= shape
        return correct, total


if __name__ == "__main__":
    # test Rn@k
    R10_at_1 = Recall_N_at_K({"N":5,"AN":2, "K":1, "skip":False})
    P_at_k = Precision_N_at_K({"N": 5, "K": 4, "skip": False})
    mrr = MRR_in_N({"N": 5, "skip": False})
    map = MAP_in_N({"N": 5, "skip": False})
    y_pred = torch.randn(10,2)
    y_true = torch.tensor([1,0,0,0,0,1,0,0,0,0])
    print (y_pred[:,1])
    print (y_true)
    print (R10_at_1.ops(y_pred,y_true))
    print (P_at_k.ops(y_pred, y_true))
    print (mrr.ops(y_pred, y_true))
    print (map.ops(y_pred, y_true))


    # test Accurary
    # acc = Accurary({})
    # y_pred = torch.randn(500000,2)
    # y_true = torch.randint(0, 2, (500000,), dtype=torch.int64)
    # # print (y_pred)
    # # print (torch.argmax(y_pred,dim=-1))
    # # print (y_true)
    # correct, total = acc.ops(y_pred, y_true)
    # print (correct, total)
    # print(correct / total)
    # print (acc.name)
