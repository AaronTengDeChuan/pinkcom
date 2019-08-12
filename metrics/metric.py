# coding=utf-8

import os
import sys

if __name__ == "__main__":
    base_work_dir = os.path.dirname(os.getcwd())
    sys.path.append(base_work_dir)

import torch
from functools import reduce
from utils import utils
from utils.utils import varname
import logging
import codecs

logger = utils.get_logger()

class Recall_N_at_K(object):
    '''
    Input:
        Designed for 2 or 1 classes
        y_pred: [x*N, 2 or 1]
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
        logger.info(
            utils.generate_module_info(self.name, "N", self.N, "AN", self.AN, "K", self.K, "skip", self.skip))

    def ops(self, y_pred, y_true):
        # TODO: check this
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        # varname(y_pred)
        # varname(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:,1].tolist() if len(y_pred.shape) == 2 else y_pred.tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        total_recall = 0.
        for i in range(x):
            p_temp = y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N]
            t_temp = y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N]
            pt_temp = list(zip(*sorted(list(zip(p_temp, t_temp)), key=lambda x: x[1], reverse=True)))

            y_p = pt_temp[0][:self.AN][::-1]
            y_t = pt_temp[1][:self.AN][::-1]
            total = sum(y_t)
            if total == 0: x -= 1; continue

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
        logger.info(
            utils.generate_module_info(self.name, "N", self.N, "AN", self.AN, "skip", self.skip))

    def ops(self, y_pred, y_true):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:, 1].tolist() if len(y_pred.shape) == 2 else y_pred.tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        APs = 0
        for i in range(x):
            p_temp = y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N]
            t_temp = y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N]
            pt_temp = list(zip(*sorted(list(zip(p_temp, t_temp)), key=lambda x: x[1], reverse=True)))

            y_p = pt_temp[0][:self.AN][::-1]
            y_t = pt_temp[1][:self.AN][::-1]
            total = sum(y_t)
            if total == 0: x -= 1; continue

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
        logger.info(
            utils.generate_module_info(self.name, "N", self.N, "AN", self.AN, "skip", self.skip))

    def ops(self, y_pred, y_true):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:, 1].tolist() if len(y_pred.shape) == 2 else y_pred.tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        RRs = 0
        for i in range(x):
            p_temp = y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N]
            t_temp = y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N]
            pt_temp = list(zip(*sorted(list(zip(p_temp, t_temp)), key=lambda x: x[1], reverse=True)))

            y_p = pt_temp[0][:self.AN][::-1]
            y_t = pt_temp[1][:self.AN][::-1]
            total = sum(y_t)
            if total == 0: x -= 1; continue

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
        logger.info(
            utils.generate_module_info(self.name, "N", self.N, "AN", self.AN, "K", self.K, "skip", self.skip))

    def ops(self, y_pred, y_true):
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.shape[0] == y_true.shape[0] and y_pred.shape[0] % self.N == 0
        y_pred = y_pred[:, 1].tolist() if len(y_pred.shape) == 2 else y_pred.tolist()
        y_true = y_true.tolist()

        x = len(y_pred) // self.N
        Ps = 0
        for i in range(x):
            p_temp = y_pred[i::x] if self.skip else y_pred[i * self.N: (i + 1) * self.N]
            t_temp = y_true[i::x] if self.skip else y_true[i * self.N: (i + 1) * self.N]
            pt_temp = list(zip(*sorted(list(zip(p_temp, t_temp)), key=lambda x: x[1], reverse=True)))

            y_p = pt_temp[0][:self.AN][::-1]
            y_t = pt_temp[1][:self.AN][::-1]
            total = sum(y_t)
            if total == 0: x -= 1; continue

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
        logger.info(
            utils.generate_module_info(self.name))

    def ops(self, y_pred, y_true):
        # TODO: check this
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        if len(y_pred.shape) < 2:
            return 1, 1
        assert y_pred.shape[:-1] == y_true.shape
        correct = torch.sum(torch.eq(torch.argmax(y_pred, dim=-1), y_true)).item()
        total = 1
        for shape in y_true.shape:
            total *= shape
        return correct, total


def calculate_metrics_with_score_file(score_file, score_format="PT"):
    with codecs.open(sys.argv[1], 'r', encoding="utf-8") as f:
        scores = [[float(x) for x in line.split('\t')] for line in f.read().strip().split('\n')]
        if len(scores) % 10 != 0: scores = scores[:-(len(scores) % 10)]

    assert score_format in ['PT', 'PP', 'PPT'], "Incorrect score format {}, and true format include '{}'.".format(
        score_format, ['PT', 'PP', 'PPT'])

    if score_format == "PT":
        scores = list(zip(*scores))
        y_pred = torch.stack([torch.tensor(scores[0])] * 2, dim=-1)
        y_true = torch.tensor(scores[1])
    elif score_format == "PP":
        y_pred = torch.tensor(scores)
        y_true = torch.tensor(reduce(lambda x, y: x + [1] + [0] * 9, range(len(scores) // 10), []))
    elif score_format == "PPT":
        scores = list(zip(*scores))
        y_pred = torch.stack([torch.tensor(scores[0]), torch.tensor(scores[1])], dim=-1)
        y_true = torch.tensor(scores[2])
    print(y_pred.shape, y_pred.dtype)
    print(y_true.shape, y_true.dtype)
    r2_at_1 = Recall_N_at_K({"N": 10, "AN": 2, "K": 1, "skip": False})  # for ubuntu
    r10_at_1 = Recall_N_at_K({"N": 10, "K": 1, "skip": False})
    r10_at_2 = Recall_N_at_K({"N": 10, "K": 2, "skip": False})
    r10_at_5 = Recall_N_at_K({"N": 10, "K": 5, "skip": False})
    map = MAP_in_N({"N": 10, "skip": False})
    mrr = MRR_in_N({"N": 10, "skip": False})
    p_at_1 = Precision_N_at_K({"N": 10, "K": 1, "skip": False})
    acc = Accurary({})

    metrics_dict = {
        r2_at_1.name: r2_at_1.ops(y_pred, y_true),  # for ubuntu
        r10_at_1.name: r10_at_1.ops(y_pred, y_true),
        r10_at_2.name: r10_at_2.ops(y_pred, y_true),
        r10_at_5.name: r10_at_5.ops(y_pred, y_true),
        map.name: map.ops(y_pred, y_true),
        mrr.name: mrr.ops(y_pred, y_true),
        p_at_1.name: p_at_1.ops(y_pred, y_true),
        acc.name: acc.ops(y_pred, y_true)
    }

    print(utils.generate_metrics_str(metrics_dict, verbose=True))


if __name__ == "__main__":
    # test actual output
    ubuntu_score_file = "/Users/aaron_teng/Documents/SCIR/papers/Dialogue/DAM/models/output/ubuntu/DAM/score.test"
    # douban_score_file = "/Users/aaron_teng/Documents/SCIR/papers/Dialogue/DAM/models/output/douban/DAM/score"
    # score_file = "/Users/aaron_teng/Documents/SCIR/HPC/score.test"
    calculate_metrics_with_score_file(sys.argv[1], sys.argv[2])
