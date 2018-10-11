# coding: utf-8

import inspect
import re
import os
import logging
import time
from functools import reduce
from copy import deepcopy
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn


# TODO: Related to Logger

LOGGER_NAME = "pinkcom"

def get_logger(logger_name=LOGGER_NAME):
    return logging.getLogger(name=logger_name)


def create_logger(log_file, logger_name=LOGGER_NAME):
    # 1. create a logger
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(logging.INFO)  # log level of logger
    # 2. create a handler for writing logs into a file
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)  # log level of handler
    # 3. output format
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 4. add the handler into logger
    logger.addHandler(fh)
    logger.info("Logger '{}' is running.".format(logger.name))
    return logger


def generate_metrics_str(results, verbose=False):
    '''
    :param results: if verbose is True, dict( {name: value, ...} ) form, otherwise dict( {name: [mv, mn], ...} )
    :return: str
    '''
    if verbose:
        beauty_line = '-' * 93
        metric_str = (beauty_line + '\n' + "| {:^20} | {:^20} | {:^20} | {:^20} |\n" + beauty_line + '\n' + "\n".join(
            ["| {:^20} | {:^20.5f} | {:^20d} | {:^20.5f} |" for i in range(len(results))]) + '\n' + beauty_line).format(
            "Metrics",
            "Accumulation",
            "Count",
            "Value", *reduce(
                lambda x, y: x + [y[0], y[1][0], y[1][1], y[1][0] / y[1][1]], results.items(), []))
    else:
        beauty_line = '-' * 47
        metric_str = (beauty_line + '\n' + "| {:^20} | {:^20} |\n" + beauty_line + '\n' + "\n".join(
            ["| {:^20} | {:^20.5f} |" for i in range(len(results))]) + '\n' + beauty_line).format(
            "Metrics", "Value", *reduce(lambda x, y: x + [y[0], y[1]], results.items(), []))
    return metric_str


def generate_module_info(*args):
    info_template = "\t| {:^15}" + " | {:^10}: {:^8}" * (len(args) // 2)
    return info_template.format(*map(str, args))


# TODO: Related to Debug

def varname(p):
    '''
    obtain the name of the variable which is only an actual parameter of function varname
    :param p: being showed variable
    '''
    if isinstance(p, tuple) or isinstance(p, list):
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
            if m:
                print("{:<40}{}".format(str(m.group(1)) + ':', ", ".join([str(x.shape) for x in p])))
    else:
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
            if m:
                var_name = str(m.group(1))
                print("{:<40}{}".format(var_name + ':', p.shape))


def inspect_parameters_update(optimizer, model, inputs, loss_fn):
    params_save = deepcopy(optimizer.param_groups[0]['params'])
    logits = model(inputs)
    loss, num_labels, batch_total_loss = loss_fn(logits, inputs["target"])
    loss.backward()
    optimizer.step()
    for sp, p in zip(params_save, optimizer.param_groups[0]['params']):
        print(torch.equal(sp, p), torch.sum(torch.eq(sp, p)), sp.shape, p.shape)


# TODO: Related to Auxiliary Functions

def name2function(f_name):
    '''
    :param f_name: the import path of function or class
    :return: function or class
    '''
    items = f_name.strip().rsplit('.', maxsplit=1)
    if len(items) == 1:
        pass
    else:
        if '.' in items[0]:
            modules = __import__(items[0], fromlist=True)
        else:
            modules = __import__(items[0])
        return getattr(modules, items[1])


def lower_dict(dic, recursive=False):
    if isinstance(dic, dict):
        dic = dict([
            (
                k.lower() if isinstance(k, str) else k,
                lower_dict(v, recursive) if recursive else v
            )
            for k, v in dic.items()])
    return dic


# TODO: Related to Padding

def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length


def multi_sequences_padding(all_sequences, max_num_utterance=10, max_sentence_len=50):
    # TODO: utterance tail padding
    PAD_SEQUENCE = [0] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences += [PAD_SEQUENCE] * (max_num_utterance - sequences_len)
            sequences_length[-1] += [0] * (max_num_utterance - sequences_len)
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length


