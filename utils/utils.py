# coding: utf-8

import inspect
import re
import os
import logging
import time
import math
from functools import reduce
from tqdm import tqdm
from copy import deepcopy
# from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import numpy as np


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
    :param results: if verbose is False, dict( {name: value, ...} ) form, otherwise dict( {name: [mv, mn], ...} )
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


def generate_module_info(*args, **kargs):
    info_template = "\t| {:^15}" + " | {:^10}: {:^8}" * (len(args) // 2 + len(kargs))
    return info_template.format(*map(str, args), *reduce(lambda x, y: x + list(map(str, y)), kargs.items(), []))


def second2hour(seconds):
    seconds = math.ceil(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 3600 % 60
    return "{}h {}m {}s".format(hours, minutes, seconds)

# TODO: Related to Debug

def varname(p, fn=None):
    '''
    obtain the name of the variable which is only an actual parameter of function varname
    :param p: being showed variable
    '''
    regular_expression = r'\bvarname\s*\(\s*([A-Za-z_]\w*)\s*(,)?\s*(?(2)(fn\s*=)?\s*(\w+)\s*)\)'
    if isinstance(p, tuple) or isinstance(p, list):
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(regular_expression, line)
            if m:
                print("{:<40}{}".format(str(m.group(1)) + ':', ", ".join([str(x.shape) for x in p])))
    else:
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(regular_expression, line)
            if m:
                var_name = str(m.group(1))
                print("{:<40}{:<40}{}".format(var_name + ':', str(p.shape), str(p.dtype)))
    if fn is not None:
        if isinstance(p, (tuple, list)):
            for x in p: fn(x)
        else: fn(p)


def output_model_params_and_grad(model, step=True):
    assert isinstance(model, nn.Module)
    for name, module in model.named_children():
        print("Module: " + name)
        for name, parameter in module.named_parameters():
            print("\nParameter: " + name)
            print(parameter)
            print("\nGradient:")
            print(parameter.grad)
            if step: input("\nnext parameter:")
        if step: input("{}\nnext module:".format('-'*50))
    print ("No more modules to be shown.")


def compare_tensors(x, y, eps=1e-8):
    def compare_tensor(tensor1, tensor2, eps=eps):
        assert tensor1.shape == tensor2.shape, "tensor1 shape: {}, tensor2 shape: {}".format(tensor1.shape,
                                                                                             tensor2.shape)
        com = torch.sum(torch.le(torch.abs(tensor1 - tensor2), eps)).item()
        print("Comparison: {} / {}".format(com, reduce(lambda a, b: a * b, tensor1.shape, 1)), end="\n\n")
    if isinstance(x, list or tuple):
        for tensor1, tensor2 in zip(x, y):
            compare_tensor(tensor1, tensor2)
    elif isinstance(x, torch.Tensor):
        compare_tensor(x, y)
    else:
        return


def inspect_parameters_update(optimizer, model, inputs, loss_fn):
    params_save = deepcopy(optimizer.param_groups[0]['params'])
    logits = model(inputs)
    loss, num_labels, batch_total_loss = loss_fn(logits, inputs["target"])
    loss.backward()
    optimizer.step()
    for sp, p in zip(params_save, optimizer.param_groups[0]['params']):
        print(torch.equal(sp, p), torch.sum(torch.eq(sp, p)), sp.shape, p.shape)


def get_tf_weights(model_path_and_prefix, save_file, ignore_suffixs=("Adam", "Adam_1")):
    import tensorflow as tf
    import numpy as np
    import pickle

    reader = tf.train.NewCheckpointReader(model_path_and_prefix)
    all_variables = reader.get_variable_to_shape_map()
    variable_names = []
    for name in all_variables.keys():
        tmp = name.rsplit("/", maxsplit=1)
        if len(tmp) == 1 or tmp[1] not in ignore_suffixs:
            variable_names.append(name)
    name2shape = dict([ (name, all_variables[name]) for name in variable_names])
    name2value = dict([ (name, reader.get_tensor(name)) for name in variable_names])

    with open(save_file, "wb") as f:
        pickle.dump(name2value, f)

    print ("Weights in {} are saved in {}:".format(model_path_and_prefix, save_file))
    for name in sorted(variable_names):
        print (name, ": ", name2shape[name])


# TODO: Related to Auxiliary Functions

def float_is_equal(a, b, eps_0=1e-6):
    return abs(a-b) <= eps_0


def clones(module, N):
    '''
    Produce N identical layers.
    '''
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


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
            modules = __import__(items[0], fromlist=[items[1]])
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

def list_process_fn(data, element_fn, list_fn_list, num_exception, exception_process=False):
    if isinstance(data, (list, tqdm)):
        res = [[] for i in range(len(list_fn_list))]
        tmp = None
        for l in data:
            tmp = l
            try:
                r = list_process_fn(l, element_fn, list_fn_list, num_exception, exception_process)
                for i in range(len(list_fn_list)): res[i].append(r[i])
            except:
                count = [0]
                print (l)
                print (list_process_fn(l, element_fn, list_fn_list, count, exception_process=True))
                print ("{} exceptions occurred.".format(count[0]))
                exit()
        if len(data) == 0 or not isinstance(tmp, list):
            res = [list_fn_list[i](res[i]) for i in range(len(list_fn_list))]
    else:
        try:
            ele = element_fn(data)
            res = [deepcopy(ele) for i in range(len(list_fn_list))]
        except:
            res = ["_unk_"] * len(list_fn_list)
            num_exception[0] += 1
            if not exception_process:
                raise
    return res


def flatten_list(data):
    flattened_data = []
    lengths = []
    if isinstance(data, (list, tqdm)):
        tmp = None
        for l in data:
            tmp = l
            l_flattened, l_lengths = flatten_list(l)
            flattened_data.extend(l_flattened)
            lengths.append(l_lengths)
        if not isinstance(tmp, list):
            lengths = len(lengths)
    else:
        flattened_data.append(data)
        lengths = 1
    return flattened_data, lengths


def restore_list(flattened_data, lengths, start_index):
    restored_data = []
    if isinstance(lengths, (list, tqdm)):
        for l in lengths:
            l_restored, start_index = restore_list(flattened_data, l, start_index)
            restored_data.append(l_restored)
    else:
        assert start_index < len(flattened_data)
        restored_data = flattened_data[start_index: start_index + lengths]
        start_index += lengths
    return restored_data, start_index


def calculate_metric_with_params(results, model_selection_params):
    '''

    :param results: dict( {name: value, ...} )
    :param model_selection_params: {
            "reduction" contains "sum", "mean" and so on,
            "mode" contain  "max" and "min",
            "metrics" is a list of names of metrics
        }
    :return: a scalar
    '''
    value = None
    if model_selection_params["reduction"] in ["sum", "mean"]:
        value = sum([results[metric_name] for metric_name in model_selection_params["metrics"]])
    if model_selection_params["reduction"] in ["mean"]:
        value /= len(model_selection_params["metrics"])

    assert model_selection_params["mode"] in ["min", "max"], "'mode' in model_selection_params must be a member of ['min', 'max']."
    if model_selection_params["mode"] == "max":
        return -value
    else:
        return value


def binary_indicator(source, source_len, target, target_len, axis=-2):
    '''
    type:   numpy.ndarray
    :param source:  shape (batch, num_turns, turn_len)
    :param source_len:  shape (batch, num_turns)
    :param target:  shape (batch, target_len)
    :param target_len:  shape (batch)
    :return: shape (batch, num_turns, target_len)
    '''
    expanded_target = np.concatenate([np.expand_dims(target, axis)] * source.shape[axis], axis=axis)
    expanded_target_len = np.concatenate([np.expand_dims(target_len, axis+1)] * source.shape[axis], axis=axis+1)
    stmp = source.reshape(-1, source.shape[-1])
    sltmp = source_len.reshape(-1)
    ttmp = expanded_target.reshape(-1, expanded_target.shape[-1])
    tltmp = expanded_target_len.reshape(-1)
    res = list(map(lambda tmp: list(map(lambda x: int(x in tmp[0][:tmp[1]]) if x != 0 else 0, tmp[2])),
                   zip(stmp, sltmp, ttmp, tltmp)))
    return np.array(res, dtype=np.int64)


# TODO: Related to Padding

def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length


def pad_3d_sequences(sequences, max_len, padding_element=0, padding='post'):
    # [num_turn, sentence_len, num_feature]
    padded_sequences = []
    for sequence in sequences:
        sequence_len = len(sequence)
        if sequence_len < max_len:
            if padding == "post":
                sequence += [padding_element] * (max_len - sequence_len)
            else:
                sequence = [padding_element] * (max_len - sequence_len) + sequence
        else:
            sequence = sequence[:max_len]
        padded_sequences.append(sequence)
    return padded_sequences


def multi_sequences_padding(all_sequences, max_num_utterance=10, max_sentence_len=50, padding_element=0):
    # TODO: utterance tail padding
    PAD_SEQUENCE = [padding_element] * max_sentence_len
    padded_sequences = []
    sequences_length = []
    for sequences in all_sequences:
        sequences_len = len(sequences)
        sequences_length.append(get_sequences_length(sequences, max_sentence_len))
        if sequences_len < max_num_utterance:
            sequences = [PAD_SEQUENCE] * (max_num_utterance - sequences_len) + sequences
            sequences_length[-1] = [0] * (max_num_utterance - sequences_len) + sequences_length[-1]
        else:
            sequences = sequences[-max_num_utterance:]
            sequences_length[-1] = sequences_length[-1][-max_num_utterance:]
        if isinstance(padding_element, list):
            sequences = pad_3d_sequences(sequences, max_sentence_len, padding_element, padding="post")
        else:
            # replace function "pad_sequences" with function "pad_3d_sequences"
            # sequences = pad_sequences(sequences, padding='post', maxlen=max_sentence_len)
            sequences = pad_3d_sequences(sequences, max_sentence_len, padding_element, padding="post")
        padded_sequences.append(sequences)
    return padded_sequences, sequences_length


# TODO: Related to RNN

def repackage_hidden(h):
    """
        Wraps hidden states in new Tensors,in order to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
