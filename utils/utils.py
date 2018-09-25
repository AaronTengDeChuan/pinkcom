# coding: utf-8

import numpy as np
import inspect
import re
import pickle
import os
import logging
import time
from functools import reduce
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

LOGGER_NAME = "pinkcom"

def get_logger(logger_name=LOGGER_NAME):
    return logging.getLogger(name=logger_name)

logger = get_logger()

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
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 4. add the handler into logger
    logger.addHandler(fh)
    logger.info("Logger '{}' is running.".format(logger.name))
    return logger

def calculate_dim_with_initialDim_conv(initial_dim, conv):
    '''
    calculate input dim of first fully connected layer
    :param initial_dim:
    :param conv: nn.Sequential containing conv, activation function, and pooling
    :return: input dim of first fully connected layer
    '''
    batch_size = 2
    inputs = torch.randn((batch_size, conv[0].in_channels, *initial_dim), dtype=torch.float32)
    outputs = conv(inputs).view(batch_size, -1)
    return outputs.shape[1]


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

def generate_metrics_str(results):
    '''
    :param results: dict( {name: value, ...} )
    :return: str
    '''
    metric_str = ("| {:^20} | {:^20} |\n" + "\n".join(["| {:^20} | {:^20.4f} |" for i in range(len(results))])).format(
        "metrics", "value", *reduce(lambda x, y: x + [y[0], y[1]], results.items(), []))
    return metric_str

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


def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length


def multi_sequences_padding(all_sequences, max_num_utterance=10, max_sentence_len=50):
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


def pack_and_pad_sequences_for_rnn(seq_embeds, seq_lens, rnn_module):
    '''
    similar to dynamic rnn
    supported rnn including GRU, LSTM
    batch_first of rnn_module must be True
    :param seq_embeds:
    :param seq_lens:
    :param rnn_module: rnn module in Pytorch
    :return: rnn_output, rnn_ht
    '''
    sorted_seq_lens, seq_indices = torch.sort(seq_lens.view(-1), dim=0, descending=True)
    sorted_seq_embeds = torch.index_select(\
                                               seq_embeds.view(-1, seq_embeds.shape[-2], seq_embeds.shape[-1]), \
                                               dim=0, index=seq_indices)
    # varname(sorted_seq_embeds) # torch.Size([None, 50, 200])
    sorted_seq_lens = sorted_seq_lens + torch.tensor(sorted_seq_lens == 0, dtype=torch.int64)

    # Packs a Tensor containing padded sequences of variable length in order to obtain a PackedSequence object
    packed_seq_input = pack_padded_sequence(sorted_seq_embeds, sorted_seq_lens, batch_first=True)
    # varname(packed_seq_input) # torch.Size([478, 200]), torch.Size([50])
    packed_seq_output, seq_ht = rnn_module(packed_seq_input)
    # varname(packed_seq_output) # torch.Size([478, 200]), torch.Size([50])
    # varname(seq_ht) # torch.Size([1, None, 200])
    
    # Pads a packed batch of variable length sequences
    # Ensure that the shape of output is not changed: check this
    seq_output, _= pad_packed_sequence(packed_seq_output, batch_first=True, total_length=seq_embeds.shape[-2])
    # varname(seq_output) # torch.Size([50, None, 200])
    
    # restore original order
    _, original_indices = torch.sort(seq_indices, dim=0, descending=False)
    seq_output = torch.index_select(seq_output, dim=0, index=original_indices)
    # varname(seq_output) # torch.Size([None, 50, 200])
    
    # restore original shape
    seq_output = seq_output.view(seq_embeds.shape[:-2] + seq_output.shape[-2:])
    # varname(seq_output)
    
    assert seq_output.shape[-2] == seq_embeds.shape[-2]
    
    if isinstance(seq_ht, torch.Tensor):
        seq_ht = seq_ht.view(seq_ht.shape[:-2] + seq_embeds.shape[:-2] + seq_ht.shape[-1:])
    else:
        seq_ht = tuple([ht.view(ht.shape[:-2] + seq_embeds.shape[:-2] + ht.shape[-1:]) for ht in seq_ht])
    # varname(seq_ht)
    return seq_output, seq_ht


def stack_channels_for_conv2d(channels, conv2d_module):
    """
    :param channels: tuple or tensor
    :param conv2d_module:
    :return: conv output
    """
    stack_dim = -3
    if isinstance(channels, tuple):
        channels = torch.stack(channels, dim=stack_dim)
    # varname(channels)
    conv_input = channels.view(tuple([-1]) + channels.shape[-3:])
    # varname(conv_input)
    conv_output = conv2d_module(conv_input)
    # varname(conv_output)

    # restore original shape of conv_output
    conv_output = conv_output.view(channels.shape[:-3] + conv_output.shape[-3:])
    # varname(conv_output)

    return conv_output

def ubuntu_emb_load(params):
    '''
    :param params:
    :return: numpy.ndarray
    '''
    default_params = {
        "path": None
    }
    default_params.update(params)
    assert default_params["path"] and os.path.isfile(default_params["path"])

    with open(default_params["path"], "rb") as f:
        embeddings = pickle.load(f, encoding="bytes")
    logger.info("embeddings:\t\t\t{}".format(embeddings.shape))
    return embeddings

def ubuntu_data_load(params):
    '''
    :param dataset_dir:
    :param max_sentence_len:
    :param max_num_utterance:
    :return: dict{str: numpy.ndarray}
    '''
    default_params = {
        "dataset_dir": None,
        "phase": "training",
        "training_files": ["responses.pkl", "utterances.pkl"],
        "evaluate_files": ["Evaluate.pkl"],
        "max_sentence_len": 50
    }
    default_params.update(params)
    assert default_params["dataset_dir"] and os.path.exists(default_params["dataset_dir"])

    np_dtype = np.int64
    if default_params["phase"] in ["training", "validation"]:
        training_files = default_params["training_files"]
        with open(os.path.join(default_params["dataset_dir"], training_files[0]), 'rb') as f:
            actions = pickle.load(f)
        with open(os.path.join(default_params["dataset_dir"], training_files[1]), 'rb') as f:
            history, true_utt = pickle.load(f)

        # prepare tf dataset
        history, history_len = multi_sequences_padding(history, max_sentence_len=default_params["max_sentence_len"])
        true_utt_len = np.array(get_sequences_length(true_utt, maxlen=default_params["max_sentence_len"]),
                                dtype=np_dtype)
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=default_params["max_sentence_len"]),
                            dtype=np_dtype)
        actions_len = np.array(get_sequences_length(actions, maxlen=default_params["max_sentence_len"]), dtype=np_dtype)
        actions = np.array(pad_sequences(actions, padding='post', maxlen=default_params["max_sentence_len"]),
                           dtype=np_dtype)
        history, history_len = np.array(history, dtype=np_dtype), np.array(history_len, dtype=np_dtype)

        return {
            "history": {"data": history, "type": "normal"},
            "history_len": {"data": history_len, "type": "normal"},
            "true_utt": {"data": true_utt, "type": "normal"},
            "true_utt_len": {"data": true_utt_len, "type": "normal"},
            "actions": {"data": actions, "type": "share"},
            "actions_len": {"data": actions_len, "type": "share"}
        }
    else:
        evaluate_files = default_params["evaluate_files"]
        with open(os.path.join(default_params["dataset_dir"], evaluate_files[0]), 'rb') as f:
            history, true_utt, labels = pickle.load(f)
        history, history_len = multi_sequences_padding(history, max_sentence_len=default_params["max_sentence_len"])
        true_utt_len = np.array(get_sequences_length(true_utt, maxlen=default_params["max_sentence_len"]),
                                dtype=np_dtype)
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=default_params["max_sentence_len"]),
                            dtype=np_dtype)
        history, history_len = np.array(history, dtype=np_dtype), np.array(history_len, dtype=np_dtype)
        labels = np.array(labels, dtype=np_dtype)

        return {
            "history": history,
            "history_len": history_len,
            "true_utt": true_utt,
            "true_utt_len": true_utt_len,
            "labels": labels
        }


def ubuntu_dataloader_gen(data_dict, params):
    default_params = {
        "device": None,
        "phase": "training",
        "batch_size": 10,
        "shuffle": False,
    }
    default_params.update(params)
    device = default_params["device"]
    # TODO: torch.tensor() or torch.from_numpy()
    # torch.tensor(): copy
    # torch.from_numpy(): share the storage
    if default_params["phase"] in ["training", "validation"]:
        utt_res = TensorDataset(
            torch.tensor(data_dict["history"], device=device),
            torch.tensor(data_dict["history_len"], device=device),
            torch.tensor(data_dict["true_utt"], device=device),
            torch.tensor(data_dict["true_utt_len"], device=device),
        )
        actions = TensorDataset(
            torch.tensor(data_dict["actions"], device=device),
            torch.tensor(data_dict["actions_len"], device=device)
        )
        return tuple([
            DataLoader(utt_res,
                       batch_size=default_params["batch_size"],
                       shuffle=default_params["shuffle"]),
            actions
        ])
    else:
        utt_res = TensorDataset(
            torch.tensor(data_dict["history"], device=device),
            torch.tensor(data_dict["history_len"], device=device),
            torch.tensor(data_dict["true_utt"], device=device),
            torch.tensor(data_dict["true_utt_len"], device=device),
            torch.tensor(data_dict["labels"], device=device)
        )
        return tuple([
            DataLoader(utt_res,
                       batch_size=default_params["batch_size"],
                       shuffle=False),
        ])


def ubuntu_data_gen(datas, params):
    default_params = {
        "phase": "training",
        "negative_samples": {
            "training": 1,
            "validation": 9
        }
    }
    default_params.update(params)
    device = datas[0][0].device
    if default_params["phase"] in ["training", "validation"]:
        assert len(datas) == 2
        negative_samples = default_params["negative_samples"][default_params["phase"]]
        utt_inputs, utt_len_inputs = [torch.cat([tensor] * (negative_samples + 1), dim=0) for tensor in
                                      datas[0][:2]]
        # varname(utt_inputs)
        # varname(utt_len_inputs)
        negative_indices = torch.randint(0, len(datas[1]), (negative_samples, datas[0][0].shape[0]),
                                         dtype=torch.int64, device=device)
        actions = [tensor.view(tuple([-1]) + tensor.shape[2:]) for tensor in datas[1][negative_indices]]
        # varname(actions)
        resp_inputs, resp_len_inputs = [torch.cat((tensor1, tensor2), dim=0) for tensor1, tensor2 in
                                        zip(datas[0][2:], actions)]
        # varname(resp_inputs)
        # varname(resp_len_inputs)
        targets = torch.cat((torch.ones(datas[0][0].shape[0], dtype=torch.int64, device=device),
                             torch.zeros(datas[0][0].shape[0] * negative_samples, dtype=torch.int64, device=device)),
                            dim=0)
        # varname(targets)

        return {
            "utt": utt_inputs,
            "utt_len": utt_len_inputs,
            "resp": resp_inputs,
            "resp_len": resp_len_inputs,
            "target": targets
        }
    else:
        assert len(datas) == 1
        return {
            "utt": datas[0][0],
            "utt_len": datas[0][1],
            "resp": datas[0][2],
            "resp_len": datas[0][3],
            "target": datas[0][4]
        }