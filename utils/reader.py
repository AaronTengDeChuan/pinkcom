# coding: utf-8

import numpy as np
import pickle
import os

from utils import utils

from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = utils.get_logger()


# TODO: Related to Auxiliary Functions

def split_dialogue_history(data, EOS_ID):
    new_data = []
    for context in data:
        turns = [[]]
        for _id in context:
            if _id != EOS_ID:
                turns[-1].append(_id)
            else:
                turns.append([])
        if turns[-1] == [] and len(turns) > 1:
            turns.pop()
        new_data.append(turns)
    return new_data


# TODO: Related to Embedding Loading

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
    logger.info("embeddings:\t{}\t{}".format(embeddings.shape, embeddings.dtype))
    return embeddings


# TODO: Related to DAM

def DAM_ubuntu_data_load(params):
    default_params = {
        "dataset_dir": None,
        "phase": "training",
        "training_files": ["training_data.pkl"],
        "evaluate_files": ["evaluate_data.pkl"],
        "eos_id": 28270,
        "max_sentence_len": 50
    }
    default_params.update(params)
    assert default_params["dataset_dir"] and os.path.exists(default_params["dataset_dir"])
    np_dtype = np.int64
    training_files = default_params["training_files"]
    evaluate_files = default_params["evaluate_files"]
    inputs = {}
    if default_params["phase"] in ["training", "validation"]:
        with open(os.path.join(default_params["dataset_dir"], training_files[0]), 'rb') as f:
            inputs = pickle.load(f)
       #      training_data, validation_data, evaluate_data = pickle.load(f)
       #  with open(os.path.join(default_params["dataset_dir"], "evaluate_data.pkl"), 'wb') as f:
       #      pickle.dump(evaluate_data, f)
       #  inputs['c'] = validation_data['c'] + training_data['c']
       #  inputs['c'] = split_dialogue_history(inputs['c'], default_params["eos_id"])
       #  inputs['r'] = validation_data['r'] + training_data['r']
       #  inputs['y'] = validation_data['y'] + training_data['y']
       #  with open(os.path.join(default_params["dataset_dir"], "training_data.pkl"), 'wb') as f:
       #      pickle.dump(inputs, f)
    else:
        with open(os.path.join(default_params["dataset_dir"], evaluate_files[0]), 'rb') as f:
            inputs = pickle.load(f)
       #      evaluate_data = pickle.load(f)
       #  inputs['c'] = evaluate_data['c']
       #  inputs['c'] = split_dialogue_history(inputs['c'], default_params["eos_id"])
       #  inputs['r'] = evaluate_data['r']
       #  inputs['y'] = evaluate_data['y']
       #  with open(os.path.join(default_params["dataset_dir"], "evaluate_data.pkl"), 'wb') as f:
       #      pickle.dump(inputs, f)

    # prepare tf dataset
    history, history_len = utils.multi_sequences_padding(inputs['c'], max_sentence_len=default_params["max_sentence_len"])
    true_utt_len = np.array(utils.get_sequences_length(inputs['r'], maxlen=default_params["max_sentence_len"]),
                            dtype=np_dtype)
    true_utt = np.array(pad_sequences(inputs['r'], padding='post', maxlen=default_params["max_sentence_len"]),
                        dtype=np_dtype)
    history, history_len = np.array(history, dtype=np_dtype), np.array(history_len, dtype=np_dtype)
    labels = np.array(inputs['y'], dtype=np_dtype)

    if default_params["phase"] in ["training", "validation"]:
        return {
            "history": {"data": history, "type": "normal"},
            "history_len": {"data": history_len, "type": "normal"},
            "true_utt": {"data": true_utt, "type": "normal"},
            "true_utt_len": {"data": true_utt_len, "type": "normal"},
            "labels": {"data": labels, "type": "normal"}
        }
    else:
        return {
            "history": history,
            "history_len": history_len,
            "true_utt": true_utt,
            "true_utt_len": true_utt_len,
            "labels": labels
        }


def DAM_ubuntu_dataloader_gen(data_dict, params):
    default_params = {
        "device": None,
        "phase": "training",
        "batch_size": {},
        "shuffle": {
            "training": True,
            "validation": False
        }
    }
    default_params.update(params)
    if "validation" not in default_params["batch_size"]:
        default_params["batch_size"]["validation"] = default_params["batch_size"]["training"]
    if "evaluate" not in default_params["batch_size"]:
        default_params["batch_size"]["evaluate"] = default_params["batch_size"]["training"]
    utt_res_labels = TensorDataset(
        torch.tensor(data_dict["history"]),
        torch.tensor(data_dict["history_len"]),
        torch.tensor(data_dict["true_utt"]),
        torch.tensor(data_dict["true_utt_len"]),
        torch.tensor(data_dict["labels"])
    )

    shuffle = False
    if default_params["phase"] in ["training", "validation"]:
        shuffle = default_params["shuffle"][default_params["phase"]]

    return tuple([
        DataLoader(utt_res_labels,
                   batch_size=default_params["batch_size"][default_params["phase"]],
                   shuffle=shuffle),
    ])


def DAM_ubuntu_data_gen(datas, params):
    default_params = {
        "device": None
    }
    default_params.update(params)
    device = default_params["device"]
    assert len(datas) == 1
    return {
        "utt": datas[0][0].to(device=device),
        "utt_len": datas[0][1].to(device=device),
        "resp": datas[0][2].to(device=device),
        "resp_len": datas[0][3].to(device=device),
        "target": datas[0][4].to(device=device)
    }


# TODO: Related to SMN

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
        history, history_len = utils.multi_sequences_padding(history, max_sentence_len=default_params["max_sentence_len"])
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=default_params["max_sentence_len"]),
                                dtype=np_dtype)
        true_utt = np.array(pad_sequences(true_utt, padding='post', maxlen=default_params["max_sentence_len"]),
                            dtype=np_dtype)
        actions_len = np.array(utils.get_sequences_length(actions, maxlen=default_params["max_sentence_len"]), dtype=np_dtype)
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
        history, history_len = utils.multi_sequences_padding(history, max_sentence_len=default_params["max_sentence_len"])
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=default_params["max_sentence_len"]),
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
        "batch_size": {},
        "shuffle": {
            "training": True,
            "validation": False
        }
    }
    default_params.update(params)
    if "validation" not in default_params["batch_size"]:
        default_params["batch_size"]["validation"] = default_params["batch_size"]["training"]
    if "evaluate" not in default_params["batch_size"]:
        default_params["batch_size"]["evaluate"] = default_params["batch_size"]["training"]
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
                       batch_size=default_params["batch_size"][default_params["phase"]],
                       shuffle=default_params["shuffle"][default_params["phase"]]),
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
                       batch_size=default_params["batch_size"][default_params["phase"]],
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