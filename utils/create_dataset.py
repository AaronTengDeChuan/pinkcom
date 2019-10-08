# coding=utf-8

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import pickle

if __name__ == "__main__":
    base_work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.path.dirname(os.path.abspath(__file__))
    print (base_work_dir)
    print (cwd)
    sys.path.remove(cwd)
    sys.path.append(base_work_dir)
    print (sys.path)


import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from functools import reduce
from utils import utils
from utils.utils import varname
from utils.bert_reader import MtrsProcessor, convert_examples_to_features
import logging
import codecs

from utils.bert_tokenization import BertTokenizer
from nets.BertDownstream import BertForSequenceRepresentation

np_dtype = np.int64
np_float_dtype = np.float32
cuda_device = torch.device("cuda")
cpu_device = torch.device("cpu")
batch_size = None

def convert_examples_to_features_for_last_utterance(examples, max_seq_length, tokenizer, padding=True):
    inputs = {
        "id":[],
        "mask": [],
        "segment": [],
        "len": [],
    }
    for (ex_index, example) in enumerate(tqdm(examples, desc="Extract Last Utterance")):
        feature = convert_examples_to_features(example.context[-1:], max_seq_length, tokenizer, padding=padding)[0]
        inputs["id"].append(feature.input_ids)
        inputs["mask"].append(feature.input_mask)
        inputs["segment"].append(feature.segment_ids)
        inputs["len"].append(feature.input_length)

        if ex_index == 0:
            logger.info(" ".join(feature.input_tokens))
            logger.info(" ".join(map(str, feature.input_ids)))
            logger.info(" ".join(map(str, feature.input_mask)))
            logger.info(" ".join(map(str, feature.segment_ids)))
            logger.info(str(feature.input_length))

    tensor_dataset = TensorDataset(
        torch.tensor(np.array(inputs["id"], dtype=np_dtype)),
        torch.tensor(np.array(inputs["mask"], dtype=np_dtype)),
        torch.tensor(np.array(inputs["segment"], dtype=np_dtype)),
        torch.tensor(np.array(inputs["len"], dtype=np_dtype)),
    )
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    return dataloader

@torch.no_grad()
def predict(dataloader, model):
    preds = []
    keys = ['id', 'mask', 'segment', 'len']
    for batch, inputs in enumerate(tqdm(dataloader, desc="Get Representation")):
        inputs = dict([[key, value.to(device=cuda_device)] for key, value in zip(keys, inputs)])
        pred = model(inputs)
        preds.append(pred.to(device=cpu_device))
    return torch.cat(preds, dim=0)


def calculate_cosine_similarity(tensor_1, tensor_2, diag_mask=False):
    result = []
    argmax = []
    max = []
    step1 = batch_size
    step2 = batch_size
    for i in tqdm(range(0, len(tensor_1), step1), desc="Calculate Cosine Similarity"):
        left = tensor_1[i:i + step1]
        result.append([])
        for j in range(0, len(tensor_2), step2):
            result[-1].append(F.cosine_similarity(left.unsqueeze(1).to(device=cuda_device),
                                                  tensor_2[j:j + step2].unsqueeze(0).to(device=cuda_device), dim=-1))
        result[-1] = torch.cat(result[-1], dim=-1)
        if diag_mask:
            for j in range(0, len(left)):
                result[-1][j][i + j] = -1
        value, idx = torch.max(result[-1], dim=-1)
        max.extend(value.tolist())
        argmax.extend(idx.tolist())
        result[-1] = result[-1].to(device=cpu_device)

    return result, argmax, max


def write_new_file(argmax, max, input_examples, source_examples, params):
    for i in range(len(argmax)):
        output_file = params["outs"][i]
        step = params["steps"][i]
        with open(output_file, 'w', encoding="utf-8") as fo:
            for j in tqdm(range(len(argmax[i])), desc="Write New File"):
                source_example = source_examples[argmax[i][j]]
                assert int(source_example.label) == 1
                for k in range(step):
                    example = input_examples[i][j * step + k]
                    fo.write("\t".join(
                        [example.label] + example.context + [source_example.response, example.response] +
                        [str(argmax[i][j]), str(max[i][j]), source_example.context[-1]]) + "\n")


def path_join(dir, files):
    for i in range(len(files)):
        files[i] = os.path.join(dir, files[i])


def create_dataset_for_context_retrieval(config):
    global batch_size
    logger.info("Create model")
    model = BertForSequenceRepresentation({"bert_model_dir": config["data"]["bert_model_dir"]}).to(device=cuda_device)
    tokenizer = BertTokenizer.from_pretrained(config["data"]["bert_model_dir"], do_lower_case=False)

    logger.info("Read files")
    processor = MtrsProcessor()
    dataset_dir = config["data"]["dataset_dir"]
    target_dir = config["data"]["target_dir"]
    max_seq_length = config["data"]["max_seq_length"]
    batch_size = config["global"]["batch_size"]

    training_params = config["data"]["training"]
    validation_params = config["data"]["validation"]
    evaluation_params = config["data"]["evaluation"]
    path_join(dataset_dir, training_params["files"])
    path_join(dataset_dir, validation_params["files"])
    path_join(dataset_dir, evaluation_params["files"])
    path_join(target_dir, training_params["vecs"])
    path_join(target_dir, validation_params["vecs"])
    path_join(target_dir, evaluation_params["vecs"])
    path_join(target_dir, training_params["outs"])
    path_join(target_dir, validation_params["outs"])
    path_join(target_dir, evaluation_params["outs"])

    train_examples = [processor.get_train_examples(train_file) for train_file in training_params["files"]]
    dev_examples = [processor.get_dev_examples(dev_file) for dev_file in validation_params["files"]]
    test_examples = [processor.get_test_examples(test_file) for test_file in evaluation_params["files"]]

    logger.info("Construct dataloaders")
    train_dataloaders = [convert_examples_to_features_for_last_utterance(examples[::step], max_seq_length, tokenizer)
                         for examples, step in zip(train_examples, training_params["steps"])]
    dev_dataloaders = [convert_examples_to_features_for_last_utterance(examples[::step], max_seq_length, tokenizer)
                         for examples, step in zip(dev_examples, validation_params["steps"])]
    test_dataloaders = [convert_examples_to_features_for_last_utterance(examples[::step], max_seq_length, tokenizer)
                       for examples, step in zip(test_examples, evaluation_params["steps"])]

    logger.info("Predict and save representations")
    train_preds = [predict(dataloader, model) for dataloader in train_dataloaders]
    logger.info("train_preds: {}".format(str(train_preds[0].shape)))
    for i in range(len(train_preds)):
        pickle.dump(train_preds[i], open(training_params["vecs"][i], "wb"))
    dev_preds = [predict(dataloader, model) for dataloader in dev_dataloaders]
    logger.info("dev_preds: {}".format(str(dev_preds[0].shape)))
    for i in range(len(dev_preds)):
        pickle.dump(dev_preds[i], open(validation_params["vecs"][i], "wb"))
    test_preds = [predict(dataloader, model) for dataloader in test_dataloaders]
    logger.info("test_preds: {}".format(str(test_preds[0].shape)))
    for i in range(len(test_preds)):
        pickle.dump(test_preds[i], open(evaluation_params["vecs"][i], "wb"))

    logger.info("Calculate cosine similarity and write new files")
    assert len(train_preds) == 1 and len(dev_preds) == 1
    train_step = training_params["steps"][0]
    _, train_argmax, train_max = calculate_cosine_similarity(train_preds[0], train_preds[0], diag_mask=True)
    logger.info("train_argmax: {} \ttrain_max: {}".format(len(train_argmax), len(train_max)))
    write_new_file([train_argmax], [train_max], train_examples, train_examples[0][::train_step], training_params)
    _, dev_argmax, dev_max = calculate_cosine_similarity(dev_preds[0], train_preds[0])
    logger.info("dev_argmax: {} \tdev_max: {}".format(len(dev_argmax), len(dev_max)))
    write_new_file([dev_argmax], [dev_max], dev_examples, train_examples[0][::train_step], validation_params)
    _, test_argmax, test_max = zip(*[calculate_cosine_similarity(test_preds[i], train_preds[0]) for i in range(len(test_preds))])
    logger.info("test_argmax: {} \ttest_max: {}".format(len(test_argmax[0]), len(test_max[0])))
    write_new_file(test_argmax, test_max, test_examples, train_examples[0][::train_step], evaluation_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Dataset.')
    parser.add_argument('--config', type=str, default="", help='Location of config file')
    args = parser.parse_args()
    config = utils.read_config(args.config)

    logger = utils.create_logger(config["global"]["log_file"])

    create_dataset_for_context_retrieval(config)


