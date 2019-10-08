# coding: utf-8

import numpy as np
import pickle
import os
import sys
import math
import random
from collections import Counter
from functools import reduce
from tqdm import tqdm

import spacy
from spacy.tokens import Doc

from utils.bert_tokenization import BertTokenizer

if __name__ == "__main__":
    base_work_dir = os.path.dirname(os.getcwd())
    print (base_work_dir)
    sys.path.append(base_work_dir)
    sys.path.remove(os.getcwd())

from utils import utils
from utils.utils import pad_3d_sequences as pad_sequences
# from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

logger = utils.get_logger()

# global variables
LANGUAGE = None
SpaCy_LANGUAGES = ["en"]
USE_SpaCy = None


# TODO: Related to Auxiliary Functions

def split_dialogue_history(data, EOS_ID):
    new_data = []
    for context in tqdm(data, desc="Split Dialog History"):
        turns = [[]]
        for _id in context:
            if _id != EOS_ID:
                turns[-1].append(_id)
            else:
                if len(turns[-1]) > 0: turns.append([])
        if turns[-1] == [] and len(turns) > 1:
            turns.pop()
        new_data.append(turns)
    return new_data


def load_vocab(word2id_file):
    word2id = {}
    id2word = {}
    lines = []
    with open(word2id_file, "r", encoding="utf-8") as f:
        line_number = 1
        line = f.readline().strip()
        lines.append(line.split("\t"))
        while line:
            try:
                line_number += 1
                line = f.readline().strip()
                if line: lines.append(line.split("\t"))
            except:
                print (line_number)
    count_field = len(lines[0])
    lines = list(map(lambda x: x[0] if len(x) == 1 else x, lines))
    if count_field == 1:
        assert len(lines) % 2 == 0
        word2id = dict(zip(lines[0::2], list(map(int, lines[1::2]))))
        id2word = dict(zip(list(map(int, lines[1::2])), lines[0::2]))
    elif count_field == 2:
        for line in lines:
            assert len(line) == 2
            word2id[line[0]] = int(line[1])
            id2word[int(line[1])] = line[0]
    else:
        assert False, "Incorrect form in vocabulary file '{}'.".format(word2id_file)
    return word2id, id2word


def doc_gen(ids, id2word, name="document", use_element_fn=True):
    nlp = spacy.load(LANGUAGE if USE_SpaCy else "en", disable=["parser"])
    if use_element_fn:
        element_fn = lambda id: id2word[id] if id in id2word else ""
    else:
        element_fn = lambda token: token
    list_fn = [lambda l: Doc(nlp.vocab, words=l)]

    print ("Processing {} ...".format(name))

    num_exception = [0]
    docs = utils.list_process_fn(tqdm(ids, desc="Generate Initial Docs from IDs"), element_fn, list_fn,
                                 num_exception=num_exception, exception_process=True)[0]
    logger.info("During generating docs for {}, {} exceptions occurred.".format(name, num_exception[0]))

    if not USE_SpaCy: return docs
    # tagging, entity recognization, and lemmazation
    lengths = None
    flatten_docs = docs
    if not isinstance(docs[0], Doc):
        flatten_docs, lengths = utils.flatten_list(tqdm(docs, desc="Flatten the list of Docs"))
    flatten_docs = [doc for doc in nlp.tagger.pipe(tqdm(flatten_docs, desc="Tag Docs"), batch_size=64, n_threads=20)]
    flatten_docs = [doc for doc in
                    nlp.entity.pipe(tqdm(flatten_docs, desc="Recognize Name Entity"), batch_size=64, n_threads=20)]
    restored_docs = flatten_docs
    if not isinstance(docs[0], Doc):
        restored_docs, start_index = utils.restore_list(flatten_docs, tqdm(lengths, desc="Restore the list of Docs"), 0)
        assert start_index == len(flatten_docs)
    return restored_docs


def feature_gen(C_docs, R_docs, no_match=False):
    """
    :param C_docs:  [batch, num_turn, sentence_len]
    :param R_docs:  [batch, sentence_len]
    :param no_match:
    :return:
        :R_tags:    [batch, sentence_len]
        :R_ents:    [batch, sentence_len]
        :R_features:    [batch, num_turn, sentence_len, num_feature]
    """
    R_tags = [[w.tag_ for w in doc] for doc in R_docs]
    R_ents = [[w.ent_type_ for w in doc] for doc in R_docs]
    R_features = []
    print_count = 1

    phar = tqdm(total=len(C_docs), desc="Generate Features")

    for context, response in zip(C_docs, R_docs):
        counter_ = Counter(w.text.lower() for w in response)
        total = sum(counter_.values())
        term_freq = [counter_[w.text.lower()] / total for w in response]

        if no_match:
            R_features.append(list(zip(term_freq)))
        else:
            turn_features = []
            for turn in context:
                turn_word = {w.text for w in turn}
                turn_lower = {w.text.lower() for w in turn}
                turn_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in turn}
                match_origin = [w.text in turn_word for w in response]
                match_lower = [w.text.lower() in turn_lower for w in response]
                match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in turn_lemma for w in response]
                turn_features.append(list(zip(match_origin, match_lower, match_lemma, term_freq)))
            if print_count > 0:
                for id, turn in enumerate(context):
                    print ("\nTurn {}: ".format(id), [w.text for w in turn])
                    print ("Origin Set: ", {w.text for w in turn})
                    print ("Lower Set: ", {w.text.lower() for w in turn})
                    print ("Lemma Set: ", {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in turn})
                    for i in range(len(response)):
                        print("{:<15}|{:<15}|{:<15}| {:^5} | {:^5} | {:^5} | {:^5}".format(response[i].text,
                                                                                           response[i].text.lower(),
                                                                                           response[i].lemma_,
                                                                                           *turn_features[id][i]))
                print ("\n" + "-" * 50)
                print_count -= 1
            R_features.append(turn_features)

        phar.update(1)

    phar.close()

    return R_tags, R_ents, R_features


def generate_bert_input_with_alignment(ids, id2word, params):
    assert "bert_model_dir" in params and params["bert_model_dir"] is not None
    assert "do_lower_case" in params and params["do_lower_case"] is not None
    tokenizer = BertTokenizer.from_pretrained(params["bert_model_dir"], do_lower_case=params["do_lower_case"])
    bert_max_sentence_len = params["bert_max_sentence_len"]

    element_fn = lambda id: tokenizer.tokenize(id2word[id]) if id in id2word else ["[UNK]"]
    # a set of functions for type 'list': id, mask, segment, alignment
    list_fn_list = [
        lambda l: tokenizer.convert_tokens_to_ids(reduce(lambda x, y: x + y, l, ["[CLS]"])[:bert_max_sentence_len - 1] + ["[SEP]"]),
        lambda l: [1] * min(reduce(lambda x, y: x + len(y), l, 2), bert_max_sentence_len),
        lambda l: [0] * min(reduce(lambda x, y: x + len(y), l, 2), bert_max_sentence_len),
        lambda l: reduce(lambda x, y: x + [min(x[-1] + len(y), bert_max_sentence_len - 1)], l[:-1], [1]) if len(l) != 0  else []
    ]

    num_exception = [0]
    res = utils.list_process_fn(tqdm(ids, desc="Generate four bert inputs of id, mask, segment, and alignment."),
                                element_fn, list_fn_list, num_exception=num_exception, exception_process=True)
    logger.info("During generating bert id, {} exceptions occurred.".format(num_exception[0]))
    inputs = {}
    inputs["id"] = res[0]
    inputs["mask"] = res[1]
    inputs["segment"] = res[2]
    inputs["alignment"] = res[3]

    return inputs


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
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float64)
    logger.info("embeddings:\t{}\t{}".format(embeddings.shape, embeddings.dtype))
    return embeddings


# TODO: Label Smoothing

def DAM_ubuntu_data_load(params):
    default_params = {
        "dataset_dir": None,
        "phase": "training",
        "training_files": ["data.pkl"],
        "evaluate_files": ["test_data.pkl"],
        "vocabulary_file": "word2id",
        "vocabulary_size": None,
        "eos_id": 28270,
        "empty_sequence_length": 0,
        "max_num_utterance": 10,
        "max_sentence_len": 50,
        "language": "en",

        "use_bert_embeddings": None,
        "bert_model_dir": None,
        "do_lower_case": None,
        "bert_max_sentence_len": 50
    }
    default_params.update(params)
    global LANGUAGE, USE_SpaCy
    LANGUAGE = default_params["language"]
    USE_SpaCy = LANGUAGE in SpaCy_LANGUAGES
    assert default_params["dataset_dir"] and os.path.exists(default_params["dataset_dir"])
    np_dtype = np.int64
    np_float_dtype = np.float32
    training_files = default_params["training_files"]
    evaluate_files = default_params["evaluate_files"]
    word2id_file = os.path.join(default_params["dataset_dir"], default_params["vocabulary_file"])

    use_bert_embeddings = default_params["use_bert_embeddings"]

    def process():
        inputs = {}
        keys = ["c", "r", "y", "r_tag", "r_ent", "r_feature", "c_bert", "r_bert"]
        save_flag = True
        cache_file = None
        if default_params["phase"] in ["training", "validation"]:
            cache_file = os.path.join(default_params["dataset_dir"], "training_" + training_files[0])
            if os.path.isfile(cache_file):
                inputs = pickle.load(open(cache_file, "rb"))
                save_flag = not ((use_bert_embeddings and all(key in inputs for key in keys))
                                 or (not use_bert_embeddings and all(key in inputs for key in keys[:6])))
            if not all(key in inputs for key in keys[:3]):
                with open(os.path.join(default_params["dataset_dir"], training_files[0]), 'rb') as f:
                    training_data, validation_data, evaluate_data = pickle.load(f)
                    assert len(training_data['c']) == len(training_data['r']) \
                           and len(training_data['r']) == len(training_data['y'])
                    assert len(validation_data['c']) == len(validation_data['r']) \
                           and len(validation_data['r']) == len(validation_data['y'])
                    assert len(evaluate_data['c']) == len(evaluate_data['r']) \
                           and len(evaluate_data['r']) == len(evaluate_data['y'])
                with open(os.path.join(default_params["dataset_dir"], evaluate_files[0]), 'wb') as f:
                    pickle.dump(evaluate_data, f)
                inputs['c'] = validation_data['c'] + training_data['c']
                inputs['c'] = split_dialogue_history(inputs['c'], default_params["eos_id"])
                inputs['r'] = validation_data['r'] + training_data['r']
                inputs['y'] = validation_data['y'] + training_data['y']
        else:
            cache_file = os.path.join(default_params["dataset_dir"], "evaluate_" + evaluate_files[0][5:])
            if os.path.isfile(cache_file):
                inputs = pickle.load(open(cache_file, "rb"))
                save_flag = not ((use_bert_embeddings and all(key in inputs for key in keys))
                                 or (not use_bert_embeddings and all(key in inputs for key in keys[:6])))
            if not all(key in inputs for key in keys[:3]):
                with open(os.path.join(default_params["dataset_dir"], evaluate_files[0]), 'rb') as f:
                    evaluate_data = pickle.load(f)
                inputs['c'] = evaluate_data['c']
                inputs['c'] = split_dialogue_history(inputs['c'], default_params["eos_id"])
                inputs['r'] = evaluate_data['r']
                inputs['y'] = evaluate_data['y']

        if save_flag: word2id, id2word = load_vocab(word2id_file)
        if not all(key in inputs for key in keys[3:6]):
            inputs["r_tag"], inputs["r_ent"], inputs["r_feature"] = feature_gen(
                doc_gen(inputs['c'], id2word, name=default_params["phase"]+"/context", use_element_fn=True),
                doc_gen(inputs['r'], id2word, name=default_params["phase"]+"/response", use_element_fn=True),
                no_match=False)
        if use_bert_embeddings:
            if "c_bert" not in inputs: inputs["c_bert"] = generate_bert_input_with_alignment(inputs['c'], id2word, params)
            if "r_bert" not in inputs: inputs["r_bert"] = generate_bert_input_with_alignment(inputs['r'], id2word, params)

        if save_flag:
            with open(cache_file, 'wb') as f: pickle.dump(inputs, f)

        return inputs


    def numpy_process():
        max_batch_size = 100000
        np_cache_file = None
        results = {}
        keys = ["history_np_list", "history_len", "response_features_np_list", "true_utt", "true_utt_len", "labels",
                "history_bert_id_np_list", "history_bert_len", "history_bert_mask_np_list",
                "history_bert_segment_np_list", "history_alignment_np_list",
                "true_utt_bert_id", "true_utt_bert_len", "true_utt_bert_mask",
                "true_utt_bert_segment", "true_utt_alignment"]

        if default_params["phase"] in ["training", "validation"]:
            np_cache_file = os.path.join(default_params["dataset_dir"], "numpy_training_" + training_files[0])
        else:
            np_cache_file = os.path.join(default_params["dataset_dir"], "numpy_evaluate_" + evaluate_files[0][5:])

        if os.path.isfile(np_cache_file):
            results = pickle.load(open(np_cache_file, 'rb'))
        if (use_bert_embeddings and all(key in results for key in keys)) or (not use_bert_embeddings and all(key in results for key in keys[:6])):
            pass
        else:
            inputs = process()
            if "history_np_list" not in results:
                history, history_len = utils.multi_sequences_padding(tqdm(inputs['c'], desc="Sequence Padding"),
                                                                     max_num_utterance=default_params[
                                                                         "max_num_utterance"],
                                                                     max_sentence_len=default_params[
                                                                         "max_sentence_len"])
                results["history_np_list"] = [np.array(history[i: i + max_batch_size], dtype=np_dtype) for i in
                                              range(0, len(history), max_batch_size)]
                results["history_len"] = np.array(history_len, dtype=np_dtype)
            if "response_features_np_list" not in results:
                feature_len = len(inputs["r_feature"][0][0][0])
                response_features, _ = utils.multi_sequences_padding(tqdm(inputs["r_feature"], desc="Feature Sequence Padding"),
                                                                     max_num_utterance=default_params["max_num_utterance"],
                                                                     max_sentence_len=default_params["max_sentence_len"],
                                                                     padding_element=[0] * feature_len)
                results["response_features_np_list"] = [
                    np.array(response_features[i:i + max_batch_size], dtype=np_float_dtype) for i in
                    range(0, len(response_features), max_batch_size)]

            if "true_utt_len" not in results: results["true_utt_len"] = np.array(utils.get_sequences_length(inputs['r'], maxlen=default_params["max_sentence_len"]), dtype=np_dtype)
            if "true_utt" not in results: results["true_utt"]= np.array(pad_sequences(inputs['r'], default_params["max_sentence_len"], padding='post'), dtype=np_dtype)
            if "labels" not in results: results["labels"]= np.array(inputs['y'], dtype=np_dtype)

            if use_bert_embeddings:
                if "history_bert_id_np_list" not in results:
                    history_bert_id, history_bert_len = utils.multi_sequences_padding(
                        tqdm(inputs["c_bert"]["id"], desc="Bert Sequence Padding"),
                        max_num_utterance=default_params["max_num_utterance"],
                        max_sentence_len=default_params["bert_max_sentence_len"])
                    results["history_bert_id_np_list"] = [
                        np.array(history_bert_id[i: i + max_batch_size], dtype=np_dtype)
                        for i in range(0, len(history_bert_id), max_batch_size)]
                    results["history_bert_len"] = np.array(history_bert_len, dtype=np_dtype)
                if "history_bert_mask_np_list" not in results:
                    history_bert_mask, _ = utils.multi_sequences_padding(
                        tqdm(inputs["c_bert"]["mask"], desc="Bert Mask Padding"),
                        max_num_utterance=default_params["max_num_utterance"],
                        max_sentence_len=default_params["bert_max_sentence_len"])
                    results["history_bert_mask_np_list"] = [
                        np.array(history_bert_mask[i: i + max_batch_size], dtype=np_dtype)
                        for i in range(0, len(history_bert_mask), max_batch_size)]
                if "history_bert_segment_np_list" not in results:
                    history_bert_segment, _ = utils.multi_sequences_padding(
                        tqdm(inputs["c_bert"]["segment"], desc="Bert Segment Padding"),
                        max_num_utterance=default_params["max_num_utterance"],
                        max_sentence_len=default_params["bert_max_sentence_len"])
                    results["history_bert_segment_np_list"] = [
                        np.array(history_bert_segment[i: i + max_batch_size], dtype=np_dtype)
                        for i in range(0, len(history_bert_segment), max_batch_size)]
                if "history_alignment_np_list" not in results:
                    history_alignment, _ = utils.multi_sequences_padding(
                        tqdm(inputs["c_bert"]["alignment"], desc="Alignment Padding"),
                        max_num_utterance=default_params["max_num_utterance"],
                        max_sentence_len=default_params["max_sentence_len"])
                    results["history_alignment_np_list"] = [
                        np.array(history_alignment[i: i + max_batch_size], dtype=np_dtype)
                        for i in range(0, len(history_alignment), max_batch_size)]

                if "true_utt_bert_id" not in results: results["true_utt_bert_id"] = np.array(pad_sequences(inputs["r_bert"]["id"], default_params["bert_max_sentence_len"], padding='post'), dtype=np_dtype)
                if "true_utt_bert_len" not in results: results["true_utt_bert_len"] = np.array(utils.get_sequences_length(inputs["r_bert"]["id"], default_params["bert_max_sentence_len"]), dtype=np_dtype)
                if "true_utt_bert_mask" not in results: results["true_utt_bert_mask"] = np.array(pad_sequences(inputs["r_bert"]["mask"], default_params["bert_max_sentence_len"], padding='post'), dtype=np_dtype)
                if "true_utt_bert_segment" not in results: results["true_utt_bert_segment"] = np.array(pad_sequences(inputs["r_bert"]["segment"], default_params["bert_max_sentence_len"], padding='post'), dtype=np_dtype)
                if "true_utt_alignment" not in results: results["true_utt_alignment"] = np.array(pad_sequences(inputs["r_bert"]["alignment"], default_params["max_sentence_len"], padding='post'), dtype=np_dtype)

            with open(np_cache_file, 'wb') as f:
                pickle.dump(results, f)

        results["history"] = np.concatenate(results["history_np_list"], axis=0)
        results["response_features"] = np.concatenate(results["response_features_np_list"], axis=0)
        if use_bert_embeddings:
            results["history_bert_id"] = np.concatenate(results["history_bert_id_np_list"], axis=0)
            results["history_bert_mask"] = np.concatenate(results["history_bert_mask_np_list"], axis=0)
            results["history_bert_segment"] = np.concatenate(results["history_bert_segment_np_list"], axis=0)
            results["history_alignment"] = np.concatenate(results["history_alignment_np_list"], axis=0)

        return results


    # prepare numpy dataset
    results = numpy_process()
    history, history_len, response_features, true_utt, true_utt_len, labels = \
        results["history"], results["history_len"], results["response_features"], results["true_utt"], \
        results["true_utt_len"], results["labels"]

    vocabulary_size = default_params["vocabulary_size"]
    if vocabulary_size is not None:
        logger.warn("Replace ids exceeding {} with the unknown word id.".format(vocabulary_size -1 ))
        history = np.where(history < vocabulary_size, history, 1)
        true_utt = np.where(true_utt < vocabulary_size, true_utt, 1)

    utt_num = np.sum(history_len != 0, axis=-1, dtype=np_dtype)

    utils.varname(history)
    utils.varname(utt_num)
    utils.varname(history_len)
    utils.varname(response_features)
    utils.varname(true_utt)
    utils.varname(true_utt_len)
    utils.varname(labels)
    if use_bert_embeddings:
        history_bert_id, history_bert_len, history_bert_mask, history_bert_segment, history_alignment, \
        true_utt_bert_id, true_utt_bert_len, true_utt_bert_mask, true_utt_bert_segment, true_utt_alignment = \
            results["history_bert_id"], results["history_bert_len"], results["history_bert_mask"], \
            results["history_bert_segment"], results["history_alignment"], \
            results["true_utt_bert_id"], results["true_utt_bert_len"], results["true_utt_bert_mask"], \
            results["true_utt_bert_segment"], results["true_utt_alignment"]

        utils.varname(history_bert_id)
        utils.varname(history_bert_len)
        utils.varname(history_bert_mask)
        utils.varname(history_bert_segment)
        utils.varname(history_alignment)
        utils.varname(true_utt_bert_id)
        utils.varname(true_utt_bert_len)
        utils.varname(true_utt_bert_mask)
        utils.varname(true_utt_bert_segment)
        utils.varname(true_utt_alignment)

    # address an issue where sent_len == 0
    empty_sequence_length = default_params["empty_sequence_length"]
    if empty_sequence_length != 0:
        assert empty_sequence_length > 0
        history_len = np.ma.array(history_len, mask=history_len==0, fill_value=empty_sequence_length).filled()

    print ("Number of empty sequences in context:   {}".format(np.sum(history_len == 0)))
    print ("Number of empty sequences in response:  {}".format(np.sum(true_utt_len == 0)))

    if default_params["phase"] in ["training", "validation"]:
        dict = {
            "utt": {"data": history, "type": "normal"},
            "utt_num": {"data": utt_num, "type": "normal"},
            "utt_len": {"data": history_len, "type": "normal"},
            "response_features": {"data": response_features, "type": "normal"},
            "resp": {"data": true_utt, "type": "normal"},
            "resp_len": {"data": true_utt_len, "type": "normal"},
            "target": {"data": labels, "type": "normal"}
        }
        if use_bert_embeddings:
            dict.update({
                "history_bert_id": {"data": history_bert_id, "type": "normal"},
                "history_bert_len": {"data": history_bert_len, "type": "normal"},
                "history_bert_mask": {"data": history_bert_mask, "type": "normal"},
                "history_bert_segment": {"data": history_bert_segment, "type": "normal"},
                "history_alignment": {"data": history_alignment, "type": "normal"},
                "true_utt_bert_id": {"data": true_utt_bert_id, "type": "normal"},
                "true_utt_bert_len": {"data": true_utt_bert_len, "type": "normal"},
                "true_utt_bert_mask": {"data": true_utt_bert_mask, "type": "normal"},
                "true_utt_bert_segment": {"data": true_utt_bert_segment, "type": "normal"},
                "true_utt_alignment": {"data": true_utt_alignment, "type": "normal"}
            })
        return dict
    else:
        dict = {
            "utt": history,
            "utt_num": utt_num,
            "utt_len": history_len,
            "response_features": response_features,
            "resp": true_utt,
            "resp_len": true_utt_len,
            "target": labels
        }
        if use_bert_embeddings:
            dict.update({
                "history_bert_id": history_bert_id,
                "history_bert_len": history_bert_len,
                "history_bert_mask": history_bert_mask,
                "history_bert_segment": history_bert_segment,
                "history_alignment": history_alignment,
                "true_utt_bert_id": true_utt_bert_id,
                "true_utt_bert_len": true_utt_bert_len,
                "true_utt_bert_mask": true_utt_bert_mask,
                "true_utt_bert_segment": true_utt_bert_segment,
                "true_utt_alignment": true_utt_alignment
            })
        return dict


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

    assert "data_names" in default_params and default_params["data_names"] is not None
    utt_res_labels = TensorDataset(*[torch.tensor(data_dict[data_name]) for data_name in default_params["data_names"]])

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

    assert "data_names" in default_params and default_params["data_names"] is not None

    return dict(
        [[data_name, datas[0][i].to(device=device)] for i, data_name in enumerate(default_params["data_names"])])


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
        "max_num_utterance": 10,
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
        history, history_len = utils.multi_sequences_padding(history,
                                                             max_num_utterance=default_params["max_num_utterance"],
                                                             max_sentence_len=default_params["max_sentence_len"])
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=default_params["max_sentence_len"]),
                                dtype=np_dtype)
        true_utt = np.array(pad_sequences(true_utt, default_params["max_sentence_len"], padding='post'),
                            dtype=np_dtype)
        actions_len = np.array(utils.get_sequences_length(actions, maxlen=default_params["max_sentence_len"]), dtype=np_dtype)
        actions = np.array(pad_sequences(actions, default_params["max_sentence_len"], padding='post'),
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
        history, history_len = utils.multi_sequences_padding(history,
                                                             max_num_utterance=default_params["max_num_utterance"],
                                                             max_sentence_len=default_params["max_sentence_len"])
        true_utt_len = np.array(utils.get_sequences_length(true_utt, maxlen=default_params["max_sentence_len"]),
                                dtype=np_dtype)
        true_utt = np.array(pad_sequences(true_utt, default_params["max_sentence_len"], padding='post'),
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
    # TODO: torch.tensor() or torch.from_numpy()
    # torch.tensor(): copy
    # torch.from_numpy(): share the storage
    if default_params["phase"] in ["training", "validation"]:
        utt_res = TensorDataset(
            torch.tensor(data_dict["history"]),
            torch.tensor(data_dict["history_len"]),
            torch.tensor(data_dict["true_utt"]),
            torch.tensor(data_dict["true_utt_len"]),
        )
        actions = TensorDataset(
            torch.tensor(data_dict["actions"]),
            torch.tensor(data_dict["actions_len"])
        )
        return tuple([
            DataLoader(utt_res,
                       batch_size=default_params["batch_size"][default_params["phase"]],
                       shuffle=default_params["shuffle"][default_params["phase"]]),
            actions
        ])
    else:
        utt_res = TensorDataset(
            torch.tensor(data_dict["history"]),
            torch.tensor(data_dict["history_len"]),
            torch.tensor(data_dict["true_utt"]),
            torch.tensor(data_dict["true_utt_len"]),
            torch.tensor(data_dict["labels"])
        )
        return tuple([
            DataLoader(utt_res,
                       batch_size=default_params["batch_size"][default_params["phase"]],
                       shuffle=False),
        ])


def ubuntu_data_gen(datas, params):
    default_params = {
        "device": None,
        "phase": "training",
        "negative_samples": {
            "training": 1,
            "validation": 9
        }
    }
    default_params.update(params)
    device = default_params["device"]
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
            "utt": utt_inputs.to(device=device),
            "utt_len": utt_len_inputs.to(device=device),
            "resp": resp_inputs.to(device=device),
            "resp_len": resp_len_inputs.to(device=device),
            "target": targets.to(device=device)
        }
    else:
        assert len(datas) == 1
        return {
            "utt": datas[0][0].to(device=device),
            "utt_len": datas[0][1].to(device=device),
            "resp": datas[0][2].to(device=device),
            "resp_len": datas[0][3].to(device=device),
            "target": datas[0][4].to(device=device)
        }


# TODO: Generate fake data for SMN Ubuntu

def ubuntu_fake_gen(history, true_utt, actions=None):
    validation_num = 50000
    assert validation_num < len(history)
    new_history = []
    new_true_utt = []
    new_actions = []
    for i in range(validation_num, len(history)):
        utt_len = len(history[i])
        t_utt_idx = math.ceil((utt_len - 1) / 2) + 1

        if utt_len > 2:
            assert t_utt_idx > 0 and t_utt_idx < utt_len, "utt_len={}, t_utt_idx={}".format(utt_len, t_utt_idx)
            new_history.append(history[i][:t_utt_idx])
            new_true_utt.append(history[i][t_utt_idx])

        new_history.append(history[i])
        new_true_utt.append(true_utt[i])

    print("{} original examples, {} new examples.".format(len(history) - validation_num,
                                                              len(new_history) - len(history) + validation_num))
    new_history = history[:validation_num] + new_history
    new_true_utt = true_utt[:validation_num] + new_true_utt
    return new_history, new_true_utt, new_actions


# TODO: Generate fake data for DAM Ubuntu

def DAM_ubuntu_fake_gen(inputs):
    new_history = []
    new_true_utt = []
    new_labels = []
    validation_num = 500000
    assert validation_num < len(
        inputs['c']), "The number of validation set {} is larger than that of entire set {}.".format(validation_num,
                                                                                                     len(inputs['c']))

    for i in range(validation_num, len(inputs['c']), 2):
        utt_len = len(inputs['c'][i])
        '''
        if the length of history is odd, t_utt_idx = math.ceil((utt_len + 1) / 2) - 1 + math.ceil((utt_len + 1) / 2)%2,
        otherwise, t_utt_idx = math.ceil((utt_len + 1) / 2) - 1 + 1-math.ceil((utt_len + 1) / 2)%2
        In particular, utt_len >= 7
            0123456 7
            01234567 8
        '''
        mid_utt_idx = math.ceil((utt_len + 1) / 2)
        t_utt_idx = mid_utt_idx - int(mid_utt_idx%2 == (0 if utt_len % 2 else 1))
        if utt_len >= 7:
            assert t_utt_idx > 0 and t_utt_idx < utt_len, "utt_len={}, t_utt_idx={}".format(utt_len, t_utt_idx)
            # positive response
            new_history.append(inputs['c'][i][:t_utt_idx])
            new_true_utt.append(inputs['c'][i][t_utt_idx])
            new_labels.append(1)
            # negative response
            new_history.append(inputs['c'][i][:t_utt_idx])
            neg_idx = random.choice(range(t_utt_idx+2, utt_len+1, 2))
            new_true_utt.append(inputs['c'][i][neg_idx] if neg_idx < utt_len else inputs['r'][i])
            new_labels.append(0)

        new_history.append(inputs['c'][i]); new_history.append(inputs['c'][i+1])
        new_true_utt.append(inputs['r'][i]); new_true_utt.append(inputs['r'][i+1])
        new_labels.append(inputs['y'][i]); new_labels.append(inputs['y'][i+1])

    print("{} original examples, {} new examples.".format(len(inputs['c']) - validation_num,
                                                          len(new_history) - len(inputs['c']) + validation_num))
    new_inputs = {}
    new_inputs['c'] = inputs['c'][:validation_num] + new_history
    new_inputs['r'] = inputs['r'][:validation_num] + new_true_utt
    new_inputs['y'] = inputs['y'][:validation_num] + new_labels
    return new_inputs


def id2text(ids, word_dict):
    texts = []
    for item in ids:
        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
            texts.append(id2text(item, word_dict))
        elif isinstance(item, list):
            texts.append(" ".join(list(map(lambda x: word_dict[x], item))))
        else:
            raise ValueError

    if len(texts) > 0 and isinstance(texts[0], list):
        return list(map(lambda text: "\n".join(text), texts))
    else:
        return texts

if __name__ == "__main__":
    output_dir = os.path.dirname(sys.argv[1])
    # for smn data
    # actions = pickle.load(open(sys.argv[1], 'rb'))
    # history, true_utt = pickle.load(open(sys.argv[2], 'rb'))
    # word_dict = pickle.load(open(sys.argv[3], 'rb'))
    # word_dict = dict([(v, k) for k, v in word_dict.items()])

    # for dam data
    inputs = pickle.load(open(sys.argv[1], 'rb'))
    lines = open(sys.argv[2], 'r', encoding='utf-8').read().strip().split('\n')
    word_dict = dict([(int(lines[i+1]), lines[i]) for i in range(0, len(lines), 2)])
    word_dict[0] = "__padding__"

    ## test function fake_gen
    # for smn data
    # new_history, new_true_utt, new_actions = ubuntu_fake_gen(history, true_utt, actions)
    # with open(os.path.join(output_dir, "fake_utterances.pkl"), 'wb') as f:
    #     pickle.dump((new_history, new_true_utt), f)
    # with open(os.path.join(output_dir, "fake_responses.pkl"), 'wb') as f:
    #     pickle.dump(new_actions, f)

    # for dam data
    new_inputs = DAM_ubuntu_fake_gen(inputs)
    with open(os.path.join(output_dir, "fake_training_data.pkl"), 'wb') as f:
        pickle.dump(new_inputs, f)

    ## test function id2text
    # for smn data
    # actions = pickle.load(open(os.path.join(output_dir, "fake_responses.pkl"), 'rb'))
    # history, true_utt = pickle.load(open(os.path.join(output_dir, "fake_utterances.pkl"), 'rb'))
    # action_text = id2text(actions, word_dict)
    # history_text = id2text(history, word_dict)
    # true_utt_text = id2text(true_utt, word_dict)
    # assert len(history_text) == len(true_utt_text)
    # with open(os.path.join(output_dir, "his_res.txt"), 'w', encoding="utf-8") as f:
    #     for his, res in zip(history_text, true_utt_text):
    #         f.write(his + "\n\n" + res)
    #         f.write("\n\n\n")
    # with open(os.path.join(output_dir, "act.txt"), 'w', encoding="utf-8") as f:
    #     f.write('\n'.join(action_text))
    #     f.write('\n')

    # for dam data
    inputs = pickle.load(open(os.path.join(output_dir, "fake_training_data.pkl"), 'rb'))
    histoty_text = id2text(inputs['c'], word_dict)
    true_utt_text = id2text(inputs['r'], word_dict)
    with open(os.path.join(output_dir, "his_res_label.txt"), 'w', encoding="utf-8") as f:
        for his, res, label in zip(histoty_text, true_utt_text, inputs['y']):
            f.write(his + "\n\n" + res + "\n" + str(label))
            f.write("\n\n\n")