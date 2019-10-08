# coding: utf-8

import os
import pickle
import numpy as np

from utils import reader
from utils import utils

from tqdm import tqdm

from utils.utils import pad_3d_sequences as pad_sequences
# from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.bert_tokenization import BertTokenizer

logger = utils.get_logger()


class InputExample4MultiTurnResponseSelection(object):
    """
    A single training/test example for multi-turn response selection task.
    """
    def __init__(self, guid, context, response, label=None):
        '''
        Construct a InputExample
        :param guid: Unique id for the example
        :param context: list. The untokenized texts of the dialog history.
        :param responses: string. The untokenized text of a response candidate.
        :param label: (Optional) string. The label of the example.
            This should be specified for train and dev examples, but not neccessary for test examples.
        '''
        self.guid = guid
        self.context = context
        self.response = response
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_tokens, input_ids, input_mask, segment_ids, input_length=None, label_id=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_length = input_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for various data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MtrsProcessor(DataProcessor):

    def get_train_examples(self, input_file):
        return self._create_examples(self._read(input_file), "train")

    def get_dev_examples(self, input_file):
        return self._create_examples(self._read(input_file), "dev")

    def get_test_examples(self, input_file):
        return self._create_examples(self._read(input_file), "test")

    def get_labels(self):
        return ["0", "1"]

    def _read(self, input_file):
        with open(input_file, 'r', encoding="utf-8") as f:
            lines = [line.split('\t') for line in f.read().strip().split('\n')]
        return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(tqdm(lines, desc="Process Raw {} Text".format(set_type))):
            guid = "%s-%s" % (set_type, i)
            response = line[-1]
            context = line[1: -1]
            label = line[0]
            examples.append(
                InputExample4MultiTurnResponseSelection(guid=guid, context=context, response=response, label=label))
        return examples


def convert_examples_to_features(texts, max_seq_length, tokenizer, padding=True):
    features = []
    for (ex_index, text) in enumerate(texts):
        tokens_text = tokenizer.tokenize(text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_text) > max_seq_length - 2:
            tokens_text = tokens_text[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_text:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        input_length = len(input_ids)
        if padding:
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_tokens=tokens,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          input_length=input_length))
    return features


def convert_examples_to_features_for_multi_turn_response_selection(
        examples, max_num_utterance, max_seq_length, tokenizer, padding=True):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Bert Tokenization")):
        example.response = convert_examples_to_features([example.response], max_seq_length, tokenizer, padding=padding)[0]
        example.context = convert_examples_to_features(example.context, max_seq_length, tokenizer, padding=padding)
        example.label = int(example.label)

        if ex_index == 0:
            logger.info("\n*** Example ***")
            logger.info("\nguid: %s" % (example.guid))
            logger.info("\ncontext: %s" % "\n".join([" ".join([x for x in feature.input_tokens]) for feature in example.context]))
            logger.info("\ncontext_ids: %s" % "\n".join([" ".join([str(x) for x in feature.input_ids]) for feature in example.context]))
            logger.info("\ncontext_mask: %s" % "\n".join([" ".join([str(x) for x in feature.input_mask]) for feature in example.context]))
            logger.info("\ncontext_segment: %s" % "\n".join([" ".join([str(x) for x in feature.segment_ids]) for feature in example.context]))
            logger.info("\ncontext_length: %s" % " ".join([str(feature.input_length) for feature in example.context]))
            logger.info("\nresponse: %s" % " ".join([x for x in example.response.input_tokens]))
            logger.info("\nresponse_ids: %s" % " ".join([str(x) for x in example.response.input_ids]))
            logger.info("\nresponse_mask: %s" % " ".join([str(x) for x in example.response.input_mask]))
            logger.info("\nresponse_segment: %s" % " ".join([str(x) for x in example.response.segment_ids]))
            logger.info("\nresponse_length: %s" % str(example.response.input_length))
            logger.info("\nlabel: %d" % (example.label))

        features.append(example)
    return features


def Ubuntu_data_load(params):

    default_params = {
        "dataset_dir": None,
        "phase": "training",
        "training_files": ["train.txt, valid.txt"],
        "evaluate_files": ["test.txt"],
        "bert_model_dir": None,
        "do_lower_case": None,
        "empty_sequence_length": 0,
        "max_num_utterance": 10,
        "max_sentence_len": 50
    }
    default_params.update(params)
    assert default_params["dataset_dir"] and os.path.exists(default_params["dataset_dir"])

    np_dtype = np.int64
    np_float_dtype = np.float32

    training_files = default_params["training_files"]
    evaluate_files = default_params["evaluate_files"]

    processor = MtrsProcessor()

    max_num_utterance = default_params["max_num_utterance"]
    max_sentence_len = default_params["max_sentence_len"]

    def bert_process():
        assert "bert_model_dir" in default_params and default_params["bert_model_dir"] is not None
        assert "do_lower_case" in default_params and default_params["do_lower_case"] is not None
        tokenizer = BertTokenizer.from_pretrained(default_params["bert_model_dir"],
                                                  do_lower_case=default_params["do_lower_case"])

        features = None
        cache_file = None

        if default_params["phase"] in ["training", "validation"]:
            cache_file = os.path.join(default_params["dataset_dir"], training_files[0].rsplit(".", 1)[0]+".pkl")
            if os.path.isfile(cache_file): return pickle.load(open(cache_file, "rb"))
            training_examples = processor.get_train_examples(os.path.join(default_params["dataset_dir"], training_files[0]))
            features = convert_examples_to_features_for_multi_turn_response_selection(
                training_examples, max_num_utterance, max_sentence_len, tokenizer, padding=False)
            validation_examples = processor.get_dev_examples(os.path.join(default_params["dataset_dir"], training_files[1]))
            features = convert_examples_to_features_for_multi_turn_response_selection(
                validation_examples, max_num_utterance, max_sentence_len, tokenizer, padding=False) + features
        else:
            cache_file = os.path.join(default_params["dataset_dir"], evaluate_files[0].rsplit(".", 1)[0] + ".pkl")
            if os.path.isfile(cache_file): return pickle.load(open(cache_file, "rb"))
            evaluate_examples = processor.get_test_examples(os.path.join(default_params["dataset_dir"], evaluate_files[0]))
            features = convert_examples_to_features_for_multi_turn_response_selection(
                evaluate_examples, max_num_utterance, max_sentence_len, tokenizer, padding=False)

        inputs = {}

        inputs["context_token"] = [ [sent.input_tokens for sent in example.context] for example in features]
        inputs["context_id"]= [ [sent.input_ids for sent in example.context] for example in features]
        inputs["context_mask"] = [ [sent.input_mask for sent in example.context] for example in features]
        inputs["context_segment"] = [ [sent.segment_ids for sent in example.context] for example in features]

        inputs["response_token"] = [example.response.input_tokens for example in features]
        inputs["response_id"] = [example.response.input_ids for example in features]
        inputs["response_mask"] = [example.response.input_mask for example in features]
        inputs["response_len"] = [example.response.input_length for example in features]
        inputs["response_segment"] = [example.response.segment_ids for example in features]

        inputs["r_tag"], inputs["r_ent"], inputs["r_feature"] = reader.feature_gen(
            reader.doc_gen(inputs['context_token'], None, name="(train, valid)/context", use_element_fn=False),
            reader.doc_gen(inputs['response_token'], None, name="(train, valid)/response", use_element_fn=False),
            no_match=False)

        inputs["labels"] = [example.label for example in features]

        with open(cache_file, "wb") as f:
            pickle.dump(inputs, f)

        return inputs


    def numpy_process():
        max_batch_size = 100000
        np_cache_file = None
        if default_params["phase"] in ["training", "validation"]:
            np_cache_file = os.path.join(default_params["dataset_dir"], "numpy_" + training_files[0].rsplit(".", 1)[0]+".pkl")
        else:
            np_cache_file = os.path.join(default_params["dataset_dir"], "numpy_" + evaluate_files[0].rsplit(".", 1)[0] + ".pkl")

        if os.path.isfile(np_cache_file):
            context_id_np_list, context_len, context_mask_np_list, context_segment_np_list, \
            response_features_np_list, response_id, response_len, response_mask, response_segment, labels = pickle.load(
                open(np_cache_file, 'rb'))
        else:
            inputs = bert_process()
            context_id, context_len = utils.multi_sequences_padding(
                tqdm(inputs["context_id"], desc="Sequence Padding"),
                max_num_utterance=max_num_utterance,
                max_sentence_len=max_sentence_len)
            context_mask, _ = utils.multi_sequences_padding(
                tqdm(inputs["context_mask"], desc="Sequence Mask Padding"),
                max_num_utterance=max_num_utterance,
                max_sentence_len=max_sentence_len)
            context_segment, _ = utils.multi_sequences_padding(
                tqdm(inputs["context_segment"], desc="Sequence Segment Padding"),
                max_num_utterance=max_num_utterance,
                max_sentence_len=max_sentence_len)
            feature_len = len(inputs["r_feature"][0][0][0])
            response_features, _ = utils.multi_sequences_padding(
                tqdm(inputs["r_feature"], desc="Feature Sequence Padding"),
                max_num_utterance=default_params["max_num_utterance"],
                max_sentence_len=default_params["max_sentence_len"],
                padding_element=[0] * feature_len)

            context_id_np_list = [np.array(context_id[i: i + max_batch_size], dtype=np_dtype) for i in
                               range(0, len(context_id), max_batch_size)]
            context_len = np.array(context_len, dtype=np_dtype)
            context_mask_np_list = [np.array(context_mask[i: i + max_batch_size], dtype=np_dtype) for i in
                            range(0, len(context_mask), max_batch_size)]
            context_segment_np_list = [np.array(context_segment[i: i + max_batch_size], dtype=np_dtype) for i in
                               range(0, len(context_segment), max_batch_size)]
            response_features_np_list = [np.array(response_features[i: i + max_batch_size], dtype=np_float_dtype) for i in
                                         range(0, len(response_features), max_batch_size)]

            response_id = np.array(pad_sequences(inputs["response_id"], max_sentence_len, padding='post'), dtype=np_dtype)
            response_len = np.array(inputs["response_len"], dtype=np_dtype)
            response_mask = np.array(pad_sequences(inputs["response_mask"], max_sentence_len, padding='post'), dtype=np_dtype)
            response_segment = np.array(pad_sequences(inputs["response_segment"], max_sentence_len, padding='post'), dtype=np_dtype)
            labels = np.array(inputs["labels"], dtype=np_dtype)

            with open(np_cache_file, 'wb') as f:
                pickle.dump([context_id_np_list, context_len, context_mask_np_list, context_segment_np_list,
                             response_features_np_list, response_id, response_len, response_mask, response_segment,
                             labels], f)

        context_id = np.concatenate(context_id_np_list, axis=0)
        context_mask = np.concatenate(context_mask_np_list, axis=0)
        context_segment = np.concatenate(context_segment_np_list, axis=0)
        response_features = np.concatenate(response_features_np_list, axis=0)

        return [context_id, context_len, context_mask, context_segment, response_features, response_id, response_len, response_mask, response_segment, labels]

    context_id, context_len, context_mask, context_segment, response_features, \
    response_id, response_len, response_mask, response_segment, labels = numpy_process()

    utils.varname(context_id)
    utils.varname(context_len)
    utils.varname(context_mask)
    utils.varname(context_segment)
    utils.varname(response_features)
    utils.varname(response_id)
    utils.varname(response_len)
    utils.varname(response_mask)
    utils.varname(response_segment)
    utils.varname(labels)

    # address an issue where sent_len == 0
    empty_sequence_length = default_params["empty_sequence_length"]
    if empty_sequence_length != 0:
        assert empty_sequence_length > 0
        context_len = np.ma.array(context_len, mask=context_len == 0, fill_value=empty_sequence_length).filled()

    print("Number of empty sequences in context:   {}".format(np.sum(context_len == 0)))
    print("Number of empty sequences in response:  {}".format(np.sum(response_len == 0)))

    if default_params["phase"] in ["training", "validation"]:
        return {
            "context_id": {"data": context_id, "type": "normal"},
            "context_len": {"data": context_len, "type": "normal"},
            "context_mask": {"data": context_mask, "type": "normal"},
            "context_segment": {"data": context_segment, "type": "normal"},
            "response_features": {"data": response_features, "type": "normal"},
            "response_id": {"data": response_id, "type": "normal"},
            "response_len": {"data": response_len, "type": "normal"},
            "response_mask": {"data": response_mask, "type": "normal"},
            "response_segment": {"data": response_segment, "type": "normal"},
            "target": {"data": labels, "type": "normal"}
        }
    else:
        return {
            "context_id": context_id,
            "context_len": context_len,
            "context_mask": context_mask,
            "context_segment": context_segment,
            "response_features": response_features,
            "response_id": response_id,
            "response_len": response_len,
            "response_mask": response_mask,
            "response_segment": response_segment,
            "target": labels
        }


def Ubuntu_dataloader_gen(data_dict, params):

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
                   shuffle=shuffle)
    ])


def Ubuntu_data_gen(datas, params):
    default_params = {
        "device": None
    }
    default_params.update(params)
    device = default_params["device"]
    assert len(datas) == 1

    assert "data_names" in default_params and default_params["data_names"] is not None

    return dict([[data_name, datas[0][i].to(device=device)] for i, data_name in enumerate(default_params["data_names"])])


def convert_examples_to_features_for_bert_sequence_classification(examples, label_list, max_seq_length, max_response_length,
                                                         tokenizer, separation):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Bert Tokenization")):
        tokens_a = tokenizer.tokenize(example.response)

        if len(tokens_a) > max_response_length:
            tokens_a = tokens_a[0:max_response_length]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(tokens_a) - 3

        tokens = []
        segment_ids = []
        tokens.append(tokenizer.cls_token)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        if tokenizer.eos_token:
            tokens.append(tokenizer.eos_token)
            segment_ids.append(0)
            max_tokens_for_doc -= 1
        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)

        if separation:
            count = -1
            insert_index = len(tokens)
            context = example.context
            for sent in context[::-1]:
                tokens_sent = tokenizer.tokenize(sent)
                if len(tokens_sent) + count + 1 > max_tokens_for_doc:
                    break
                tokens.insert(insert_index, tokenizer.sep_token)
                segment_ids.insert(insert_index, 1)
                count += 1
                for index in range(len(tokens_sent)):
                    tokens.insert(insert_index, tokens_sent.pop())
                    segment_ids.insert(insert_index, 1)
                    count += 1
        else:
            tokens_b = tokenizer.tokenize(" ".join(example.context))
            if len(tokens_b) > max_tokens_for_doc:
                tokens_b = tokens_b[-max_tokens_for_doc:]
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(tokenizer.sep_token)
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_tokens=tokens,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def Ubuntu_data_load_for_bert_sequence_classification(params):

    default_params = {
        "dataset_dir": None,
        "phase": "training",
        "training_files": ["train.txt, valid.txt"],
        "evaluate_files": ["test.txt"],
        "bert_model_dir": None,
        "do_lower_case": None,
        "separation": False,
        "max_seq_length": 384,
        "max_response_length": 50,
        "tokenizier": "utils.bert_tokenization.BertTokenizer"
    }
    default_params.update(params)
    assert default_params["dataset_dir"] and os.path.exists(default_params["dataset_dir"])

    np_dtype = np.int64

    training_files = default_params["training_files"]
    evaluate_files = default_params["evaluate_files"]

    processor = MtrsProcessor()

    label_list = processor.get_labels()
    max_seq_length = default_params["max_seq_length"]
    max_response_length = default_params["max_response_length"]
    separation = default_params["separation"]

    def bert_process():
        assert "bert_model_dir" in default_params and default_params["bert_model_dir"] is not None
        assert "do_lower_case" in default_params and default_params["do_lower_case"] is not None
        tokenizer = utils.name2function(default_params["tokenizier"])
        tokenizer = tokenizer.from_pretrained(default_params["bert_model_dir"],
                                                  do_lower_case=default_params["do_lower_case"])

        features = None
        cache_file = None

        if default_params["phase"] in ["training", "validation"]:
            cache_file = os.path.join(default_params["dataset_dir"], training_files[0].rsplit(".", 1)[0]+"_for_bert_sequence_classfication.pkl")
            if os.path.isfile(cache_file): return pickle.load(open(cache_file, "rb"))
            training_examples = processor.get_train_examples(os.path.join(default_params["dataset_dir"], training_files[0]))
            features = convert_examples_to_features_for_bert_sequence_classification(
                training_examples, label_list, max_seq_length, max_response_length, tokenizer, separation)
            validation_examples = processor.get_dev_examples(os.path.join(default_params["dataset_dir"], training_files[1]))
            features = convert_examples_to_features_for_bert_sequence_classification(
                validation_examples, label_list, max_seq_length, max_response_length, tokenizer, separation) + features
        else:
            cache_file = os.path.join(default_params["dataset_dir"], evaluate_files[0].rsplit(".", 1)[0] + "_for_bert_sequence_classfication.pkl")
            if os.path.isfile(cache_file): return pickle.load(open(cache_file, "rb"))
            evaluate_examples = processor.get_test_examples(os.path.join(default_params["dataset_dir"], evaluate_files[0]))
            features = convert_examples_to_features_for_bert_sequence_classification(
                evaluate_examples, label_list, max_seq_length, max_response_length, tokenizer, separation)

        inputs = {}

        inputs["input_ids"]= [ f.input_ids for f in features ]
        inputs["input_mask"] = [ f.input_mask for f in features ]
        inputs["segment_ids"] = [ f.segment_ids for f in features ]
        inputs["labels"] = [ f.label_id for f in features ]

        with open(cache_file, "wb") as f:
            pickle.dump(inputs, f)

        return inputs


    def numpy_process():
        max_batch_size = 100000
        np_cache_file = None
        results = {}

        if default_params["phase"] in ["training", "validation"]:
            np_cache_file = os.path.join(default_params["dataset_dir"], "numpy_" + training_files[0].rsplit(".", 1)[0] + "_for_bert_sequence_classfication.pkl")
        else:
            np_cache_file = os.path.join(default_params["dataset_dir"], "numpy_" + evaluate_files[0].rsplit(".", 1)[0] + "_for_bert_sequence_classfication.pkl")

        if os.path.isfile(np_cache_file):
            results = pickle.load(open(np_cache_file, 'rb'))
        else:
            inputs = bert_process()
            results["input_ids_np_list"] = [np.array(inputs["input_ids"][i: i + max_batch_size], dtype=np_dtype) for i
                                            in range(0, len(inputs["input_ids"]), max_batch_size)]
            results["input_mask_np_list"] = [np.array(inputs["input_mask"][i: i + max_batch_size], dtype=np_dtype) for i
                                            in range(0, len(inputs["input_mask"]), max_batch_size)]
            results["segment_ids_np_list"] = [np.array(inputs["segment_ids"][i: i + max_batch_size], dtype=np_dtype) for i
                                             in range(0, len(inputs["segment_ids"]), max_batch_size)]
            results["labels"] = np.array(inputs["labels"], dtype=np_dtype)

            with open(np_cache_file, 'wb') as f:
                pickle.dump(results, f)

        results["input_ids"] = np.concatenate(results["input_ids_np_list"], axis=0)
        results["input_mask"] = np.concatenate(results["input_mask_np_list"], axis=0)
        results["segment_ids"] = np.concatenate(results["segment_ids_np_list"], axis=0)
        del results["input_ids_np_list"]
        del results["input_mask_np_list"]
        del results["segment_ids_np_list"]

        return results

    results = numpy_process()
    input_ids, input_mask, segment_ids, labels = results["input_ids"], results["input_mask"], results["segment_ids"], \
                                                 results["labels"]

    utils.varname(input_ids)
    utils.varname(input_mask)
    utils.varname(segment_ids)
    utils.varname(labels)

    if default_params["phase"] in ["training", "validation"]:
        return {
            "input_ids": {"data": input_ids, "type": "normal"},
            "input_mask": {"data": input_mask, "type": "normal"},
            "segment_ids": {"data": segment_ids, "type": "normal"},
            "target": {"data": labels, "type": "normal"}
        }
    else:
        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "target": labels
        }
