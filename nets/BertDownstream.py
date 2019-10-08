# coding: utf-8

import os

import torch
import torch.nn as nn

from layers.bert_modeling import BertConfig, PreTrainedBertModel, BertModel, BertForSequenceClassification
import layers.operations as op
from utils.bert_tokenization import BertTokenizer
from utils import utils
from utils.utils import varname

logger = utils.get_logger()

class BertModelWrapper(PreTrainedBertModel):
    def __init__(self, config):
        super(BertModelWrapper, self).__init__(config)
        # config.hidden_dropout_prob = 0
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, output_embeddings=True, only_embeddings=True):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=output_all_encoded_layers,
                output_embeddings=output_embeddings, only_embeddings=only_embeddings)


def optimizer_grouped_parameters(model, optimizer_params, model_params):

    grouped_parameters = []
    parameters = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    grouped_parameters.append(
        {"params": [p for n, p in parameters if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01})
    grouped_parameters.append(
        {"params": [p for n, p in parameters if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0})

    return grouped_parameters


def output_result(predictions, config):
    file_name = config["file_name"]
    tokenizer = BertTokenizer.from_pretrained(config["bert_model_dir"], do_lower_case=config["do_lower_case"])
    separator = " [SEP] "

    inputs = predictions[2]
    score = predictions[0][0]
    score_length = max(
        (len(str(score[0])) * len(score) + len(score) - 1) if isinstance(score, list) else len(str(score)),
        len("SCORE"))
    prediction_length = max(len(str(predictions[1][0])), len("PREDICTION"))
    gold_length = max(len(str(predictions[2][0])), len("GOLD"))

    template = "{}\t\t{}\t\t{}\t\t{}\n"

    with open(file_name, 'w', encoding="utf-8") as f:
        for i in range(len(predictions[0])):
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i][:inputs["input_mask"][i].count(1)])
            sentences = " ".join(tokens[1:-1]).split(separator)
            if i % 10 == 0:
                f.write("CONTEXT:\n" + "\n".join(sentences[1:]) + "\n")
                f.write(
                    template.format("SCORE".ljust(score_length), "PREDICTION".ljust(prediction_length),
                                                  "GOLD".ljust(gold_length), "RESPONSE"))
            scores = predictions[0][i] if isinstance(score, list) else [predictions[0][i]]
            f.write(
                template.format(" ".join([str(s) for s in scores]), str(predictions[1][i]).center(prediction_length),
                                str(inputs["target"][i]).center(gold_length), sentences[0]))
            if i % 10 == 9:
                f.write("\n")


class BertForMultiTurnResponseSelection(nn.Module):
    def __init__(self, config):
        super(BertForMultiTurnResponseSelection, self).__init__()
        self.final_out_features = config["final_out_features"] if "final_out_features" in config else 2
        assert "bert_model_dir" in config
        self.bert_model_dir = config["bert_model_dir"]

        self.bert_model = BertForSequenceClassification.from_pretrained(self.bert_model_dir, cache_dir=None,
                                                                        num_labels=self.final_out_features)

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["segment_ids"]
        attention_mask = inputs["input_mask"]
        logits = self.bert_model(input_ids, token_type_ids, attention_mask)

        return logits.squeeze(-1)


class MultiTurnBert(nn.Module):
    def __init__(self, config):
        super(MultiTurnBert, self).__init__()
        self.max_num_utterance = config["max_num_utterance"] if "max_num_utterance" in config else 10
        self.max_sentence_len = config["max_sentence_len"] if "max_sentence_len" in config else 50
        self.bert_hidden_size = config["bert_hidden_size"] if "bert_hidden_size" in config else 768
        self.rnn_units = config["rnn_units"] if "rnn_units" in config else 768
        self.final_out_features = config["final_out_features"] if "final_out_features" in config else 2

        self.rnn = nn.GRU(self.max_num_utterance * self.bert_hidden_size, self.rnn_units, bidirectional=True,
                          batch_first=True)
        self.classifier = nn.Linear(2 * self.rnn_units, self.final_out_features)

        # self.bert_model = BertModelWrapper(config["bert_config"])
        assert "bert_model_dir" in config
        self.bert_model_dir = config["bert_model_dir"]
        bert_config = BertConfig.from_json_file(os.path.join(self.bert_model_dir, 'bert_config.json'))
        self.bert_model = BertModelWrapper(bert_config)
        self.apply(self.bert_model.init_bert_weights)
        self.bert_model = BertModelWrapper.from_pretrained(self.bert_model_dir, cache_dir=None)

    def forward(self, inputs):
        context_id, context_len, context_mask, context_segment, context_alignment = \
            inputs["history_bert_id"], inputs["history_bert_len"], inputs["history_bert_mask"], \
            inputs["history_bert_segment"], inputs["history_alignment"]
        response_id, response_len, response_mask, response_segment, response_alignment = \
            inputs["true_utt_bert_id"], inputs["true_utt_bert_len"], inputs["true_utt_bert_mask"], \
            inputs["true_utt_bert_segment"], inputs["true_utt_alignment"]

        # context_id: [batch, max_num_utterance, max_sentence_len]
        # response_id: [batch, max_sentence_len]
        response_id_full = response_id.unsqueeze(1).expand(context_id.size(0), context_id.size(1),
                                                           response_id.size(1)).contiguous()
        response_segment_full = response_segment.unsqueeze(1).expand(context_segment.size(0), context_segment.size(1),
                                                                     response_segment.size(1)).contiguous()
        response_mask_full = response_mask.unsqueeze(1).expand(context_mask.size(0), context_mask.size(1),
                                                               response_mask.size(1)).contiguous()

        input_id = torch.cat((response_id_full, context_id[:, :, 1:]), dim=2)
        input_segment = torch.cat((response_segment_full, 1 - context_segment[:, :, 1:]), dim=2)
        input_mask = torch.cat((response_mask_full, context_mask[:, :, 1:]), dim=2)

        encoded_layers, pooled_output = self.bert_model(
            input_id.view(-1, input_id.shape[-1]),
            input_segment.view(-1, input_segment.shape[-1]),
            input_mask.view(-1, input_mask.shape[-1]),
            output_all_encoded_layers=False,
            output_embeddings=False,
            only_embeddings=False)  # torch.Size([batch * max_num_utterance, 2 * max_sentence_len - 1, hidden_size]),
                                    # torch.Size([batch * max_num_utterance, hidden_size])

        contextual_response = encoded_layers[:, 1: self.max_sentence_len - 1, :]  # torch.Size([batch * max_num_utterance, max_sentence_len - 2, hidden_size])

        # rnn_input: [batch, max_sentence_len - 2, max_num_utterance * hidden_size]
        rnn_input = contextual_response.view(context_id.size(0), context_id.size(1), contextual_response.size(1), -1).\
            transpose(1, 2).contiguous().view(context_id.size(0), contextual_response.size(1), -1)

        # response_output:  [batch, max_sentence_len - 2, num_directions * rnn_units]
        # response_hn:      [num_layers * num_directions, batch, rnn_units]
        response_output, response_hn = op.pack_and_pad_sequences_for_rnn(rnn_input, response_len - 2, self.rnn)

        logits = self.classifier(response_hn.transpose(0, 1).view(response_output.size(0), -1))

        return logits.squeeze(-1)


class BertForSequenceRepresentation(nn.Module):
    def __init__(self, config):
        super(BertForSequenceRepresentation, self).__init__()
        self.bert_model = BertModelWrapper.from_pretrained(config["bert_model_dir"], cache_dir=None)

    def forward(self, inputs):
        utterance_id, utterance_mask, utterance_segment, utterance_len = \
            inputs["id"], inputs["mask"], inputs["segment"],inputs["len"]
        encoded_layers, pooled_output = self.bert_model(
            utterance_id,
            utterance_segment,
            utterance_mask,
            output_all_encoded_layers=False,
            output_embeddings=False,
            only_embeddings=False)
        return pooled_output


if __name__ == "__main__":
    vocab_size = 1000
    batch_size = 2
    max_num_utterance = 3
    max_sentence_len = 4

    config = {
        "max_num_utterance": max_num_utterance,
        "max_sentence_len": max_sentence_len,
        "rnn_units": 5
    }
    config["bert_config"] = BertConfig(vocab_size)

    model = MultiTurnBert(config)

    inputs = {}
    inputs["context_id"] = torch.randint(0, vocab_size, (batch_size, max_num_utterance, max_sentence_len), dtype=torch.int64)
    inputs["context_len"] = torch.randint(2, max_sentence_len + 1, (batch_size, max_num_utterance), dtype=torch.int64)
    context_mask = op.sequence_mask(inputs["context_len"], max_sentence_len).to(dtype=torch.int64)
    inputs["context_mask"] = context_mask
    varname(context_mask)
    inputs["context_segment"] = torch.zeros(batch_size, max_num_utterance, max_sentence_len, dtype=torch.int64)

    inputs["response_id"] = torch.randint(0, vocab_size, (batch_size, max_sentence_len), dtype=torch.int64)
    inputs["response_len"] = torch.randint(2, max_sentence_len + 1, (batch_size, ), dtype=torch.int64)
    response_mask = op.sequence_mask(inputs["response_len"], max_sentence_len).to(dtype=torch.int64)
    inputs["response_mask"] = response_mask
    varname(response_mask)
    inputs["response_segment"] = torch.zeros(batch_size, max_sentence_len, dtype=torch.int64)

    logits = model(inputs)
    print (logits)



