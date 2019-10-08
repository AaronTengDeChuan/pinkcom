# coding: utf-8
import os

from utils import utils
from utils.utils import varname

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import layers
import layers.operations as op
from layers.bert_modeling import BertConfig, PreTrainedBertModel, BertModel, BertLayerNorm, gelu

logger = utils.get_logger()


class BertModelWrapper(PreTrainedBertModel):
    def __init__(self, config):
        super(BertModelWrapper, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, output_embeddings=True, only_embeddings=True):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=output_all_encoded_layers,
                output_embeddings=output_embeddings, only_embeddings=only_embeddings)


def tensor_hook(grad):
    print('grad:', grad)
    input("\nnext hook:")

def tensor_info(tensor):
    print (tensor)
    tensor.register_hook(tensor_hook)
    input("\nnext tensor:")

class FlowQAModel(nn.Module):
    """FlowQA contains ."""

    def __init__(self, config):
        super(FlowQAModel, self).__init__()
        SpaCy_LANGUAGES = ["en"]
        # hyperparameters
        embeddings = config["embeddings"] if "embeddings" in config else None
        self.language = config["language"] if "language" in config else "en"
        self.use_spacy = self.language in SpaCy_LANGUAGES
        self.vocabulary_size = config["vocabulary_size"] if "vocabulary_size" in config else 434511
        self.word_embedding_size = config["embedding_dim"] if "embedding_dim" in config else 200
        self.hidden_size = config["hidden_size"] if "hidden_size" in config else 200

        self.bidirectional_dialog_flow = config["bidirectional_dialog_flow"] if "bidirectional_dialog_flow" in config else False

        # TODO: word features
        self.num_word_features = config["num_word_features"] if "num_word_features" in config else 4
        self.no_em = config["no_em"] if "no_em" in config else False
        self.no_dialog_flow = config["no_dialog_flow"] if "no_dialog_flow" in config else False

        self.do_prealign = config["do_prealign"] if "do_prealign" in config else True
        self.prealign_hidden = config["prealign_hidden"] if "prealign_hidden" in config else 200

        self.deep_inter_att_do_similar = config["deep_inter_att_do_similar"] if "deep_inter_att_do_similar" in config else False
        self.deep_att_hidden_size_per_abstr = config["deep_att_hidden_size_per_abstr"] if "deep_att_hidden_size_per_abstr" in config else 250
        self.self_attention_opt = config["self_attention_opt"] if "self_attention_opt" in config else True

        self.do_hierarchical_query = config["do_hierarchical_query"] if "do_hierarchical_query" in config else True

        self.max_num_utterance = config["max_num_utterance"] if "max_num_utterance" in config else 10
        self.max_sentence_len = config["max_sentence_len"] if "max_sentence_len" in config else 50

        self.do_seq_dropout = config["do_seq_dropout"] if "do_seq_dropout" in config else True
        self.my_dropout_p = config["my_dropout_p"] if "my_dropout_p" in config else 0.0
        self.dropout_emb = config["dropout_emb"] if "dropout_emb" in config else 0.0

        self.final_out_features = config["final_out_features"] if "final_out_features" in config else 2
        self.last_score = config["last_score"] if "last_score" in config else False
        self.embeddings_trainable = config["emb_trainable"] if "emb_trainable" in config else True

        doc_input_size, que_input_size = 0, 0

        layers.set_my_dropout_prob(self.my_dropout_p)
        layers.set_seq_dropout(self.do_seq_dropout)

        # build model
        ## Embedding TODO: with GloVe, CoVE, and ELMo embeddings
        self.embeddings = op.init_embedding(self.vocabulary_size, self.word_embedding_size, embeddings=embeddings,
                                            embeddings_trainable=self.embeddings_trainable)
        ## use bert embedding
        self.use_bert_embeddings = config["use_bert_embeddings"] if "use_bert_embeddings" in config else False
        if self.use_bert_embeddings:
            self.bert_hidden_size = config["bert_hidden_size"] if "bert_hidden_size" in config else 768
            assert "bert_model_dir" in config
            self.bert_model_dir = config["bert_model_dir"]
            self.bert_trainable = config["bert_trainable"]
            self.bert_config = BertConfig.from_json_file(os.path.join(self.bert_model_dir, 'bert_config.json'))

            self.word_embedding_size += self.bert_hidden_size

        doc_input_size, que_input_size = self.word_embedding_size, self.word_embedding_size

        if self.no_em:
            doc_input_size += self.num_word_features - 3
        else:
            if self.use_spacy:
                doc_input_size += self.num_word_features
            else:
                doc_input_size += self.num_word_features - 2

        # Attention (on Question):
        # For each question, we compute attention in the word level to enhance context word embeddings with question
        if self.do_prealign:
            self.pre_align = layers.GetAttentionHiddens(self.word_embedding_size, self.prealign_hidden,
                                                        similarity_attention=True)
            doc_input_size += self.word_embedding_size

        # Setup the vector size for [doc, question]
        # they will be modified in the following code
        doc_hidden_size, que_hidden_size = doc_input_size, que_input_size
        logger.info(
            "Initially, the vector_sizes [doc, question] are [{}, {}].".format(doc_hidden_size, que_hidden_size))

        flow_size = self.hidden_size
        if self.bidirectional_dialog_flow: flow_size *= 2
        # RNN document encoder: Intergration-Flow * 2
        self.doc_rnn1 = layers.StackedBRNN(doc_hidden_size, self.hidden_size, num_layers=1)
        self.dialog_flow1 = layers.StackedBRNN(self.hidden_size * 2, self.hidden_size, num_layers=1,
                                               rnn_type=nn.GRU, bidir=self.bidirectional_dialog_flow)
        self.doc_rnn2 = layers.StackedBRNN(self.hidden_size * 2 + flow_size, self.hidden_size, num_layers=1)
        self.dialog_flow2 = layers.StackedBRNN(self.hidden_size * 2, self.hidden_size, num_layers=1,
                                               rnn_type=nn.GRU, bidir=self.bidirectional_dialog_flow)
        doc_hidden_size = self.hidden_size * 2

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(que_hidden_size, self.hidden_size, num_layers=2)
        que_hidden_size = self.hidden_size * 2
        # Output sizes of rnn encoders
        logger.info('After Input LSTM, the vector_sizes [doc, query] are [{}, {}] * 2.'.format(doc_hidden_size,
                                                                                               que_hidden_size))
        # Question understanding and compression
        self.high_lvl_qrnn = layers.StackedBRNN(que_hidden_size * 2, self.hidden_size, num_layers=1, concat_layers=True)
        que_hidden_size = self.hidden_size * 2

        # Deep inter-attention
        self.deep_attn = layers.DeepAttention(
            {"embedding_dim": self.word_embedding_size, "hidden_size": self.hidden_size}, abstr_list_cnt=2,
            deep_att_hidden_size_per_abstr=self.deep_att_hidden_size_per_abstr,
            do_similarity=self.deep_inter_att_do_similar, no_rnn=True)

        self.deep_attn_rnn = layers.StackedBRNN(self.deep_attn.att_final_size + flow_size, self.hidden_size,
                                                num_layers=1)
        doc_hidden_size = self.hidden_size * 2
        self.dialog_flow3 = layers.StackedBRNN(doc_hidden_size, self.hidden_size, num_layers=1, rnn_type=nn.GRU,
                                               bidir=self.bidirectional_dialog_flow)

        # Self attention on context
        att_size = doc_hidden_size + 2 * self.hidden_size * 2

        if self.self_attention_opt > 0:
            self.highlvl_self_att = layers.GetAttentionHiddens(att_size, self.deep_att_hidden_size_per_abstr)
            self.high_lvl_crnn = layers.StackedBRNN(doc_hidden_size * 2 + flow_size, self.hidden_size, num_layers=1, concat_layers=False)
            doc_hidden_size = self.hidden_size * 2
            logger.info('Self deep-attention {} rays in {}-dim space'.format(self.deep_att_hidden_size_per_abstr, att_size))
        elif self.self_attention_opt == 0:
            self.high_lvl_crnn = layers.StackedBRNN(doc_hidden_size * 2 + flow_size, self.hidden_size, num_layers=1, concat_layers=False)
            doc_hidden_size = self.hidden_size * 2

        logger.info('Before answer span finding, hidden size are [{}, {}]'.format(doc_hidden_size, que_hidden_size))

        # Question merging
        self.self_attn = layers.LinearSelfAttn(que_hidden_size)
        if self.do_hierarchical_query:
            self.hier_query_rnn = layers.StackedBRNN(que_hidden_size, self.hidden_size, num_layers=1, rnn_type=nn.GRU,
                                                     bidir=self.bidirectional_dialog_flow)
            que_hidden_size = self.hidden_size
            if self.bidirectional_dialog_flow: que_hidden_size *= 2

        # Matching score calculation
        self.separate_matching_score = layers.BilinearLayer(doc_hidden_size * 2, que_hidden_size,
                                                            class_num=self.final_out_features if self.last_score else 1)
        if not self.last_score:
            self.overall_matching_score = nn.Linear(self.max_num_utterance, self.final_out_features)

        ## Bert pretrained model
        if self.use_bert_embeddings:
            self.apply(self.init_weights)
            self.bert = BertModelWrapper.from_pretrained(self.bert_model_dir, cache_dir=None)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            module.gamma.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, inputs):
        """inputs:
            x1 = document word indices             [batch * len_d]
                # x1_c = document char indices           [batch * len_d * len_w] or [1]
            x1_f = document word features indices  [batch * q_num * len_d * nfeat]
                # x1_pos = document POS tags             [batch * len_d]
                # x1_ner = document entity tags          [batch * len_d]
            x1_mask = document padding mask        [batch * len_d]
            x2_full = question word indices        [batch * q_num * len_q]
                # x2_c = question char indices           [(batch * q_num) * len_q * len_w]
            x2_full_mask = question padding mask   [batch * q_num * len_q]
        """
        device = inputs["target"].device
        dtype = torch.get_default_dtype()

        x1 = inputs["resp"] # torch.Size([batch, max_sentence_len])
        x1_len = inputs["resp_len"]    # torch.Size([batch])
        x1_mask = op.sequence_mask(x1_len, self.max_sentence_len, mask_value=1) # torch.Size([batch, max_sentence_len])
        x1_f = inputs["response_features"]   #TODO

        # if self.num_updates == 1425: utils.varname(x1)
        # if self.num_updates == 1425: utils.varname(x1_len)
        # if self.num_updates == 1425: utils.varname(x1_mask)
        # if self.num_updates == 1425: utils.varname(x1_f)

        x2_full = inputs["utt"] # torch.Size([batch, max_num_utterance, max_sentence_len])
        x2_num = inputs["utt_num"]  # torch.Size([batch])
        x2_full_num = x2_num.unsqueeze(1).expand(x2_full.size(0), x1.size(1)).contiguous()  # torch.Size([batch, context_length])
        x2_full_len = inputs["utt_len"] # torch.Size([batch, max_num_utterance])
        x2_full_mask = op.sequence_mask(x2_full_len, self.max_sentence_len, mask_value=1)   # torch.Size([batch, max_num_utterance, max_sentence_len])

        # if self.num_updates == 1425: utils.varname(x2_full)
        # if self.num_updates == 1425: utils.varname(x2_full_len)
        # if self.num_updates == 1425: utils.varname(x2_full_mask)

        """
            x1_full = document word indices        [batch * q_num * len_d]
            x1_full_mask = document padding mask   [batch * q_num * len_d]
        """
        x1_full = x1.unsqueeze(1).expand(x2_full.size(0), x2_full.size(1), x1.size(1)).contiguous() # torch.Size([batch, max_num_utterance, max_sentence_len])
        x1_full_len = x1_len.unsqueeze(1).expand(x2_full.size(0), x2_full.size(1)).contiguous() # torch.Size([batch, max_num_utterance])
        x1_full_mask = x1_mask.unsqueeze(1).expand(x2_full.size(0), x2_full.size(1), x1.size(1)).contiguous()   # torch.Size([batch, max_num_utterance, max_sentence_len])

        # if self.num_updates == 1425: utils.varname(x1_full)
        # if self.num_updates == 1425: utils.varname(x1_full_mask)

        drnn_input_list, qrnn_input_list = [], []

        x2 = x2_full.view(-1, x2_full.size(-1)) # torch.Size([batch * max_num_utterance, max_sentence_len])
        x2_len = x2_full_len.view(-1)   # torch.Size([batch * max_num_utterance])
        x2_mask = x2_full_mask.view(-1, x2_full.size(-1))   # torch.Size([batch * max_num_utterance, max_sentence_len])

        # Word embedding for both document and question
        x1_emb = self.embeddings(x1)    # torch.Size([batch, max_sentence_len, embedding_dim])
        x2_emb = self.embeddings(x2)    # torch.Size([batch * max_num_utterance, max_sentence_len, embedding_dim])
        # Bert Embedding
        if self.use_bert_embeddings:
            history_bert_id, history_bert_len, history_bert_mask, history_bert_segment, history_alignment = \
                inputs["history_bert_id"], inputs["history_bert_len"], inputs["history_bert_mask"], \
                inputs["history_bert_segment"], inputs["history_alignment"]
            response_bert_id, response_bert_len, response_bert_mask, response_bert_segment, response_alignment = \
                inputs["true_utt_bert_id"], inputs["true_utt_bert_len"], inputs["true_utt_bert_mask"], \
                inputs["true_utt_bert_segment"], inputs["true_utt_alignment"]

            with torch.set_grad_enabled(self.bert_trainable):
                # if not self.bert_trainable: self.bert.eval()
                ## push context into bert
                context_encoded_layers, context_pooled_output, context_embeds = self.bert(
                    history_bert_id.view(-1, history_bert_id.shape[-1]),
                    history_bert_segment.view(-1, history_bert_segment.shape[-1]),
                    history_bert_mask.view(-1, history_bert_mask.shape[-1]),
                    output_all_encoded_layers=False,
                    output_embeddings=True,
                    only_embeddings=True)  # torch.Size([None * 10, 50, hidden_size]), torch.Size([None * 10, hidden_size]), torch.Size([None * 10, 50, hidden_size])

                ## push response into bertresponse_output
                response_encoded_layers, response_pooled_output, response_embeds = self.bert(response_bert_id,
                                                                                             response_bert_segment,
                                                                                             response_bert_mask,
                                                                                             output_all_encoded_layers=False,
                                                                                             output_embeddings=True,
                                                                                             only_embeddings=True)  # torch.Size([None, 50, hidden_size]), torch.Size([None, hidden_size]), torch.Size([None, 50, hidden_size])
            responses_indices = torch.cat([response_alignment.unsqueeze(-1)] * response_embeds.shape[-1], dim=-1)
            x1_emb = torch.cat([x1_emb, torch.gather(response_embeds, -2, responses_indices)], dim=-1)
            context_indices = torch.cat([history_alignment.view(-1, history_alignment.shape[-1]).unsqueeze(-1)] * context_embeds.shape[-1], dim=-1)
            x2_emb = torch.cat([x2_emb, torch.gather(context_embeds, -2, context_indices)], dim=-1)

        # if self.num_updates == 1425: utils.varname(x1_emb, fn=tensor_info)
        # if self.num_updates == 1425: utils.varname(x2_emb, fn=tensor_info)
        # Dropout on embeddings
        if self.dropout_emb > 0:
            x1_emb = layers.dropout(x1_emb, p=self.dropout_emb, training=self.training)
            x2_emb = layers.dropout(x2_emb, p=self.dropout_emb, training=self.training)
            # if self.num_updates == 1425: utils.varname(x1_emb, fn=tensor_info)
            # if self.num_updates == 1425: utils.varname(x2_emb, fn=tensor_info)

        drnn_input_list.append(x1_emb)
        qrnn_input_list.append(x2_emb)

        # TODO: More word features

        x1_input = torch.cat(drnn_input_list, dim=2)    # torch.Size([batch, max_sentence_len, *])
        x2_input = torch.cat(qrnn_input_list, dim=2)    # torch.Size([batch * max_num_utterance, max_sentence_len, *])

        # if self.num_updates == 1425: utils.varname(x1_input, fn=tensor_info)
        # if self.num_updates == 1425: utils.varname(x2_input, fn=tensor_info)

        def expansion_for_doc(z):
            # expand torch.Size([batch, max_sentence_len, *]) into # torch.Size([batch * max_num_utterance, max_sentence_len, *])
            return z.unsqueeze(1).expand(z.size(0), x2_full.size(1), z.size(1), z.size(2)).contiguous().view(-1, z.size(1), z.size(2))

        x1_emb_expand = expansion_for_doc(x1_emb)   # torch.Size([batch * max_num_utterance, max_sentence_len, embedding_dim])

        # if self.num_updates == 1425: utils.varname(x1_emb_expand, fn=tensor_info)

        # TODO: test the performance of model with or without binary indicator em_(i,j)
        # Binary indicator em_(i,j), whether the j-th context word occurs in the i-th question
        # [batch, num_turns, context_len, 1]
        if self.no_em:
            x1_f = x1_f[:, :, :, 3:]
        elif not self.use_spacy:
            x1_f = torch.cat([x1_f[:, :, :, 0:1], x1_f[:, :, :, 3:]], dim=-1)

        x1_input = torch.cat([expansion_for_doc(x1_input), x1_f.view(-1, x1_f.size(-2), x1_f.size(-1))], dim=2) # torch.Size([batch * max_num_utterance, max_sentence_len, *])
        x1_len = x1_full_len.view(-1)  # torch.Size([batch * max_num_utterance])
        x1_mask = x1_full_mask.view(-1, x1_full_mask.size(-1))  # torch.Size([batch * max_num_utterance, max_sentence_len])

        # if self.num_updates == 1425: utils.varname(x1_input, fn=tensor_info)
        # if self.num_updates == 1425: utils.varname(x1_mask)

        if self.do_prealign:
            x1_atten = self.pre_align(x1_emb_expand, x2_emb, x2_mask)
            # if self.num_updates == 1425: utils.varname(x1_atten, fn=tensor_info)
            x1_input = torch.cat([x1_input, x1_atten], dim=2)
            # if self.num_updates == 1425: utils.varname(x1_input, fn=tensor_info)

        # === Start processing the dialog ===
        # cur_h: [batch_size * max_qa_pair, context_length, hidden_state]
        # flow : fn (rnn)
        # x1_full: [batch_size, max_qa_pair, context_length]
        # x2_full_num: [batch_size, context_length]
        no_dialog_flow = self.no_dialog_flow
        def flow_operation(cur_h, flow):
            flow_in = cur_h.view(x1_full.size(0), x1_full.size(1), x1_full.size(2), -1).transpose(1, 2).contiguous().view(
                x1_full.size(0) * x1_full.size(2), x1_full.size(1), -1) # [bsz * context_length, max_qa_pair, hidden_state]

            # if self.bidirectional_dialog_flow:
            #     flow_in_len = x2_full_num.view(-1)
            #     flow_in = torch.flip(flow_in, (1,))
            #     flow_out = flow(flow_in, x_len=flow_in_len)   # [bsz * context_length, max_qa_pair, flow_hidden_state_dim (hidden_state/2)]
            #     flow_out = torch.flip(flow_out, (1,))
            # else:
            #     flow_out = flow(flow_in)    # [bsz * context_length, max_qa_pair, flow_hidden_state_dim (hidden_state/2)]

            flow_out = flow(flow_in)  # [bsz * context_length, max_qa_pair, flow_hidden_state_dim (hidden_state/2)]

            if no_dialog_flow:
                flow_out = flow_out * 0

            flow_out = flow_out.view(x1_full.size(0), x1_full.size(2), x1_full.size(1), -1).transpose(1, 2).contiguous().view(
                x1_full.size(0) * x1_full.size(1), x1_full.size(2), -1) # [bsz * max_qa_pair, context_length, flow_hidden_state_dim]

            return flow_out

        # Encode document with RNN
        doc_abstr_ls = []

        doc_hiddens = self.doc_rnn1(x1_input, x_len=x1_len, x_mask=x1_mask)
        # if self.num_updates == 1425: utils.varname(doc_hiddens, fn=tensor_info)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow1)
        # if self.num_updates == 1425: utils.varname(doc_hiddens_flow, fn=tensor_info)

        doc_abstr_ls.append(doc_hiddens)

        doc_hiddens = self.doc_rnn2(torch.cat((doc_hiddens, doc_hiddens_flow), dim=2), x_len=x1_len, x_mask=x1_mask)
        # if self.num_updates == 1425: utils.varname(doc_hiddens, fn=tensor_info)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow2)
        # if self.num_updates == 1425: utils.varname(doc_hiddens_flow, fn=tensor_info)

        doc_abstr_ls.append(doc_hiddens)

        # Encode question with RNN
        _, que_abstr_ls = self.question_rnn(x2_input, x_len=x2_len, x_mask=x2_mask, return_list=True)

        # if self.num_updates == 1425: utils.varname(que_abstr_ls, fn=tensor_info)

        # Final question layer
        question_hiddens = self.high_lvl_qrnn(torch.cat(que_abstr_ls, 2), x_len=x2_len, x_mask=x2_mask)

        # if self.num_updates == 1425: utils.varname(question_hiddens, fn=tensor_info)

        que_abstr_ls += [question_hiddens]

        # Main Attention Fusion Layer
        doc_info = self.deep_attn([x1_emb_expand], doc_abstr_ls, [x2_emb], que_abstr_ls, x1_mask, x2_mask)
        # if self.num_updates == 1425: utils.varname(doc_info, fn=tensor_info)
        doc_hiddens = self.deep_attn_rnn(torch.cat((doc_info, doc_hiddens_flow), dim=2), x_len=x1_len, x_mask=x1_mask)
        # if self.num_updates == 1425: utils.varname(doc_hiddens, fn=tensor_info)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow3)
        # if self.num_updates == 1425: utils.varname(doc_hiddens_flow, fn=tensor_info)

        doc_abstr_ls += [doc_hiddens]

        # Self Attention Fusion Layer
        x1_att = torch.cat(doc_abstr_ls, 2)

        # if self.num_updates == 1425: utils.varname(x1_att, fn=tensor_info)

        if self.self_attention_opt > 0:
            highlvl_self_attn_hiddens = self.highlvl_self_att(x1_att, x1_att, x1_mask, x3=doc_hiddens,
                                                              drop_diagonal=True)
            # if self.num_updates == 1425: utils.varname(highlvl_self_attn_hiddens, fn=tensor_info)
            doc_hiddens = self.high_lvl_crnn(
                torch.cat([doc_hiddens, highlvl_self_attn_hiddens, doc_hiddens_flow], dim=2), x_len=x1_len, x_mask=x1_mask)

        elif self.self_attention_opt == 0:
            doc_hiddens = self.high_lvl_crnn(torch.cat([doc_hiddens, doc_hiddens_flow], dim=2), x_len=x1_len, x_mask=x1_mask)

        # if self.num_updates == 1425: utils.varname(doc_hiddens, fn=tensor_info)

        doc_abstr_ls += [doc_hiddens]

        # Merge the question hidden vectors
        q_merge_weights = self.self_attn(question_hiddens, x2_mask) # torch.Size([batch * max_num_utterance, max_sentence_len])
        # if self.num_updates == 1425: utils.varname(q_merge_weights, fn=tensor_info)
        question_avg_hidden = op.weighted_sum(q_merge_weights.unsqueeze(1), question_hiddens).squeeze(1)    # [bsz * max_qa_pair, hidden_state]
        # if self.num_updates == 1425: utils.varname(question_avg_hidden, fn=tensor_info)

        if self.do_hierarchical_query:
            # if self.bidirectional_dialog_flow:
            #     question_avg_hidden = torch.flip(question_avg_hidden.view(x1_full.size(0), x1_full.size(1), -1), (1,))  # [bsz, max_qa_pair, hidden_state]
            #     question_avg_hidden = self.hier_query_rnn(question_avg_hidden, x_len=x2_num)  # torch.Size([batch, max_num_utterance, *])
            #     question_avg_hidden = torch.flip(question_avg_hidden, (1,))  # [bsz, max_qa_pair, hidden_state]
            # else:
            #     question_avg_hidden = question_avg_hidden.view(x1_full.size(0), x1_full.size(1), -1)
            #     question_avg_hidden = self.hier_query_rnn(question_avg_hidden)  # torch.Size([batch, max_num_utterance, *])

            question_avg_hidden = question_avg_hidden.view(x1_full.size(0), x1_full.size(1), -1)
            question_avg_hidden = self.hier_query_rnn(question_avg_hidden)  # torch.Size([batch, max_num_utterance, *])

            # if self.num_updates == 1425: utils.varname(question_avg_hidden)
            question_avg_hidden = question_avg_hidden.contiguous().view(-1, question_avg_hidden.size(-1))   # torch.Size([batch * max_num_utterance, *])

        # if self.num_updates == 1425: utils.varname(question_avg_hidden, fn=tensor_info)
        # Get matching scores
        doc_avg_hidden = torch.cat((torch.max(doc_hiddens, dim=1)[0], torch.mean(doc_hiddens, dim=1)), dim=1)
        # if self.num_updates == 1425: utils.varname(doc_avg_hidden, fn=tensor_info)
        class_scores = self.separate_matching_score(doc_avg_hidden, question_avg_hidden)   # [batch * q_num, class_num]
        all_class_scores = class_scores.view(x1_full.size(0), x1_full.size(1), -1)  # [batch, q_num, class_num]
        all_class_scores = all_class_scores.squeeze(-1)  # when class_num = 1, [batch, q_num]

        # if self.num_updates == 1425: utils.varname(class_scores)
        # if self.num_updates == 1425: utils.varname(all_class_scores, fn=tensor_info)

        if self.last_score or not self.training:
        # TODO: Last score
            logits = all_class_scores[:, -1]
        else:
        # TODO: Linear combination
        #     utterance_mask = torch.min(x2_full_mask, dim=2)[0]
        #     all_class_scores.data.masked_fill_(utterance_mask, 0)
        #     logits = self.overall_matching_score(all_class_scores)  # [batch, final_out_features]
            logits = all_class_scores

        # if self.num_updates == 1425: utils.varname(logits, fn=tensor_info)

        return logits   # when final_out_features = 1, [batch]