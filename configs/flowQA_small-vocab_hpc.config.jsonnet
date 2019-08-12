# small vocabulary training data follows the modified ESIM designed for multi-turn response selection

local dataset_dir = "/users3/dcteng/work/Dialogue/ESIM-Response-Selection/data/pickle_ubuntu_data";
local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/uncased_L-12_H-768_A-12";
local use_bert_embeddings = false;
local bert_trainable = false;
local optimizer_grouped_parameters_gen = {
    "optimizer_grouped_parameters_gen":{
        "function": "nets.BERT_SMN.optimizer_grouped_parameters",
        "params": {
            "bert": {
                "lr": 2e-5
            }
        }
    }
};

local log_dir = "logs/flowqa";
local model_dir = "models/FlowQA";
local test_model = "model.pkl";

local language = "zh";
local emb_trainable = false;
local bidirectional_dialog_flow = true;

local batch_size = 100;
local num_validation_examples = 500000;
local num_training_examples = 1000000;
//local num_validation_examples = 10000;
//local num_training_examples = 10000;
local num_log = 100;
local log_interval = num_training_examples / (batch_size * num_log);
local num_validation = 5;
local validation_interval = num_training_examples / (batch_size * num_validation);
local num_lr_step = 5;
local lr_step_size = std.floor(num_training_examples / (batch_size * num_lr_step));
local lr_gamma = 0.9;
local dropout = 0.0;

local model_name = "flowqa_small-vocab_emb-trainable-%s_use-bert-emb-%s_bidir-flow-%s_last-score-true_dropout-%s_valid%s_LS%s-%s"
    % [emb_trainable, use_bert_embeddings, bidirectional_dialog_flow, std.toString(dropout)[:3], num_validation, num_lr_step, std.toString(lr_gamma)[:3]];

{
    "data_manager":
        {
            "data_load": {
                "function": "utils.reader.DAM_ubuntu_data_load",
                "params": {
                    "dataset_dir": dataset_dir,
                    "phase": "training",
                    "training_files": ["data.pkl"],
                    "evaluate_files": ["test_data.pkl"],
                    "vocabulary_file": "word2id.txt",
//                    "training_files": ["data_small.pkl"],
//                    "evaluate_files": ["test_data_small.pkl"],
//                    "vocabulary_file": "word2id_small.txt",
                    "eos_id": 100000,
                    "empty_sequence_length": 1,
                    "max_num_utterance": 10,
                    "max_sentence_len": 50,
                    "use_bert_embeddings": use_bert_embeddings,
                    "bert_model_dir": bert_model_dir,
                    "do_lower_case": true,
                    "bert_max_sentence_len": 50
                }
            },
            "dataloader_gen": {
                "function": "utils.reader.DAM_ubuntu_dataloader_gen",
                "params": {
                    "phase": "training",
                    "batch_size": {
                        "validation": batch_size,
                        "evaluate": batch_size
                    },
                    "shuffle": {
                        "training": true,
                        "validation": false
                    }
                }
            },
            "data_gen": {
                "function": "utils.reader.DAM_ubuntu_data_gen",
                "params": {}
            }
        },
    "model":
        {
            "model_path": "nets.FlowQA.FlowQAModel",
            "params": {
                "vocabulary_size": 100000,
                "embedding_dim": 300,
                "hidden_size": 200,
                "bidirectional_dialog_flow": bidirectional_dialog_flow,
                "num_word_features": 4,
                "no_em": false,
                "no_dialog_flow": false,
                "do_prealign": true,
                "prealign_hidden": 200,
                "deep_inter_att_do_similar": false,
                "deep_att_hidden_size_per_abstr": 200,
                "self_attention_opt": true,
                "do_hierarchical_query": true,
                "max_num_utterance": 10,
                "max_sentence_len": 50,
                "do_seq_dropout": true,
                "my_dropout_p": dropout,
                "dropout_emb": dropout,
                "final_out_features": 1,
                "last_score": true,
                "emb_trainable": emb_trainable,
                "language": language,
                "use_bert_embeddings": use_bert_embeddings,
                "bert_hidden_size": 768,
                "bert_model_dir": bert_model_dir,
                "bert_trainable": bert_trainable
            },
            "loss": {
                "function": "losses.loss.BCEWithLogitsLoss",
                "params": {
                    "reduction": "elementwise_mean"
                }
            }
        },
    "global":
        {
            "device": "cuda",
            "batch_size": batch_size,
            "log_file": "%s/%s.log" % [log_dir, model_name]
        },
    "training":
        {
            "emb_load": {
                "function": "utils.reader.ubuntu_emb_load",
                "params": {
                    "path": "%s/%s" % [dataset_dir, "word_embedding.pkl"]
//                    "path": "%s/%s" % [dataset_dir, "word_embedding_small.pkl"],
                }
            },
            "optimizer": {
                "function": "optimizers.optimizer.AdamOptimizer",
                "params": {
                    "lr": 0.002,
                    "lr_scheduler": {
                        "function": "torch.optim.lr_scheduler.StepLR",
                        "params": {
                            "step_size": lr_step_size,
                            "gamma": lr_gamma
                        }
                    }
                }
            } + if use_bert_embeddings then optimizer_grouped_parameters_gen else {},
            "grad_clipping": 10,
            "continue_train": false,
            "start_epoch": 1,
            "training_num": num_training_examples,
            "validation_num": num_validation_examples,
            "num_epochs": 10,
            "early_stopping": num_validation,
            "log_interval": log_interval,
            "validation_interval": validation_interval,
            "model_selection": {
                "reduction": "sum",
                "mode": "max",
                "metrics": ["R10@1", "R10@2"]
            },
            "only_save_best": true,
            "model_save_path": "%s/%s" % [model_dir, model_name]
        },
    "metrics":
        {
            "training": [
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 2, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Accurary", "params": {} }
            ],
            "validation": [
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "AN": 2, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "K": 2, "skip": false} },
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "K": 5, "skip": false} },
                {   "function": "metrics.metric.MAP_in_N", "params": {"N": 10, "skip": false} },
                {   "function": "metrics.metric.MRR_in_N", "params": {"N": 10, "skip": false} },
                {   "function": "metrics.metric.Precision_N_at_K", "params": {"N": 10, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Accurary", "params": {} }
            ],
            "evaluate": [
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "AN": 2, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "K": 2, "skip": false} },
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 10, "K": 5, "skip": false} },
                {   "function": "metrics.metric.MAP_in_N", "params": {"N": 10, "skip": false} },
                {   "function": "metrics.metric.MRR_in_N", "params": {"N": 10, "skip": false} },
                {   "function": "metrics.metric.Precision_N_at_K", "params": {"N": 10, "K": 1, "skip": false} },
                {   "function": "metrics.metric.Accurary", "params": {} }
            ]
        },
    "evaluate":
        {
            "test_model_file": model_dir + "/" + model_name + "/" + test_model,
            "output_result": {
                "function": "nets.DAM.output_result",
                "params": {
                    "file_name": "%s/%s/%s" % [model_dir, model_name, "result.txt"]
                }
            }
        }
}
