local dataset_dir = "data/dam/douban";
local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/chinese_L-12_H-768_A-12";
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

local log_dir = "logs/dam";
local model_dir = "models/DAM";
local test_model = "model.pkl";

local language = "zh";

local batch_size = 100;
local num_validation_examples = 50000;
local num_training_examples = 1000000;
local num_log = 100;
local log_interval = num_training_examples / (batch_size * num_log);
local num_validation = 5;
local validation_interval = num_training_examples / (batch_size * num_validation);
local num_lr_step = 15;
local lr_step_size = std.floor(num_training_examples / (batch_size * num_lr_step));
local lr_gamma = 0.9;
local dropout = 0.0;

local model_name = "dam_douban_use-bert-emb-%s_last-score-true_dropout-%s_LS%s-%s_minus1example_gc-0" % [use_bert_embeddings, std.toString(dropout)[:3], num_lr_step, std.toString(lr_gamma)[:3]];

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
                    "vocabulary_file": "word2id_modified.txt",
                    "eos_id": 1,
//                    "empty_sequence_length": 1,
                    "max_num_utterance": 10,
                    "max_sentence_len": 50,
                    "language": language,
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
            "model_path": "nets.DAM.DAMModel",
            "params": {
                "vocabulary_size": 172131,
                "embedding_dim": 200,
                "hidden_size": 200,
                "max_num_utterance": 10,
                "max_sentence_len": 50,
                "is_positional": false,
                "head_num": 0,
                "stack_num": 5,
                "is_layer_norm": true,
                "attention_type": "dot",
                "is_mask": true,
                "final_out_features": 1,
                "emb_trainable": true,
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
                    "path": "data/dam/douban/word_embedding.pkl"
                }
            },
            "optimizer": {
                "function": "optimizers.optimizer.AdamOptimizer",
                "params": {
                    "lr": 0.001,
                    "lr_scheduler": {
                        "function": "torch.optim.lr_scheduler.StepLR",
                        "params": {
                            "step_size": lr_step_size,
                            "gamma": lr_gamma
                        }
                    }
                }
            } + if use_bert_embeddings then optimizer_grouped_parameters_gen else {},
            "grad_clipping": 0,
            "continue_train": false,
            "start_epoch": 1,
            "training_num": num_training_examples,
            "validation_num": num_validation_examples,
            "num_epochs": 2,
            "early_stopping": num_validation,
            "log_interval": log_interval,
            "validation_interval": validation_interval,
            "model_selection": {
                "reduction": "sum",
                "mode": "max",
                "metrics": ["R2@1"]
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
                {   "function": "metrics.metric.Recall_N_at_K", "params": {"N": 2, "K": 1, "skip": false} },
                {   "function": "metrics.metric.MAP_in_N", "params": {"N": 2, "skip": false} },
                {   "function": "metrics.metric.MRR_in_N", "params": {"N": 2, "skip": false} },
                {   "function": "metrics.metric.Precision_N_at_K", "params": {"N": 2, "K": 1, "skip": false} },
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
