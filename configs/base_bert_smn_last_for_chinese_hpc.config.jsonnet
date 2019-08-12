local dataset_dir = "/users3/dcteng/work/PengChengProject/data";
local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/chinese_L-12_H-768_A-12";
local log_dir = "/users3/dcteng/work/PengChengProject/logs/bert_smn";
local model_dir = "/users3/dcteng/work/PengChengProject/models/BertSMN";
local test_model = "model.pkl";

local batch_size = 100;
local num_validation_examples = 100000;
local num_training_examples = 160000;
local num_log = 100;
local log_interval = num_training_examples / (batch_size * num_log);
local num_validation = 4;
local validation_interval = num_training_examples / (batch_size * num_validation);
local num_lr_step = 5;
local lr_step_size = std.floor(num_training_examples / (batch_size * num_lr_step));
local lr_gamma = 1.0;

local model_name = "chinese_bert_smn";

{
    "data_manager":
        {
            "data_load": {
                "function": "utils.bert_reader.Ubuntu_data_load",
                "params": {
                    "dataset_dir": dataset_dir,
                    "phase": "training",
                    "training_files": ["train_small.txt", "valid_small.txt"],
                    "evaluate_files": ["test_small.txt"],
                    "bert_model_dir": bert_model_dir,
                    "do_lower_case": true,
                    "max_num_utterance": 10,
                    "max_sentence_len": 50
                }
            },
            "dataloader_gen": {
                "function": "utils.bert_reader.Ubuntu_dataloader_gen",
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
                "function": "utils.bert_reader.Ubuntu_data_gen",
                "params": {
                    "phase": "training"
                }
            }
        },
    "model":
        {
            "model_path": "nets.BERT_SMN.BERTSMNModel",
            "params": {
                "bert_hidden_size": 768,
                "hidden_size": 200,
                "rnn_units": 200,
                "bert_layers": [11],
                "feature_maps": 8,
                "dense_out_dim": 50,
                "drop_prob": 0.0,
                "max_num_utterance": 10,
                "max_sentence_len": 50,
                "final_out_features": 1,
                "bert_model_dir": bert_model_dir,
                "bert_trainable": true
            },
            "loss": {
                "function": "losses.loss.BCEWithLogitsLoss",
                "params": {
                    "ignore_index": -100,
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
            "optimizer": {
                "optimizer_grouped_parameters_gen": {
                    "function": "nets.BERT_SMN.optimizer_grouped_parameters",
                    "params": {
                        "bert": {
                            "lr": 2e-5
                        }
                    }
                },
                "function": "optimizers.optimizer.AdamOptimizer",
                "params":{
                    "lr": 5e-4,
                    "warmup": 0.1,
                    "t_total": 100000,
                    "lr_scheduler": {
                        "function": "torch.optim.lr_scheduler.StepLR",
                        "params": {
                            "step_size": lr_step_size,
                            "gamma": lr_gamma
                        }
                    }
                }
            },
            "continue_train": false,
            "start_epoch": 1,
            "validation_num": num_validation_examples,
            "num_epochs": 12,
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
            "test_model_file": "%s/%s/%s" % [model_dir, model_name, test_model],
            "test_result_file": "result_smn_last"
        }
}
