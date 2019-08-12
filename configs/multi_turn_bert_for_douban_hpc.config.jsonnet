local dataset_dir = "/users3/dcteng/work/pinkcom/data/dam/douban";
local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/chinese_L-12_H-768_A-12";
local use_bert_embeddings = true;

local training_shuffle = true;
local final_out_features = 1;
local loss_fn = if final_out_features == 1 then "losses.loss.BCEWithLogitsLoss" else "losses.loss.CrossEntropyLoss";

local log_dir = "logs/multi-turn-bert";
local model_dir = "models/MultiTurnBert";
local test_model = "model.pkl";

local language = "zh";

local batch_size = 10;
//local num_validation_examples = 1000;
//local num_training_examples = 10000;
local num_validation_examples = 50000;
local num_training_examples = 1000000;
local t_total = num_training_examples / batch_size;
local num_log = 100;
local log_interval = num_training_examples / (batch_size * num_log);
local num_validation = 4;
local validation_interval = num_training_examples / (batch_size * num_validation);
local num_lr_step = 5;
local lr_step_size = std.floor(num_training_examples / (batch_size * num_lr_step));
local lr_gamma = 1.0;

local model_name = "multi-turn-bert_for_douban-multi-turn-response-selection";

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
                    "empty_sequence_length": 0,
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
                        "validation": 10,
                        "evaluate": 10
                    },
                    "shuffle": {
                        "training": training_shuffle,
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
            "model_path": "nets.BertDownstream.MultiTurnBert",
            "params": {
                "max_num_utterance": 10,
                "max_sentence_len": 50,
                "bert_hidden_size": 768,
                "rnn_units": 768,
                "final_out_features": final_out_features,
                "bert_model_dir": bert_model_dir
            },
            "loss": {
                "function": loss_fn,
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
            "optimizer": {
                "optimizer_grouped_parameters_gen": {
                    "function": "nets.BertDownstream.optimizer_grouped_parameters",
                    "params": {}
                },
                "function": "optimizers.bert_optimization.BertAdam",
                "params":{
                    "lr": 1e-6,
                    "warmup": 0.1,
                    "t_total": t_total,
                    "lr_scheduler": {
                        "function": "torch.optim.lr_scheduler.StepLR",
                        "params": {
                            "step_size": lr_step_size,
                            "gamma": lr_gamma
                        }
                    }
                }
            },
            "grad_clipping": 10,
            "continue_train": false,
            "start_epoch": 1,
            "training_num": num_training_examples,
            "validation_num": num_validation_examples,
            "num_epochs": 3,
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
            "test_model_file": "%s/%s/%s" % [model_dir, model_name, test_model]
        }
}
