local dataset_dir = "data/dam/ubuntu";
local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/uncased_L-12_H-768_A-12";
local log_dir = "logs/bert_flowqa";
local model_dir = "models/Bert_FlowQA";
local test_model = "model.pkl";

local batch_size = 10;
local num_validation_examples = 500000;
local num_training_examples = 1000000;
local num_log = 100;
local log_interval = num_training_examples / (batch_size * num_log);
local num_validation = 3;
local validation_interval = num_training_examples / (batch_size * num_validation);
local num_lr_step = 5;
local lr_step_size = std.floor(num_training_examples / (batch_size * num_lr_step));
local lr_gamma = 1.0;

// local model_name = "flowqa_last-score-true" + "_ls" + lr_step_size + "-" + std.toString(lr_gamma)[:3];
local model_name = "flowqa_with-trainable-bert_last-score-true_bs-10";

{
    "data_manager":
        {
            "data_load": {
                "function": "utils.bert_reader.Ubuntu_data_load",
                "params": {
                    "dataset_dir": dataset_dir,
                    "phase": "training",
                    "training_files": ["train.txt", "valid.txt"],
                    "evaluate_files": ["test.txt"],
                    "bert_model_dir": bert_model_dir,
                    "do_lower_case": true,
                    "empty_sequence_length": 1,
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
                "params": {}
            }
        },
    "model":
        {
            "model_path": "nets.Bert_FlowQA.BertFlowQAModel",
            "params": {
                "bert_hidden_size": 768,
                "word_embedding_size": 768,
                "hidden_size": 200,
                "num_word_features": 4,
                "no_em": false,
                "no_dialog_flow": false,
                "do_prealign": true,
                "prealign_hidden": 200,
                "deep_inter_att_do_similar": false,
                "deep_att_hidden_size_per_abstr": 250,
                "self_attention_opt": true,
                "do_hierarchical_query": true,
                "max_num_utterance": 10,
                "max_sentence_len": 50,
                "do_seq_dropout": true,
                "my_dropout_p": 0.0,
                "dropout_emb": 0.0,
                "final_out_features": 1,
                "last_score": true,
                "bert_model_dir": bert_model_dir,
                "bert_trainable": true
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
                "params": {
                    "lr": 2e-5,
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
            "validation_num": num_validation_examples,
            "num_epochs": 5,
            "early_stopping": 5,
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
            "test_result_file": "result_" + model_name
        }
}
