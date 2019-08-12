local dataset_dir = "/users3/dcteng/work/PengChengProject/data";
local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/chinese_L-12_H-768_A-12";
local log_dir = "/users3/dcteng/work/PengChengProject/logs/bert";
local model_dir = "/users3/dcteng/work/PengChengProject/models/Bert";
local test_model = "model.pkl";

local batch_size = 10;
local num_validation_examples = 100000;
local num_training_examples = 160000;
local t_total = num_training_examples / batch_size;
local num_log = 100;
local log_interval = num_training_examples / (batch_size * num_log);
local num_validation = 2;
local validation_interval = num_training_examples / (batch_size * num_validation);
local num_lr_step = 5;
local lr_step_size = std.floor(num_training_examples / (batch_size * num_lr_step));
local lr_gamma = 1.0;

local model_name = "chinese_bert_for_sequence_classification";

{
    "data_manager":
        {
            "data_load": {
                "function": "utils.bert_reader.Ubuntu_data_load_for_bert_sequence_classification",
                "params": {
                    "dataset_dir": dataset_dir,
                    "phase": "training",
                    "training_files": ["train_small.txt", "valid_small.txt"],
                    "evaluate_files": ["test_small.txt"],
                    "bert_model_dir": bert_model_dir,
                    "do_lower_case": true,
                    "separation": false,
                    "max_seq_length": 512,
                    "max_response_length": 64
                }
            },
            "dataloader_gen": {
                "function": "utils.bert_reader.Ubuntu_dataloader_gen",
                "params": {
                    "phase": "training",
                    "batch_size": {
                        "validation": 10,
                        "evaluate": 10
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
            "model_path": "nets.BertDownstream.BertForMultiTurnResponseSelection",
            "params": {
                "final_out_features": 2,
                "bert_model_dir": bert_model_dir
            },
            "loss": {
                "function": "losses.loss.CrossEntropyLoss",
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
                    "function": "nets.BertDownstream.optimizer_grouped_parameters",
                    "params": {}
                },
                "function": "optimizers.bert_optimization.BertAdam",
                "params":{
                    "lr": 2e-5,
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
            "validation_num": num_validation_examples,
            "num_epochs": 2,
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
            "test_model_file": "%s/%s/%s" % [model_dir, model_name, test_model],
            "output_result": {
                "function": "nets.BertDownstream.output_result",
                "params": {
                    "file_name": "%s/%s/%s" % [model_dir, model_name, "result.txt"],
                    "bert_model_dir": bert_model_dir,
                    "do_lower_case": true
                }
            }
        }
}
