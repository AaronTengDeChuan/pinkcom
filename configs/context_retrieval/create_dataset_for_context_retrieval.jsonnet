local bert_model_dir = "/users3/dcteng/work/Dialogue/bert/bert-pretrained-models/chinese_L-12_H-768_A-12";
local dataset_dir = "/users3/dcteng/work/pinkcom/data/dua/e-commerce";
local target_dir = "/users3/dcteng/work/pinkcom/data/dua/e-commerce_lu_augumentation";


{
    "data": {
        "dataset_dir": dataset_dir,
        "target_dir": target_dir,
        "training": {
            "files": ["train.txt"],
            "vecs": ["train.vec"],
            "outs": ["train.cr.txt"],
            "steps": [2]
        },
        "validation": {
            "files": ["dev.txt"],
            "vecs": ["dev.vec"],
            "outs": ["dev.cr.txt"],
            "steps": [2]
        },
        "evaluation": {
            "files": ["test.txt", "labeled_test.txt"],
            "vecs": ["test.vec", "labeled_test.vec"],
            "outs": ["test.cr.txt", "labeled_test.cr.txt"],
            "steps": [10, 10]
        },
        "bert_model_dir": bert_model_dir,
        "max_seq_length": 50,
    },
    "global": {
        "batch_size": 500,
        "log_file": "/users3/dcteng/work/pinkcom/logs/context_retrieval/create_dataset.log"
    }
}
