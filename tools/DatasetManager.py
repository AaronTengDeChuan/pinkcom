# coding: utf-8

import torch
from torch.utils.data import TensorDataset, DataLoader
import json
from copy import deepcopy
from utils import utils
from utils.utils import varname


class DatasetManager(object):
    def __init__(self, data_dict=None, dataManagerParams={}):
        self.dataManagerParams = deepcopy(dataManagerParams)

        if data_dict is not None:
            self.data_dict = data_dict
            if "dataloader_gen" in dataManagerParams:
                self.dataloader_gen_fn = \
                    utils.name2function(dataManagerParams["dataloader_gen"]["function"])
                self.dataloaders = self.dataloader_gen_fn(
                        self.data_dict,
                        dataManagerParams["dataloader_gen"]["params"]
                    )

        if "data_gen" in dataManagerParams:
            self.data_gen_fn = \
                utils.name2function(dataManagerParams["data_gen"]["function"])
        
        self.dataiters = None
        self.iter_call_flag = False

    def _build_iter(self):
        self.dataiters = [iter(dataloader) \
                              if not isinstance(dataloader, torch.utils.data.Dataset) \
                              else dataloader \
                          for dataloader in self.dataloaders]
        
    def _getitem(self):
        return self.data_gen_fn(
            [next(iterator) \
                 if not isinstance(iterator, torch.utils.data.Dataset) \
                 else iterator \
             for iterator in self.dataiters],
            self.dataManagerParams["data_gen"]["params"] \
                if "data_gen" in self.dataManagerParams else {}
        )
    
    def set_dataloaders(self, dataloaders):
        assert isinstance(dataloaders, list) or isinstance(dataloaders, tuple) 
        self.dataloaders = dataloaders
        
    def set_data_gen_fn(self, data_gen_fn):
        self.data_gen_fn = data_gen_fn
    
    def __iter__(self):
        self.iter_call_flag = True
        self._build_iter()
        return self
    
    def __next__(self):
        if self.dataiters == None:
            self._build_iter()
        try:
            return self._getitem()
        except StopIteration:
            if self.iter_call_flag:
                self.iter_call_flag = False
                raise StopIteration
            else:
                self._build_iter()
                return self._getitem()


if __name__ == "__main__":
    # data Manager Params
    dataManagerParams = {
        "data_load": {
            "function": "utils.utils.ubuntu_data_load",
            "params": {
                "dataset_dir": "/Users/aaron_teng/Documents/SCIR/papers/Dialogue/SMN/Ubuntu",
                "training": True,
                "training_files": ["responses.pkl", "utterances.pkl"],
                "evaluate_files": ["Evaluate.pkl"],
                "max_sentence_len": 50
            }
        },
        "dataloader_gen": {
            "function": "utils.utils.ubuntu_dataloader_gen",
            "params": {
                "training": True,
                "batch_size": 10,
                "shuffle": False
            }
        },
        "data_gen": {
            "function": "utils.utils.ubuntu_data_gen",
            "params": {
                "training": True,
                "negative_samples": 1
            }
        }
    }


    ubuntu_manager = DatasetManager(dataManagerParams)

    for inputs in ubuntu_manager:
        for name, data in inputs.items():
            print(name + ':')
            varname(data)
        break