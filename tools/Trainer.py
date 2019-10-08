from tools.DatasetManager import DatasetManager
from utils import utils
import time
import os
import json
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from optimizers.optimizer import Optimizer

logger = utils.get_logger()

class Trainer(object):
    '''
    A Guide to Trainer:
    * mode: Training
        1. Initialization
            trainer = Trainer(trainerParams)
        2. Load training data (and evaluate data if required)
            trainer.load_data()
            optional: trainer.load_data(training_data=False)
        3. Generate data managers
            trainer.generate_data_managers()
        4. Start training
            trainer.train()
        5. Evaluate trained model
            trainer.load_model()
            trainer.evaluate()

    * mode: Evaluate
        1. Initialization
            trainer = Trainer(trainerParams, training=False)
        2. Load evaluate data
            trainer.load_data(training_data=False)
        3. Generate data managers
            trainer.generate_data_managers()
        4. Start evaluating
            trainer.evaluate()
    '''

    def __init__(self, trainerParams, training=True):
        # TODO: computation in cuda
        self.trainerParams = trainerParams
        self.saved_config = deepcopy(trainerParams)
        self.training = training
        self.fine_tune = False
        self.continue_train = False

        self.training_data_dict = {}
        self.validation_data_dict = {}
        self.evaluate_data_dict = {}

        self.training_data_manager = None
        self.validation_data_manager = None
        self.evaluate_data_manager = None

        self._set_torch_variables()
        self._load_metrics()

        # Set the random seed manually for reproducibility.
        # torch.manual_seed(args.seed)

        if self.training:
            self.best_epoch = -1
            self.start_epoch = self.trainerParams["training"]["start_epoch"] \
                if "start_epoch" in self.trainerParams["training"] else 1
            # load embedding
            embeddings = None
            if "emb_load" in self.trainerParams["training"]:
                emb_load_params = self.trainerParams["training"]["emb_load"]
                logger.info("Loading embedding file '{}'...".format(emb_load_params["params"]["path"]))
                embeddings = utils.name2function(emb_load_params["function"])(emb_load_params["params"])
                logger.info("loading embedding has been completed.")
            # create model
            model_params = self.trainerParams["model"]
            self._creat_model(embeddings=embeddings)
            # define loss function
            logger.info("Defining loss function '{}' with params:\n{}".format(model_params["loss"]["function"],
                                                                              json.dumps(model_params["loss"]["params"],
                                                                                         indent=4)))
            self.loss_fn = utils.name2function(model_params["loss"]["function"])(model_params["loss"]["params"])
            logger.info("Defining loss function has been completed.")
            # define optimizer
            optimizer_params = deepcopy(self.trainerParams["training"]["optimizer"])
            logger.info("Defining optimizer '{}' with params:\n{}".format(optimizer_params["function"],
                                                                          json.dumps(optimizer_params["params"],
                                                                                     indent=4)))
            optimizer_grouped_parameters = [{"params": self.model.parameters()}]
            if "optimizer_grouped_parameters_gen" in optimizer_params:
                gen_params = optimizer_params["optimizer_grouped_parameters_gen"]
                optimizer_grouped_parameters = utils.name2function(gen_params["function"])(
                    self.model.module, gen_params["params"], model_params["params"])
            self.lr_scheduler, self.optimizer = Optimizer(optimizer_params).ops(optimizer_grouped_parameters)
            logger.info("Defining optimizer has been completed.")
        else:
            # create model
            self._creat_model()
            # load model
            self.load_model()

    def _set_torch_variables(self):
        # set default dtype as torch.float64
        # torch.set_default_dtype(torch.float64)

        global_params = self.trainerParams["global"]

        # set the random seed manually for reproducibility
        random_seed = global_params["random_seed"] if "random_seed" in global_params else 1111
        np.random.seed(random_seed)
        logger.info("Set numpy random seed: {}.".format(random_seed))
        torch.manual_seed(random_seed)
        logger.info("Set torch random seed: {}.".format(random_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            logger.info("Set torch cuda random seed: {}.".format(random_seed))

        # set computing device
        if "device" in global_params and global_params["device"].lower() in ["cuda", "cpu"]:
            if torch.cuda.is_available():
                if global_params["device"].lower() == "cuda":
                    self.device = torch.device("cuda")
                    logger.info("Cuda is avaliable and use computing device 'cuda'.")
                else:
                    logger.warning("Cuda is now avaliable. Instead of specifying 'cpu', you can use 'cuda' for computing.")
                    self.device = torch.device("cpu")
            else:
                logger.info("Cuda is not avaliable and use computing device 'cpu'.")
                self.device = torch.device("cpu")
        else:
            logger.warning(
                "No computing device is specified or the name of specified device is incorrect.\nUse the default device 'cpu'")
            self.device = torch.device("cpu")

        # torch.backends.cudnn.enabled = False

    def _creat_model(self, embeddings=None):
        model_params = deepcopy(self.trainerParams["model"])
        ensemble = 'model_0' in model_params
        if not ensemble:
            logger.info("Creating model '{}' with params:\n{}".format(model_params["model_path"],
                                                                      json.dumps(model_params["params"], indent=4)))
            model_params["params"]["device"] = self.device
            model_params["params"]["embeddings"] = embeddings
            # put model on specified computing device
            self.model = utils.name2function(model_params["model_path"])(model_params["params"]).to(device=self.device)
            self.model = nn.DataParallel(self.model)
            logger.info("Creating model has been completed.")
        else:
            self.model = []
            for model_name in model_params:
                logger.info("Creating model '{}' with params:\n{}".format(model_params[model_name]["model_path"],
                                                                          json.dumps(model_params[model_name]["params"], indent=4)))
                model_params[model_name]["params"]["device"] = self.device
                model_params[model_name]["params"]["embeddings"] = embeddings
                # put model on specified computing device
                model = utils.name2function(model_params[model_name]["model_path"])(model_params[model_name]["params"]).to(
                    device=self.device)
                model = nn.DataParallel(model)
                self.model.append(model)
                logger.info("Creating model has been completed.")

    def _load_metrics(self):
        metrics_params = self.trainerParams["metrics"]
        self.metrics = {}
        for mode, metrics in metrics_params.items():
            self.metrics[mode] = [utils.name2function(metric["function"])(metric["params"]) for metric in metrics]

    def _split_data_into_training_validation(self, data_dict, validation_num, training_num=None):
        # TODO: check this
        for key in data_dict.keys():
            if data_dict[key]["type"].lower() == "normal":
                assert validation_num <= data_dict[key]["data"].shape[0]
                if validation_num > 0: self.validation_data_dict[key] = data_dict[key]["data"][:validation_num]
                if training_num is None:
                    self.training_data_dict[key] = data_dict[key]["data"][validation_num:]
                else:
                    assert training_num <= data_dict[key]["data"].shape[0] - validation_num
                    self.training_data_dict[key] = data_dict[key]["data"][-training_num:]
            elif data_dict[key]["type"].lower() == "share":
                if validation_num > 0: self.validation_data_dict[key] = data_dict[key]["data"]
                self.training_data_dict[key] = data_dict[key]["data"]
            else:
                assert False, "{} is not a valid type.".format(key)

    def load_data(self, training_data=True):
        if "data_load" in self.trainerParams["data_manager"]:
            data_load_params = deepcopy(self.trainerParams["data_manager"]["data_load"])
            data_load_fn = utils.name2function(data_load_params["function"])
            if self.training and training_data:
                logger.info("Loading training and validation data...")
                data_load_params["params"]["phase"] = "training"  # data_load.params.phase is changed
                training_num = self.trainerParams["training"]["training_num"] \
                    if "training_num" in self.trainerParams["training"] else None
                validation_num = self.trainerParams["training"]["validation_num"]
                data_dict = data_load_fn(data_load_params["params"])
                self._split_data_into_training_validation(data_dict, validation_num, training_num)
                logger.info("Loading training and validation data has been completed.")
            else:
                logger.info("Loading evaluate data...")
                data_load_params["params"]["phase"] = "evaluate"  # data_load.params.phase is changed
                self.evaluate_data_dict = data_load_fn(data_load_params["params"])
                logger.info("Loading evaluate data has been completed.")
        else:
            logger.error("Please speicfy data managers for training, validation or test set.")

    def set_data_dict(self, training_data_dict=None, validation_data_dict=None, evaluate_data_dict=None):
        """
        {
            "name_str": data
            ...
        }
        """
        self.training_data_dict = training_data_dict
        self.validation_data_dict = validation_data_dict
        self.evaluate_data_dict = evaluate_data_dict

    def generate_data_managers(self):
        data_manager_params = deepcopy(self.trainerParams["data_manager"])
        data_manager_params["dataloader_gen"]["params"]["batch_size"]["training"] = self.trainerParams["global"]["batch_size"]
        # put data on specified computing device
        data_manager_params["data_gen"]["params"]["device"] = self.device
        # training data manager
        if len(self.training_data_dict):
            logger.info("Generating training data manager...")
            data_manager_params["dataloader_gen"]["params"]["phase"] = "training"
            data_manager_params["data_gen"]["params"]["phase"] = "training"
            self.training_data_manager = DatasetManager(self.training_data_dict, data_manager_params)
            logger.info("Generating training data manager has been completed")
        # validation data manager
        if len(self.validation_data_dict):
            logger.info("Generating validation data manager...")
            data_manager_params["dataloader_gen"]["params"]["phase"] = "validation"
            data_manager_params["data_gen"]["params"]["phase"] = "validation"
            self.validation_data_manager = DatasetManager(self.validation_data_dict, data_manager_params)
            logger.info("Generating validation data manager has been completed")
        # evaluate data manager
        if len(self.evaluate_data_dict):
            logger.info("Generating evaluate data manager...")
            data_manager_params["dataloader_gen"]["params"]["phase"] = "evaluate"
            data_manager_params["data_gen"]["params"]["phase"] = "evaluate"
            self.evaluate_data_manager = DatasetManager(self.evaluate_data_dict, data_manager_params)
            logger.info("Generating evaluate data manager has been completed")

    def _generate_model_name(self):
        model_path = self.trainerParams["training"]["model_save_path"]
        only_save_best = self.trainerParams["training"]["only_save_best"] \
            if "only_save_best" in self.trainerParams["training"] else True
        model_files = {}
        if self.best_epoch >= 1:
            suffix = ".pkl" if only_save_best else "_epoch{}.pkl".format(self.best_epoch)
            model_files["model"] = os.path.join(model_path, "model") + suffix
            model_files["optimizer"] = os.path.join(model_path, "optimizer") + suffix
            model_files["lr_scheduler"] = os.path.join(model_path, "lr_scheduler") + suffix
            model_files["config"] = os.path.join(model_path, "config.json")
            return model_files
        else:
            logger.error("Best epoch is None.")

    def load_model(self):
        if self.training:
            if self.fine_tune:
                # load fine-tuned model: load only model's parameters
                logger.info("Loading model from {} ...".format(self.fine_tune_model_path))
                self.model.load_state_dict(torch.load(self.fine_tune_model_path))
            else:
                # load model in training
                logger.info("Loading model from {} ...".format(self.trainerParams["training"]["model_save_path"]))
                model_files = self._generate_model_name()
                self.model.load_state_dict(torch.load(model_files["model"]))
                self.lr_scheduler.load_state_dict(torch.load(model_files["lr_scheduler"]))
                self.optimizer.load_state_dict(torch.load(model_files["optimizer"]))
        else:
            # load model in evaluate
            model_file = self.trainerParams["evaluate"]["test_model_file"]
            ensemble = 'file_0' in model_file
            if not ensemble:
                logger.info("Loading model from {} ...".format(model_file))
                self.model.load_state_dict(torch.load(model_file))
            else:
                for model_idx, file_name in enumerate(model_file):
                    logger.info("Loading model from {} ...".format(model_file[file_name]))
                    self.model[model_idx].load_state_dict(torch.load(model_file[file_name]))

    def save_model(self):
        model_path = self.trainerParams["training"]["model_save_path"]
        os.makedirs(model_path, exist_ok=True)
        model_files = self._generate_model_name()
        # Save only the model parameters
        torch.save(self.model.state_dict(), model_files["model"])
        torch.save(self.lr_scheduler.state_dict(), model_files["lr_scheduler"])
        torch.save(self.optimizer.state_dict(), model_files["optimizer"])
        # Save model config
        logger.info("Whether TrainerParams is changed during training: {}".format(self.saved_config != self.trainerParams))
        json.dump(self.saved_config, open(model_files["config"], 'w'), indent=4)


    def _get_validation_metric(self, results, loss):
        '''
        model_selection_params = {
            "reduction" contains "sum", "mean" and so on,
            "mode" contain  "max", "min", and "loss",
            "metrics" is a list of names of metrics
        }
        '''
        model_selection_params = self.trainerParams["training"]["model_selection"] \
            if "model_selection" in self.trainerParams["training"] else None

        if model_selection_params == None \
                or "mode" not in model_selection_params or model_selection_params["mode"] == "loss" \
                or "metrics" not in model_selection_params or not isinstance(model_selection_params["metrics"], list) \
                or len(model_selection_params["metrics"]) == 0 or "reduction" not in model_selection_params:
            return loss
        return utils.calculate_metric_with_params(results, model_selection_params)


    def train(self):
        if not self.training: logger.error("Mode: Evaluate.")
        training_start_time = time.time()
        training_params = deepcopy(self.trainerParams["training"])
        training_params["num_epochs"] = int(training_params["num_epochs"])
        training_params["log_interval"] = int(training_params["log_interval"])
        training_params["validation_interval"] = int(training_params["validation_interval"])
        training_params["gradient_accumulation"] = int(
            training_params["gradient_accumulation"]) if "gradient_accumulation" in training_params else 1
        best_valid_metric = None
        self.early_stopping_count = 0
        self.early_stopping_flag = False
        # Whether continuing training
        if "continue_train" in training_params and training_params["continue_train"]:
            self.continue_train = True
            self.best_epoch = self.start_epoch
            self.load_model()
            # evaluate existing model
            valid_loss, results, best_valid_metric = self.evaluate(
                validation=True if self.validation_data_manager is not None else False)
            self.continue_train = False  # change continue train flag for evaluating trained model

        if "fine_tune" in training_params and training_params["fine_tune"] is not None:
            self.fine_tune = True
            self.fine_tune_model_path = training_params["fine_tune"]
            self.load_model()
            # evaluate existing model
            valid_loss, results, best_valid_metric = self.evaluate(
                validation=True if self.validation_data_manager is not None else False)
            self.save_model()
            self.fine_tune = False  # change fine-tune flag for evaluating trained model

        data_parallel = isinstance(self.model, nn.DataParallel)
        if not hasattr(self.model.module if data_parallel else self.model, "num_updates"):
            setattr(self.model.module if data_parallel else self.model, "num_updates", 0)

        self.epoch_total_samples = self.training_data_manager.num_samples
        self.epoch_total_batches = int(self.epoch_total_samples / self.training_data_manager.batch_size)
        self.epoch_total_labels = 0

        # self.model = nn.DataParallel(self.model)

        for epoch in range(self.start_epoch, training_params["num_epochs"] + 1):
            if self.early_stopping_flag: break
            # epoch level
            epoch_start_time = time.time()
            epoch_total_samples = 0
            epoch_total_batches = 0
            epoch_total_labels = 0
            epoch_total_loss = 0
            # log level
            log_start_time = epoch_start_time
            log_total_labels = 0
            log_total_loss = 0

            self.optimizer.zero_grad()
            for batch_in_epoch, inputs in enumerate(self.training_data_manager):
                if self.early_stopping_flag: break
                self.model.train()

                pred = self.model(inputs)
                loss, num_labels, batch_total_loss = self.loss_fn(pred, inputs["target"])

                if isinstance(loss, torch.Tensor): loss.backward()

                if (batch_in_epoch+1) % training_params["gradient_accumulation"] == 0:
                    # utils.output_model_params_and_grad(self.model)
                    if "grad_clipping" in training_params and training_params["grad_clipping"] > 0:
                        # nn.utils.clip_grad_norm_(self.model.parameters(), training_params["grad_clipping"])
                        nn.utils.clip_grad_value_(self.model.parameters(), training_params["grad_clipping"])
                    self.lr_scheduler.step()
                    self.optimizer.step()

                    self.optimizer.zero_grad()

                if data_parallel: self.model.module.num_updates += 1
                else: self.model.num_updates += 1

                del pred, loss

                # epoch level
                epoch_total_samples += inputs["target"].shape[0]
                epoch_total_batches += 1
                epoch_total_labels += num_labels
                epoch_total_loss += batch_total_loss
                # log level
                log_total_labels += num_labels
                log_total_loss += batch_total_loss

                if (batch_in_epoch + 1) % training_params["log_interval"] < 1:
                    # training log
                    cur_loss = log_total_loss / log_total_labels
                    elapsed = time.time() - log_start_time
                    speed = elapsed * 1000 / training_params["log_interval"]
                    logger.info(
                        "\n| epoch {:3d} | {:5d}/{:5d} batches | time {:5.2f}s | {:5.2f} ms/batch | loss {:8.5f} "
                        "| lr {} | best model in epoch {:3d}".format(
                            epoch, batch_in_epoch + 1, self.epoch_total_batches, elapsed, speed, cur_loss,
                            self.optimizer.get_lr()[0], self.best_epoch))
                    # log level
                    log_start_time = time.time()
                    log_total_labels = 0
                    log_total_loss = 0

                if (batch_in_epoch + 1) % training_params["validation_interval"] == 0:
                    # validation
                    valid_loss, results, valid_metric = self.evaluate(
                        validation=True if self.validation_data_manager is not None else False)
                    if best_valid_metric == None or valid_metric < best_valid_metric:
                        self.early_stopping_count = 0
                        self.best_epoch = epoch
                        best_valid_metric = valid_metric
                        logger.info("| epoch {:3d} | batch {:5d} | New record has been achieved. |".format(epoch,
                                                                                                           batch_in_epoch + 1))
                        logger.info("Saving model...")
                        self.save_model()
                        logger.info(
                            "Best model is saved in \n'{}'".format(json.dumps(self._generate_model_name(), indent=4)))
                    elif "early_stopping" in training_params and training_params["early_stopping"] > 0:
                        # early stopping
                        self.early_stopping_count += 1
                        if self.early_stopping_count >= training_params["early_stopping"]:
                            self.early_stopping_flag = True
                            logger.warning("\n| Early stopping mechanism start up. "
                                           "| Recent {} times, the performance in validation set has not been improved."
                                           "".format(self.early_stopping_count))
                            logger.warning("\n| Best model is saved in \n'{}'".format(
                                json.dumps(self._generate_model_name(), indent=4)))

                            # epoch level
            self.epoch_total_samples = epoch_total_samples
            self.epoch_total_batches = epoch_total_batches
            self.epoch_total_labels = epoch_total_labels

            cur_loss = epoch_total_loss / epoch_total_labels
            elapsed = time.time() - epoch_start_time
            speed = elapsed * 1000 / epoch_total_batches
            if (batch_in_epoch + 1) % training_params["validation_interval"] / training_params[
                "validation_interval"] >= 0.5:
                valid_loss, results, valid_metric = self.evaluate(
                    validation=True if self.validation_data_manager is not None else False)
            logger.info(
                "\n| end of epoch {:3d} | {:8d} samples, {:8d} batches | time {} | {:5.2f} ms/batch "
                "| training loss {:8.5f} | validation loss {} | validation metric {}".format(epoch, epoch_total_samples,
                                                                      epoch_total_batches, utils.second2hour(elapsed),
                                                                      speed, cur_loss, valid_loss, valid_metric))
        elapsed = time.time() - training_start_time
        logger.info("\n| Training has been completed.| {} epochs | time {} | {}/epoch |\nBest model is saved in \n'{}'".format(
            epoch, utils.second2hour(elapsed), utils.second2hour(elapsed / epoch),json.dumps(self._generate_model_name(), indent=4)))


    def _accumulate_metrics(self, mode, results_dict, y_pred, y_true):
        # TODO: check this
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_pred = F.softmax(y_pred, dim=-1)
        for metric in self.metrics[mode]:
            mv, mn = metric.ops(y_pred, y_true)
            results_dict[metric.name][0] += mv
            results_dict[metric.name][1] += mn


    def _accumulate_predictions(self, predictions, y_pred, inputs):
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_scores = F.softmax(y_pred, dim=-1)
            y_pred = torch.argmax(y_scores, dim=-1)
        else:
            y_scores = y_pred
        predictions[0] += y_scores.tolist()
        predictions[1] += y_pred.tolist()
        for key, value in inputs.items(): predictions[2][key] += value.tolist()


    def output_predictions(self, predictions, config):
        utils.name2function(config["function"])(predictions, config["params"])


    @torch.no_grad()
    def evaluate(self, validation=False):

        ensemble = isinstance(self.model, list)
        # utils.output_model_params_and_grad(self.model.module)
        if not ensemble:
            self.model.eval()
        else:
            for model in self.model:
                model.eval()
        # metric result
        verbose_results = {}
        # predictions
        predictions = [[], [], defaultdict(list)]  # [scores, prediction, inputs]
        start_time = time.time()
        total_samples = 0
        total_batches = 0
        if self.training and validation:
            if self.validation_data_manager == None: logger.error("Validation data manager is None.")
            verbose_results.update([(metric.name, [0., 0]) for metric in self.metrics["validation"]])
            total_loss = 0.
            total_labels = 0
            # validation
            for batch, inputs in enumerate(self.validation_data_manager):
                pred = self.model(inputs)
                _, num_labels, batch_total_loss = self.loss_fn(pred, inputs["target"])
                # accumulate metrics
                self._accumulate_metrics("validation", verbose_results, pred, inputs["target"])

                total_samples += inputs["target"].shape[0]
                total_loss += batch_total_loss
                total_labels += num_labels
                total_batches += 1
            # related to loss
            cur_loss = total_loss / total_labels
            # related to time
            elapsed = time.time() - start_time
            speed = elapsed * 1000 / total_batches
            # related to metrics
            results = dict([(name, l[0] / l[1]) for name, l in verbose_results.items()])
            valid_metric = self._get_validation_metric(results, cur_loss)
            logger.info(
                "\n| end of validation | {:8d} samples, {:8d} batches | time {:5.2f}s | {:5.2f} ms/batch "
                "| validation loss {:8.5f} | validation metric {:.5f} | early stopping {}/{}\n{}".format(
                    total_samples, total_batches, elapsed, speed, cur_loss, valid_metric, self.early_stopping_count,
                    self.trainerParams["training"]["early_stopping"] if "early_stopping" in self.trainerParams["training"] else 0,
                    utils.generate_metrics_str(verbose_results, verbose=True)))
            return cur_loss, results, valid_metric
        else:
            if self.evaluate_data_manager == None:
                logger.error("Evaluate data manager is None.")
                return None, None
            verbose_results.update([(metric.name, [0., 0]) for metric in self.metrics["evaluate"]])
            # self.model = nn.DataParallel(self.model)
            # evaluate and output predictions
            for batch, inputs in tqdm(enumerate(self.evaluate_data_manager)):
                if not ensemble:
                    pred = self.model(inputs)
                else:
                    preds = [model(inputs).unsqueeze(-1) for model in self.model]
                    pred = torch.mean(torch.cat(preds, dim=-1), dim=-1)
                # calculate metrics
                self._accumulate_metrics("evaluate", verbose_results, pred, inputs["target"])
                self._accumulate_predictions(predictions, pred, inputs)

                total_samples += inputs["target"].shape[0]
                total_batches += 1
            # related to time
            elapsed = time.time() - start_time
            speed = elapsed * 1000 / total_batches
            # related to metrics
            results = dict([(name, l[0] / l[1]) for name, l in verbose_results.items()])
            # output predictions
            if "output_result" in self.trainerParams["evaluate"]:
                self.output_predictions(predictions, self.trainerParams["evaluate"]["output_result"])

            logger.info(
                "\n| end of evaluation | {:8d} samples, {:8d} batches | time {:5.2f}s | {:5.2f} ms/batch\n{}".format(
                    total_samples, total_batches, elapsed, speed, utils.generate_metrics_str(verbose_results, verbose=True)))
            # TODO: write prediction result
            return None, results, None
