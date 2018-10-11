from tools.DatasetManager import DatasetManager
from utils import utils
import time
import os
import json
from copy import deepcopy

import torch
import torch.nn as nn

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
        self.training = training
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

        self.best_epoch = -1

        if self.training:
            # load embedding
            embeddings = None
            if "emb_load" in self.trainerParams["training"]:
                emb_load_params = self.trainerParams["training"]["emb_load"]
                logger.info("Loading embedding file '{}'...".format(emb_load_params["params"]["path"]))
                embeddings = utils.name2function(emb_load_params["function"])(emb_load_params["params"])
                logger.info("loading embedding has been completed.")
            # create model
            model_params = self.trainerParams["model"]
            self._creat_model(embeddings)
            # define loss function
            logger.info("Defining loss function '{}' with params:\n{}".format(model_params["loss"]["function"],
                                                                              json.dumps(model_params["loss"]["params"],
                                                                                         indent=4)))
            self.loss_fn = utils.name2function(model_params["loss"]["function"])(model_params["loss"]["params"])
            logger.info("Defining loss function has been completed.")
            # define optimizer
            optimizer_params = self.trainerParams["training"]["optimizer"]
            logger.info("Defining optimizer '{}' with params:\n{}".format(optimizer_params["function"],
                                                                          json.dumps(optimizer_params["params"],
                                                                                     indent=4)))
            self.optimizer = utils.name2function(optimizer_params["function"])(optimizer_params["params"]).ops(
                [{"params": self.model.parameters()}])
            logger.info("Defining optimizer has been completed.")
        else:
            # create model
            self._creat_model()
            # load model
            self.load_model()

    def _set_torch_variables(self):
        # set default dtype as torch.float64
        # torch.set_default_dtype(torch.float64)
        # set computing device
        global_params = self.trainerParams["global"]
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

    def _creat_model(self, embeddings=None):
        model_params = deepcopy(self.trainerParams["model"])
        logger.info("Creating model '{}' with params:\n{}".format(model_params["model_path"],
                                                                  json.dumps(model_params["params"], indent=4)))
        model_params["params"]["device"] = self.device
        model_params["params"]["embeddings"] = embeddings
        # put model on specified computing device
        self.model = utils.name2function(model_params["model_path"])(model_params["params"]).to(device=self.device)
        logger.info("Creating model has been completed.")

    def _load_metrics(self):
        metrics_params = self.trainerParams["metrics"]
        self.metrics = {}
        for mode, metrics in metrics_params.items():
            self.metrics[mode] = [utils.name2function(metric["function"])(metric["params"]) for metric in metrics]

    def _split_data_into_training_validation(self, data_dict, validation_num):
        # TODO: check this
        for key in data_dict.keys():
            if data_dict[key]["type"].lower() == "normal":
                assert validation_num <= data_dict[key]["data"].shape[0]
                if validation_num > 0: self.validation_data_dict[key] = data_dict[key]["data"][:validation_num]
                self.training_data_dict[key] = data_dict[key]["data"][validation_num:]
            elif data_dict[key]["type"].lower() == "share":
                if validation_num > 0: self.validation_data_dict[key] = data_dict[key]["data"]
                self.training_data_dict[key] = data_dict[key]["data"]
            else:
                assert False, "{} is not a valid type.".format(key)

    def load_data(self, training_data=True):
        if "data_load" in self.trainerParams["data_manager"]:
            data_load_params = self.trainerParams["data_manager"]["data_load"]
            data_load_fn = utils.name2function(data_load_params["function"])
            if self.training and training_data:
                logger.info("Loading training and validation data...")
                data_load_params["params"]["phase"] = "training"  # data_load.params.phase is changed
                validation_num = self.trainerParams["training"]["validation_num"]
                data_dict = data_load_fn(data_load_params["params"])
                self._split_data_into_training_validation(data_dict, validation_num)
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
        # put model on specified computing device
        data_manager_params["dataloader_gen"]["params"]["device"] = self.device
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
        if self.best_epoch:
            model_file = os.path.join(model_path,
                                      self.trainerParams["training"]["model_prefix"]) + "_epoch{}.pkl".format(
                self.best_epoch)
            return model_file
        else:
            logger.error("Best epoch is None.")

    def load_model(self):
        if self.training and not self.continue_train:
            # load model in training
            model_file = self._generate_model_name()
            self.model.load_state_dict(torch.load(model_file))
        else:
            # load model in evaluate or continue_train mode
            model_file = self.trainerParams["evaluate"]["test_model_file"]
            self.model.load_state_dict(torch.load(model_file))

    def save_model(self):
        model_path = self.trainerParams["training"]["model_save_path"]
        os.makedirs(model_path, exist_ok=True)
        model_file = self._generate_model_name()
        # Save only the model parameters
        torch.save(self.model.state_dict(), model_file)

    def train(self):
        if not self.training: logger.error("Mode: Evaluate.")
        training_params = self.trainerParams["training"]
        min_loss = None
        # Whether continuing training
        if "continue_train" in training_params and training_params["continue_train"]:
            self.continue_train = True
            self.load_model()
            # evaluate existing model
            min_loss, _ = self.evaluate(validation=True)
            self.save_model()
            self.continue_train = False  # change continue train flag for evaluating trained model
        self.epoch_total_samples = 0
        self.epoch_total_batches = 0
        self.epoch_total_labels = 0

        self.early_stopping_count = 0
        self.early_stopping_flag = False

        for epoch in range(1, training_params["num_epochs"] + 1):
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

            for batch_in_epoch, inputs in enumerate(self.training_data_manager):
                if self.early_stopping_flag: break
                self.model.train()
                self.optimizer.zero_grad()

                pred = self.model(inputs)
                loss, num_labels, batch_total_loss = self.loss_fn(pred, inputs["target"])

                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 1)
                self.optimizer.step()

                # epoch level
                epoch_total_samples += inputs["target"].shape[0]
                epoch_total_batches += 1
                epoch_total_labels += num_labels
                epoch_total_loss += batch_total_loss
                # log level
                log_total_labels += num_labels
                log_total_loss += batch_total_loss

                if (batch_in_epoch + 1) % training_params["log_interval"] == 0:
                    # training log
                    cur_loss = log_total_loss / log_total_labels
                    elapsed = time.time() - log_start_time
                    speed = elapsed * 1000 / training_params["log_interval"]
                    logger.info(
                        "\n| epoch {:3d} | {:5d}/{:5d} batches | time {:5.2f}s | {:5.2f} ms/batch | loss {:8.5f} | best model in epoch {:3d}".format(
                            epoch, batch_in_epoch + 1, self.epoch_total_batches, elapsed, speed, cur_loss, self.best_epoch))
                    # log level
                    log_start_time = time.time()
                    log_total_labels = 0
                    log_total_loss = 0

                if (batch_in_epoch + 1) % training_params["validation_interval"] == 0:
                    # validation
                    valid_loss, results = self.evaluate(validation=True)
                    if min_loss == None or valid_loss < min_loss:
                        self.early_stopping_count = 0
                        self.best_epoch = epoch
                        min_loss = valid_loss
                        logger.info("\n| epoch {:3d} | batch {:5d} | New record has been achieved. |".format(epoch,
                                                                                                             batch_in_epoch + 1))
                        logger.info("Saving model...")
                        self.save_model()
                        logger.info("Best model is saved in '{}'".format(self._generate_model_name()))
                    elif "early_stopping" in training_params and training_params["early_stopping"] > 0:
                        # early stopping
                        self.early_stopping_count += 1
                        if self.early_stopping_count >= training_params["early_stopping"]:
                            self.early_stopping_flag = True
                            logger.warning("\n| Early stopping mechanism start up. "
                                           "| Recent {} times, the performance in validation set has not been improved."
                                           "".format(
                                self.early_stopping_count))
                            logger.warning("\n| Best model is saved in '{}'".format(self._generate_model_name()))

                            # epoch level
            self.epoch_total_samples = epoch_total_samples
            self.epoch_total_batches = epoch_total_batches
            self.epoch_total_labels = epoch_total_labels

            cur_loss = epoch_total_loss / epoch_total_labels
            elapsed = time.time() - epoch_start_time
            speed = elapsed * 1000 / epoch_total_batches
            valid_loss, results = self.evaluate(validation=True)
            logger.info(
                "\n| end of epoch {:3d} | {:8d} samples, {:8d} batches | time {:5.2f}s | {:5.2f} ms/batch "
                "| training loss {:8.5f} | validation loss {:8.5f}".format(epoch, epoch_total_samples,
                                                                           epoch_total_batches, elapsed, speed,
                                                                           cur_loss, valid_loss))
        logger.info(
            "\n| Training has been completed. | Best model is saved in '{}'".format(self._generate_model_name()))

    def _accumulate_metrics(self, mode, results_dict, y_pred, y_true):
        # TODO: check this
        y_pred = torch.nn.functional.softmax(y_pred, dim=-1)
        for metric in self.metrics[mode]:
            mv, mn = metric.ops(y_pred, y_true)
            results_dict[metric.name][0] += mv
            results_dict[metric.name][1] += mn

    def evaluate(self, validation=False):
        self.model.eval()
        results = {}
        start_time = time.time()
        total_samples = 0
        total_batches = 0
        if self.training and validation:
            if self.validation_data_manager == None: logger.error("Validation data manager is None.")
            results.update([(metric.name, [0., 0]) for metric in self.metrics["validation"]])
            total_loss = 0.
            total_labels = 0
            # validation
            for batch, inputs in enumerate(self.validation_data_manager):
                pred = self.model(inputs)
                _, num_labels, batch_total_loss = self.loss_fn(pred, inputs["target"])
                # accumulate metrics
                self._accumulate_metrics("validation", results, pred, inputs["target"])

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
            results.update([(name, l[0] / l[1]) for name, l in results.items()])

            logger.info(
                "\n| end of validation | {:8d} samples, {:8d} batches | time {:5.2f}s | {:5.2f} ms/batch "
                "| validation loss {:8.5f} | early stopping {}/{}\n{}".format(
                    total_samples, total_batches, elapsed, speed, cur_loss, self.early_stopping_count,
                    self.trainerParams["training"]["early_stopping"] if "early_stopping" in self.trainerParams["training"] else 0,
                    utils.generate_metrics_str(results)))
            return cur_loss, results
        else:
            if self.evaluate_data_manager == None: logger.error("Evaluate data manager is None.")
            results.update([(metric.name, [0., 0]) for metric in self.metrics["evaluate"]])
            # evaluate
            for batch, inputs in enumerate(self.evaluate_data_manager):
                pred = self.model(inputs)
                # calculate metrics
                self._accumulate_metrics("evaluate", results, pred, inputs["target"])

                total_samples += inputs["target"].shape[0]
                total_batches += 1
            # related to time
            elapsed = time.time() - start_time
            speed = elapsed * 1000 / total_batches
            # related to metrics
            results.update([(name, l[0] / l[1]) for name, l in results.items()])

            logger.info(
                "\n| end of evaluate | {:8d} samples, {:8d} batches | time {:5.2f}s | {:5.2f} ms/batch\n{}".format(
                    total_samples, total_batches, elapsed, speed, utils.generate_metrics_str(results)))
            # TODO: write prediction result
            return results
