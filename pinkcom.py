# coding=utf-8

from tools.Trainer import Trainer
from utils import utils
import os
import json
import argparse

VERSION = "0.1.0"

def train(config, evaluate=False):
    trainer = Trainer(config)
    trainer.load_data()
    if evaluate:
        trainer.load_data(training_data=False)
    trainer.generate_data_managers()
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning('Exiting from training early.')
    if evaluate:
        trainer.load_model()
        trainer.evaluate()

def evaluate(config):
    trainer = Trainer(config, training=False)
    trainer.load_data(training_data=False)
    trainer.generate_data_managers()
    trainer.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pinkcom based on PyTorch')
    parser.add_argument("-v", "--version", action='version', version='| Version | {:^8} |'.format(VERSION))
    parser.add_argument('--train', action='store_true',
                        help='Train a new model or continue training an existing model.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model.')
    parser.add_argument('--config', type=str, default="configs/smn_last.config.json",
                        help='Location of config file')
    args = parser.parse_args()

    config_file = args.config
    trainerParams = json.load(open(config_file, 'r'))

    logger = utils.create_logger(trainerParams["global"]["log_file"])

    if args.train:
        train(trainerParams, evaluate=args.evaluate)
    elif args.evaluate:
        evaluate(trainerParams)
    else:
        parser.print_help()
        logger.error("\n"+parser.format_help())
else:
    logger = utils.get_logger()