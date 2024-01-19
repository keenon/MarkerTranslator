import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb
import numpy as np
import logging
import subprocess
import torch.optim.lr_scheduler as lr_scheduler
from loss.RegressionLossEvaluator import RegressionLossEvaluator


# Utility to get the current repo's git hash, which is useful for replicating runs later
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Git hash could not be found."


# Utility to check if the current repo has uncommited changes, which is useful for debugging why we can't replicate
# runs later, and also yelling at people if they run experiments with uncommited changes.
def has_uncommitted_changes():
    try:
        # The command below checks for changes including untracked files.
        # You can modify this command as per your requirement.
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        return bool(status)
    except subprocess.CalledProcessError:
        return "Could not determine if there are uncommitted changes."


class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        self.register_standard_options(subparser)
        subparser.add_argument('--no-wandb', action='store_true', default=False,
                               help='Log this run to Weights and Biases.')
        subparser.add_argument('--learning-rate', type=float, default=1e-5,
                               help='The learning rate for weight updates.')
        subparser.add_argument('--epochs', type=int, default=10, help='The number of epochs to run training for.')
        subparser.add_argument('--opt-type', type=str, default='rmsprop',
                               help='The optimizer to use when adapting the weights of the model during training.')
        subparser.add_argument('--batch-size', type=int, default=4,
                               help='The batch size to use when training the model.')
        subparser.add_argument('--data-loading-workers', type=int, default=5,
                               help='Number of separate processes to spawn to load data in parallel.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        model_type: str = args.model_type
        opt_type: str = args.opt_type
        checkpoint_dir: str = os.path.join(os.path.abspath(args.checkpoint_dir), model_type)
        learning_rate: float = args.learning_rate
        epochs: int = args.epochs
        batch_size: int = args.batch_size
        log_to_wandb: bool = not args.no_wandb
        data_loading_workers: int = args.data_loading_workers
        overfit: bool = args.overfit

        has_uncommitted = has_uncommitted_changes()
        if has_uncommitted:
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error(
                "ERROR: UNCOMMITTED CHANGES IN REPO! THIS WILL MAKE IT HARD TO REPLICATE THIS EXPERIMENT LATER")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if log_to_wandb:
            # Grab all cmd args and add current git hash
            config = args.__dict__
            config["git_hash"] = get_git_hash

            logging.info('Initializing wandb...')
            wandb.init(
                # set the wandb project where this run will be logged
                project="marker-translator",

                # track hyperparameters and run metadata
                config=config
            )

        # Create an instance of the dataset
        logging.info('## Loading datasets with skeletons:')

        train_dataset = self.get_dataset(args, 'train')
        train_loss_evaluator = RegressionLossEvaluator(dataset=train_dataset, split='train')
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=data_loading_workers,
                                      persistent_workers=True)

        dev_dataset = self.get_dataset(args, 'dev')
        dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset, split='dev')
        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=data_loading_workers,
                                    persistent_workers=True)

        mp.set_start_method('spawn')  # 'spawn' or 'fork' or 'forkserver'

        # Create an instance of the model
        model = self.get_model(args)

        # Define the optimizer
        if opt_type == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif opt_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif opt_type == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        elif opt_type == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        else:
            logging.error('Invalid optimizer type: ' + opt_type)
            assert False

        # Define a learning rate scheduler (e.g., StepLR)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.5)

        # self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir, optimizer=optimizer)

        for epoch in range(epochs):
            # Iterate over the entire training dataset

            # print(f'Evaluating Dev Set before {epoch=}')
            # with torch.no_grad():
            #     model.eval()  # Turn dropout off
            #     for i, batch in enumerate(dev_dataloader):
            #         # print(f"batch iter: {i=}")
            #         inputs: Dict[str, torch.Tensor]
            #         labels: Dict[str, torch.Tensor]
            #         batch_subject_indices: List[int]
            #         batch_trial_indices: List[int]
            #         data_time = time.time()
            #         inputs, labels, batch_subject_indices, batch_trial_indices = batch
            #         data_time = time.time() - data_time
            #         forward_time = time.time()
            #         outputs = model(inputs)
            #         forward_time = time.time() - forward_time
            #         loss_time = time.time()
            #         dev_loss_evaluator(outputs,
            #                                 labels,
            #                                 split = 'dev',
            #                                  log_reports_to_wandb = log_to_wandb,
            #                                  args=args)
            #         loss_time = time.time() - loss_time
            #         # logging.info(f"{data_time=}, {forward_time=}, {loss_time}")
            #         if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
            #             print('  - Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
            # # Report dev loss on this epoch
            # logging.info('Dev Set Evaluation: ')
            # dev_loss_evaluator.print_report(reset=True)

            print('Running Train Epoch ' + str(epoch))
            model.train()  # Turn dropout back on
            for i, batch in enumerate(train_dataloader):
                # print(f"batch iter: {i=}")
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                batch_trial_indices: List[int]
                inputs, labels, batch_subject_indices, batch_trial_indices = batch

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = train_loss_evaluator(outputs,
                                            labels,
                                            split='train',
                                            log_reports_to_wandb=log_to_wandb,
                                            args=args)

                if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                    logging.info('  - Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))
                    train_loss_evaluator.print_report(reset=False)

                if (i + 1) % 1000 == 0 or i == len(train_dataloader) - 1:
                    model_path = f"{checkpoint_dir}/epoch_{epoch}_batch_{i}.pt"
                    if not os.path.exists(os.path.dirname(model_path)):
                        os.makedirs(os.path.dirname(model_path))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)

                # Backward pass
                loss.backward()

                # Update the model's parameters
                optimizer.step()

                # Update the learning rate scheduler
                scheduler.step()
            # Report training loss on this epoch
            logging.info(f"{epoch=} / {epochs}")
            logging.info('Training Set Evaluation: ')
            train_loss_evaluator.print_report(reset=True)
        return True

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm True --dropout True --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "../data" --epochs 500
