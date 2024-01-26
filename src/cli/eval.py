import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import logging


class EvalCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('eval', help='Evaluate a model on the AddBiomechanics dataset')
        self.register_standard_options(subparser)

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'eval':
            return False
        device: str = args.device

        # Create an instance of the dataset
        logging.info('## Loading datasets with skeletons:')

        dev_dataset = self.get_dataset(args, 'dev')
        dev_dataset.pad_with_random_unknown_markers = True
        dev_dataset.randomly_hide_markers_prob = 0.0
        dev_loss_evaluator = self.get_loss(args, 'dev')
        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    persistent_workers=True,
                                    pin_memory=device != 'cpu',
                                    pin_memory_device=device,
                                    drop_last=True)

        mp.set_start_method('spawn')  # 'spawn' or 'fork' or 'forkserver'

        # Create an instance of the model
        model = self.get_model(args)
        self.load_latest_checkpoint(model, checkpoint_dir=args.checkpoint_dir, device=device)

        print(f'Evaluating Dev Set')
        with torch.no_grad():
            model.eval()  # Turn dropout off
            for i, batch in enumerate(dev_dataloader):
                # print(f"batch iter: {i=}")
                inputs: torch.Tensor
                labels: torch.Tensor
                mask: torch.Tensor
                batch_subject_indices: List[int]
                batch_trial_indices: List[int]
                inputs, labels, mask, batch_subject_indices, batch_trial_indices = batch
                print(f"batch subject: {dev_dataset.subject_paths[batch_subject_indices[0]]}")
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                assert (labels.shape == mask.shape)

                outputs = model(inputs, mask)

                # Compute the loss
                dev_loss_evaluator(outputs,
                                   labels,
                                   mask,
                                   split='dev',
                                   log_reports_to_wandb=False,
                                   args=args)

                # logging.info(f"{data_time=}, {forward_time=}, {loss_time}")
                if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
                    print('  - Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
                    dev_loss_evaluator.print_report(reset=True)
        # Report dev loss on this epoch
        logging.info('Dev Set Evaluation: ')
        dev_loss_evaluator.print_report(reset=True)

        return True

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm True --dropout True --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "../data" --epochs 500
