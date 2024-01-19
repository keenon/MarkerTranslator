import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset
from typing import Dict, List, Optional
import numpy as np
import wandb
import logging
import matplotlib.pyplot as plt
import os
import argparse


class RegressionLossEvaluator:
    dataset: AddBiomechanicsDataset

    losses: List[torch.Tensor]

    def __init__(self, dataset: AddBiomechanicsDataset, split: str):
        self.dataset = dataset
        self.split = split

        # Aggregating losses across batches for dev set evaluation
        self.losses = []

    @staticmethod
    def get_mean_norm_error(output_tensor: torch.Tensor, label_tensor: torch.Tensor, vec_size: int = 3) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 3:
            raise ValueError('Output and label tensors must be 3-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_tensor.shape[-1] % vec_size != 0:
            raise ValueError('Tensors must have a final dimension divisible by vec_size=' + str(vec_size))

        diffs = output_tensor - label_tensor

        # Reshape the tensor so that the last dimension is split into chunks of `vec_size`
        reshaped_tensor = diffs.view(diffs.shape[0], diffs.shape[1], -1, vec_size)

        # Compute the norm over the last dimension
        norms = torch.norm(reshaped_tensor[:,-1:,:,:], dim=3)

        # Compute the mean norm over all the dimensions
        mean_norm = torch.mean(norms)

        return mean_norm

    def __call__(self,
                 outputs: torch.Tensor,
                 labels: torch.Tensor,
                 split: str = 'dev',
                 log_reports_to_wandb: bool = False,
                 args: argparse.Namespace = None) -> torch.Tensor:

        # Step 1: Compute the loss
        loss = RegressionLossEvaluator.get_mean_norm_error(outputs, labels)
        self.losses.append(loss)

        # Step 2: Log reports to wandb and plot results, if requested
        if log_reports_to_wandb:
            self.log_to_wandb(split, args, loss)

        return loss

    def log_to_wandb(self,
                     split: str,
                     args: argparse.Namespace,
                     loss: torch.Tensor):

        report: Dict[str, float] = {
            f'{self.split}/loss': loss.item()
        }
        wandb.log(report)

    def print_report(self,
                     reset: bool = True):

        avg_loss = torch.mean(torch.vstack(self.losses), dim=0)
        print(f'\tAvg Marker Err: {avg_loss} m')

        # Reset
        if reset:
            self.losses = []
