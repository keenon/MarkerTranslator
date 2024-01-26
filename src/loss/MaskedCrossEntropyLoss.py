import torch
import torch.nn.functional as F
import argparse
from typing import Dict
import wandb
import numpy as np


class MaskedCrossEntropyLoss:
    split: str
    correct_predictions: int
    total_predictions: int
    confusion: np.ndarray
    num_classes: int

    def __init__(self, split: str, num_classes: int):
        self.correct_predictions = 0
        self.total_predictions = 0
        self.split = split
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.num_classes = num_classes

    def __call__(self,
                 logits: torch.Tensor,
                 target: torch.Tensor,
                 mask: torch.Tensor,
                 split: str = 'dev',
                 log_reports_to_wandb: bool = False,
                 args: argparse.Namespace = None):
        # logits: Predictions from the model, shape [batch_size, N, num_classes]
        # target: True labels, shape [batch_size, N]
        # mask: Mask tensor, shape [batch_size, N]
        batch_size, n = target.shape
        if logits.shape[1] < n:
            # If we have fewer frames than expected, pad the logits with zeros
            logits = F.pad(logits, (0, 0, 0, n - logits.shape[1]))
        assert mask.shape == (batch_size, n)

        # Flatten the tensors
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        mask_flat = mask.view(-1) == 1

        logits_masked = logits_flat[mask_flat]
        target_masked = target_flat[mask_flat]

        assert not torch.any(torch.isnan(logits_masked))

        # Compute cross entropy loss
        loss = F.cross_entropy(logits_masked, target_masked, reduction='mean')

        # Assert loss is not NaN
        assert not torch.isnan(loss)

        # # Apply the mask
        # masked_loss = loss * mask_flat

        # Update accuracy counters
        with torch.no_grad():
            predictions = torch.argmax(logits_masked, dim=1)
            for i, j in zip(target_masked, predictions):
                self.confusion[i, j] += 1
            correct = predictions == target_masked
            num_correct = correct.sum().item()
            num_total = len(predictions)
            self.correct_predictions += correct.sum().item()
            self.total_predictions += mask_flat.sum().item()

        # Assert total_loss is not NaN
        assert not torch.isnan(loss)

        # Step 2: Log reports to wandb and plot results, if requested
        if log_reports_to_wandb:
            self.log_to_wandb(split, args, loss, num_correct / num_total)

        # Average the loss
        return loss

    def log_to_wandb(self,
                     split: str,
                     args: argparse.Namespace,
                     loss: torch.Tensor,
                     accuracy: float):
        report: Dict[str, float] = {
            f'{self.split}/loss': loss.item(),
            f'{self.split}/accuracy': accuracy
        }
        wandb.log(report)

    def print_report(self, reset: bool = True):
        accuracy = 100.0 * self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        # print(f"Confusion Matrix:")
        # print(self.confusion)
        print(f"Current Accuracy: {accuracy:.2f}%")
        if reset:
            self.correct_predictions = 0
            self.total_predictions = 0
            self.confusion = np.zeros_like(self.confusion)