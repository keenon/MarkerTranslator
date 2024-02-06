import unittest
import torch
import torch.optim as optim
import numpy as np
from src.loss.MaskedCrossEntropyLoss import MaskedCrossEntropyLoss


class TestMaskedCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.num_classes = 3
        self.loss_function = MaskedCrossEntropyLoss(split='test', num_classes=self.num_classes)

    def test_initialization(self):
        self.assertEqual(self.loss_function.split, 'test')
        self.assertEqual(self.loss_function.correct_predictions, 0)
        self.assertEqual(self.loss_function.total_predictions, 0)
        self.assertTrue((self.loss_function.confusion == np.zeros((self.num_classes, self.num_classes))).all())

    def test_call(self):
        # Create dummy data
        logits = torch.randn(10, 5, self.num_classes)  # Example size
        target = torch.randint(0, self.num_classes, (10, 5))
        mask = torch.randint(0, 2, (10, 5)).float()

        # Call the loss function
        loss = self.loss_function(logits, target, mask)

        # Check loss type and value
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)

        # Check if internal state is updated correctly
        self.assertGreaterEqual(self.loss_function.total_predictions, mask.sum().item())
        self.assertGreaterEqual(self.loss_function.correct_predictions, 0)

        # Check confusion matrix
        self.assertIsInstance(self.loss_function.confusion, np.ndarray)

    def test_reset(self):
        # Call the loss function with some dummy data to change internal state
        logits = torch.randn(10, 5, self.num_classes)
        target = torch.randint(0, self.num_classes, (10, 5))
        mask = torch.randint(0, 2, (10, 5)).float()
        _ = self.loss_function(logits, target, mask)

        # Reset and test if internal state is reset
        self.loss_function.print_report(reset=True)
        self.assertEqual(self.loss_function.correct_predictions, 0)
        self.assertEqual(self.loss_function.total_predictions, 0)
        self.assertTrue((self.loss_function.confusion == np.zeros((self.num_classes, self.num_classes))).all())

    def test_sgd_on_logits(self):
        num_classes = 3
        num_samples = 50  # Number of samples
        num_steps = 500  # Number of SGD iterations
        learning_rate = 0.5

        # Initialize a tensor of logits
        logits = torch.randn(num_samples, num_classes, requires_grad=True)

        # Random target labels and mask
        targets = torch.randint(0, num_classes, (num_samples,))
        mask = torch.ones(num_samples)

        # Loss function and optimizer
        loss_function = MaskedCrossEntropyLoss(split='train', num_classes=num_classes)
        optimizer = optim.SGD([logits], lr=learning_rate)

        for _ in range(num_steps):
            # Calculate loss
            loss = loss_function(logits.unsqueeze(0), targets.unsqueeze(0), mask.unsqueeze(0))
            loss_function.print_report(reset=True)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check if loss has decreased
        final_loss = loss.item()
        self.assertLess(final_loss, 1.0)  # Threshold depends on the specific task


if __name__ == '__main__':
    unittest.main()
