import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import nimblephysics as nimble
import logging

ACTIVATION_FUNCS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}



class MaxPoolLSTMPointCloudRegressor(nn.Module):
    history_len: int
    num_input_markers: int
    num_output_markers: int

    def __init__(self,
                 history_len: int,
                 num_input_markers: int,
                 num_output_markers: int,
                 in_mlp_hidden_dims: List[int] = [512],
                 time_hidden_dim: int = 1024,
                 out_mlp_hidden_dims: List[int] = [512],
                 dropout: bool = False,
                 dropout_prob: float = 0.0,
                 dropout_marker_prob: float = 0.3,
                 batchnorm: bool = False,
                 activation: str = 'sigmoid',
                 device: str = 'cpu'):
        super(MaxPoolLSTMPointCloudRegressor, self).__init__()

        self.history_len = history_len
        self.num_input_markers = num_input_markers
        self.num_output_markers = num_output_markers

        self.in_dropout = nn.Dropout(dropout_marker_prob)

        self.in_mlp = []
        dims = [3] + in_mlp_hidden_dims + [time_hidden_dim]
        for i, (h0, h1) in enumerate(zip(dims[:-1], dims[1:])):
            if dropout:
                self.in_mlp.append(nn.Dropout(dropout_prob))
            if batchnorm:
                self.in_mlp.append(nn.BatchNorm1d(h0))
            self.in_mlp.append(nn.Linear(h0, h1, dtype=torch.float32, device=device))
            self.in_mlp.append(ACTIVATION_FUNCS[activation])
        self.in_mlp = nn.Sequential(*self.in_mlp)

        self.time_model = nn.LSTM(input_size=time_hidden_dim,
                                  hidden_size=(time_hidden_dim // 2),
                                  num_layers=2,
                                  batch_first=True,
                                  bidirectional=True)

        self.out_mlp = []
        dims = [time_hidden_dim] + out_mlp_hidden_dims + [3 * num_output_markers]
        out_mlp = []
        for i, (h0, h1) in enumerate(zip(dims[:-1], dims[1:])):
            if dropout:
                out_mlp.append(nn.Dropout(dropout_prob))
            if batchnorm:
                out_mlp.append(nn.BatchNorm1d(h0))
            out_mlp.append(nn.Linear(h0, h1, dtype=torch.float32, device=device))
            if i < len(dims) - 2:
                out_mlp.append(ACTIVATION_FUNCS[activation])
            else:
                out_mlp.append(nn.Tanh())
        self.out_mlp = nn.Sequential(*out_mlp)

        logging.info(f"{self.in_mlp=}")
        logging.info(f"{self.out_mlp=}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Reshape tensor: (batch_size * seq_len * num_points, 3)
        batch_size, seq_len, _ = x.size()
        num_points = _ // 3
        x = x.view(-1, 3)

        tensor_norms = torch.norm(x, dim=1)
        tensor_norms_nonzero = tensor_norms > 0
        tensor_norms_indicator = tensor_norms_nonzero.float()
        tensor_norms_indicator = self.in_dropout(tensor_norms_indicator)

        # Process each point independently
        x = self.in_mlp(x)

        # Zero out the points that are zero on inputs
        x = x * tensor_norms_indicator.unsqueeze(1)

        # Reshape back and max-pool
        hidden_size = x.size(-1)
        x = x.view(batch_size, seq_len, num_points, hidden_size)
        x = torch.mean(x, dim=2)  # Mean-pool over the points

        # Run through time model
        x, _ = self.time_model(x)

        # Run through output models

        # Return the outputs
        outputs = self.out_mlp(x)
        return outputs