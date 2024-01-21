import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSequenceClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 d_model=512,
                 nhead=4,
                 num_transformer_layers=2,
                 dim_feedforward=512,
                 device: str = 'cpu'):
        super(TransformerSequenceClassifier, self).__init__()

        # Embedding layer to match the transformer's input dimensions
        self.embedding = nn.Linear(4, d_model, device=device)

        # Transformer Layer
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, device=device)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x is expected to be of shape [batch_size, N, 4]

        # Pass through embedding layer
        x = self.embedding(x)

        # Transformer Encoder
        # The mask for the transformer needs to be different, it should be a boolean mask where True values are ignored
        transformer_mask = mask == 1  # Assuming mask is of shape [batch_size, N]

        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        assert not torch.any(torch.isnan(x))

        # Pass through output layer and apply softmax
        assert not torch.any(torch.isnan(x))
        x = self.output_layer(x)

        assert not torch.any(torch.isnan(x))
        output = F.softmax(x, dim=-1)

        assert not torch.any(torch.isnan(output))
        return output
