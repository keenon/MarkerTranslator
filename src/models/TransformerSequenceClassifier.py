import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSequenceClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 d_model=128,
                 nhead=4,
                 num_transformer_layers=6,
                 dim_feedforward=128,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 dropout: float = 0.1):
        super(TransformerSequenceClassifier, self).__init__()

        # Embedding layer to match the transformer's input dimensions
        self.embedding = nn.Linear(4, d_model, device=device, dtype=dtype)

        self.dropout_1 = nn.Dropout(dropout)

        # Transformer Layer
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, device=device, dtype=dtype, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        self.dropout_2 = nn.Dropout(dropout)

        # Output layer
        self.output_layer = nn.Linear(d_model, num_classes, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x is expected to be of shape [batch_size, N, 4]
        batch_size, n, _ = x.shape

        # Pass through embedding layer
        x = self.embedding(x)
        assert not torch.any(torch.isnan(x))

        # Pass through dropout layer
        x = self.dropout_1(x)

        # Transformer Encoder
        # The mask for the transformer needs to be different, it should be a boolean mask where True values are ignored
        transformer_mask = mask == 0  # Assuming mask is of shape [batch_size, N]
        assert transformer_mask.shape == (batch_size, n)
        for b in range(batch_size):
            # check that no row of transformer_mask is entirely masked, which means all values are True
            assert not torch.all(transformer_mask[b, :])

        # x = self.transformer_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        assert not torch.any(torch.isnan(x))

        # Pass through dropout layer
        x = self.dropout_2(x)

        # Pass through output layer and apply softmax
        assert not torch.any(torch.isnan(x))
        x = self.output_layer(x)

        assert not torch.any(torch.isnan(x))
        # output = F.softmax(x, dim=-1)
        output = x

        assert not torch.any(torch.isnan(output))
        return output
