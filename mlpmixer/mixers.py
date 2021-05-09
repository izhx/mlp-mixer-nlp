"""
https://arxiv.org/pdf/2105.01601.pdf
"""


import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    """
    def __init__(self, in_features: int, expansion_factor: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.intermediate_size = in_features * expansion_factor
        self.fc_1 = nn.Linear(in_features, self.intermediate_size, False)
        self.fc_2 = nn.Linear(self.intermediate_size, in_features, False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc_2(self.dropout(self.gelu(self.fc_1(inputs))))


class MixerInTimeStepAndHidden(nn.Module):
    """
    """
    def __init__(
        self, hidden_size: int, num_tokens: int = 128, expansion_factor: int = 1,
        dropout: float = 0.0, layer_norm_eps: float = 0.00001
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens

        self.mlp_1 = MLP(num_tokens, expansion_factor, dropout)
        self.mlp_2 = MLP(hidden_size, expansion_factor, dropout)
        self.ln_1 = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.ln_2 = nn.LayerNorm(hidden_size, layer_norm_eps)

    def get_input_dim(self) -> int:
        return self.hidden_size

    def get_output_dim(self) -> int:
        return self.hidden_size

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # shape (b, h, t)
        transposed_normed = self.ln_1(inputs * mask).transpose(-1, -2)
        # shape (b, t, h)
        transposed_token_mixed = self.mlp_1(transposed_normed).transpose(-1, -2)
        intermediate = (inputs + transposed_token_mixed) * mask
        channel_mixed_normed = self.mlp_2(self.ln_2(intermediate))
        outputs = channel_mixed_normed + intermediate
        return outputs
