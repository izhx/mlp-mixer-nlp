"""
https://arxiv.org/pdf/2105.01601.pdf
"""


import torch
import torch.nn as nn

# from allennlp.common.checks import ConfigurationError
# from allennlp.common.params import Params
from allennlp.modules import Seq2VecEncoder
# from allennlp.nn import util

from .mixers import MixerInTimeStepAndHidden


@Seq2VecEncoder.register("mixer_in_timestep_hidden")
class MixerInTimeStepAndHiddenSeq2VecEncoder(Seq2VecEncoder):
    """
    """
    def __init__(
        self, hidden_size: int, num_layers: int, num_tokens: int,
        dropout: float = 0.0, **kwargs
    ):
        # Seq2VecEncoders cannot be stateful.
        super().__init__(stateful=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tokens = num_tokens

        kwargs.update(hidden_size=hidden_size, num_tokens=num_tokens, dropout=dropout)
        self._mixer_layers = nn.ModuleList(
            MixerInTimeStepAndHidden(**kwargs) for _ in range(num_layers)
        )
        self._pooler = nn.Linear(hidden_size * num_tokens, hidden_size)
        self._dropout = nn.Dropout(dropout)

    def get_input_dim(self) -> int:
        return self.hidden_size

    def get_output_dim(self) -> int:
        return self.hidden_size

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        pad or cut input sequence and run mixers.
        """
        batch_size = inputs.size(0)
        # Shape (batch_size, num_tokens, 1)
        mixer_mask = inputs.new_ones(batch_size, self.num_tokens).unsqueeze(-1)

        diffence = self.num_tokens - inputs.size(1)
        if diffence > 0:
            padding_size = (inputs.size(0), diffence)
            zeros = inputs.new_zeros(*padding_size, inputs.size(-1))
            # Shape (batch_size, num_tokens, hidden_size)
            hidden_states = torch.cat((inputs, zeros), dim=1)
            if mask is not None:
                falses = inputs.new_zeros(padding_size)
                # Shape (batch_size, num_tokens, 1)
                mixer_mask = torch.cat((mask.float(), falses), dim=1).unsqueeze(-1)
        elif diffence < 0:
            hidden_states = inputs[:, :self.num_tokens]
            if mask is not None:
                mixer_mask = mask[:, :self.num_tokens].unsqueeze(-1).float()

        for _, layer in enumerate(self._mixer_layers):
            hidden_states = layer(hidden_states, mixer_mask)

        # Shape (batch_size, num_tokens*hidden_size)
        flatten_hidden_states = (hidden_states * mixer_mask).view(batch_size, -1)
        # Shape (batch_size, hidden_size)
        outputs = self._dropout(self._pooler(flatten_hidden_states))
        return outputs
