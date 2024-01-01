from torch import nn
import torch
import torch.nn.functional as F


class Projector(nn.Module):
        """Fully-Connected 2-layer Linear Model with skip-connections, GELU activation an layer norm."""

        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear_1 = nn.Linear(input_size, input_size)
            self.linear_2 = nn.Linear(input_size, input_size)
            self.norm1 = nn.LayerNorm(input_size)
            self.norm2 = nn.LayerNorm(input_size)
            self.final = nn.Linear(input_size, output_size)
            self.orthogonal_initialization()

        def orthogonal_initialization(self):
            for l in [self.linear_1, self.linear_2]:
                torch.nn.init.xavier_uniform_(l.weight)

        def forward(self, x):
            x = x + F.gelu(self.linear_1(self.norm1(x)))
            x = x + F.gelu(self.linear_2(self.norm2(x)))

            return F.normalize(self.final(x), dim=1)
