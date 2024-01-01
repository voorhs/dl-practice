from torch import nn
from .config import CRNNConfig


class Predictor(nn.Module):
    """Convert RNN outputs to characters logits"""
    def __init__(self, config: CRNNConfig):
        super().__init__()

        self.config = config

        in_features = config.rnn_hidden_size
        if config.rnn_bidirectional:
            in_features *= 2

        self.fc = nn.Linear(
            in_features=in_features,
            out_features=len(config.alphabet)+1
        )

        # n_features = config.rnn_hidden_size
        # self.norm = nn.BatchNorm1d(n_features)
        # self.ff = nn.Linear(n_features, n_features)

        # nn.init.zeros_(self.norm.bias)
        # nn.init.ones_(self.norm.weight)
    
    def forward(self, x):
        """
        Input shape: (T,B,F')
        Output shape: (T,B,n_chars)
        """

        # x = x.permute(1,2,0)    # (B,F,T)
        # x = self.norm(x)
        # x = x.permute(0,2,1)    # (B,T,F)
        # x = x + self.ff(x)
        # x = x.permute(1,0,2)    # (T,B,F)

        return self.fc(x)
