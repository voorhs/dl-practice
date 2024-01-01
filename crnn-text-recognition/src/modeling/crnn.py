from torch import nn
from .config import CRNNConfig
from .predictor import Predictor
from .connector import Connector


class CRNN(nn.Module):
    def __init__(self, config: CRNNConfig, cnn, rnn):
        super().__init__()

        self.config = config

        self.cnn = cnn
        self.connector = Connector(config)
        self.rnn = rnn
        self.predictor = Predictor(config)

    def forward(self, x):
        """
        Input shape: (B,C,H,W)
        Output shape: (T,B,n_chars)
        """
        x = self.cnn(x)         # (B,C,H,W)
        x = self.connector(x)   # (T,B,F)
        x, _ = self.rnn(x)      # (T,B,F')
        x = self.predictor(x)   # (T,B,n_chars)

        return x
