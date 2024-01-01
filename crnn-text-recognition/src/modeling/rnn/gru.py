from torch import nn
from ..config import CRNNConfig


def get_gru(config: CRNNConfig):
    return nn.GRU(
        input_size=config.cnn_output_features,
        hidden_size=config.rnn_hidden_size,
        num_layers=config.rnn_n_layers,
        dropout=config.rnn_dropout,
        bidirectional=config.rnn_bidirectional
    )
