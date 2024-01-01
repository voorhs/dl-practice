from dataclasses import dataclass
from typing import Tuple


@dataclass
class CRNNConfig:
    cnn_input_size: Tuple[int, int] = (64, 320)
    rnn_sequence_len: int = 20
    cnn_output_features: int = 1792
    rnn_hidden_size: int = 128
    rnn_n_layers: int = 3
    alphabet: str = '0123456789ABEKMHOPCTYX'
    rnn_dropout: float = 0
    rnn_bidirectional: bool = False

    rnn_variational_dropout: float = 0.3
