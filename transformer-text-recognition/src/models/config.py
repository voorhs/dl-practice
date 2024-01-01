from dataclasses import dataclass
from typing import Tuple


@dataclass
class OCRConfig:
    cnn_input_size: Tuple[int, int] = (32, 160)
    sequence_len: int = 15
    cnn_output_features: int = 512
    max_generated_len: int = 15
    transformer_hidden_size: int = None
