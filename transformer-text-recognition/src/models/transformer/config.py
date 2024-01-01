from dataclasses import dataclass
from typing import Tuple


@dataclass
class TransformerConfig:
    hidden_size: int = 512
    n_heads: int = 4
    att_dropout: float = .0
    hidden_dropout: float = .2
    intermediate_size: int = 2048
    max_len: int = 5000
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    alphabet: str = '0123456789ABEKMHOPCTYX'
    extended_alphabet: str = '0123456789ABEKMHOPCTYXpbe'
    pad_token_id: int = 22
    bos_token_id: int = 23
    eos_token_id: int = 24
    pad_token: str = 'p'
    bos_token: str = 'b'
    eos_token: str = 'e'
    special_token_ids: Tuple[int, int, int] = (22, 23, 24)
