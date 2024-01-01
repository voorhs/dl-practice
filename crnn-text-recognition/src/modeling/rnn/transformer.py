from dataclasses import dataclass
from torch import nn
import torch
import math


@dataclass
class myTransformerConfig:
    hidden_size: int = 128
    num_attention_heads: int = 1
    attention_probs_dropout_prob: float = .1
    intermediate_size: int = 256
    n_layers: int = 2


class mySelfAttention(nn.Module):
    def __init__(
            self,
            config: myTransformerConfig
        ):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.config = config

        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.norm = nn.LayerNorm(config.hidden_size)
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        change view from (B, T, H) to (B, n, T, h)
        - B batch size
        - T longest sequence size
        - H hidden size
        - n number of att heads
        - h single att head size
        """
        new_x_shape = x.size()[:-1] + (self.config.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask):
        """
        x: (B, T, H)
        attention_mask: (B, T) or (B, T, T), if 0 then ignore corresponding token
        """
        # (B, T, H)
        hidden_states = self.norm(x)

        # (B, T, H)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # (B, n, T, h)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        if len(attention_mask.shape) == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif len(attention_mask.shape) == 3:
            attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f'strange shape of attention mask: {attention_mask.shape}')

        # (B, n, T, T)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores.masked_fill(attention_mask==0, -torch.inf)

        # (B, n, T, T)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # (B, n, T, h)
        c = torch.matmul(attention_probs, v)

        # (B, T, H)
        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.config.hidden_size,)
        c = c.view(*new_c_shape)

        # (B, T, H)
        return x + self.o(c)


class myFFBlock(nn.Module):
    def __init__(self, config: myTransformerConfig):
        super().__init__()
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.nonlinear = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, x):
        return x + self.linear2(self.nonlinear(self.linear1(self.norm(x))))


class myTransformerBlock(nn.Module):
    def __init__(self, config: myTransformerConfig):
        super().__init__()
        
        self.att = mySelfAttention(config)
        self.ff = myFFBlock(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, attention_mask=None):
        if attention_mask is None:
            B, T, _ = x.shape
            attention_mask = x.new_ones(size=(B,T))
        x = self.att(x, attention_mask)
        x = self.ff(x)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, in_features, config: myTransformerConfig = None):
        super().__init__()

        if config is None:
            config = myTransformerConfig()
        self.config = config
        
        self.projector = nn.Linear(in_features, config.hidden_size)
        self.layers = nn.Sequential(*[myTransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        """
        x: (T,B,F)
        """

        x = self.projector(x)
        x = x.permute(1,0,2)    # (B,T,F)
        x = self.layers(x)
        x = x.permute(1,0,2)    # (T,B,F)

        return x, None
