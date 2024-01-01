from torch import nn
import torch
import math
from .config import TransformerConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.hidden_size % config.n_heads == 0, "hidden_size must be divisible by n_heads"
        
        self.config = config
        H = config.hidden_size
        d = config.hidden_size // config.n_heads

        self.norm = nn.LayerNorm(H)
        
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

        self.dropout = nn.Dropout(config.att_dropout)
                
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q,K,V: (B,n,T,h)
        result: (B,n,T,h)
        """
        
        Q = self.q(Q)
        K = self.k(K)
        V = self.v(V)

        # (B,n,T,h) x (B,n,h,T) -> (B,n,T,T)
        head_size = self.config.hidden_size // self.config.n_heads
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(head_size)
        
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :]
            elif len(mask.shape) == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f'strange shape of attention mask: {mask.shape}')
            attn_scores = attn_scores.masked_fill(mask == 0, -torch.inf)
           
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # (B,n,T,T) x (B,n,T,h) -> (B,n,T,h)
        output = torch.matmul(attn_probs, V)
        
        return output
        
    def split_heads(self, x):
        """
        x: (B,T,H)
        result: (B,n,T,h)
        """

        B, T, H = x.size()
        head_size = H // self.config.n_heads
        x = x.view(B, T, self.config.n_heads, head_size)
        x = x.transpose(1, 2)
        
        return x
        
    def combine_heads(self, x):
        """
        x: (B,n,T,h)
        result: (B,T,H)
        """
        
        B, _, T, _ = x.size()
        x = x.transpose(1, 2).contiguous()          # (B,T,n,h)
        x = x.view(B, T, self.config.hidden_size)   # (B,T,H)

        return x
        
    def forward(self, Q, K, V, mask=None):
        """
        Q,K,V: (B,T,H)
        output: updated Q with info pooled from V based on similarity with K
        """

        # pre-norm strategy
        q = self.norm(Q)
        k = self.norm(K)
        v = self.norm(V)

        # (B,n,T,h)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # (B,n,T,h)
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # (B,T,H)
        output = Q + self.combine_heads(self.o(attn_output))
        
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.nonlinear = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x):
        return x + self.dropout(self.linear2(self.nonlinear(self.linear1(self.norm(x)))))


class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.hidden_size % 2 == 0, "hidden size must be even"
        
        self.config = config

        H = config.hidden_size
        freq = torch.exp(-(torch.arange(0, H, 2, dtype=torch.float) / H * math.log(1e4)))   # (H/2,)
        phase = torch.arange(config.max_len, dtype=torch.float)    # (T,)
        
        pe = torch.zeros(config.max_len, H)  # (T, H)
        pe[:, 0::2] = torch.sin(phase[:, None] * freq[None, :])
        pe[:, 1::2] = torch.cos(phase[:, None] * freq[None, :])

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x):
        """
        x: (B,T,H)
        result: (B,T,H)
        """
        B, T, H = x.shape

        p = self.pe[torch.arange(0, T)]         # (T,H)
        p = p.unsqueeze(dim=0).expand(B,T,H)    # (B,T,H)
        p = self.dropout(p)

        return p


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config

        self.attn = MultiHeadAttention(config)
        self.ff = PositionWiseFeedForward(config)
                
    def forward(self, x):
        x = self.attn(x, x, x)
        x = self.ff(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config

        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = PositionWiseFeedForward(config)
        
    def forward(self, x, enc_output, tgt_mask):
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.cross_attn(x, enc_output, enc_output)
        x = self.feed_forward(x)
        return x


class EncoderInput(nn.Module):
    def __init__(self, positional_encoding, config: TransformerConfig):
        super().__init__()

        self.positional_encoding = positional_encoding
        self.config = config

        self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        return self.norm(x + self.positional_encoding(x))


class DecoderInput(nn.Module):
    def __init__(self, positional_encoding, config: TransformerConfig):
        super().__init__()

        self.positional_encoding = positional_encoding
        self.config = config
        
        self.embedding = nn.Embedding(
            num_embeddings=len(config.extended_alphabet),
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, token_ids, attention_mask=None):
        """
        token_ids: (B,T)
        attention_mask: (B,T)
        """
        x = self.embedding(token_ids)
        x = x + self.positional_encoding(x)
        x = self.norm(x)
        if attention_mask is not None:
            x = x * attention_mask[:, :, None]
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        
        positional_encoding = PositionalEncoding(config)
        self.encoder_input = EncoderInput(positional_encoding, config)
        self.decoder_input = DecoderInput(positional_encoding, config)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_decoder_layers)])

    def forward(self, encoder_input, token_ids, attention_mask):
        """
        encoder_input: (B,T,H)
        token_ids: (B,T)
        attention_mask: (B,T)
        """

        encoder_outputs = self.encoder(encoder_input)
        decoder_outputs = self.decoder(token_ids, encoder_outputs, attention_mask)
        return encoder_outputs, decoder_outputs

    def encoder(self, encoder_input):
        x = self.encoder_input(encoder_input)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decoder(self, token_ids, encoder_outputs, attention_mask):
        extended_attention_mask = self._extended_attention_mask(attention_mask)
        x = self.decoder_input(token_ids, attention_mask)
        for layer in self.decoder_layers:
            x = layer(x, encoder_outputs, extended_attention_mask)
        return x

    def _extended_attention_mask(self, attention_mask):
        B, T = attention_mask.shape

        casual_mask = torch.tril(attention_mask.new_ones(T, T))
        return casual_mask[None, :, :] * attention_mask[:, None, :]
