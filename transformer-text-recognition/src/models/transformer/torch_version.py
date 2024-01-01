"""this implementation doesn't work"""

from torch import nn
import torch
from .modules import EncoderInput, DecoderInput, PositionalEncoding
from .config import TransformerConfig


class TorchTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config

        pos = PositionalEncoding(config)
        self.encoder_input = EncoderInput(pos, config)
        self.decoder_input = DecoderInput(pos, config)
        # self.transformer = nn.Transformer(
        #     d_model=config.hidden_size,
        #     nhead=config.n_heads,
        #     num_encoder_layers=config.n_encoder_layers,
        #     num_decoder_layers=config.n_decoder_layers,
        #     dim_feedforward=config.intermediate_size,
        #     dropout=config.hidden_dropout,
        #     activation='relu',
        #     batch_first=True,
        #     norm_first=True
        # )
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        ) for _ in range(config.n_encoder_layers)])
        self.decoder_layers = nn.ModuleList([nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        ) for _ in range(config.n_decoder_layers)])
    
    def forward(self, encoder_inputs, token_ids, attention_mask):
        '''
        params
        ---
        encoder_inputs: (B,T,H)
        token_ids: (B,T)
        '''
        
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(token_ids, encoder_outputs, attention_mask)
        
        return encoder_outputs, decoder_outputs
    
    def encoder(self, encoder_input):
        x = self.encoder_input(encoder_input)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def decoder(self, token_ids, encoder_outputs, attention_mask):
        x = self.decoder_input(token_ids)
        extended_attention_mask = self._extended_attention_mask(attention_mask)
        x = self.decoder_input(token_ids, attention_mask)
        for layer in self.decoder_layers:
            x = layer(x, encoder_outputs, attention_mask, tgt_is_casual=True)
        return x

    def _extended_attention_mask(self, attention_mask):
        B, T = attention_mask.shape
        n_heads = self.config.n_heads

        casual_mask = torch.tril(attention_mask.new_ones(T, T))
        mask = casual_mask[None, :, :] * attention_mask[:, None, :]
        # mask = mask[None, ...].expand(n_heads, B, T, T).reshape(n_heads * B, T, T)
        return (mask == 0)
