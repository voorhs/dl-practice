from torch import nn
import torch
from .transformer import Transformer, Tokenizer
from .connector import Connector
from .config import OCRConfig


class TransformerOCR(nn.Module):
    def __init__(self, cnn, transformer: Transformer, config: OCRConfig):
        super().__init__()

        self.config = config

        self.cnn = cnn
        self.connector = Connector(config)
        self.transformer = transformer
        self.clf = nn.Linear(transformer.config.hidden_size, len(transformer.config.extended_alphabet))

    def forward(self, images, token_ids, attention_mask):
        '''
        images: (B,C,H,W)
        token_ids: (B,T)
        attention_mask: (B,T)
        extended_attention_mask: (B,T,T)
        '''
        x = self.cnn(images)   # (B,C,H,W)
        x = self.connector(x)   # (B,T,d)
        encoder_outputs, decoder_outputs = self.transformer(x, token_ids, attention_mask)  # (B,T,d)
        char_logits = self.clf(decoder_outputs)     # (B,T,n_chars)
        return encoder_outputs, char_logits

    def decode(self, encoder_outputs, tokenizer: Tokenizer):
        B = encoder_outputs.shape[0]
        device = self.clf.weight.data.device
        token_ids = torch.full(size=(B,1), fill_value=self.transformer.config.bos_token_id, device=device)
        attention_mask = torch.ones_like(token_ids)
        
        for _ in range(self.config.max_generated_len):
            decoder_outputs = self.transformer.decoder(token_ids, encoder_outputs, attention_mask)
            
            new_decoder_outputs = decoder_outputs[:, -1, :]     # (B,d)
            logits = self.clf(new_decoder_outputs)              # (B,n_chars)
            pred_token_ids = torch.argmax(logits, dim=1)        # (B,)
            
            not_eos = (pred_token_ids != self.transformer.config.eos_token_id).int()
            token_ids = torch.cat([token_ids, pred_token_ids[:, None]], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask[:, -1][:, None] * not_eos[:, None]], dim=1)

        return tokenizer.decode(token_ids)
    
    def predict(self, images, tokenizer):
        x = self.cnn(images)   # (B,C,H,W)
        x = self.connector(x)   # (B,T,d)
        encoder_outputs = self.transformer.encoder(x)  # (B,T,d)

        return self.decode(encoder_outputs, tokenizer)
