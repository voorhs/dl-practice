from .config import TransformerConfig
import torch


class Tokenizer:
    def __init__(self, config: TransformerConfig):
        self.config = config

        self.char_to_idx = {c: i for i, c in enumerate(config.extended_alphabet)}
        self.pad = self.config.pad_token_id
        self.bos = self.config.bos_token_id
        self.eos = self.config.eos_token_id

    def __call__(self, texts):
        """
        texts: list[str] of length B
        outputs:
            token_ids: (B,T)
            attention_mask: (B,T)
            extended_attention_mask: (B,T,T)
        """

        tokens = [[self.bos] + [self.char_to_idx[c] for c in txt] + [self.eos] for txt in texts]
        
        T = max(len(x) for x in tokens)
        padded_tokens = [x + [self.pad] * (T - len(x)) for x in tokens]
        attention_mask = [[1] * len(x) + [0] * (T - len(x)) for x in tokens]

        padded_tokens = torch.LongTensor(padded_tokens)
        attention_mask = torch.LongTensor(attention_mask)

        return {
            'token_ids': padded_tokens,
            'attention_mask': attention_mask,
        }
    
    def decode(self, token_ids):
        """
        token_ids: (B,T)
        result: list of strings
        """

        vocab = self.config.extended_alphabet
        spec = self.config.special_token_ids
        eos = self.config.eos_token_id
        
        res = []
        for ids in token_ids:
            i = 0
            cur_res = []
            while i < len(ids) and ids[i] != eos:
                if ids[i] not in spec:
                    cur_res.append(vocab[ids[i]])
                i += 1
            res.append(''.join(cur_res))
        
        return res
