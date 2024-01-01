from torch import nn
from .config import OCRConfig


class Predictor(nn.Module):
    """Convert Transformer outputs to characters logits"""
    def __init__(self, config: OCRConfig):
        super().__init__()

        self.config = config

        self.fc = nn.Linear(
            in_features=config.hidden_size,
            out_features=len(config.alphabet)+1
        )
    
    def forward(self, x):
        """
        Input shape: (B,T,F')
        Output shape: (B,T,n_chars)
        """
        return self.fc(x)
