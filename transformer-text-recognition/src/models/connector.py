from torch import nn
from .config import OCRConfig


class Connector(nn.Module):
    """Converts output from CNN to input for Transformer."""
    def __init__(self, config: OCRConfig):
        super().__init__()

        h, w = config.cnn_input_size
        self.length_projector = nn.Conv2d(w // 8, config.sequence_len, kernel_size=1)
        self.projector = nn.Linear(config.cnn_output_features, config.transformer_hidden_size)
    
    def forward(self, x):
        """
        Input shape: (B,C,H,W)
        Output shape: (B,T,d)
        """
        
        x = x.mean(dim=2, keepdim=True)         # (B,F,1,W)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,W,1,F)
        x = self.length_projector(x)            # (B,T,1,F)
        x = x.squeeze(dim=2)                    # (B,T,F)
        x = self.projector(x)                   # (B,T,d)

        return x
