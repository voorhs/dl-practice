from torch import nn
from .config import CRNNConfig


class Connector(nn.Module):
    """Converts output from CNN to input for RNN."""
    def __init__(self, config: CRNNConfig):
        super().__init__()

        h, w = config.cnn_input_size
        self.length_projector = nn.Conv2d(w // 16, config.rnn_sequence_len, kernel_size=1)
    
    def forward(self, x):
        """
        Input shape: (B,C,H,W)
        Output shape: (T,B,F)
        """
        
        x = x.mean(dim=2, keepdim=True)         # (B,F,1,W)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,W,1,F)
        x = self.length_projector(x)            # (B,T,1,F)
        x = x.squeeze(dim=2)                    # (B,T,F)
        # x = x.permute(0, 2, 1)                  # (B,F,T)
        # x = self.norm(x)
        x = x.permute(1, 0, 2).contiguous()     # (T,B,F)
        # x = x + self.ff(x)

        return x
