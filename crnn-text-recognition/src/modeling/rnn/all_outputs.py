from torch import nn
import warnings
from ..config import CRNNConfig
import torch


class AllOutputsGRU(nn.Module):
    def __init__(
            self,
            config: CRNNConfig
        ):
        super().__init__()

        self.config = config
        
        self.cells = nn.ModuleList([nn.GRUCell(
            config.cnn_output_features,
            config.rnn_hidden_size,
        ) for _ in range(config.rnn_n_layers)])

        self.layer_names = [n for n, p in self.cells.named_parameters() if 'weights' in n]

        # add postfix '_raw' to each layer name
        for layer in self.layer_names:
            w = getattr(self.cells, layer)
            delattr(self.cells, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))

    def _dropout_on_weights(self):
        """
        Apply dropout to raw weights.
        """
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')

            # raw_w has shape (out_features, in_features), so in order to 
            # zero out ith entry of input, we need to zero out ith column of matrix
            p = self.config.rnn_variational_dropout
            if self.training:
                mx = raw_w.new_empty(1, raw_w.shape[1]).bernoulli_(1 - p)
            else:
                mx = raw_w.new_full((1,), 1 - p)
            
            masked_raw_w = raw_w * mx
            
            setattr(self.cells, layer, masked_raw_w)

    def forward(self, x):
        """
        x: (T,B,F)
        output: (T,B,F*n_layers)
        """

        self._dropout_on_weights()

        time_steps = x.shape[0]
        outputs = []        
        for cell in self.cells:
            layer_outputs = []
            h = None
            for i in range(time_steps):
                h = cell(x[i], h)
                layer_outputs.append(h)
            outputs.append(torch.stack(layer_outputs, dim=0))
        
        return torch.stack(outputs, dim=0).mean(dim=0), None
