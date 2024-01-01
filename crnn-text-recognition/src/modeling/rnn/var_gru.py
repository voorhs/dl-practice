from torch import nn
import warnings
from ..config import CRNNConfig


class VariationalDropoutGRU(nn.Module):
    def __init__(
            self,
            config: CRNNConfig
        ):
        super().__init__()

        self.config = config
        
        self.module = nn.GRU(
            config.cnn_output_features,
            config.rnn_hidden_size,
            dropout=config.rnn_dropout,
            num_layers=config.rnn_n_layers
        )

        self.layer_names = []
        for i_layer in range(self.config.rnn_n_layers):
            self.layer_names += [f'weight_hh_l{i_layer}', f'weight_ih_l{i_layer}']

        # add postfix '_raw' to each layer name
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            delattr(self.module, layer)
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
            
            setattr(self.module, layer, masked_raw_w)

    def forward(self, x):
        """
        :param x: tensor containing the features of the input sequence.
        :param Tuple[torch.Tensor, torch.Tensor] h_c: initial hidden state and initial cell state
        """
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")

            self._dropout_on_weights()
            output = self.module(x)
        
        return output
