import torch
import warnings

class RNNClassifier(torch.nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, output_size, vocab,
        rec_layer=torch.nn.LSTM, dropout=0, **kwargs
    ):
        """`rec_layer` is for further experiments"""
        super().__init__()

        self.dropout = dropout

        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.embedding_dim = embedding_dim

        # simple lookup table that stores embeddings of a fixed dictionary and size.
        self.word_embeddings = torch.nn.Embedding(
            num_embeddings=len(self.vocab),
            embedding_dim=embedding_dim,
            padding_idx=self.vocab['<pad>'],
            max_norm=None
        )

        # recurrent neural network (default is LSTM)
        self.rnn = rec_layer(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout,
            **kwargs
        )

        # linear layer for classification
        self.output = torch.nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.output_size
        )

    def forward(self, tokens, tokens_lens):
        """
        :param torch.Tensor(dtype=torch.long) tokens: Batch of texts represented with tokens.
        :param torch.Tensor(dtype=torch.long) tokens_lens: Number of non-padding tokens for each object in batch.
        :return torch.Tensor(dtype=torch.long): Vector representation for each sequence in batch
        """
        # `tokens` has shape (seq_length, batch_size)

        # Evaluate embeddings -> `emb` has shape (seq_length, batch_size, embedding_dim)
        emb = self.word_embeddings(tokens)

        # Make forward pass through recurrent network ->
        # `output` is y_1, ..., y_T -> has shape (seq_length, batch_size, hidden_size), because in LSTM shape(y) == shape(h)
        # `hn` is hidden state from last LSTM box, i.e. h_T -> has shape (batch_size, hidden_size)
        # `cn` is long memory from last LSTM box, i.e. c_T -> has shape (batch_size, hidden_size)
        output, (hn, cn) = self.rnn(emb)

        # take rnn hidden state after the last token which is not '<pad>'
        # (each object in batch has the same length T because of torch.nn.utils.rnn.pad_sequence)
        # (in LSTM y_t == h_t)
        # works only for batch_first=False
        # `last_hidden`: shape (batch_size, hidden_dim)
        last_hidden = output[tokens_lens-1, torch.arange(len(tokens_lens))]

        # Pass through linear layer
        # output: shape (output_size,)
        return self.output(last_hidden)


class RNNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super().__init__()

        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn_cell = torch.nn.LSTMCell(self.input_size, self.hidden_size)

    def forward(self, x):
        # x has shape (seq_length, batch_size, hidden_size)

        # Initialize h_0, c_0 with both having shape (batch_size, hidden_size)
        h, c = init_h0_c0(
            batch_size=x.shape[1],
            hidden_size=self.hidden_size,
            some_existing_tensor=x
        )

        # Gen dropout masks for input and hidden state
        mx, mh = gen_dropout_mask(
            self.input_size, self.hidden_size,
            self.training,
            p=self.dropout,
            some_existing_tensor=x
        )

        # apply dropout for all inputs in batch
        x = x * mx

        # iterate along `sequence_length` axis
        output = []
        for xt in x:
            # apply one LSTM cell
            # xt: shape (batch, input_size)
            # h, c:  shape (batch, hidden_size)
            h, c = self.rnn_cell(xt, (h, c))

            # apply variational dropout
            h = h * mh

            # collect outputs (in LSTM: y_t == h_t)
            output.append(h)

        # convert to tensor
        # output is y1, ..., yT, shape (seq_length, batch_size, hidden_size)
        output = torch.stack(output, dim=0)

        # h, c are from last element of sequence
        return output, (h, c)
   

class FastRNNLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, layers_dropout=0.0, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.dropout = dropout
        self.layers_dropout = layers_dropout
        self.module = torch.nn.LSTM(input_size, hidden_size, dropout=layers_dropout, num_layers=num_layers)

        self.layer_names = []
        for i_layer in range(self.num_layers):
            # it is the torch build-in way give names to LSTM params
            self.layer_names += [f'weight_hh_l{i_layer}', f'weight_ih_l{i_layer}']

        # add postfix '_raw' to each layer name
        # (raw layers will be stored in model,
        # we will apply dropout to them
        # and then use in forward)
        for layer in self.layer_names:
            # Get torch.nn.Parameter with weights from torch.nn.LSTM instance
            w = getattr(self.module, layer)

            # Remove it from model
            delattr(self.module, layer)

            # And create new torch.nn.Parameter with the same data but different name
            self.register_parameter(f'{layer}_raw', torch.nn.Parameter(w.data))

    def _setweights(self, x):
        """
            Apply dropout to the raw weights.
        """
        for layer in self.layer_names:
            # Get torch.nn.Parameter with weights
            raw_w = getattr(self, f'{layer}_raw')

            # make dropout mask for raw_w
            # (it has shape (out_features, in_features), so in order to 
            # zero out ith entry of input, we need to zero out ith column of matrix)
            mx, _ = gen_dropout_mask(
                # in_features
                input_size=raw_w.shape[1],
                # we dont need mask for hidden state vector
                hidden_size=0,
                is_training=self.training,
                p=self.dropout,
                some_existing_tensor=raw_w
            )
            
            # Apply dropout mask
            masked_raw_w = raw_w * mx
            
            # Set modified weights in its place
            setattr(self.module, layer, masked_raw_w)

    def forward(self, x, h_c=None):
        """
        :param x: tensor containing the features of the input sequence.
        :param Tuple[torch.Tensor, torch.Tensor] h_c: initial hidden state and initial cell state
        """
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")

            # Set new weights of self.module and call its forward
            self._setweights(x)
            output = self.module(x, h_c)  # h_c=None by default
        
        return output
    
    def reset(self):
        if hasattr(self.module, 'reset'):
            self.module.reset()


class HandmadeLSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_size, dropout=0.0):
    super().__init__()
    
    self.dropout = dropout
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    # stacked U^i, U^f, U^o, U^g
    self.input_weights = torch.nn.Linear(input_size, 4 * hidden_size)
    
    # stacked W^i, W^f, W^o, W^g
    self.hidden_weights = torch.nn.Linear(hidden_size, 4 * hidden_size)
    
    self.reset_params()

  def reset_params(self):
    """
    Initialization as in Pytorch. 
    Do not forget to call this method!
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
    """
    stdv = 1.0 / self.hidden_size ** 0.5
    for weight in self.parameters():
        torch.nn.init.uniform_(weight, -stdv, stdv)

  def forward(self, x):
    # initialize hidden state and memory
    # (tensors on same device and with same dtype)
    # h, c: shape (batch_size, hidden_size)
    h, c = init_h0_c0(x.shape[1], self.hidden_size, x)

    # generate dropout mask for input and hidden vectors
    # (actually latter one will be applied to information gate)
    mx, mh = gen_dropout_mask(
        self.input_size, self.hidden_size,
        self.training, 
        p=self.dropout,
        some_existing_tensor=x
    )

    # apply dropout for all inputs in batch
    x = x * mx

    # Implement recurrent logic to mimic torch.nn.LSTM
    output = []
    for xt in x:
      # calculate W @ h + U @ x + b for each of four gate
      # stacked_stuff: shape (batch, 4 * hidden_size)
      stacked_stuff = self.input_weights(xt) + self.hidden_weights(h)
      
      # split gates i, f, o and g apart
      # gates_i_f_o: shape (batch_size, 3 * hidden_size),
      # gate_g: shape (batch_size, hidden_size)
      gates_i_f_o = stacked_stuff[:, :3*self.hidden_size]
      gate_g = stacked_stuff[:, 3*self.hidden_size:]
      # print(gates_i_f_o.shape, gate_g.shape, self.hidden_size)
      
      # apply non-linearity
      gates_i_f_o = torch.sigmoid(gates_i_f_o)
      gate_g = torch.tanh(gate_g)

      # split gates apart
      # gate_i, gate_f, gate_o: shape (batch_size, hidden_size)
      gate_i, gate_f, gate_o = gates_i_f_o.split(self.hidden_size, dim=1)
      
      # apply dropout mask to 
      gate_g = gate_g * mh

      # c, h: (batch, hidden_size)
      c = gate_f * c + gate_i * gate_g
      h = gate_o * torch.tanh(c)
      output.append(h)
    
    # output: shape (seq_length, batch_size, hidden_size)
    output = torch.stack(output)

    # h, c are from last time step
    return output, (h, c)


def init_h0_c0(batch_size, hidden_size, some_existing_tensor):
    """
    We won't iterate along batch axis, so we have vectors h_0, c_0 
    as concatenations of such vectors for all objects in batch

    Params
    ------
    some_existing_tensor (torch.Tensor): tensor to copy device and dtype from

    Return
    ------
    h0, c0: torch.Tensor of shape (num_objects, hidden_size)
    """
    size = (batch_size, hidden_size)
    h0 = some_existing_tensor.new_zeros(size)
    c0 = some_existing_tensor.new_zeros(size)
    return h0, c0

def gen_dropout_mask(input_size, hidden_size, is_training, p, some_existing_tensor):
    """
    Params
    ------
    is_training (bool): current mode of nn
    True -> masks from Bernoulli
    False -> tensor consisting of one value 1-p
    
    p (float 0..1):
    float value stands for the probability that we zero out a neuron's output
    i.e. 0 -> we don't zero out any neuron (tensor consisting of one value 1 is returned)
    
    some_existing_tensor (torch.Tensor): tensor to copy device and dtype from

    Return
    ------
    mx, mh (torch.tensor): binary masks
    mx is a mask applied to input of rnn cell (x_t), shape (input_size,)
    mh is a mask applied to previous hidden state of rnn cell (h_{t-1}), shape (hidden_size,)
    """
    if p == 0:
        # it's no matter to multiply on scalar or vector of same scalars
        mx = some_existing_tensor.new_ones((1,))
        mh = some_existing_tensor.new_ones((1,))
    elif is_training:
        mx = some_existing_tensor.new_empty(input_size).bernoulli_(1 - p)
        mh = some_existing_tensor.new_empty(hidden_size).bernoulli_(1 - p)
    else:
        # it's no matter to multiply on scalar or vector of same scalars
        mx = some_existing_tensor.new_full((1,), 1 - p)
        mh = some_existing_tensor.new_full((1,), 1 - p)
    return mx, mh
