from model_clf import RNNClassifier, FastRNNLayer
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack


class RNNLM(RNNClassifier):
    def __init__(
        self, embedding_dim, hidden_dim, vocab, dropout=0.5, layers_dropout=0.5, num_layers=1
    ):
        super().__init__(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_size=len(vocab), vocab=vocab,
            rec_layer=FastRNNLayer, dropout=dropout, layers_dropout=layers_dropout, num_layers=num_layers
        )

    def forward(self, tokens, tokens_lens):
        """
        :param torch.Tensor(dtype=torch.long) tokens: 
            Batch of texts represented with tokens. Shape: [T, B]
        :param torch.Tensor(dtype=torch.long) tokens_lens: 
            Number of non-padding tokens for each object in batch. Shape: [B]
        :return torch.Tensor: 
            Distribution of next token for each time step. Shape: [T, B, V], V — size of vocabulary
        """
        # Make embeddings for all tokens
        # inputs: shape (seq_length, batch_size, embedding_dim)
        inputs = self.word_embeddings(tokens)

        # Forward pass embeddings through network
        # output is y_1, ..., y_T
        # output: shape (seq_length, batch_size, hidden_dim)
        output, _ = self.rnn(inputs)

        # Take all hidden states from the last layer of LSTM for each step and perform linear transformation
        # result: shape (seq_length, batch_size, len(vocab))
        return self.output(output)


class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, outputs, tokens, tokens_lens):
        """
        :param torch.Tensor outputs: Output from RNNLM.forward. Shape: [T, B, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [T, B]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :return torch.Tensor: CrossEntropyLoss between corresponding logits and tokens
        """
        # remove padding and flatten logits
        packed_outputs = pack(
            outputs, tokens_lens.cpu() + 1,
            batch_first=False, enforce_sorted=False
        ).data

        # remove padding and flatten tokens
        packed_tokens = pack(
            tokens[1:], tokens_lens.cpu() + 1,
            batch_first=False, enforce_sorted=False
        ).data

        # now there is one-to-one relation between tokens and outputs
        # and we can put it into loss function
        return super().forward(packed_outputs, packed_tokens)


class LMAccuracy(torch.nn.Module):
    def forward(self, outputs, tokens, tokens_lens):
        """
        :param torch.Tensor outputs: Output from RNNLM.forward. Shape: [T, B, V]
        :param torch.Tensor tokens: Batch of tokens. Shape: [T, B]
        :param torch.Tensor tokens_lens: Length of each sequence in batch
        :return torch.Tensor: Accuracy for given logits and tokens
        """
        # remove padding and flatten logits
        packed_outputs = pack(
            outputs, tokens_lens.cpu() + 1,
            batch_first=False, enforce_sorted=False
        ).data

        # remove padding and flatten tokens
        packed_tokens = pack(
            tokens[1:], tokens_lens.cpu() + 1,
            batch_first=False, enforce_sorted=False
        ).data

        # now there is one-to-one relation between tokens and outputs
        # and we can put it into loss function
        return (packed_outputs.argmax(axis=1) == packed_tokens).sum()


@torch.no_grad()
def final_h_c(model, start_tokens, start_tokens_lens):
    # Get embedding for start_tokens
    embedding = model.word_embeddings(start_tokens)

    # Pass embedding through rnn and collect hidden states and cell states for each time moment
    # all_h, all_c: shape (seq_length, num_layers, batch_size, hidden_dim)
    all_h, all_c = [], []
    h = embedding.new_zeros(
        [model.rnn.num_layers, start_tokens.shape[1], model.hidden_dim])
    c = embedding.new_zeros(
        [model.rnn.num_layers, start_tokens.shape[1], model.hidden_dim])
    for time_step in range(start_tokens.shape[0]):
        _, (h, c) = model.rnn(embedding[time_step].unsqueeze(0), (h, c))
        all_h.append(h)
        all_c.append(c)

    # all_h, all_c: shape (num_layers, seq_length, batch_size, hidden_dim)
    all_h = torch.stack(all_h, dim=1)
    all_c = torch.stack(all_c, dim=1)

    # Take final hidden state and cell state for each start sequence in batch
    # We will use them as h_0, c_0 for generation new tokens

    # индексирование числом убирает ось, но индексирование тензором дает тензор этого же размера
    # поэтому
    #   ":" оставляет num_layers ось,
    #   `start_tokens_lens` убирает T ось,
    #   arange убирает B ось,
    #   ось hidden остается

    # h, c: shape (num_layers, batch_size, hidden_dim)
    h = all_h[:, start_tokens_lens - 1,
              torch.arange(start_tokens_lens.shape[0])]
    c = all_c[:, start_tokens_lens - 1,
              torch.arange(start_tokens_lens.shape[0])]

    return h, c

@torch.no_grad()
def decode(model, start_tokens, start_tokens_lens, max_generated_len=20, top_k=None):
    """
    :param RNNLM model: Model
    :param torch.Tensor start_tokens: Batch of seed tokens. Shape: [T, B]
    :param torch.Tensor start_tokens_lens: Length of each sequence in batch. Shape: [B]
    :param int max_generated_len: Maximum lenght of generated samples
    :param Optional[int] top_k: Number of tokens with the largest probability to sample from
    :return Tuple[torch.Tensor, torch.Tensor]. 
        Newly predicted tokens and length of generated part. Shape [T*, B], [B]
    """
    # Take final hidden state and cell state for each start sequence in batch
    # We will use them as h_0, c_0 for generation new tokens
    # h, c: shape (num_layers, batch_size, hidden_dim)
    h, c = final_h_c(model, start_tokens, start_tokens_lens)

    # List of predicted tokens for each time step
    predicted_tokens = []
    
    # Length of generated part for each object in the batch
    # decoded_lens: shape (batch_size,)
    decoded_lens = torch.zeros_like(start_tokens_lens, dtype=torch.long)
    
    # boolean mask where we store if the sequence has already generated
    # i.e. `<eos>` was selected on any step
    # is_finished_decoding: shape (batch_size,)
    is_finished_decoding = torch.zeros_like(start_tokens_lens, dtype=torch.bool)
    
    # until <eos> is generated or max len is reached
    while not torch.all(is_finished_decoding) and torch.max(decoded_lens) < max_generated_len:
        # evaluate next token distribution using hidden state h from last LSTM layer.
        # logits: shape (batch_size, len(vocab))
        logits = model.output(h[-1])
        
        if top_k is not None:
            # top-k largest values in each batch
            # values: shape (batch_size,)
            topk_values, _ = torch.topk(logits, top_k, dim=1, largest=True)
            
            # kth top value
            # values: shape (batch_size)
            threshold = topk_values[:, -1]

            # find every logit that is less then kth top value
            # indices_to_remove: shape (batch_size, len(vocab)) 
            indices_to_remove = logits < threshold[..., None]
            
            # zero them out
            logits[indices_to_remove] = -1e10
            
            # descrete distribution with probabilities proportional to logits
            sampler = torch.distributions.categorical.Categorical(logits=logits)
            
            # sample next token
            next_token = sampler.sample()
        else:
            # select most probable token
            next_token = logits.argmax(dim=1)
            
        predicted_tokens.append(next_token)
        
        decoded_lens += (~is_finished_decoding)
        is_finished_decoding |= (next_token == model.vocab['<eos>'])

        # Compute embedding for next token
        embedding = model.word_embeddings(next_token)

        # Update hidden and cell states (LSTMCell like)
        _, (h, c) = model.rnn(embedding.unsqueeze(0), (h, c))
        
    return torch.stack(predicted_tokens), decoded_lens
