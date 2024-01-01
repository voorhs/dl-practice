import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from collections import deque


LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 64
PROJECTION_SIZE = 512
RESPONSE_SELECTION_CONTEXT_LENGTH = 2


class Projector(nn.Module):
        """Fully-Connected 2-layer Linear Model. Taken from linking prediction paper code."""

        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear_1 = nn.Linear(input_size, input_size)
            self.linear_2 = nn.Linear(input_size, input_size)
            self.norm1 = nn.LayerNorm(input_size)
            self.norm2 = nn.LayerNorm(input_size)
            self.final = nn.Linear(input_size, output_size)
            self.orthogonal_initialization()

        def orthogonal_initialization(self):
            for l in [self.linear_1, self.linear_2]:
                torch.nn.init.xavier_uniform_(l.weight)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            else:
                x = x.to(torch.float32)
            x = x.cuda()
            x = x + F.gelu(self.linear_1(self.norm1(x)))
            x = x + F.gelu(self.linear_2(self.norm2(x)))

            return F.normalize(self.final(x), dim=1)
        

class MAP(pl.LightningModule):
    def __init__(self, context_emb_size, response_emb_size, projection_size) -> None:
        super().__init__()

        self.context_mapper = Projector(context_emb_size, projection_size)
        self.response_mapper = Projector(response_emb_size, projection_size)
        self.i_epoch = 0
    
    def forward(self, context, response):
        """
        Params
        ------
            torch Tensors of size (batch_size, n_emb), embeddings of context and response respectively
        Return
        ------
            contrastive loss
        """
        context = self.context_mapper(context)
        response = self.response_mapper(response)
        
        logits = context @ response.T
        labels = torch.arange(context.shape[0]).cuda()

        loss_c = F.cross_entropy(logits, labels, reduction='mean')
        loss_r = F.cross_entropy(logits.T, labels, reduction='mean')

        return (loss_c + loss_r) / 2
    
    def training_step(self, batch, batch_idx):
        context, response = batch
        loss = self.forward(context, response)
        self.log(
            name='train_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        context, response = batch
        loss = self.forward(context, response)
        self.log(
            name='val_loss',
            value=loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
    
    @torch.no_grad()
    def compute_similarities(self, responses, context):
        """
        Is designed to be used during response selection. Compute similarity for single context and set of responses.

        Params
        ------
            context: torch.Tensor of size (n_emb,)
            responses: torch.Tensor of size (n_responses, n_emb)
        
        Return
        ------
            similarities: torch.Tensor of size (n_respones,)
        """
        # responses = torch.FloatTensor(responses)
        # context = torch.FloatTensor(context)

        responses = self.response_mapper(responses)

        if len(context.shape) == 1:
            context = self.context_mapper(context.unsqueeze(0)).squeeze()
            return responses @ context
        else:
            context = self.context_mapper(context)
            return context @ responses.T


@torch.no_grad()
def get_accuracy_k(model: MAP, k_list, context, response, cluster_labels, dialogues_rle):
    """
    Two steps of aggregation:
        - mean acc@k for each dialogue
        - mean acc@k for whole set of dialogues
    
    Warning: this is a fairly heavy function and may take a while to compute.
        
    Params
    ------
        - `model`: trained response selector
        - `k_list`: list of all k for which acc@k to calculate
        - `context`, response: outputs of data_utils.make_data_for_response_selection
        - `cluster_labels`: cluster labels of provided data
        - `dialogues_rle`: run-length encoding of dialogue dataset
    
    Return
    ------
        list of acc@k for each k in `k_list`
    """
    
    # these are labels of correct responses
    global_indices = torch.arange(len(cluster_labels))

    i = 0
    metric = defaultdict(list)

    for dia_len in tqdm(dialogues_rle, desc='Computing acc@k'):
        # to store zeros and ones for each utterance in current dialogue
        dia_metric = defaultdict(list)
        
        for _ in range(dia_len):
            mask = cluster_labels == cluster_labels[i]
            
            # we use this as mapping: in cluster (local) indices -> in dataset (global) indices
            local_to_global = global_indices[mask]

            # search top k among responses from predicted cluster
            candidate_responses = response[mask]
            probabilities = model.compute_similarities(candidate_responses, context).cpu()
            
            _, local_indices = torch.topk(probabilities, max(k_list), sorted=True)
            predicted_labels = local_to_global[local_indices]

            # assign zeros and ones
            # note: `i` is a label of correct response
            for k in k_list:
                dia_metric[k].append(i in predicted_labels[:k])
            
            i += 1
        
        for k in k_list:
            metric[k].append(np.mean(dia_metric[k]))

    return [np.mean(metric[k]) for k in k_list]


def make_data(
        dialogues_rle,
        context_length,
        utterance_embeddings
    ):    
    '''
    Transform list of dialogues to training data and target values for retrieving responses from predicted cluster.
    For each utterance concatenate sentence embeddings of previous `context_length` utterances.
    
    Note: Embeddings of absent utterances are replaced with zero vectors.

    Params
    ------
        - dialogues_rle, list[int]: list of dialogues lengths
        - context_length
        - utterance_embeddings, np.ndarray: (n_utterances, sentence_emb_size)
    
    Return
    ------
        - data
        - data_target
    '''
    
    # resulting arrays
    context = []
    response = []

    # first utterance in dialogue has no context so we fill it with zero embeddings 
    null_sentence_embedding = torch.zeros(utterance_embeddings.shape[1], dtype=torch.double)

    i = 0

    # for each dialogue in dataset
    for dia_len in tqdm(dialogues_rle, desc='Making data for response selection'):
        
        # queue for prev_k utterances
        dialogue_context = deque(
            [null_sentence_embedding for _ in range(context_length)],
            maxlen=context_length
        )

        # for each utter in dialogue
        for _ in range(dia_len):
            # make context of prev_k utterances
            context_embeddings = torch.cat(
                [utterance_embedding for utterance_embedding in dialogue_context]
            )

            # update resulting arrays
            context.append(context_embeddings)
            response.append(torch.from_numpy(utterance_embeddings[i]))

            # update context with current utterance
            dialogue_context.append(torch.from_numpy(utterance_embeddings[i]))
            i += 1

    return list(zip(context, response))


if __name__ == '__main__':
    import json
    import numpy as np
    from torch.utils.data import DataLoader
    from lightning.pytorch.callbacks import ModelCheckpoint
    from datetime import datetime
    import argparse

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(0)

    ap = argparse.ArgumentParser()
    ap.add_argument('--task', dest='task', required=True)
    args = ap.parse_args()

    directory = f'{args.task}_stuff'

    if args.task == 'persona_chat':
        train_response_embeddings = np.load(f'{directory}/train_embeddings.npy')
        val_response_embeddings = np.load(f'{directory}/val_embeddings.npy')
    elif args.task == 'imad':
        train_response_embeddings = np.load(f'{directory}/train_response_embeddings.npy')
        val_response_embeddings = np.load(f'{directory}/val_response_embeddings.npy')
    else:
        raise ValueError(f'Not supported: {args.task}. Supported tasks: imad, persona_chat.')

    train_data = make_data(
        json.load(open(f'{directory}/train_rle.json', 'r')),
        context_length=RESPONSE_SELECTION_CONTEXT_LENGTH,
        utterance_embeddings=train_response_embeddings
    )

    val_data = make_data(
        json.load(open(f'{directory}/val_rle.json', 'r')),
        context_length=RESPONSE_SELECTION_CONTEXT_LENGTH,
        utterance_embeddings=val_response_embeddings
    )
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10
    )

    context_emb_size = len(train_data[0][0])
    response_emb_size = len(train_data[0][1])
    model = MAP(context_emb_size, response_emb_size, projection_size=PROJECTION_SIZE)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_last=True,
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        # max_epochs=100,
        max_time={'minutes': 90},

        # hardware settings
        accelerator='gpu',
        deterministic=True,  # for reproducibility
        precision="16-mixed",

        # logging and checkpointing
        logger=True,
        enable_progress_bar=False,
        profiler=None,
        callbacks=[checkpoint_callback],

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fails
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    # do magic!
    trainer.fit(model, train_loader, val_loader)

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))
