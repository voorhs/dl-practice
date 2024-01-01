import numpy as np
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass


LINKING_PREDICTION_CONTEXT_LENGTH = 5


def get_accuracy_k(k, labels, probabilities, dialogues):
    """
    Two steps of aggregation:
        - mean acc@k for each dialogue
        - mean acc@k for whole set of dialogues
    
    Params
    ------
        - k: int
        - labels: np.ndarray of size (n_utterances,)
        - probabilities: np.ndarray of size (n_utterances, n_clusters)
        - dialogues: list[list[str]] where list[str] is a single dialogue
    """
    i = 0
    metric = []

    for obj in dialogues:
        utterance_metric = []
        for _ in obj:
            cur_cluster = labels[i]
            top = np.argpartition(probabilities[i], kth=-k)[-k:]
            utterance_metric.append(cur_cluster in top)
            i += 1
        metric.append(np.mean(utterance_metric))
    
    return np.mean(metric)


# helper for make_data function
@dataclass
class Utterance:
    sentence_embedding: np.ndarray
    cluster_embedding: np.ndarray


def make_data(
        cluster_embeddings,
        cluster_labels,
        dialogues_rle,
        context_length,
        utterance_embeddings
    ):    
    '''
    Transform list of dialogues to training data and target values for predicting next cluster (linking prediction).
    For each utterance following vectors are concatenated and treated as features:
        - sbert embeddings of previous `context_length` utterances
        - cluster embeddings of previous `context_length` utterances
    
    Note: Embeddings of absent utterances are replaced with zero vectors.

    Params
    ------
        - cluster_embeddings, np.ndarray: (n_clusters, clust_emb_size)
        - cluster_labels, pd.Series: (n_utterances,) with clust labels for each utterance
        - dialogues_rle, list[int]: list of dialogues lengths
        - context_length
        - utterance_embeddings, np.ndarray: (n_utterances, sentence_emb_size)
    
    Return
    ------
        - data
        - data_target
    '''
    
    # resulting arrays
    data = []
    target = []

    # first utterance in dialogue has no context so we fill it with zero embeddings 
    null_sentence_embedding = np.zeros(utterance_embeddings.shape[1])
    null_cluster_embedding = np.zeros(cluster_embeddings.shape[1])

    i = 0

    # for each dialogue in dataset
    for dia_len in tqdm(dialogues_rle, desc='Preparing data'):
        
        # queue for prev_k utterances
        dialogue_context = deque(
            [Utterance(null_sentence_embedding, null_cluster_embedding) for _ in range(context_length)],
            maxlen=context_length
        )

        # for each utter in dialogue
        for _ in range(dia_len):
            # make context of prev_k utterances
            context_embeddings = np.concatenate(
                [np.r_[ut.sentence_embedding, ut.cluster_embedding] for ut in dialogue_context]
            )

            i_cluster = cluster_labels[i]

            # update resulting arrays
            data.append(context_embeddings)
            target.append(i_cluster)

            # update context with current utterance
            dialogue_context.append(Utterance(utterance_embeddings[i], cluster_embeddings[i_cluster]))
            i += 1
            
    data = np.stack(data, axis=0)
    target = np.array(target)

    return data, target


if __name__ == "__main__":
    import argparse
    import json
    import pickle
    import torch
    from dgac_clustering import Clusters
    from catboost import CatBoostClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument('--task', dest='task', required=True)
    args = ap.parse_args()

    directory = f'{args.task}_stuff'

    print('GPU count:', torch.cuda.device_count())

    with open(f'{directory}/dgac_clusterer.pickle', 'rb') as file:
        clusterer: Clusters = pickle.load(file)

    print('train')
    train_X, train_y = make_data(
        clusterer._second_embeddings,
        clusterer._second_labels,
        json.load(open(f'{directory}/train_dialogues.json', 'r')),
        LINKING_PREDICTION_CONTEXT_LENGTH,
        np.load(f'{directory}/train_embeddings.npy')
    )

    print('test')
    test_dialogues = json.load(open(f'{directory}/test_dialogues.json', 'r'))
    test_X, test_y = make_data(
        clusterer._second_embeddings,
        clusterer.predict(np.load(f'{directory}/test_embeddings.npy')),
        test_dialogues,
        LINKING_PREDICTION_CONTEXT_LENGTH,
        np.load(f'{directory}/test_embeddings.npy')
    )

    print('val')
    val_X, val_y = make_data(
        clusterer._second_embeddings,
        clusterer.predict(np.load(f'{directory}/val_embeddings.npy')),
        json.load(open(f'{directory}/val_dialogues.json', 'r')),
        LINKING_PREDICTION_CONTEXT_LENGTH,
        np.load(f'{directory}/val_embeddings.npy')
    )

    print('Training catboost...')
    classif = CatBoostClassifier(
        iterations = 500,
        learning_rate = 0.1,
        random_seed = 43,
        loss_function = 'MultiClass',
        task_type = 'GPU',
        early_stopping_rounds=50
    )
    classif.fit(train_X, train_y, eval_set = [(val_X, val_y)], verbose = 10)

    print('Predicting...')
    test_pred = classif.predict_proba(test_X)

    print("Accuracy metric\n")

    for k in [1,3,5,10]:
        print(f"Acc@{k}: {get_accuracy_k(k, test_y, test_pred, test_dialogues):.4f}")

    classif.save_model(f'{directory}/cluster_predictor')
