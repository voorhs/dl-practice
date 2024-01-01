if __name__ == '__main__':
    import argparse
    import json
    import numpy as np
    import torch
    from response_selection import MAP, PROJECTION_SIZE, get_accuracy_k, make_data
    from dgac_clustering import Clusters
    import pickle


    ap = argparse.ArgumentParser()
    ap.add_argument('--task', dest='task', required=True)
    args = ap.parse_args()

    directory = f'{args.task}_stuff'

    if args.task == 'persona_chat':
        response_embeddings = np.load(f'{directory}/test_embeddings.npy')
    elif args.task == 'imad':
        response_embeddings = np.load(f'{directory}/test_response_embeddings.npy')
    else:
        raise ValueError(f'Not supported: {args.task}. Supported tasks: imad, persona_chat.')


    test_data = make_data(
        json.load(open(f'{directory}/test_dialogues.json', 'r')),
        context_length=2,
        utterance_embeddings=response_embeddings
    )

    context_emb_size = len(test_data[0][0])
    response_emb_size = len(test_data[0][1])

    context = []
    response = []
    for c, r in test_data:
        context.append(c)
        response.append(r)
    context = torch.stack(context)
    response = torch.stack(response)

    model = MAP.load_from_checkpoint(
        f'{directory}/response_selector.ckpt',
        context_emb_size=context_emb_size,
        response_emb_size=response_emb_size,
        projection_size=PROJECTION_SIZE
    )

    clusterer: Clusters = pickle.load(open(f'{directory}/dgac_clusterer.pickle', 'rb'))
    cluster_labels = clusterer.predict(response_embeddings)

    dialogues_rle = json.load(open(f'{directory}/test_rle.json', 'r'))

    k_list = [1,3,5,10,20,50]
    acc_k_list = get_accuracy_k(model, k_list, context.cuda(), response.cuda(), cluster_labels, dialogues_rle)
    for k, acc in zip(k_list, acc_k_list):
        print(f'Acc@{k}: {acc:.4f}')