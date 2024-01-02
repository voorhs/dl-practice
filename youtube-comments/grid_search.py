import numpy as np
import optuna
from sklearn.cluster import DBSCAN


search_space = {
    'eps': np.linspace(0.2, 1.2, 20).tolist(),
    'min_samples': np.arange(1, 21).tolist()
}

def dbscan_n_clusters(trial: optuna.trial.Trial, embeddings):
    hparams = {param: trial.suggest_categorical(param, vals) for param, vals in search_space.items()}

    clusterer = DBSCAN(
        n_jobs=4,
        **hparams
    ).fit(embeddings)

    return len(np.unique(clusterer.labels_))


if __name__ == "__main__":
    from optuna.samplers import GridSampler
    import argparse
    from functools import partial
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', dest='encoder', required=True, choices=['mpnet', 'tfidf', 'bow', 'labse'])
    ap.add_argument('--study_name', dest='study_name', required=True)
    args = ap.parse_args()

    storage_name = f"sqlite:///{args.study_name}.db"

    study = optuna.create_study(
        sampler=GridSampler(search_space, seed=0),
        directions=['maximize'],
        study_name=args.study_name,
        storage=storage_name,
        load_if_exists=True
    )
    
    # n_trials is equal to number of all possible hparams combinations
    n_trials = 1
    for val in search_space.values():
        if isinstance(val, list):
            n_trials *= len(val)

    embeddings = np.load(f'{args.encoder}_embeddings.npy')
    func = partial(dbscan_n_clusters, embeddings=embeddings)
    
    study.optimize(func, n_trials=n_trials, n_jobs=4)
