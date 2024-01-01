from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pickle

import numpy as np
import json
from scipy.spatial.distance import cdist
import argparse


class Clusters:
    ''' 
        Two-stage utternces clustering (DGAC paper).

        Attributes
        ----------
        - embeddings: trained w2v embeddings of clusters
        - centroids: mean sentence embeddings of utterances for each cluster
        - labels: cluster labels of train utterances
    '''

    def __init__(self, n_first_clusters, n_second_clusters=None):
        self.n_first_clusters = n_first_clusters
        self.n_second_clusters = n_second_clusters

    def _first_clustering(self, X):
        '''
        KMeans clustering over sentence embeddings of utterances.
        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
        Return
        ------
            centroids: np.ndarray of size (self.n_first_clusters, emb_size)
            labels: np.ndarray of size (n_utterances,)
        '''

        kmeans = KMeans(n_clusters=self.n_first_clusters, n_init=5).fit(X)
        return kmeans.cluster_centers_, kmeans.labels_

    def _cluster_embeddings(self, labels, dialogues_rle):
        """
        Train word2vec: word = cluster labels, sentence = dialogue.

        Params
        ------
            labels: np.ndarray of size (n_utterances,), cluster labels of each utterance from `dialogues`
            dialogues: list[list[str]] where list[str] is a single dialogue
        
        Return
        ------
            np.ndarray of size (n_clusters, 100)
        """
        i = 0
        array_for_word2vec = []

        for dia_len in dialogues_rle:
            array_for_word2vec.append(
                [str(clust_label) for clust_label in labels[i:i+dia_len]]
            )
            i += dia_len

        w2v_model = Word2Vec(
            sentences=array_for_word2vec,
            sg=0,
            min_count=1,
            workers=4,
            window=10,
            epochs=20
        )
        
        n_clusters = len(np.unique(labels))
        return np.stack([w2v_model.wv[str(i)] for i in range(n_clusters)])

    def _second_clustering(self, X, first_embeddings, first_labels):
        """
        KMeans clustering over word2vec embeddings of first stage clusters.

        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
            first_embeddings: np.ndarray of size (self.n_first_clusters, 100)
            first_labels: np.ndarray of size (n_utterances,)
        
        Return
        ------
            centroids: np.ndarray of size (self.n_second_clusters, emb_size)
            labels: np.ndarray of size (n_utterances,)
        """

        kmeans = KMeans(
            n_clusters=self.n_second_clusters,
            n_init=5,
            algorithm="elkan"
        ).fit(first_embeddings)

        second_labels = kmeans.labels_[first_labels]
        
        # calculating mass centers of the clusters
        centroids = []

        for i in range(self.n_second_clusters):
            centroids.append(X[second_labels == i].mean(axis=0))

        return np.stack(centroids), second_labels

    def fit(self, X, dialogues_rle):
        '''
        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
        '''
        print("First stage of clustering has begun...")
        self._first_centroids, self._first_labels = self._first_clustering(X)
        
        print('Training first stage word2vec...')
        self._first_embeddings = self._cluster_embeddings(self._first_labels, dialogues_rle)

        self.embeddings = self._first_embeddings
        self.centroids = self._first_centroids
        self.labels = self._first_labels

        if self.n_second_clusters is None:
            return self
        
        print("Second stage of clustering has begun...")
        self._second_centroids, self._second_labels = self._second_clustering(X, self._first_embeddings, self._first_labels)
        
        print('Training second stage word2vec...')
        self._second_embeddings = self._cluster_embeddings(self._second_labels, dialogues_rle)  

        self.embeddings = self._second_embeddings
        self.centroids = self._second_centroids
        self.labels = self._second_labels

        return self      

    def predict(self, X):
        """
        Predict cluster label for given utterances embeddings.
        
        Params
        ------
            X: np.ndarray of size (n_utterances, emb_size)
        
        Return
        ------
            np.ndarray of size (n_utterances,)
        """

        return cdist(X, self.centroids, metric='euclidean').argmin(axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', dest='task', required=True)
    args = ap.parse_args()

    directory = f'{args.task}_stuff'

    clusterer = Clusters(
        n_first_clusters=200,
        n_second_clusters=30,
    )

    X = np.load(f'{directory}/train_embeddings.npy')
    dialogues = json.load(open(f'{directory}/train_dialogues.json', 'r'))

    clusterer.fit(X, dialogues)

    with open(f'{directory}/dgac_clusterer.pickle', 'wb') as handle:
        pickle.dump(clusterer, handle, protocol=pickle.HIGHEST_PROTOCOL)
