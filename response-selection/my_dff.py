from dgac_clustering import Clusters
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np
import pickle
from catboost import CatBoostClassifier
from response_selection import MAP, RESPONSE_SELECTION_CONTEXT_LENGTH, PROJECTION_SIZE
from linking_prediction import LINKING_PREDICTION_CONTEXT_LENGTH
from dataclasses import dataclass
from collections import deque
import json
from torch import multinomial, from_numpy
import torch.nn.functional as F

# helper for DialogueContext
@dataclass
class Utterance:
    sentence_embedding: np.ndarray
    cluster_embedding: np.ndarray


# helper for ChatBot
class DialogueContext:
    def __init__(
            self,
            linking_context_length,
            response_context_length,
            null_sentence_embedding,
            null_cluster_embedding
        ):
        self.linking_context_length = linking_context_length
        self.response_context_length = response_context_length
        self.null_sentence_embedding = null_sentence_embedding
        self.null_cluster_embedding = null_cluster_embedding
        self.deque = self._empty_deque()
    
    def append(self, sentence_embedding, cluster_embedding):
        self.deque.append(Utterance(sentence_embedding, cluster_embedding))
    
    def features_for_linking_prediction(self):
        return np.concatenate(
            [np.r_[
                self.deque[i].cluster_embedding,
                self.deque[i].sentence_embedding
            ] for i in range(self.linking_context_length)]
        )
    
    def features_for_response_selection(self):
        return np.concatenate(
            [self.deque[i].sentence_embedding for i in range(self.response_context_length)]
        )

    def _empty_deque(self):
        return deque(
            [Utterance(
                self.null_sentence_embedding,
                self.null_cluster_embedding
            ) for _ in range(max(self.linking_context_length, self.response_context_length))]
        )
    
    def reset(self):
        self.deque = self._empty_deque()


class ChatBot:
    """
    типы эмбеддингов, которые используются:
    - distilroberta для самой реплики
    - word2vec для кластера 1 уровня
    - word2vec для кластера 2 уровня
    - средний sbert в кластере

    алгоритм:
    - clustering:
        - считать реплику и вычислить её distilroberta-эмбеддинг
        - найти кластер, средний distilroberta-эмбеддинг которого ближе всего к данной реплике 
    - linking prediction:
        - сформировать контекст из k реплик: distilroberta-эмбеддинги + w2v-эмбеддинги
        - скормить катбусту и предсказать следующий кластер
    - response selection:
        - сформировать контекст из m реплик: distilroberta-эмбеддинги
        - методом MAP спроецировать контекст и все возможные ответы в единую размерность
        - среди максимальных по косинусу выбрать ответ с помощью top k sampling
    """
    def __init__(
            self,
            linking_context_length=LINKING_PREDICTION_CONTEXT_LENGTH,
            response_context_length=RESPONSE_SELECTION_CONTEXT_LENGTH,
            clusterer_path='persona_chat_stuff/dgac_clusterer.pickle',
            cluster_predictor_path='persona_chat_stuff/cluster_predictor',
            response_embeddings_path='persona_chat_stuff/train_embeddings.npy',
            response_utterances_path='persona_chat_stuff/train_utterances.json',
            response_selector_path='persona_chat_stuff/response_selector.ckpt',
        ):
        self.linking_context_length = linking_context_length
        self.response_context_length = response_context_length
        
        self.clusterer_path = clusterer_path
        self.cluster_predictor_path = cluster_predictor_path
        self.response_embeddings_path = response_embeddings_path
        self.response_utterances_path = response_utterances_path
        self.response_selector_path = response_selector_path

        self.sentence_emb_size = np.load(self.response_embeddings_path).shape[1]
        cluster_emb_size = len(self._cluster_embedding(i_cluster=0))
        null_sentence_emb = np.zeros(self.sentence_emb_size)
        null_cluster_emb = np.zeros(cluster_emb_size)

        self.dialogue_context = DialogueContext(
            self.linking_context_length,
            self.response_context_length,
            null_sentence_emb,
            null_cluster_emb
        )
    
    def send(self, utterance):
        sentence_embedding = self._encode_utterance(utterance)
        i_cluster = self._find_nearest_cluster(sentence_embedding)
        cluster_embedding = self._cluster_embedding(i_cluster)
        self.dialogue_context.append(sentence_embedding, cluster_embedding)

    def respond(self):
        features = self.dialogue_context.features_for_linking_prediction()
        next_i_cluster = self._linking_prediction(features)
        features = self.dialogue_context.features_for_response_selection()
        sentence_embeddings, utterances = self._get_cluster(next_i_cluster)
        return self._response_selection(features, sentence_embeddings, utterances)
    
    def reset(self):
        self.dialogue_context.reset()

    def _encode_utterance(self, utterance) -> np.ndarray:
        """Encode received utterance with some sentence encoder."""

        sbert_encoder = SentenceTransformer('sentence-transformers/all-distilroberta-v1').to('cuda')
        return sbert_encoder.encode(utterance)
    
    def _find_nearest_cluster(self, utterance_embedding) -> int:
        """Find nearest cluster of received utterance"""
        centroids = pickle.load(open(self.clusterer_path, 'rb'))._second_centroids
        return cdist(utterance_embedding[None, :], centroids, metric='euclidean')[0].argmin()
    
    def _linking_prediction(self, features) -> int:
        """Solve problem of linking prediction with some classifier."""

        cluster_predictor = CatBoostClassifier()
        cluster_predictor.load_model(self.cluster_predictor_path)
        return cluster_predictor.predict(features[None, :])[0]
    
    def _cluster_embedding(self, i_cluster):
        return pickle.load(open(self.clusterer_path, 'rb')).embeddings[i_cluster]

    def _get_cluster(self, i_cluster):
        labels = pickle.load(open(self.clusterer_path, 'rb')).labels
        mask = (labels == i_cluster)
        utterances = np.array(json.load(open(self.response_utterances_path, 'r')))[mask]
        embeddings = np.load(self.response_embeddings_path)[mask]
        return embeddings, utterances
        
    def _response_selection(self, context_embedding, response_embeddings, response_utterances) -> int:
        """Get appropriate utterance from train_dialogues using topk sampling"""

        model = MAP.load_from_checkpoint(
            self.response_selector_path,
            context_emb_size=self.sentence_emb_size * RESPONSE_SELECTION_CONTEXT_LENGTH,
            response_emb_size=self.sentence_emb_size,
            projection_size=PROJECTION_SIZE
        )
        model.eval()

        response_embeddings = from_numpy(response_embeddings)
        context_embedding = from_numpy(context_embedding)
        
        similarities = model.compute_similarities(response_embeddings, context_embedding)
        weights, indices = similarities.topk(k=7)
        i_response = indices[multinomial(F.softmax(weights, dim=0), num_samples=1)]
        return response_utterances[i_response]
        