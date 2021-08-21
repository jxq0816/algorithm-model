import numpy as np
from typing import Iterable, List
from gensim.models.keyedvectors import BaseKeyedVectors
from sklearn.decomposition import PCA

Sentence = List[str]


def word_vec(wv: BaseKeyedVectors, s: str):
    try:
        return wv.get_vector(s)
    except KeyError:
        return np.zeros(wv.vector_size)


class SentenceVec:
    wv: BaseKeyedVectors
    u: np.array
    a: float

    def __init__(self, sentences: Iterable[Sentence], wv: BaseKeyedVectors, a: float = 1e-3):
        self.wv = wv
        self.a = a
        embedding_size = wv.vector_size
        sentence_set = []
        for sentence in sentences:
            vs = self.weighted_average(sentence)
            sentence_set.append(vs)  # add to our existing re-calculated set of sentences

        # calculate PCA of this sentence set
        pca = PCA(n_components=embedding_size)
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))

        self.u = u
        self.vec = sentence_vecs

    def feature(self, sentence: Sentence):
        vs = self.weighted_average(sentence)
        return vs - vs * self.u

    def get_word_frequency(self, s) -> float:
        vocab = self.wv.vocab.get(s)
        return vocab.count / 10000000 if vocab else 0

    def weighted_average(self, sentence: Sentence):
        dim = self.wv.vector_size
        a = self.a
        vs = np.zeros(dim)  # add all word2vec values into one vector for the sentence
        for word in sentence:
            a_value = a / (a + self.get_word_frequency(word))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word_vec(self.wv, word)))  # vs += sif * word_vector

        vs = np.divide(vs, len(sentence))  # weighted average
        return vs
