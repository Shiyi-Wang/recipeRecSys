import pandas as pd
import nltk
import string
import ast
import re
import unidecode
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import time
import redis
from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
import io
import string
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path



class TfidfEmbeddingVectorizer(object):
    def __init__(self, model_cbow):

        self.model_cbow = model_cbow
        self.word_idf_weight = None
        self.vector_size = model_cbow.wv.vector_size

    def fit(self, docs):
        """
	Build a tfidf model to compute each word's idf as its weight.
	"""

        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)
        # if a word was never seen it is given idf of the max of known idf value
        max_idf = max(tfidf.idf_)
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs):
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):

        mean = []
        for word in doc:
            if word in self.model_cbow.wv.index_to_key:
                mean.append(
                    self.model_cbow.wv.get_vector(word) * self.word_idf_weight[word]
                )

        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.doc_average(doc) for doc in docs])