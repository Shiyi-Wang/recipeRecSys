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





class MeanEmbeddingVectorizer(object):
    def __init__(self, model_cbow):
        self.model_cbow = model_cbow
        self.vector_size = model_cbow.wv.vector_size
        self.p=False

    def fit(self):
        return self

    def transform(self, docs):
        doc_vector = self.doc_average_list(docs)
        return doc_vector

    def doc_average(self, doc):
        mean = []
        for word in doc:

            if word in self.model_cbow.wv.index_to_key:
                mean.append(self.model_cbow.wv.get_vector(word))

        if not mean:

            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.doc_average(doc) for doc in docs])