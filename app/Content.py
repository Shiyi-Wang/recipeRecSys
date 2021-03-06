from MeanEmbeddingVectorizer import MeanEmbeddingVectorizer
from TfidfEmbeddingVectorizer import TfidfEmbeddingVectorizer
from pathlib import Path
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import io
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import current_app
import redis
import time
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import unidecode
import re
import ast
import string
import os.path
import pathlib

import pandas as pd
import nltk
nltk.download('omw-1.4')
# nltk.download('wordnet')


nltk.download('wordnet')


class Content:

    def __init__(self, data_path='RAW_recipes.csv'):
        data = pd.read_csv(data_path)
        # parse the ingredients for each recipe
        data['parsed'] = data.ingredients.apply(self.ingredient_parser)
        self.data = data

        # get corpus
        corpus = self.get_and_sort_corpus(data)
        print(f"Length of corpus: {len(corpus)}")

        # train and save CBOW Word2Vec model
        model_path = pathlib.Path('model_cbow.bin')
        if model_path.is_file():
            print('Find trained model.')
            self.model_cbow = Word2Vec.load("model_cbow.bin")
        else:
            model_cbow = Word2Vec(
                corpus, sg=0, workers=1, window=self.get_window(corpus), min_count=1,
            )
            filepath = Path('model_cbow.model')
            filepath.parent.mkdir(parents=True, exist_ok=True)
            MODELPATH = 'model_cbow.model'
            if model_cbow.save('model_cbow.bin'):
                print("Word2Vec model successfully trained")
            self.model_cbow = model_cbow

    def ingredient_parser(self, ingreds):
        '''

        This function takes in a list (but it is a string as it comes from pandas dataframe) of
           ingredients and performs some preprocessing.
           For example:

           input = '['1 x 1.6kg whole duck', '2 heaped teaspoons Chinese five-spice powder', '1 clementine',
                     '6 fresh bay leaves', 'GRAVY', '', '1 bulb of garlic', '2 carrots', '2 red onions',
                     '3 tablespoons plain flour', '100 ml Marsala', '1 litre organic chicken stock']'

           output = ['duck', 'chinese five spice powder', 'clementine', 'fresh bay leaf', 'gravy', 'garlic',
                     'carrot', 'red onion', 'plain flour', 'marsala', 'organic chicken stock']

        '''
        measures = ['teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'tbl.', 'tb', 'tbsp.', 'fluid ounce', 'fl oz', 'gill',
                    'cup', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'ml',
                    'milliliter', 'millilitre', 'cc', 'mL', 'l', 'liter', 'litre', 'L', 'dl', 'deciliter', 'decilitre',
                    'dL', 'bulb', 'level', 'heaped', 'rounded', 'whole', 'pinch', 'medium', 'slice', 'pound', 'lb', '#',
                    'ounce', 'oz', 'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram',
                    'kilogramme', 'x', 'of', 'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre', 'm',
                    'meter', 'metre', 'inch', 'in', 'milli', 'centi', 'deci', 'hecto', 'kilo']
        words_to_remove = ['fresh', 'oil', 'a', 'red', 'bunch', 'and', 'clove', 'or', 'leaf', 'chilli', 'large',
                           'extra', 'sprig', 'ground', 'handful', 'free', 'small', 'pepper', 'virgin', 'range', 'from',
                           'dried', 'sustainable', 'black', 'peeled', 'higher', 'welfare', 'seed', 'for', 'finely',
                           'freshly', 'sea', 'quality', 'white', 'ripe', 'few', 'piece', 'source', 'to', 'organic',
                           'flat', 'smoked', 'ginger', 'sliced', 'green', 'picked', 'the', 'stick', 'plain', 'plus',
                           'mixed', 'mint', 'bay', 'basil', 'your', 'cumin', 'optional', 'fennel', 'serve', 'mustard',
                           'unsalted', 'baby', 'paprika', 'fat', 'ask', 'natural', 'skin', 'roughly', 'into', 'such',
                           'cut', 'good', 'brown', 'grated', 'trimmed', 'oregano', 'powder', 'yellow', 'dusting',
                           'knob', 'frozen', 'on', 'deseeded', 'low', 'runny', 'balsamic', 'cooked', 'streaky',
                           'nutmeg', 'sage', 'rasher', 'zest', 'pin', 'groundnut', 'breadcrumb', 'turmeric', 'halved',
                           'grating', 'stalk', 'light', 'tinned', 'dry', 'soft', 'rocket', 'bone', 'colour', 'washed',
                           'skinless', 'leftover', 'splash', 'removed', 'dijon', 'thick', 'big', 'hot', 'drained',
                           'sized', 'chestnut', 'watercress', 'fishmonger', 'english', 'dill', 'caper', 'raw',
                           'worcestershire', 'flake', 'cider', 'cayenne', 'tbsp', 'leg', 'pine', 'wild', 'if', 'fine',
                           'herb', 'almond', 'shoulder', 'cube', 'dressing', 'with', 'chunk', 'spice', 'thumb', 'garam',
                           'new', 'little', 'punnet', 'peppercorn', 'shelled', 'saffron', 'other''chopped', 'salt',
                           'olive', 'taste', 'can', 'sauce', 'water', 'diced', 'package', 'italian', 'shredded',
                           'divided', 'parsley', 'vinegar', 'all', 'purpose', 'crushed', 'juice', 'more', 'coriander',
                           'bell', 'needed', 'thinly', 'boneless', 'half', 'thyme', 'cubed', 'cinnamon', 'cilantro',
                           'jar', 'seasoning', 'rosemary', 'extract', 'sweet', 'baking', 'beaten', 'heavy', 'seeded',
                           'tin', 'vanilla', 'uncooked', 'crumb', 'style', 'thin', 'nut', 'coarsely', 'spring', 'chili',
                           'cornstarch', 'strip', 'cardamom', 'rinsed', 'honey', 'cherry', 'root', 'quartered', 'head',
                           'softened', 'container', 'crumbled', 'frying', 'lean', 'cooking', 'roasted', 'warm',
                           'whipping', 'thawed', 'corn', 'pitted', 'sun', 'kosher', 'bite', 'toasted', 'lasagna',
                           'split', 'melted', 'degree', 'lengthwise', 'romano', 'packed', 'pod', 'anchovy', 'rom',
                           'prepared', 'juiced', 'fluid', 'floret', 'room', 'active', 'seasoned', 'mix', 'deveined',
                           'lightly', 'anise', 'thai', 'size', 'unsweetened', 'torn', 'wedge', 'sour', 'basmati',
                           'marinara', 'dark', 'temperature', 'garnish', 'bouillon', 'loaf', 'shell', 'reggiano',
                           'canola', 'parmigiano', 'round', 'canned', 'ghee', 'crust', 'long', 'broken', 'ketchup',
                           'bulk', 'cleaned', 'condensed', 'sherry', 'provolone', 'cold', 'soda', 'cottage', 'spray',
                           'tamarind', 'pecorino', 'shortening', 'part', 'bottle', 'sodium', 'cocoa', 'grain', 'french',
                           'roast', 'stem', 'link', 'firm', 'asafoetida', 'mild', 'dash', 'boiling']
        # The ingredient list is now a string so we need to turn it back into a list. We use ast.literal_eval

        if isinstance(ingreds, list):
            ingredients = ingreds
        else:
            ingredients = ast.literal_eval(ingreds)
        # We first get rid of all the punctuation. We make use of str.maketrans. It takes three input
        # arguments 'x', 'y', 'z'. 'x' and 'y' must be equal-length strings and characters in 'x'
        # are replaced by characters in 'y'. 'z' is a string (string.punctuation here) where each character
        #  in the string is mapped to None.
        translator = str.maketrans('', '', string.punctuation)
        lemmatizer = WordNetLemmatizer()
        ingred_list = []
        for i in ingredients:
            i.translate(translator)
            # We split up with hyphens as well as spaces
            items = re.split(' |-', i)
            # Get rid of words containing non alphabet letters
            items = [word for word in items if word.isalpha()]
            # Turn everything to lowercase
            items = [word.lower() for word in items]
            # remove accents
            items = [unidecode.unidecode(word) for word in
                     items]  # ''.join((c for c in unicodedata.normalize('NFD', items) if unicodedata.category(c) != 'Mn'))
            # Lemmatize words so we can compare words to measuring words
            items = [lemmatizer.lemmatize(word) for word in items]
            # Gets rid of measuring words/phrases, e.g. heaped teaspoon
            items = [word for word in items if word not in measures]
            # Get rid of common easy words
            items = [word for word in items if word not in words_to_remove]
            if items:
                ingred_list.append(' '.join(items))

        return ingred_list

    def remove(self, x):
        return x.replace('[', '').replace('\'', '').replace(']', '')

    def get_and_sort_corpus(self, data):
        corpus_sorted = []
        c = 0
        for doc in data.parsed.values:

            doc = list(map(self.remove, doc))
            corpus_sorted.append(doc)

        return corpus_sorted

    # calculate average length of each document
    def get_window(self, corpus):
        lengths = [len(doc) for doc in corpus]
        avg_len = float(sum(lengths)) / len(lengths)
        return round(avg_len)

    def ingredient_parser_final(self, ingredient):
        if isinstance(ingredient, list):
            ingredients = ingredient
        else:
            ingredients = ast.literal_eval(ingredient)

        ingredients = ','.join(ingredients)
        ingredients = unidecode.unidecode(ingredients)
        return ingredients

    def title_parser(self, title):
        title = unidecode.unidecode(title)
        return title

    def get_recommendations(self, N, scores):
        """
        Rank scores and output a pandas data frame containing all the details of the top N recipes.
        :param scores: list of cosine similarities
        """
        # load in recipe dataset
        df_recipes = self.data
        # order the scores with and filter to get the highest N scores
        top = sorted(range(len(scores)),
                     key=lambda i: scores[i], reverse=True)[:N]
        # create dataframe to load in recommendations
        recommendation = pd.DataFrame(
            columns=["id", "recipe", "ingredients", "score", "n_steps", "steps"])
        count = 0
        for i in top:
            recommendation.at[count, "id"] = df_recipes["id"][i]
            recommendation.at[count, "recipe"] = self.title_parser(
                df_recipes["name"][i])
            recommendation.at[count, "ingredients"] = self.ingredient_parser_final(
                df_recipes["ingredients"][i]
            )
            # recommendation.at[count, "url"] = df_recipes["recipe_urls"][i]
            recommendation.at[count, "score"] = f"{scores[i]}"
            recommendation.at[count, "n_steps"] = df_recipes["n_steps"][i]
            recommendation.at[count, "steps"] = df_recipes["steps"][i]
            count += 1
        return recommendation

    def get_recs(self, ingredients, N=5, mean=False):
        """
        Get the top N recipe recomendations.
        :param ingredients: comma seperated string listing ingredients
        :param N: number of recommendations
        :param mean: False if using tfidf weighted embeddings, True if using simple mean
        """
        # load in word2vec model
        model = self.model_cbow
        # normalize embeddings
        model.init_sims(replace=True)
        if model:
            print("Successfully loaded model")
        # # load in data
        # data = pd.read_csv(self.data_path)
        # # parse ingredients
        # data["parsed"] = data.ingredients.apply(self.ingredient_parser)
        # create corpus
        corpus = self.get_and_sort_corpus(self.data)

        if mean:
            # get average embdeddings for each document
            mean_vec_tr = MeanEmbeddingVectorizer(model)
            doc_vec = mean_vec_tr.transform(corpus)
            doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
            assert len(doc_vec) == len(corpus)
        else:
            # use TF-IDF as weights for each word embedding
            tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
            tfidf_vec_tr.fit(corpus)
            doc_vec = tfidf_vec_tr.transform(corpus)
            doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
            assert len(doc_vec) == len(corpus)

        # create embeddings for input text
        input = ingredients
        # create tokens with elements
        input = input.split(",")
        # parse ingredient list
        input = self.ingredient_parser(input)
        # get embeddings for ingredient doc

        if mean:
            input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
        else:
            input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

        # get cosine similarity between input embedding and all the document embeddings

        cos_sim = map(lambda x: cosine_similarity(
            input_embedding, x)[0][0], doc_vec)
        scores = list(cos_sim)
        # Filter top N recommendations
        recommendations = self.get_recommendations(N, scores)
        return recommendations
