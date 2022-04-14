import json
import random
import pickle

file = open("../data/recipes_names.pkl", 'rb')
rep_names = pickle.load(file)

with open('../data/top_ingredients.json') as f:
    random_generate_ingredients_range = json.load(f)


def translate_recipe_names(results, rep_names=rep_names):
    print("succeed")
    return [pretty_text(rep_names[r]) for r in results]


def pretty_text(text):
    ''' This function takes in text and try to put it in a human readable format by putting back \' and making it capitalize
    '''
    text = text.replace(" s ", "\'s ")
    text_split = text.split(" ")
    # print(text_split)
    text_split = [t.strip().capitalize() for t in text_split if t != '']
    # print(text_split)
    return " ".join(text_split)


def generate_a_random_ingredient():
    return random.choice(random_generate_ingredients_range)


if __name__ == '__main__':
    print(generate_a_random_ingredient())
    print(translate_recipe_names([23.0, 56.0, 34.0, 88.0]))
