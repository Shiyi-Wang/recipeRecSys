import json
import random

with open('../data/top_ingredients.json') as f:
    random_generate_ingredients_range = json.load(f)


def generate_a_random_ingredient():
    return random.choice(random_generate_ingredients_range)


if __name__ == '__main__':
    print(generate_a_random_ingredient())
