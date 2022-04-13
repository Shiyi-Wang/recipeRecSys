import json

'''
Generate top 1000 popular ingredients in a total of 6714 ingredients 
from 39774 recipes. Most popular ingredient is on the top.
'''

with open('../data/ingredients.json') as f:
    data = json.load(f)

ingredients = {}
for item in data:
    for i in item['ingredients']:
        if i not in ingredients:
            ingredients[str(i)] = 1
        else:
            ingredients[str(i)] += 1

sorted_ingredients_in_frequency = sorted(
    ingredients, key=ingredients.get, reverse=True)
random_generate_ingredients_range = sorted_ingredients_in_frequency[:1000]

# data is generated, do not run this again. Set generating to True if regeneration needed.
generating = False
if generating:
    with open('../data/top_ingredients.json', 'w') as f:
        json.dump(random_generate_ingredients_range, f, indent=2)
