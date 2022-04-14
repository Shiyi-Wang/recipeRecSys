import json
import random
import pickle

file = open("../data/recipes_names.pkl", 'rb')
rep_names = pickle.load(file)

with open('../data/top_ingredients.json') as f:
    random_generate_ingredients_range = json.load(f)

# helper


def translate_recipe_names(results, rep_names=rep_names):
    cleaned_results = set()
    for r in results:
        try:
            rep_names[r]
        except KeyError:
            print(
                "Key error in translating recipe names for content_based result index of " + str(r))
        else:
            cleaned_results.add(r)
    cleaned_results = list(cleaned_results)
    print(cleaned_results)
    return [pretty_text(rep_names[r]) for r in cleaned_results]

# helper


def pretty_text(text):
    ''' This function takes in text and try to put it in a human readable format by putting back \' and making it capitalize
    '''
    text = text.replace(" s ", "\'s ")
    text_split = text.split(" ")
    # print(text_split)
    text_split = [t.strip().capitalize() for t in text_split if t != '']
    # print(text_split)
    return " ".join(text_split)

# for generating a random might want ingredient in frontend


def generate_a_random_ingredient():
    return random.choice(random_generate_ingredients_range)


'''
Three input:
1. I might want ingredients...
2. Ingredients that I absolutely don't want
3. How likely I want to see other users' recipe with a taste similar to me today

weight based system on knn, svd, content_based
If three input provided by user:
1. I may be want ingredients: Onions
2. Ingredients that I absolutely don't want: garlic
3. similar taste: 0.6

content_based(onion) -> 0.4
knn() -> 0.3
svd() -> 0.3
Remove result with garlic

result: Content_based_result -> knn_result -> svd_result

----------------------------------------------------------------------------------
If three input provided by user:
1. I may be want ingredients: any ingredients works for me
   -> Random generate from all potential ingredients e.g. tomato, cheese
2. Ingredients that I absolutely don't want: No
3. similar taste: 0.8

knn() -> 0.4
svd() -> 0.4
content_based(tomato, cheese) -> 0.2
** No remove step 

result: knn_result -> svd_result -> content_based_result
'''


def combine_results(knn, svd, content_based, similar_taste_weight):
    # calculate weight
    similar_taste_weight = 0.5
    knn_weight = similar_taste_weight / 2
    svd_weight = similar_taste_weight / 2
    content_based_weight = 1 - knn_weight - svd_weight

    # call knn
    knn = ['simple peach pie',
           'gluten free chocolate cake',
           'chevy s salsa   original recipe',
           'hooters buffalo shrimp',
           'kittencal s greek moussaka']
    knn_selected = random.sample(knn, round(len(knn) * knn_weight))
    print(knn_selected)
    # call svd
    svd = ['Outback Copycat Alice Springs Chicken',
           'Ensalada Criolla',
           'Amish Triple Butter Biscuits',
           'One Pot Chicken Bacon Spinach Parmesan Pasta',
           'Thai Peanut Coconut Chicken',
           'Banana Coffee Cake',
           "Nif's Cajun Macaroni And Cheese",
           'Coco Oatmeal Honey Cookies',
           'Creole Watermelon Feta Salad',
           'Grilled Cajun Green Beans']
    svd_selected = random.sample(knn, round(len(svd) * svd_weight))
    print(svd_selected)
    # call content-based
    # assume it comes back as name of recipes
    content_based = [137739.0, 35397.0, 42195.0, 261482.0, 112444.0]
    content_based = translate_recipe_names(content_based)
    copy_content_based = content_based.copy()
    unwanted_ingredients = ['biscuits', 'po']
    for unwanted in unwanted_ingredients:
        lowercased_unwanted = str(unwanted).lower()
        for c in copy_content_based:
            words = c.split(" ")
            lower_words = [w.lower() for w in words]
            if lowercased_unwanted in lower_words:
                content_based.remove(c)
        copy_content_based = content_based.copy()

    # print(random.choices(list(set(['Coco Oatmeal Honey Cookies'])), k=round(
    #     len(content_based) * content_based_weight)))

    print(content_based)
    return[]


if __name__ == '__main__':
    print(generate_a_random_ingredient())
    print(translate_recipe_names([23.0, 56.0, 34.0, 88.0]))
    combine_results([], [], [], 0)
