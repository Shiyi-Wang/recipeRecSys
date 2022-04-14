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
    return [pretty_text(rep_names[r]) for r in cleaned_results]

# helper


def pretty_text(text):
    ''' This function takes in text and try to put it in a human readable format by putting back \' and making it capitalize
    '''
    text = text.replace(" s ", "\'s ")
    text_split = text.split(" ")
    text_split = [t.strip().capitalize() for t in text_split if t != '']
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

**Remove result with garlic
content_based(onion) -> 0.4
knn() -> 0.3
svd() -> 0.3

result order: Content_based_result -> knn_result -> svd_result

----------------------------------------------------------------------------------
If three input provided by user:
1. I may be want ingredients: any ingredients works for me
   -> Random generate from all potential ingredients e.g. tomato, cheese
2. Ingredients that I absolutely don't want: No
3. similar taste: 0.8

** No remove step
knn() -> 0.4
svd() -> 0.4
content_based(tomato, cheese) -> 0.2

result: knn_result -> svd_result -> content_based_result
'''


def remove_unwanted_ingredients(combined_list, unwanted_ingredients):
    copy_combined_list = combined_list.copy()
    for unwanted in unwanted_ingredients:
        lowercased_unwanted = str(unwanted).lower()
        for c in copy_combined_list:
            words = c.split(" ")
            lower_words = [w.lower() for w in words]
            if lowercased_unwanted in lower_words:
                combined_list.remove(c)
        copy_combined_list = combined_list.copy()
    return


def remove_duplicates(combined_list):
    result = []
    for c in combined_list:
        if c not in result:
            result.append(c)
    return result


def combine_results(knn, svd, content_based, similar_taste_weight, unwanted_ingredients):
    # calculate weights for different model based on similar_taste_weight specified by the user
    knn_weight = similar_taste_weight / 2
    svd_weight = similar_taste_weight / 2
    content_based_weight = 1 - knn_weight - svd_weight

    # remove "definately unwanted ingredients" from the combined result
    content_based = translate_recipe_names(content_based)
    print("--")
    print(content_based)
    combined_list = knn + svd + content_based
    remove_unwanted_ingredients(combined_list, unwanted_ingredients)

    # select recipes from each list based on the list weight
    # Note we have to make sure we have 10 result from each list
    knn_selected = knn[:round(10 * knn_weight)]
    svd_selected = svd[:round(10 * svd_weight)]
    content_based_selected = content_based[:round(10 * content_based_weight)]

    # combine the three lists in the order of list weights
    pairing = [(knn_selected, knn_weight), (svd_selected, svd_weight),
               (content_based_selected, content_based_weight)]
    combined_result = []
    for i in range(len(pairing)):
        max_v = 0
        selected = []
        for pair in pairing:
            if pair[1] > max_v:
                selected, max_v = pair[0], pair[1]
        combined_result += selected
        pairing.remove((selected, max_v))

    # make sure we remove duplicated recipes - small probability colliding
    combined_result = remove_duplicates(combined_result)
    return combined_result


if __name__ == '__main__':
    knn = ['simple peach pie',
           'gluten free chocolate cake',
           'chevy s salsa   original recipe',
           'hooters buffalo shrimp',
           'kittencal s greek moussaka']
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
    content_based = [137739.0, 35397.0, 42195.0, 35.0, 112444.0]
    '''
    After translation:
    content_based = ['Buttermilk Oat Bread',
                     'Perfect Ham And Bean Soup',
                     '7 Up Biscuits',
                     "Wyatt Cafeteria's Baked Eggplant Aubergine",
                     'Simple Arroz Con Pollo']
    '''
    print(";)")
    print(combine_results(knn=knn, svd=svd, content_based=content_based,
                          similar_taste_weight=0.5, unwanted_ingredients=["buffalo"]))
