import json
import pickle

'''
This hybrid process mainly works in this way:
Given three input from the frontend:
1. I might want ingredients... -> list of ingredients
2. Ingredients that I absolutely don't want -> list of ingredients
3. How likely I want to see other users' recipe with a taste similar to me today -> rating bar

Example 1:
If three input provided by user:
1. I may be want ingredients: Onions
2. Ingredients that I absolutely don't want: garlic
3. similar taste: 0.6

First, we generate three result lists by method 
pass_to_models(may_want_ingredient='Onions', similar_taste_weight, unwanted_ingredients)

Then, we get the result of knn, svd, and content_based

Second, we combine the results with appropriate weights and other user requirements by method
combine_results(knn, svd, content_based, similar_taste_weight, unwanted_ingredients):
    Remove recipes using garlic as ingredient
    Weight assignment:
        content_based(onion) -> 0.4
        knn() -> 0.3
        svd() -> 0.3
    final result order: content_based_result(0.4) -> knn_result(0.3) -> svd_result(0.3)
    Remove duplicate recipes

----------------------------------------------------------------------------------
Example 2:
If three input provided by user:
1. I may be want ingredients: any ingredients works for me
   -> Random generate from all potential ingredients e.g. tomato, cheese
2. Ingredients that I absolutely don't want: No -> empty list
3. similar taste: 0.8

First, if the user is okay with any ingredients, randomly generate a potential 
ingredient by method generate_a_random_ingredient().
    e.g. a frontend button "choose one for me" that can randomly generate an 
    ingredient and fill the textbox. If the user is okay with the generate 
    result, they can press "looks good to me" button. 

Then, we generate three result lists by method 
pass_to_models(may_want_ingredient=generated_ingredient, similar_taste_weight, unwanted_ingredients)

Therefore, we get the result of knn, svd, and content_based

Next, we combine the results with appropriate weights and other user requirements by method
combine_results(knn, svd, content_based, similar_taste_weight, unwanted_ingredients):
    Remove nothing because no ingredient is specified as unwanted
    Weight assignment:
        knn() -> 0.4
        svd() -> 0.4
        content_based(tomato, cheese) -> 0.2
    final result order: knn_result -> svd_result -> content_based_result
'''

file = open("../data/recipes_names.pkl", 'rb')
rep_names = pickle.load(file)

with open('../data/top_ingredients.json') as f:
    random_generate_ingredients_range = json.load(f)


def translate_recipe_names(results, rep_names=rep_names):
    """ 
    ** No need to interact with this in front-end
    This is a helper method used for translating the iid of recipes to 
    its recipe names for the content based list
    """
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


def pretty_text(text):
    """
    ** No need to interact with this in front end
    This is a helper method used in the process of translating recipe names
    """
    text = text.replace(" s ", "\'s ")
    text_split = text.split(" ")
    text_split = [t.strip().capitalize() for t in text_split if t != '']
    return " ".join(text_split)


def remove_unwanted_ingredients(process_list, unwanted_ingredients):
    """
    ** No need to interact with this in front end
    This is a helper method removing unwanted ingredients in a list
    """
    copy_process_list = process_list.copy()
    for unwanted in unwanted_ingredients:
        lowercased_unwanted = str(unwanted).lower()
        for c in copy_process_list:
            words = c.split(" ")
            lower_words = [w.lower() for w in words]
            if lowercased_unwanted in lower_words:
                process_list.remove(c)
        copy_process_list = process_list.copy()
    return


def remove_duplicates(combined_list):
    """
    ** No need to interact with this in front end
    This is a helper method removing all duplicates in the final result list
    """
    result = []
    for c in combined_list:
        if c not in result:
            result.append(c)
    return result


def generate_a_random_ingredient():
    """
    *** Used in frontend for randomly generate one ingredient ***
    """
    return random.choice(random_generate_ingredients_range)


def pass_to_models(may_want_ingredient, similar_taste_weight, unwanted_ingredients):
    """
    *** Main interact hit point for calling knn, svd, content_based api to 
    to generate three lists and pass to combine_results(..) for an optimal
    combined result ***

    Note: 
    may_want_ingredient should be a list of string values coming from
    frontend. It is used in calling content based model.

    similar_taste_weight should be a value coming from frontend. The user 
    drag a rating bar to generate the value. The value is between 0 and 1.

    unwanted_ingredients should be a list of string values coming from 
    frontend. User specify ingredients they don't want. 

    Args:
        may_want_ingredient (list): a frontend feedback value
        similar_taste_weight (double): a frontend feedback value
        unwanted_ingredients (list): a frontend feedback value

    Returns:
        list of recipe names: a final recommendation list that should be 
        sent to the frontend for user to see their recommended recipes by the system.
    """
    # not yet finished here
    knn, svd, content_based = [], [], []
    # current design is every model return 10 results, please take this into account when calling them
    # call knn api -> generate one list
    # call svd api -> generate one list
    # call contentbased(may_want_ingredient) api -> generate one list
    return combine_results(knn, svd, content_based, similar_taste_weight, unwanted_ingredients)


def combine_results(knn, svd, content_based, similar_taste_weight, unwanted_ingredients):
    """
    *** Main interact hit point for generating a hybrid result given three 
    lists from three models ***
    Called by pass_to_models

    Args:
        knn (list): list of recipe names by knn
        svd (list): list of recipe names by svd
        content_based (list): list of recipe iids by content based
        similar_taste_weight (_type_): a frontend feedback value
        unwanted_ingredients (_type_): a frontend feedback value

    Returns:
        list of recipe names: a final recommendation list that should be 
        sent to the frontend for user to see their recommended recipes by the system.
    """

    # calculate weights for different model based on similar_taste_weight specified by the user
    knn_weight = similar_taste_weight / 2
    svd_weight = similar_taste_weight / 2
    content_based_weight = 1 - knn_weight - svd_weight

    # remove "definately unwanted ingredients" from all lists
    content_based = translate_recipe_names(content_based)
    for process_list in [knn, svd, content_based]:
        remove_unwanted_ingredients(process_list, unwanted_ingredients)

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
                          similar_taste_weight=0.5, unwanted_ingredients=['cake', 'shirmp']))
