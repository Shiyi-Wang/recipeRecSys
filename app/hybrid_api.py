import json
import pickle
import ast
# import svd_api
# import knn_api
from Content import Content

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

file = open("../data/id_name.pkl", 'rb')
id_name = pickle.load(file)

file = open("../data/id_ingredients.pkl", 'rb')
id_ingredients = pickle.load(file)

file = open("../data/id_steps.pkl", 'rb')
id_steps = pickle.load(file)

with open('../data/top_ingredients.json') as f:
    random_generate_ingredients_range = json.load(f)


def remove_unwanted_ingredients_by_list_of_ids(rep_ids, unwanted_ingredients):
    unwanted_ingredients = [u.lower() for u in unwanted_ingredients]
    result = []
    for r in rep_ids:
        try:
            ingredients = id_ingredients[r]
        except KeyError:
            print(
                "Key error in getting ingredients recipe id: " + str(r))
        else:
            ingredients = ast.literal_eval(ingredients)
            for i in ingredients:
                word = i.split(' ')
                error = False
                for w in word:
                    if w.lower() in unwanted_ingredients:
                        error = True
                        break
                if error:
                    break
            if not error:
                result.append(r)
    return result


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


def get_name_ingredients_and_steps_by_id(recipe_ids):
    return_list = []
    for r in recipe_ids:
        try:
            name = id_name[r]
            ingredients = id_ingredients[r]
            steps = id_steps[r]
        except KeyError:
            print(
                "Key error in getting name, ingredients and steps for recipe id: " + str(r))
        else:
            ingredients = ast.literal_eval(ingredients)
            steps = ast.literal_eval(steps)
            return_list.append([name, ingredients, steps])
    return return_list


def generate_a_random_ingredient():
    """
    *** Used in frontend for randomly generate one ingredient ***
    """
    return random.choice(random_generate_ingredients_range)


def advanced_pass_to_models(user_id, user_rated_iids, may_want_ingredient, similar_taste_weight, unwanted_ingredients, num_of_recommendation):
    knn, svd, content_based = [], [], []
    # optimization on speed
    if (similar_taste_weight != 0):
        # call knn api -> generate one list
        knn = knn_api.recommend(
            userId=user_id, num_similar_users=5, num_recipes_recommended=num_of_recommendation)
        print("knn result")
        print(knn)
        # call svd api -> generate one list
        svd = svd_api.get_n_predictions(iids=user_rated_iids,
                                        algo=svd_api.SVD_algo, n=num_of_recommendation, uid=user_id)
        print("svd result")
        print(svd)
    # call contentbased(may_want_ingredient) api -> generate one list
    c = Content('../data/RAW_recipes.csv')
    content_based = c.get_recs(
        may_want_ingredient, N=num_of_recommendation).id.values.tolist()
    print('content based result')
    print(content_based)
    return combine_results(knn, svd, content_based, similar_taste_weight, unwanted_ingredients)


def pass_to_models(user_id, user_rated_iids, may_want_ingredient, similar_taste_weight, unwanted_ingredients):
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
    knn, svd, content_based = [], [], []
    # optimization on speed
    if (similar_taste_weight != 0):
        # call knn api -> generate one list
        knn = knn_api.recommend(
            userId=user_id, num_similar_users=5, num_recipes_recommended=10)
        print("knn result")
        print(knn)
        # call svd api -> generate one list
        svd = svd_api.get_n_predictions(iids=user_rated_iids,
                                        algo=svd_api.SVD_algo, uid=user_id)
        print("svd result")
        print(svd)
    # call contentbased(may_want_ingredient) api -> generate one list
    c = Content('../data/RAW_recipes.csv')
    content_based = c.get_recs(may_want_ingredient, N=10).id.values.tolist()
    print("content_based result")
    print(content_based)
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
        unwanted_ingredients (_type_): a frontend feedback value - string

    Returns:
        list of recipe names: a final recommendation list that should be 
        sent to the frontend for user to see their recommended recipes by the system.
    """
    unwanted_ingredients = unwanted_ingredients.split(', ')

    # calculate weights for different model based on similar_taste_weight specified by the user
    knn_weight = similar_taste_weight / 2
    svd_weight = similar_taste_weight / 2
    content_based_weight = 1 - knn_weight - svd_weight

    # remove "definately unwanted ingredients" from all lists
    knn = remove_unwanted_ingredients_by_list_of_ids(
        knn, unwanted_ingredients)
    svd = remove_unwanted_ingredients_by_list_of_ids(
        svd, unwanted_ingredients)
    content_based = remove_unwanted_ingredients_by_list_of_ids(
        content_based, unwanted_ingredients)

    # select recipes from each list based on the list weight
    # Note we have to make sure we have 10 result from each list
    knn_selected = knn[:round(10 * knn_weight)]
    svd_selected = svd[:round(10 * svd_weight)]
    content_based_selected = content_based[:round(
        10 * content_based_weight)]

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
    print(combined_result)
    combined_result = get_name_ingredients_and_steps_by_id(combined_result)
    return combined_result


if __name__ == '__main__':
    knn = ['simple peach pie',
           'gluten free chocolate cake',
           'chevy s salsa   original recipe',
           'hooters buffalo shrimp',
           'kittencal s greek moussaka']
    knn = [128494, 224448, 432666, 26554, 88804]
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
    svd = [536363, 536382, 536384, 536436,
           536476, 536568, 536688, 536729, 536734]
    content_based = [137739.0, 35397.0, 42195.0, 35.0, 112444.0, 112444]
    '''
    After translation:
    content_based = ['Buttermilk Oat Bread',
                     'Perfect Ham And Bean Soup',
                     '7 Up Biscuits',
                     "Wyatt Cafeteria's Baked Eggplant Aubergine",
                     'Simple Arroz Con Pollo']
    '''
    # print(pass_to_models(user_id=3787, user_rated_iids=[
    #     16642, 5840, 16580, 13811], may_want_ingredient='tomato, chicken, celery', similar_taste_weight=0.5, unwanted_ingredients='celery, potato'))

    # result = combine_results(knn=knn, svd=svd, content_based=content_based,
    #                          similar_taste_weight=0.6, unwanted_ingredients='onion, garlic, chicken')

    # list_of_dict = []
    # for one_rep in result:
    #     dictionary = {}
    #     dictionary['name'] = one_rep[0]
    #     dictionary['inglist'] = one_rep[1]
    #     dictionary['step'] = one_rep[2]
    #     list_of_dict.append(dictionary)
    # print(len(list_of_dict))
    # print(list_of_dict)
    knn = [130403, 52424, 83971, 10291, 36581,
           43817, 43768, 43724, 43728, 43746]
    svd = [139670, 139671, 139672, 139673, 139674,
           139675, 139676, 139677, 139678, 139679]
    c = Content('../data/RAW_recipes.csv')
    content_based = c.get_recs(
        'beef, pork, egg, milk, chili, squash, bok choy, butter', N=25).id.values.tolist()

    result = combine_results(knn=knn, svd=svd, content_based=content_based,
                             similar_taste_weight=0.2, unwanted_ingredients='celery, cinnamon')
    print(len(result))
