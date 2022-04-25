import hybrid_api
import json
from Content import Content


def write_to_json(name, data):
    file_name = '../demo_result/' + name + '.json'
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)


def easy_case():
    '''
    Easy case:
        may_want_ingredient = 'winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt'
        similar_taste_weight = 0
        unwanted_ingredients = ''

        fastest because no knn model is ran through in the backend.
        no unwanted ingredients, so no removal steps needed.
    '''
    data = hybrid_api.pass_to_models(user_id=3787, user_rated_iids=[
        16642, 5840, 16580, 13811], may_want_ingredient='winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt', similar_taste_weight=0, unwanted_ingredients='')
    write_to_json('easy_case_demo', data)


def avg_case():
    '''
    Avg case:
        may_want_ingredient = 'winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt'
        similar_taste_weight = 0.3
        unwanted_ingredients = 'cucumbers, garlic'
    '''
    data = hybrid_api.pass_to_models(user_id=3787, user_rated_iids=[
        16642, 5840, 16580, 13811], may_want_ingredient='winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt', similar_taste_weight=0.3, unwanted_ingredients='cucumbers, garlic')
    write_to_json('avg_case_demo', data)


def hard_case():
    '''
    Hard case:
        may_want_ingredient = 'winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt'
        similar_taste_weight = 0.5
        unwanted_ingredients = 'cucumbers, garlic'
        num_of_recommendation = 20
    '''
    data = hybrid_api.advanced_pass_to_models(user_id=3787, user_rated_iids=[
        16642, 5840, 16580, 13811], may_want_ingredient='winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt', similar_taste_weight=0.5, unwanted_ingredients='onion, garlic', num_of_recommendation=20)
    write_to_json('hard_case_demo', data)


if __name__ == '__main__':
    easy_case()
