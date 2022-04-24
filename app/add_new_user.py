import pickle
import pandas as pd
import numpy as np

'''
Description:
    Add new user information to dataframe
Inputs:
    recipe_id: all rated recipe IDs
    ratings: corresponding recipe ratings
    recipe_names: corresponding recipe names
Output:
    Auto generated new user ID
'''
def add_new_user(data, recipe_id, ratings, recipe_names):
    
    new_user_id = data["user_id"].max() + 1
    num_rated = recipe_id.len()
    
    for i in range(num_rated):
        new_row = {'user_id':new_user_id, 'recipe_id':recipe_id[i], 'rating':ratings[i], 'name': recipe_names[i]}
        data = data.append(new_row, ignore_index=True)
    
    data.to_pickle("../data/processed_data.pkl")

    return new_user_id

if __name__ == '__main__':
    add_new_user(recipe_id = [], ratings = [], recipe_names = [])
