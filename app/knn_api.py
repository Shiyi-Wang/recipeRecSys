from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from pprint import pprint

data = pd.read_pickle('../data/processed_data.pkl')
data = data.drop(data.index[150000:])
tmat = data.pivot_table(index='user_id', columns='recipe_id',
                        values='rating').fillna(0)

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tmat.values)
user_neigh_dist, user_neigh_ind = knn.kneighbors(tmat.values, n_neighbors=6)


def findSimilarUsers(user, n=5):
    neigh_users_dist, neigh_users_ind = knn.kneighbors(
        np.asarray([tmat.values[user - 1]]), n_neighbors=n + 1)
    print('The top ' + str(n) + ' most similar users of user ' + str(user) + ' are:')
    for i in range(1, len(neigh_users_dist[0])):
        print('No.' + str(i) + ": User ID: " +
              str(neigh_users_ind[0][i]+1) + ", with distance " + str(neigh_users_dist[0][i]))

    print("\n")

    return neigh_users_dist.flatten()[1:], neigh_users_ind.flatten()[1:] + 1


def getRecommendations(num_recipes_recommended, avg_rating, userId):
    zero_rating = np.where(avg_rating == 0)[0][-1]
    ranked_ind = np.argsort(avg_rating)[::-1]
    ranked_ind = ranked_ind[:list(ranked_ind).index(zero_rating)]
    num_recipes_recommended = min(len(ranked_ind), num_recipes_recommended)
    seen = list(data[data['user_id'] == userId]['name'])
    recipes = list(tmat.columns[ranked_ind])
    count = 0
    recommended_recipes = []
    for recipe in recipes:
        if recipe not in seen:
            recommended_recipes.append(recipe)
            count += 1
        if count == num_recipes_recommended:
            break

    pprint(recommended_recipes)


def recommend(userId, num_similar_users, num_recipes_recommended):

    print("User " + str(userId) + " has rated the following recipes: ")
    pprint(list(data[data['user_id'] == userId]['name']))
    print("\n")

    neigh_users_dist, neigh_users_ind = findSimilarUsers(
        userId, num_similar_users)
    weighted_user_neigh_dist = neigh_users_dist / np.sum(neigh_users_dist)
    weighted_user_neigh_dist = weighted_user_neigh_dist[:,
                                                        np.newaxis] + np.zeros(len(tmat.columns))
    avg_rating = (weighted_user_neigh_dist *
                  tmat.values[neigh_users_ind]).sum(axis=0)
    print("Based on other users rating, we recommend:")

    getRecommendations(num_recipes_recommended, avg_rating, userId)


if __name__ == '__main__':
    recommend(userId = 3787, num_similar_users = 5, num_recipes_recommended = 10)