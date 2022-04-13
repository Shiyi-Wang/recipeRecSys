import numpy as np
import pickle

file = open("../data/SVD_algo.pkl", 'rb')
SVD_algo = pickle.load(file)

file = open("../data/recipes_names.pkl", 'rb')
rep_names = pickle.load(file)

file = open("../data/processed_data.pkl", 'rb')
rep_U = pickle.load(file)


def get_recipe_similar_score(iids, U=rep_U):
    users_to_rec = [iid for iid in range(U.shape[0]) if iid not in iids]

    user_sim_score = []

    for user in users_to_rec:
        user_sim_score.append(
            float(np.mean([np.dot(U[userid], U[user]) for userid in iids])))

    return users_to_rec, user_sim_score


def get_n_predictions(iids, algo=SVD_algo, uid=3787):

    iid_to_test = [iid for iid in range(139684) if iid not in iids]
    test_set = [[uid, iid, 4.] for iid in iid_to_test]
    predictions = algo.test(test_set)
    pred_ratings = [pred.est for pred in predictions]
    return pred_ratings


def translate_recipe_names(results, rep_names=rep_names):
    return [refactorRecipeNames(rep_names[r]) for r in results]


def refactorRecipeNames(text):
    text = text.replace(" s ", "\'s ")
    text_split = text.split(" ")
    text_split = [t.strip().capitalize() for t in text_split if t != '']
    return " ".join(text_split)


if __name__ == '__main__':
    print(get_n_predictions(iids=[16642, 5840, 16580, 13811],
                      algo=SVD_algo, uid=3787))
