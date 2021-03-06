import numpy as np
import pickle

file = open("../data/SVD_algo.pkl", 'rb')
SVD_algo = pickle.load(file)

def get_n_predictions(iids, algo=SVD_algo, n=10, uid=3787):

    iid_to_test = [iid for iid in range(139684) if iid not in iids]
    test_set = [[uid, iid, 4.] for iid in iid_to_test]
    predictions = algo.test(test_set)
    pred_ratings = [pred.est for pred in predictions]
    top_n = np.argpartition(pred_ratings, 1)[-n:]
    return top_n.tolist()

if __name__ == '__main__':
    print(get_n_predictions(iids=[16642, 5840, 16580, 13811],
                            algo=SVD_algo, uid=3787))
