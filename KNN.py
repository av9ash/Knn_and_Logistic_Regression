import numpy as np
from collections import Counter
from scipy.spatial import distance


class KNN(object):

    def __init__(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def predict(self, testing_data, k):
        dists = self.compute_distances(testing_data)

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            labels = self.training_labels[np.argsort(dists[i, :])].flatten()
            # find k nearest lables
            k_closest_y = labels[:k]
            # out of these k nearest labels which one is most common
            # break ties by selecting smaller label
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return y_pred

    def compute_distances(self, test_data):
        return distance.cdist(test_data, self.training_data, metric='euclidean')