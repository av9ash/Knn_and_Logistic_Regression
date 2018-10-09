from __future__ import division
import numpy as np


class LogisticRegressionOVA(object):
    def __init__(self, alpha=0.1, n_iter=50):
        self.alpha = alpha
        self.n_iter = n_iter
        self.weight = []

    def fit(self, data_matrix, label_matrix):
        data_matrix = np.insert(data_matrix, 0, 1, axis=1)

        for i in np.unique(label_matrix):
            label_copy = np.where(label_matrix == i, 1, 0)
            weight = np.ones(data_matrix.shape[1])
            self.grad_ascent(data_matrix, label_copy, weight, i)
        return self

    def grad_ascent(self, data_matrix, label_copy, weight, i):
        rows = data_matrix.shape[0]
        for _ in range(self.n_iter):
            output = data_matrix.dot(weight)
            errors = label_copy - self.__sigmoid__(output)
            weight += self.alpha / rows * errors.dot(data_matrix)
        self.weight.append((weight, i))

    def predict(self, data_matrix):
        return [self.__predict_one__(data) for data in np.insert(data_matrix, 0, 1, axis=1)]

    def __predict_one__(self, data):
        return max((data.dot(weight), c) for weight, c in self.weight)[1]

    def score(self, data_matrix, label):
        return sum(self.predict(data_matrix) == label) / len(label)

    def __sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))
