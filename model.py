import numpy as np

class KnnClassifier:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k: int = 1):

        N = self.x_train.shape[0]
        N_t = x_test.shape[0]

        train = self.x_train.reshape(N, -1)

        test = x_test.reshape(N_t, -1)

        train_sum = np.sum((train ** 2), dim=1).reshape(-1, 1)
        test_sum = np.sum((test ** 2), dim=1)

        train_test_mm = -2 * np.mm(train, test.T)

        dists = (train_sum + test_sum + train_test_mm) ** (1 / 2)

        num_train, num_test = dists.shape
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            label = np.argsort(dists[:, i])
            closest_y = np[label[:k]]
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred