import numpy
import numpy as np
import matplotlib.pyplot as plt


def get_scatter_matrix(X):
    """
    Scatter matrix 구현하는 함수
    """
    X : numpy.array
    N = X.shape[0]
    mean = X.mean(axis = 0)
    X = X - mean
    # numpy.cov(X, rowvar = False, bias=True)
    result = (X.T @ X)
    return result


def get_eigen_vectors(X):
    eigen_value, eigen_vector = numpy.linalg.eig(X)
    # eigen_value = np.diag(eigen_va/lue)
    index = np.argsort(eigen_value)[::-1]
    eigen_value = eigen_value[index]
    eigen_vector = eigen_vector[:, index]
    return eigen_value, eigen_vector

class Dataset():
    def __init__(self, X=None, vector=None, value=None):
        self.X = X
        self.vector = vector
        self.value = value
        self.N, self.D =X.shape

    def get_mean(self, X):
        mean = X.mean(axis=0)
        return mean


class Linear(Dataset):
    """
    projection 투영하는 클래스
    """
    def __init__(self, X=None, vector=None, value=None):
        super().__init__(X, vector, value)

    def forward(self, num=1):
        X = self.X - self.get_mean(self.X)
        return X @ self.vector[:, range(num)]

    def __call__(self, num):
        return self.forward(num)


class PCA(Linear):
    """
    PCA 구현하는 클래스
    """
    def __init__(self, X, vector, value):
        super(PCA, self).__init__(X, vector, value)
        self.explained_variance_ratio = value / value.sum()


    def reconstruct(self, num=1):
        """
        예시 :
        벡터 (2,1) @ (1,)
        reconst_test = (test_Vector.T @ valuess + 5)

        :param num:
        :return:
        """
        X = self.X - self.get_mean(self.X)
        return (X @ self.vector[:, range(num)]) @ self.vector[:, range(num)].T + self.get_mean(self.X)


class LDA():
    """
    LDA 구현하는 클래스
    """
    def __init__(self, data):
        self.data = data
        self.mean = None



    def grad_show(self):
        min_x = min(0, 11)
        max_x = max(0, 11)

        # 축 범위 설정
        plt.xlim(min_x, max_x)
        plt.ylim(min_x, max_x)
        plt.scatter(x1[:, 0], x1[:, 1], color="r")
        plt.scatter(x2[:, 0], x2[:, 1], color="b")
        plt.scatter(u1[0], u1[1], color="g")
        plt.scatter(u2[0], u2[1], color="g")

        plt.grid(True, linestyle='--', linewidth=1)
        plt.show()