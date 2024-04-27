import numpy
import numpy as np
import matplotlib.pyplot as plt


COLOR = ["r", "g", 'b']



def get_eigen_vectors(X):
    eigen_value, eigen_vector = numpy.linalg.eig(X)
    # eigen_value = np.diag(eigen_va/lue)
    index = np.argsort(eigen_value)[::-1]
    eigen_value = eigen_value[index]
    eigen_vector = eigen_vector[:, index]
    return eigen_value, eigen_vector

class PCA():
    """
    PCA 구현하는 클래스
    """
    def __init__(self, X):
        self.X = X
        self.N, self.F = X.shape
        self.v = None
        self.value = None
        self.explained_variance_ratio = None

        # self.explained_variance_ratio = value / value.sum()

    def fit(self, num=1):
        s = self.get_scatter_matrix()
        value, vector = get_eigen_vectors(s)

        self.value = value.real

        self.explained_variance_ratio = self.value.real / self.value.sum()
        print(vector.shape)
        v = vector[:, :num].real
        if num == 1:
           v = v.reshape(-1 ,1)
        self.v = v

    def get_mean(self, X):
        mean = X.mean(axis=0)
        return mean
    def get_scatter_matrix(self):
        """
        Scatter matrix 구현하는 함수
        """
        X: numpy.array
        mean = self.X.mean(axis=0)
        X = self.X - mean
        # numpy.cov(X, rowvar = False, bias=True)
        result = (X.T @ X) / self.N
        return result

    def reconstruct(self):
        """
        예시 :
        벡터 (2,1) @ (1,)
        reconst_test = (test_Vector.T @ valuess + 5)

        :param num:
        :return:
        """
        mean = self.get_mean(self.X)
        X = self.X - mean
        return (X @ self.v) @ self.v.T + mean

    def __call__(self):
        X = self.X - self.get_mean(self.X)
        return X @ self.v


class LDA():
    """
    LDA 구현하는 클래스
    """
    def __init__(self, data, target):
        super().__init__()
        #입력 데이터 정보 저장
        self.data = data
        self.N, self.F = data.shape
        
        # classes 대한 정보
        self.classes = np.unique(target)
        self.labels = target
        self.vector = None


    def fit(self, num=1):
        # print(self.data[self.labels == 0] - self.data[self.labels == 0].mean(axis=0))
        mean_list = [self.data[self.labels == c].mean(axis=0) for c in self.classes]
        SW, SB = self.get_scatter_matrix(mean_list)
        inverse = np.linalg.inv(SW).dot(SB)
        value, vector = get_eigen_vectors(inverse)
        vector = vector[:, :num].real
        self.vector = vector

    def get_scatter_matrix(self, mean_list):

        U = np.mean(self.data, axis=0)

        S_W = np.zeros((self.F, self.F))
        S_B = np.zeros((self.F, self.F))
        for index, value in enumerate(self.classes):
            u_i = self.data[self.labels == value]
            S_W += (u_i - mean_list[index]).T @ (u_i - mean_list[index])

            u = (mean_list[index] - U).reshape(-1,1)
            N = u_i.shape[0]
            S_B += N * (u @ u.T)
        return S_W, S_B


    def __call__(self):
        return  self.data @ self.vector

