import numpy
import numpy as np
import matplotlib.pyplot as plt


COLOR = ["r", "g", 'b']

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

        self.scatter_matrix = None


    def get_mean(self, X):
        mean = X.mean(axis=0)
        return mean

    def get_scatter_matrix(self):
        N = self.X.shape[0]
        mean = self.X.mean(axis=0)
        X = self.X - mean
        # numpy.cov(X, rowvar = False, bias=True)
        result = (X.T @ X) / N
        return result


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
        super().__init__()
        self.classes = data.shape[0]
        self.N, self.F = data.shape[1:]

        self.data : np.ndarray= data
        self.mean : np.ndarray = self.data.mean(axis=1).reshape(self.data.shape[0], 1, -1)

        self.U = (self.mean * 5).sum(axis=0) / (self.N * self.classes)

        self.s_w : np.ndarray = None
        self.s_b : np.ndarray = np.zeros((self.F, self.F), dtype=np.float32)


    def grad_show(self):
        if self.F >= 3:
            raise "출력 실패"
        min_x = self.data.min() - 1
        max_x = self.data.max() + 1

        # 축 범위 설정
        plt.xlim(min_x, max_x)
        plt.ylim(min_x, max_x)

        for index, value in enumerate(self.data):
            x = self.data[index, :, 0]
            y= self.data[index, :, 1]
            mean = self.mean[index, :].reshape(-1)

            plt.scatter(x, y, color=COLOR[index])
            plt.scatter(mean[0], mean[1], color="k")

        plt.grid(True, linestyle='--', linewidth=1)
        plt.show()

    def set_scatter_w_matrix(self, dim=0):
        if None:
            raise "이미 값이 존재 합니다"
        N = self.data.shape[1]
        X = np.subtract(self.data, self.mean)
        # numpy.cov(X, rowvar = False, bias=True)
        result = (X.transpose(0, 2, 1) @ X) / N
        self.s_w = result.sum(axis=0)
        return self.s_w

    def set_scatter_b_matrix(self):
        if self.classes > 2:
            for index, vector in enumerate(self.mean):
                u_i = (vector - self.U).reshape(-1 ,1)
                self.s_b += self.N * (u_i @ u_i.T)
        else:
            u12 = (self.mean[0] - self.mean[1]).reshape(-1, 1)
            self.s_b = u12 @ u12.T

        return self.s_b

    def get_lda(self, num=1):
        inverse = np.linalg.inv(self.s_w) @ self.s_b
        value, vector = get_eigen_vectors(inverse)
        vector = vector[:, :num]

        return self.data @ vector

