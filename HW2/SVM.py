import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


def get_kernel(x1, x2, kernel="linear", gamma=0.5):
    if kernel == "linear":
        return np.dot(x1, x2)
    elif kernel == "rbf":
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    else :
        return np.dot(x1, x2)

class CustomSVM():
    """
    SVM 구현하는 클래스
    """
    def __init__(self, X, y, C = 1, kernel="linear"):
        self.X = X                                  #X 데이터 저장
        self.N, self.M = X.shape                    #N, F 저장
        self.y = y
        self.C = C
        self.support_vector =None
        self.w = None
        self.b = None

        self.K = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.K[i, j] = get_kernel(X[i], X[j])


    def get_constaint(self):
        n = self.N
        G = np.concatenate([np.diag(np.ones(n) * -1), np.diag(np.ones(n))], axis=0)
        h = np.concatenate([np.zeros(n), np.ones(n) * self.C], axis=0)
        A = self.y.reshape(1, -1)
        b = np.zeros(1)
        return matrix(G), matrix(h), matrix(A, tc='d'), matrix(b)

    def fit(self):
        n = self.N
        diag_Y = np.diag(self.y.reshape(-1))
        H = diag_Y @ self.K @ diag_Y
        P =  matrix(H)
        q = matrix(np.ones(n) * -1)
        G, h, A, b = self.get_constaint()

        sol = solvers.qp(P, q, G, h, A, b)

        alpha = sol['x']
        alpha = np.array(alpha)

        zero_mask = ((alpha > 1e-4) & (alpha < self.C- 1e-4)).flatten()
        self.support_vector =self.X[zero_mask]

        # 알파는 0 < 알파 < C 사이에 값을 구해야하기 때문에  mask 통해 특정 값만 추출

        new_X, new_Y, new_alpha = self.X[zero_mask], self.y[zero_mask], alpha[zero_mask]
        w = (new_alpha * new_Y.reshape(-1, 1) * new_X).sum(axis=0)

        b = new_Y - np.dot(new_X, w)
        b= b.mean()

        self.b = b
        self.w = w

        return 
    
    def decision_boundary(self, xx, f_x, f_y, b):
        w = self.w
        return (-w[f_x] * xx - b) / w[f_y]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def score(self, X, y, verbose=False):
        N = X.shape[0]
        result = (np.sum(self.predict(X) == y) / N) * 100
        if verbose:
            print(f"정확도 : {result}%")

        return result
    def plot_svm(self, X, y, dims =[[0, 1]]):
        plt.figure(figsize=(20, 15))

        assert len(dims[0]) == 2 , "2개의 차원만 선택해 주세요"

        for i in range(len(dims)):
            plt.subplot(3, 3, i + 1)

            dim = dims[i]
            f_x, f_y = dim

            # Plot the data points
            plt.scatter(X[:, f_x], X[:, f_y], c=y, s=50, cmap=plt.cm.Paired)

            # Plot the decision boundary for cvxopt
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)

            yy_cv = self.decision_boundary(xx, f_x, f_y, self.b)
            plt.plot(xx, yy_cv, 'k-', label='decision boundary',)

            margin = self.C * 0.1
            yy_margin1 = self.decision_boundary(xx, f_x, f_y, self.b - margin)
            yy_margin2 = self.decision_boundary(xx, f_x, f_y, self.b + margin)

            # Plot the margins
            plt.plot(xx, yy_margin1, 'k--' )
            plt.plot(xx, yy_margin2, 'k--',)
            plt.legend()
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.show()