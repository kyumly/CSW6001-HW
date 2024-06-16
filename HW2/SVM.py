import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


def get_kernel(x1, x2, kernel="linear", gamma=0.5):
    """
    기본 Linear kernel
    
    :param x1: 데이터 I 
    :param x2:  데이터 J
    :param kernel: Kernel 사용할 이름
    :param gamma: 
    :return: 
    """
    if kernel == "linear":
        # 선형 커널
        return np.dot(x1, x2)
    elif kernel == "rbf":
        # Radial Basis Fuction 커널
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    else :
        return np.dot(x1, x2)

class CustomSVM():
    """
    SVM 구현하는 클래스
    """
    def __init__(self, X, y, C = 1, kernel="linear"):
        """
        
        :param X: train 데이터 
        :param y:  label 데이터
        :param C: 파라미터
        :param kernel: 커널 이름
        """
        self.X = X                                  #X 데이터 저장
        self.N, self.M = X.shape                    #N, F 저장
        self.y = y                                  # 정답 저장
        self.C = C
        self.support_vector =None                   # 분류기에 해당하는 값 (i 번째 샘플들)
        self.w = None                               # Weight 가중치
        self.b = None                               # Bias

        self.K = np.zeros((self.N, self.N))
        # 커널 구현하는 함수 
        for i in range(self.N):
            for j in range(self.N):
                self.K[i, j] = get_kernel(X[i], X[j])


    def get_constaint(self, tpye="non-linear"):
        """
        제약조건 설정하는 함수

        :param tpye: linear or  non-linear(non-separating)
        :return: 제약조건들
        """
        n = self.N

        A = self.y.reshape(1, -1)
        b = np.zeros(1)


        if tpye == "non-linear":
            # diag 행렬 만들어서 0 보다 큰 조건, 그리고 C 보다 작은 조건 만들기
            G = np.concatenate([np.diag(np.ones(n) * -1), np.diag(np.ones(n))], axis=0)
            
            #H 값은 절반은 0보다 큰값 하나는 C 보다 작은값
            h = np.concatenate([np.zeros(n), np.ones(n) * self.C], axis=0)
        else:
            
            # diag 행렬 만들어서 0이상인 조건 만들기
            G = np.diag(np.ones(n) * -1)
            # 무조건 0보다 큰값 조건
            h = (np.zeros(n))


        return matrix(G), matrix(h), matrix(A, tc='d'), matrix(b)

    def fit(self, option=True, type="non-linear", ):
        n = self.N
        diag_Y = np.diag(self.y.reshape(-1))
        # H 행렬 만들기
        H = diag_Y @ self.K @ diag_Y
        
        
        P =  matrix(H)
        q = matrix(np.ones(n) * -1)
        G, h, A, b = self.get_constaint(type)

        solvers.options['show_progress'] = option
        sol = solvers.qp(P, q, G, h, A, b)

        # QP 함수의 결과를 받음
        alpha = sol['x']
        alpha = np.array(alpha)

        if type == "non-linear":
            print("non-linear 실행")
            # non-separation 조건에서는 C보다는 작고 0보다는 커야하는 조건 달기
            zero_mask = ((alpha > 1e-4) & (alpha < self.C - 1e-4)).flatten()
        else:
            print("linear 실행")
            # 무조건 0보다는 커야한다.
            zero_mask = (alpha > 1e-4).flatten()

        self.support_vector =self.X[zero_mask]

        # 알파는 0 < 알파 < C 사이에 값을 구해야하기 때문에  mask 통해 특정 값만 추출

        new_X, new_Y, new_alpha = self.X[zero_mask], self.y[zero_mask], alpha[zero_mask]
        # W 값 계하기 
        w = (new_alpha * new_Y.reshape(-1, 1) * new_X).sum(axis=0)
        
        # B 값 내적으로 가능하면, 평균값 구하기
        b = new_Y - np.dot(new_X, w)
        b= b.mean()

        self.b = b
        self.w = w

        return 
    
    def decision_boundary(self, xx, f_x, f_y, b):
        """
        hyperplane 할때 사용하는 함수
        """
        w = self.w
        return (-w[f_x] * xx - b) / w[f_y]
    
    def predict(self, X):
        """
        클래스 예측할때 사용하는 함수
        """
        return np.sign(np.dot(X, self.w) + self.b)
    
    def score(self, X, y, verbose=False):
        """
        클래스 예측하고 정답을 얼마나 맞췄는지 알려주는 함수
        """
        N = X.shape[0]
        result = (np.sum(self.predict(X) == y) / N) * 100
        if verbose:
            print(f"정확도 : {result}%")

        return result
    def plot_svm(self, X, y, dims =[[0, 1]]):
        """
        2차원 feature 가지고 SVM 대한 hyperplane 그려주는 함수
        :param X: 데이터 
        :param y: 정답
        :param dims: 차원
        :return: 
        """
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
            plt.title(f"dimension : {f_x}, {f_y}")
            
        plt.show()