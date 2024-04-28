import numpy
import numpy as np
import matplotlib.pyplot as plt




def  get_angle_in_degrees(vector1, vector2):
    """
    서로의 각도 구하는 식
    """
    cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)) 
    truncated_arr = np.round(np.trunc(cos_sim * 100) / 100, 1)
    # 코사인 유사도를 각도로 변환
    angle_in_radians = np.arccos(truncated_arr)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def get_eigen_vectors(X):
    eigen_value, eigen_vector = numpy.linalg.eig(X)
    # eigen_value = np.diag(eigen_va/lue)
    index = np.argsort(eigen_value)[::-1]
    eigen_value = eigen_value[index]
    eigen_vector = eigen_vector[:, index]
    return eigen_value, eigen_vector


def grid_3d(test_reuslt, y_test, name, label_encode_dict):
    # 라벨별로 데이터를 모으기
    unique_labels = np.unique(y_test)
    label_data = {label: [] for label in unique_labels}
    for i, label in enumerate(y_test):
        label_data[label].append(test_reuslt[i])

    # 3D 플롯 그리기
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 각 라벨별로 데이터를 플롯에 추가
    for label, data_points in label_data.items():
        data_points = np.array(data_points)
        ax.scatter(data_points[:,0], data_points[:,1], data_points[:,2], label=f'{label_encode_dict[label]}')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.savefig(f"./fig/{name} grid_3d.png")
    plt.show()


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

    def reconstruct(self, X = None):
        """
        예시 :
        벡터 (2,1) @ (1,)
        reconst_test = (test_Vector.T @ valuess + 5)

        :param num:
        :return:
        """
        if X is not None:
            mean = self.get_mean(X)
            X = X - mean
            return (X @ self.v) @ self.v.T + mean

        mean = self.get_mean(self.X)
        X = self.X - mean
        return (X @ self.v) @ self.v.T + mean

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

    def __call__(self, X =None):
        if X is not None:
            X = X - self.get_mean(X)
            return X @ self.v
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
    
    def fit_svd(self, num=1):
        _, y_t = np.unique(self.labels, return_inverse=True)  #
        
        priors_ = np.bincount(y_t) / float(len(self.labels))


        XG = []

        n_classes = len(list(self.classes))

        mean_list = [self.data[self.labels == c].mean(axis=0) for c in self.classes]
        mean_stack = np.stack(mean_list)

        for idx, g in enumerate(self.classes):
            xg = self.data[self.labels == g]
            XG.append(xg - mean_list[idx])
        
        xbar_ = np.dot(priors_, mean_stack)    
        mean_data = np.concatenate(XG, axis=0)

        std = mean_data.std(axis=0)
        std[std == 0] = 1.0
        fac = 1.0 / (self.N - n_classes)
        
        X = np.sqrt(fac) * (mean_data / std)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        rank = np.sum(S > 0.0001)

        scalings = (Vt[:rank] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        X = np.dot(
            (
                (np.sqrt((self.N *  priors_) * fac))
                * (mean_stack -  xbar_).T
            ).T,
            scalings,
        )

        _, S, Vt = np.linalg.svd(X, full_matrices=0)
        rank = np.sum(S > 0.0001 * S[0])
        self.vector = np.dot(scalings, Vt.T[:, :num])


    def fit_eigen(self, num=1):
        mean_list = [self.data[self.labels == c].mean(axis=0) for c in self.classes]
        SW, SB = self.get_scatter_matrix(mean_list)

        inverse = np.linalg.pinv(SW).dot(SB)
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


    def __call__(self, X = None):
        if X is not None:
            return X @ self.vector
        return  self.data @ self.vector