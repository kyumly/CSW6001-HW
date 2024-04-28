import numpy
import numpy as np
import matplotlib.pyplot as plt




def  get_angle_in_degrees(vector1, vector2):
    """
    :param vector1: 비교할 벡터 1
    :param vector2:  비교할 벡터 2
    :return: 벡터1 벡터2에 대한 각도 반화
    """

    cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))  # cos similarity 실행하여, 내적 결과 파악
    truncated_arr = np.round(np.trunc(cos_sim * 100) / 100, 1)                        # 절삭 이후, 소수점 2번째자리에서 반올림

    angle_in_radians = np.arccos(truncated_arr)                                               #cos 각도 계산
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def get_eigen_vectors(X):
    """
    :param X: X 행렬
    :return: Eigen_value, Eigen_Vector
    """
    eigen_value, eigen_vector = numpy.linalg.eig(X)             #X행렬에 대한 EVD 실행
    # eigen_value = np.diag(eigen_va/lue)
    index = np.argsort(eigen_value)[::-1]                       #eigen_value에 대한 index 정렬 실행
    eigen_value = eigen_value[index]                            #정렬한 index 내림차순으로 한 뒤 value 가져임
    eigen_vector = eigen_vector[:, index]                       #vecotr index 따라 큰 순서대로 정렬
    return eigen_value, eigen_vector


def grid_3d(test_reuslt, y_test, name, label_encode_dict):

    # 라벨별로 데이터를 모으기
    unique_labels = np.unique(y_test)
    label_data = {label: [] for label in unique_labels}
    for i, label in enumerate(y_test):
        label_data[label].append(test_reuslt[i])

    # 3D plot 그리기
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
        """

        :param X: Train data 수집
        """
        self.X = X                                  #X 데이터 저장
        self.N, self.F = X.shape                    #N, F 저장
        self.vector = None                          #Vector &vector value
        self.value = None
        self.explained_variance_ratio = None        #데이터 분산 비율 저장

        # self.explained_variance_ratio = value / value.sum()

    def fit(self, num=1):
        """
        :param num: 축소할 vector 사이즈 지정 EX) 1개, 2개, 3개...
        :return: None 내부적으로 vector 저장
        """
        s = self.get_scatter_matrix()                               #scatter 행렬 계산
        value, vector = get_eigen_vectors(s)                        #S가지고 Eigenvalue, vector 계산
        
        self.value = value.real                                     #eigen value 값 저장

        self.explained_variance_ratio = self.value.real / self.value.sum()      #eigen value 전체 분산 저장
        v = vector[:, :num].real                                                #Vector 값은 num 만큼 저장하고 복소수 말고 real 값만 저장
        if num == 1:                                                            #num == 1 일때 값이 (1, -1) 저장 벡터화 시켜주기
           v = v.reshape(-1 ,1)
        self.vector = v                                                         # 벡터 저장

    def reconstruct(self, X = None):
        """
        reconstruction 실행 함수
        :param num: X 재구성할 데이터 넣기 
        :return:
        """
        # X 데이터가 있으면 Test데이터 없으면 Train 데이터 실행
        # 똑같이 차원 축소 후 평균값 더해 차원 맞추기
        if X is not None:
            mean = self.get_mean(X)
            X = X - mean
            return (X @ self.vector) @ self.vector.T + mean

        mean = self.get_mean(self.X)
        X = self.X - mean
        return (X @ self.vector) @ self.vector.T + mean # 다시 self.vector.T 내적해야 원래 데이터 차원이 나옴 (평균값 다시 더 해주기)

    def get_mean(self, X):
        """
        평균 반환하는 함수
        :param X: 데이터
        :return: 평균값 반환
        """
        mean = X.mean(axis=0) # 데이터 평균 구해서 반환
        return mean

    def get_scatter_matrix(self):
        """
        Scatter matrix 구현하는 함수
        """
        X: numpy.array
        mean = self.X.mean(axis=0)                      # 전체 Feature 대한 평균 구하기
        X = self.X - mean                               # 평균 빼기
        # numpy.cov(X, rowvar = False, bias=True)
        result = (X.T @ X) / self.N                     # Scatter 행렬 계산
        return result                                   # 행렬 반환

    def __call__(self, X =None):
        """
        차원 축소할때 사용하는 함수
        :param X: Test 데이터는 테스트 데이터 사용 용도
        :return: 차원 축소된 결과값
        """
        if X is not None:
            X = X - self.get_mean(X)            #평균값 구하기 뺴기
            return X @ self.vector              # 기존에 구한 Vector와 X 내적하기
        X = self.X - self.get_mean(self.X)
        return X @ self.vector


class LDA():
    """
    LDA 구현하는 클래스
    """
    def __init__(self, data, target):
        """
        :param data:   학습 데이터
        :param target: 학습 데이터 정답
        """
        super().__init__()
        self.data = data                    #데이터 저장
        self.N, self.F = data.shape         #데이터 갯수 and feature size 저장
        
        # classes 대한 정보
        self.classes = np.unique(target)    #저장 시킬 Classes 정보 저장
        self.labels = target                #데이터 정답 저장
        self.vector = None                  # LDA 할때 축소할 Vector 선언
    
    def fit_svd(self, num=1):
        """
        :param num: 축소할 vector 사이즈 지정 EX) 1개, 2개, 3개...
        :return: None 내부적으로 vector 저장
        """
        _, y_t = np.unique(self.labels, return_inverse=True)  #고유 Y값 추출
        
        priors_ = np.bincount(y_t) / float(len(self.labels)) # priors_ 설정하여, 표본 비율을 유지하도록 한다.


        XG = []

        n_classes = len(list(self.classes))

        mean_list = [self.data[self.labels == c].mean(axis=0) for c in self.classes]        # Classes 각 평균에 대한 값을 저장한다.
        mean_stack = np.stack(mean_list)                                                    # ex 37, 4096

        for idx, g in enumerate(self.classes):
            xg = self.data[self.labels == g]
            XG.append(xg - mean_list[idx])
        
        xbar_ = np.dot(priors_, mean_stack)    #xbar 표본 평균 값을 구한다.
        mean_data = np.concatenate(XG, axis=0)  # 평균값을 계산

        std = mean_data.std(axis=0)             # 분산 계산
        std[std == 0] = 1.0
        fac = 1.0 / (self.N - n_classes)

        #svd low rank approximation   사용한다.
        X = np.sqrt(fac) * (mean_data / std) # 값을 표준편차로 정규화
        U, S, Vt = np.linalg.svd(X, full_matrices=False) # 그 후 값을 EVD 아닌, SVD 설정해 값을 분해한다.
        rank = np.sum(S > 0.0001)

        scalings = (Vt[:rank] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)
        
        #X에 대한 값을 스케일링 실행
        X = np.dot(
            (
                (np.sqrt((self.N *  priors_) * fac))
                * (mean_stack -  xbar_).T
            ).T,
            scalings,
        )
        # 다시 X에 대한 값을 분해한다.
        _, S, Vt = np.linalg.svd(X, full_matrices=0)
        rank = np.sum(S > 0.0001 * S[0])

        # eigen vector 값 설정
        self.vector = np.dot(scalings, Vt.T[:, :num])


    def fit_eigen(self, num=1):
        """
        
        :param num: 축소할 vector 사이즈 지정 EX) 1개, 2개, 3개...
        :return: None 내부적으로 vectro 저장
        """
        
        mean_list = [self.data[self.labels == c].mean(axis=0) for c in self.classes]    # classes 별 데이터 평균 계산
        SW, SB = self.get_scatter_matrix(mean_list)                                     # Within, between scatter 구하는 메소드 호출

        inverse = np.linalg.pinv(SW).dot(SB)                                            #구한 SW inverse 구하고, SB 내적을 실행.
        value, vector = get_eigen_vectors(inverse)                                      #inverse 값을 eigen vector를 구함
        vector = vector[:, :num].real                                                   #vecotr 사이즈 만큼 벡터를 가져와 self.vector안에 저장
        self.vector = vector

    def get_scatter_matrix(self, mean_list):
        """
        :param mean_list: 클래스당 평균 list
        :return: SW, SB
        """

        U = np.mean(self.data, axis=0)                                          #값 전체 평균

        S_W = np.zeros((self.F, self.F))                                        # SW = F by F 행렬 만들기
        S_B = np.zeros((self.F, self.F))                                        # SB = F by F 행렬 만들기
        
        for index, value in enumerate(self.classes):
            u_i = self.data[self.labels == value]                               #클래스당 평균값 추출
            S_W += (u_i - mean_list[index]).T @ (u_i - mean_list[index])        #SW 행렬 계산하기 (전체 Class 분산을 더한다)

            u = (mean_list[index] - U).reshape(-1,1)                            #SB는 클래스당 평균값과 전체 클래스 평균값의 거리를 계산.
            N = u_i.shape[0]
            S_B += N * (u @ u.T)                                                #SB 행렬 다 더하기
        return S_W, S_B


    def __call__(self, X = None):
        """
        :param X: Test data에 대한 차원축소 
        :return: 차원 축소 결과
        """
        if X is not None:                                                   #Test 데이터가 있으면 if문 통과 후 Test데이터 차원 축소 실행
            return X @ self.vector
        return  self.data @ self.vector                                     #train 데이터 차원 축소 실행