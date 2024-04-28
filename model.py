import numpy as np
import statistics
import matplotlib.pyplot as plt
np.random.seed(42)



def knn_cross_validate( x_train, y_train, num_folds: int = 5, k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]):
    """
    :param x_train: 훈련 데이터 
    :param y_train: 정답 데이터
    :param num_folds:  몇개의 Fold 설정
    :param k_choices: KNN 파라미터 설정
    :return: 
    """

    # Numfolds 갯수 만큼 Train, y_train 데이터 나누기
    x_train_folds = np.array_split(x_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds, axis=0)

    k_to_accuracies = {}
    
    # KNN 파라미터 설정해 갯수만큼 FOR문 돌림
    for k in k_choices:
        accuracy = []
        # num_fold만큼 반본문 돌림
        for i in range(num_folds):
            validation_x = x_train_folds[i]
            validation_y = y_train_folds[i]
            train_x = x_train_folds[:i] + x_train_folds[i+1:]
            train_y = y_train_folds[:i] + y_train_folds[i+1:]

            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)
            
            #데이터 결과 accuracy 저장
            knn = KnnClassifier(train_x, train_y)
            accuracy.append(knn.check_accuracy(validation_x, validation_y, k,True))

        # dict 형태로 KNN 결과와 10개의 Accuracy 저장
        k_to_accuracies[k] = accuracy
    return k_to_accuracies


def grid_cross_validation(k_to_accuracies, name, axis, i):
    """
    cross_validation 결과값 구려주는 함수
    :param k_to_accuracies:  정확도
    :param name: PCA or LDA
    :param axis: axis 축
    :param i: save number
    :return:
    """
    #knn 파라미터 전체 , 평균값 분산값 저장
    ks, means, stds = [], [], []

    for k, accs in sorted(k_to_accuracies.items()):
        
        plt.scatter([k] * len(accs), accs, color='g') # accs 해당하는 값 찍기 (세로로 찍기)
        ks.append(k)                                  # K 개 적립
        means.append(statistics.mean(accs))           # 평균 저장
        stds.append(statistics.stdev(accs))           # 분산 저장
    plt.errorbar(ks, means, yerr=stds)                #Error 설정
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.title(f'Cross-validation on {name} {axis}')
    plt.savefig(f"./fig/{i}.png")
    plt.show()


def confusion_matrix(actual, predicted, labels):
    """
    :param actual: 실제값
    :param predicted:  예측값
    :param labels: 데이터 labels
    :return: confusion matrix
    """
    num_labels = len(labels)                            #데이터 라벨 저장
    matrix = np.zeros((num_labels, num_labels))         #데이터 라벨의 N by N 행렬 생성

    for a, p in zip(actual, predicted):                 #실제 값과, 예측값 계산하여 측정
        a_index = labels.index(a)
        p_index = labels.index(p)
        matrix[a_index, p_index] += 1

    return matrix


class KnnClassifier:
    def __init__(self, x_train, y_train):
        """
        
        :param x_train: train data 수집 
        :param y_train:  train label 데이터 수집
        """
        self.x_train = x_train
        self.y_train = y_train
        self.classes = list(set(y_train)) # Classes list 저장(중복 제거 데이터)


    def predict(self, X_test, k=1):
        """
        
        :param X_test:
        :param k: K 파라미터 설정
        :return: 
        """

        y_pred = np.zeros(X_test.shape[0])
        # For문 돌려면서, Train, test 데이터에 대한 L2 거리 공식 비교
        for i, x in enumerate(X_test):
            distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
            # 큰 작은 순서대로 정렬
            nearest_indices = distances.argsort()[:k]
            
            # 데이터 labels 실시
            nearest_labels = self.y_train[nearest_indices]
            
            #K안에 있는 카운트 갯수 세기
            y_pred[i] = np.bincount(nearest_labels).argmax()

        return y_pred


    def check_accuracy( self, x_test, y_test, k: int = 1, quiet: bool = False ):
        """
        :param x_test: 입력에 사용할 데이터 
        :param y_test: 입력에 사용된 정답 데이터
        :param k:  KNN 파리미터 값
        :param quiet: print문 실행
        :return: 
        """
        y_test_pred = self.predict(x_test, k=k)         # 예측 값 반환
        num_samples = x_test.shape[0]               
        num_correct = (y_test == y_test_pred).sum()     # 실제 값과 비교하여 맞는 수 반환하기


        accuracy = 100.0 * num_correct / num_samples     #% 결과값 설정
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy


def knn_get_best_k(k_to_accuracies, axis_choices):
    max_accuracy = 0
    axis = 0
    best_k= 0
    for index, accuracies in enumerate(k_to_accuracies):
        for k, accuracy in accuracies.items():
            max_acc_for_k = max(accuracy)           # 각 키에 해당하는 값들의 리스트에서 최대값 찾기
            if max_acc_for_k > max_accuracy:        # 최대 값이 비교하는 알고리즘
                max_accuracy = max_acc_for_k        # 최대값 설정
                best_k = k                          # K값 설정
                axis = axis_choices[index]          # axis 축 설정

    print("Best K:", best_k)                        #가장 큰 값 K
    print("Best axis : ", axis)                     #가장 큰 값 axis
    print("Max Accuracy:", max_accuracy)            #가장 정확도
    return best_k, axis
