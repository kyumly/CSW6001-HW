import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from SVM import CustomSVM
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def svm_cross_validate(x_train, y_train, num_folds: int = 10, C_choices=[1, 3, 5, 8, 10, 12]):
    """
    
    :param x_train: 훈련할 데이터 
    :param y_train: 훈련하는 정답 데이터
    :param num_folds: Fold 설정 기본 : 10개
    :param C_choices: C 파라미터 설정
    :return: custom_svm, sklearn_svm 정확도 반환
    """
    
    
    # Fold 나누기
    x_train_folds = np.array_split(x_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds, axis=0)

    c_custom_to_accuracies = {}
    c_sklearn_to_accuracies = {}

    #C에 따라 결과 달라지기 때문에 최적에 하이퍼파라미터를 찾는다ㄴ.
    for C in C_choices:
        # custom_svm 직접 구현한 SVM  
        # Sklearn_svm 파이썬 라이브러리로 있는 패키지, 서로 성능을 비교하기 위해 사용 
        custom_accuracy = []
        sklearn_accuracy = []

        for i in range(num_folds):
            # K Fold 만큼 반복분 실행

            validation_x = x_train_folds[i]
            validation_y = y_train_folds[i]

            train_x = x_train_folds[:i] + x_train_folds[i + 1:]
            train_y = y_train_folds[:i] + y_train_folds[i + 1:]

            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)
            #Custom SVM 파일 실행
            svm = CustomSVM(train_x, train_y, C=C)
            #bool : Option QP 훈련과정 출력
            svm.fit(option=False)
            
            # 정확도 측정
            custom_accuracy.append(svm.score(validation_x, validation_y, True))

            
            #Sklearn SVM 실행
            svm_model = SVC(kernel='linear', C=C)
            svm_model.fit(train_x, train_y)
            y_pred = svm_model.predict(validation_x)
            result = (np.sum(y_pred == validation_y) / validation_x.shape[0]) * 100
            sklearn_accuracy.append(result)

        c_custom_to_accuracies[C] = custom_accuracy
        c_sklearn_to_accuracies[C]= sklearn_accuracy
    # 결과 2개 반환
    return c_custom_to_accuracies, c_sklearn_to_accuracies


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



def get_parameter_random_forest():
    """
    
    :return: random forest 하이퍼 파라미터 값 반환
    """
    params = {
        'n_estimators': [2,4,8],
        'max_depth': [2,4, 6, 8, 10],
        'min_samples_leaf': [2,4,8, 12, 18],
        'min_samples_split': [2, 4, 8, 16, 20]
    }
    return params


def random_forest_cross_validate(x_train, y_train, num_folds: int = 10, params: dict = None):
    """

    :param x_train: 훈련할 데이터
    :param y_train: 훈련하는 정답 데이터
    :param num_folds: Fold 설정 기본 : 10개
    :param params: random forest 하이퍼파라미터 값 
    :return:
    """

    x_train_folds = np.array_split(x_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds, axis=0)

    c_custom_to_accuracies = {}

    for i in range(num_folds):
        validation_x = x_train_folds[i]
        validation_y = y_train_folds[i]

        train_x = x_train_folds[:i] + x_train_folds[i + 1:]
        train_y = y_train_folds[:i] + y_train_folds[i + 1:]
        
        # 빈 RandomForestClassifier 인스턴스 생성
        rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        # 기존에 설정한 하이퍼 파라미터 값 + 빈 RandomForestClassifier 놓고 -> GridSearchCV 실행
        grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=3, n_jobs=-1)
    
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        
        # grid_cv 훈련
        grid_cv.fit(train_x, train_y)
        
        # 가장 좋은값 반환 + 정확도 출력
        print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
        print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
        
        # 가장 좋은 값 파라미터에 놓고 다시 인스턴스 만들기
        search_rf =  RandomForestClassifier(**grid_cv.best_params_)
        search_rf.fit(train_x, train_y)
        
        # Validation 데이터 측정하기
        y_pred = search_rf.predict(validation_x)
        result = (np.sum(y_pred == validation_y) / validation_x.shape[0]) * 100
        c_custom_to_accuracies[i] = [grid_cv.best_params_, result]

    return c_custom_to_accuracies


def grid_plot_rf(rf_dict):
    """
    c_custom_to_accuracies 값 받아서 화면에 이쁘게 출력하기

    :param rf_dict: c_custom_to_accuracies 값 
    :return: None(이미지)
    """

    indices = list(rf_dict.keys())
    accuracies = [item[1] for item in rf_dict.values()]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(indices, accuracies, color='skyblue')
    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Models')
    plt.xticks(indices)
    plt.grid(True)
    plt.show()


def find_best_model(data):
    """
    위에서 결과 받은값 제일 성능 좋은 결과 찾아서 모델 파라미터, index, 정확도 반환하기
    :param data:  c_custom_to_accuracies
    :return:
    """
    
    best_accuracy = -1
    best_model_index = None

    for index, (params, accuracy) in data.items():
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_index = index

    if best_model_index is not None:
        best_model_params = data[best_model_index][0]
        return best_model_index, best_model_params, best_accuracy
    else:
        return None
