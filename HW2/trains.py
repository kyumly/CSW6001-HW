import numpy as np

from HW2.SVM import CustomSVM
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import statistics


def svm_cross_validate(x_train, y_train, num_folds: int = 10, C_choices=[1, 3, 5, 8, 10, 12]):

    x_train_folds = np.array_split(x_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds, axis=0)

    c_custom_to_accuracies = {}
    c_sklearn_to_accuracies = {}

    for C in C_choices:
        custom_accuracy = []
        sklearn_accuracy = []

        for i in range(num_folds):
            validation_x = x_train_folds[i]
            validation_y = y_train_folds[i]
            train_x = x_train_folds[:i] + x_train_folds[i + 1:]
            train_y = y_train_folds[:i] + y_train_folds[i + 1:]
            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)
            svm = CustomSVM(train_x, train_y, C=C)
            svm.fit()
            custom_accuracy.append(svm.score(validation_x, validation_y, True))


            svm_model = SVC(kernel='linear', C=C)
            svm_model.fit(train_x, train_y)
            y_pred = svm_model.predict(validation_x)
            result = (np.sum(y_pred == validation_y) / validation_x.shape[0]) * 100
            sklearn_accuracy.append(result)

        c_custom_to_accuracies[C] = custom_accuracy
        c_sklearn_to_accuracies[C]= sklearn_accuracy
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
