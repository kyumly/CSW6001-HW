import numpy as np
import statistics
import matplotlib.pyplot as plt
np.random.seed(42)



def knn_cross_validate( x_train, y_train, num_folds: int = 5, k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]):

    # x_train과 y_train을 랜덤하게 섞기

    x_train_folds = np.array_split(x_train, num_folds, axis=0)
    y_train_folds = np.array_split(y_train, num_folds, axis=0)
    k_to_accuracies = {}

    for k in k_choices:
        accuracy = []

        for i in range(num_folds):
            validation_x = x_train_folds[i]
            validation_y = y_train_folds[i]
            train_x = x_train_folds[:i] + x_train_folds[i+1:]
            train_y = y_train_folds[:i] + y_train_folds[i+1:]

            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)
            
            knn = KnnClassifier(train_x, train_y)
            accuracy.append(knn.check_accuracy(validation_x, validation_y, k,True))

        k_to_accuracies[k] = accuracy
    return k_to_accuracies


def grid_cross_validation(k_to_accuracies, name, pca, i):
    ks, means, stds = [], [], []
    for k, accs in sorted(k_to_accuracies.items()):
        plt.scatter([k] * len(accs), accs, color='g')
        ks.append(k)
        means.append(statistics.mean(accs))
        stds.append(statistics.stdev(accs))
    plt.errorbar(ks, means, yerr=stds)
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.title(f'Cross-validation on {name} {pca}')
    plt.savefig(f"./fig/{i}.png")
    plt.show()


def confusion_matrix(actual, predicted, labels):
    num_labels = len(labels)
    matrix = np.zeros((num_labels, num_labels))
    # 실제 클래스와 예측한 클래스 간의 관계를 계산하여 Confusion matrix 작성
    for a, p in zip(actual, predicted):
        a_index = labels.index(a)
        p_index = labels.index(p)
        matrix[a_index, p_index] += 1

    return matrix


class KnnClassifier:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = list(set(y_train))

    def predict2(self, x_test, k: int = 1):
        dists = self.get_distance(x_test)
        num_train, num_test = dists.shape

        y_pred = np.zeros(num_test)

        for i in range(num_test):
            label = np.argsort(dists[:, i])
            closest_y = self.y_train[label[:k]]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred



    def get_distance(self, x_test):
        num_train = self.x_train.shape[0]
        num_test = x_test.shape[0]
        dists = np.zeros((num_train, num_test))
        for i in range(num_train):
            distint = np.sqrt(((self.x_train[i] - x_test) ** 2).sum(axis=1))
            dists[i, :] = distint

        return dists


    def predict(self, X_test, k=1):
        y_pred = np.zeros(X_test.shape[0])

        for i, x in enumerate(X_test):
            distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
            nearest_indices = distances.argsort()[:k]
            nearest_labels = self.y_train[nearest_indices]
            y_pred[i] = np.bincount(nearest_labels).argmax()

        return y_pred


    def check_accuracy( self, x_test, y_test, k: int = 1, quiet: bool = False ):
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum()

        accuracy = 100.0 * num_correct / num_samples
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
    for index, accuracy in enumerate(k_to_accuracies):
        for k, accuracies in accuracy.items():
            max_acc_for_k = max(accuracies)  # 각 키에 해당하는 값들의 리스트에서 최대값 찾기
            if max_acc_for_k > max_accuracy:
                max_accuracy = max_acc_for_k
                best_k = k
                axis = axis_choices[index]

    print("Best K:", best_k)
    print("Best axis : ", axis)
    print("Max Accuracy:", max_accuracy)
    return best_k, axis
