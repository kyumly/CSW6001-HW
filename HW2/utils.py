import numpy as np

np.random.seed(30)

import numpy as np
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    return X_std


def get_split_data(X_data, y):
    data = np.column_stack((X_data, y))

    np.random.shuffle(data)

    # 최종 데이터 선정
    split_index = int(0.8 * len(data))

    # 데이터 분할
    final_x_train = data[:split_index, :-1]  
    final_y_train = data[:split_index, -1]   

    final_x_test = data[split_index:, :-1]   
    final_y_test = data[split_index:, -1]   

    return final_x_train, final_x_test, final_y_train, final_y_test