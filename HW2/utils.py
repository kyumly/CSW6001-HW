import numpy as np

np.random.seed(30)

import numpy as np
def standardize(X):
    """
    
    :param X: 정규화할 대상 데이터 
    :return:  정규화된 데이터
    """
    
    # 데이터 평균 구하기
    mean = np.mean(X, axis=0)
    
    #데이터 표준편차 구하기
    std = np.std(X, axis=0)
    
    # 데이터 정규화
    X_std = (X - mean) / std
    return X_std


def get_split_data(X_data, y):
    # column_stack 데이터 합치기 (F1, F2, F3, Y)
    data = np.column_stack((X_data, y))

    # 데이터 Shuffle
    np.random.shuffle(data)

    # 최종 데이터 선정
    split_index = int(0.8 * len(data))

    # 데이터 분할
    final_x_train = data[:split_index, :-1]  
    final_y_train = data[:split_index, -1]   

    final_x_test = data[split_index:, :-1]   
    final_y_test = data[split_index:, -1]   

    return final_x_train, final_x_test, final_y_train, final_y_test