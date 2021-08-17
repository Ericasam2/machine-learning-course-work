# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:59:08 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_all_dict(data):
    '''
    # data : DataFrame
    -----------------------
    #产生所有列名的一个字典
    #列名->该列中所有的数据
    '''
    status_dict = {}
    for i in list(data):
        status_dict[i] = data[i].unique().tolist()
    return status_dict


def transform_to_int(data):
    '''
    # data : DataFrame
    ----------------------------------
    #将数据集中的字符型数据转化为数据型
    '''
    status_dict = get_all_dict(data)
    for i in status_dict:
        if (type(status_dict[i][0]) == int or
            type(status_dict[i][0]) == float):
            continue
        data[i] = data[i].apply(lambda x : status_dict[i].index(x))  
    return data


def load_split_dataset(dataset_path, features):
    df = pd.read_csv(dataset_path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = transform_to_int(data)
    f_data = data[features]
    t_data = pd.DataFrame(data["target"])
    X_train, X_test, y_train, y_test = train_test_split(f_data, t_data, test_size=0.2, random_state=4)
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(data):
    '''
    # data : DataFrame, 测试集+预测结果
    ---------------------
    # result : None
    function : 绘制预测结果的混淆矩阵
    '''
    TP = 0
    TN = 0
    P = data[data["predict"] == 0]
    len_P = len(P)
    for i in range(len_P):
        if (data["predict"].iloc[i] == data["target"].iloc[i]):
            TP = TP + 1
    FP = len_P - TP
    N = data[data["predict"] == 1]
    len_N = len(N)
    for i in range(len_N):
        if (data["predict"].iloc[i] == data["target"].iloc[i]):
            TN = TN + 1
    FN = len_N - TN
    plt.figure()
    x_axis = [x for x in range(16)]
    y_axis = [y for y in range(16)]
    matrix = np.ones([16, 16])
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if (x < matrix.shape[0]/2 and y < matrix.shape[1]/2):
                matrix[x][y] = matrix[x][y] * 255
            if (x >= matrix.shape[0]/2 and y >= matrix.shape[1]/2):
                matrix[x][y] = matrix[x][y] * 205
            if (x < matrix.shape[0]/2 and y >= matrix.shape[1]/2):
                matrix[x][y] = matrix[x][y] * 155
            if (x >= matrix.shape[0]/2 and y < matrix.shape[1]/2):
                matrix[x][y] = matrix[x][y] * 105
    # plt.plot(matrix)
    plt.text(1,4, "TP:"+str(TP), fontsize=20, color="red")  # 红色，图片左上角
    plt.text(1,12, "FP:"+str(FP), fontsize=20, color="red")  # 红色，图片右上角
    plt.text(9,4, "TN:"+str(TN), fontsize=20, color="red")  # 红色，图片左下角
    plt.text(9,12, "FN:"+str(FN), fontsize=20, color="red")  # 红色，图片右下角
    plt.title("confusion matrix")
    plt.imshow(matrix)
    print("准确率为：{}%".format(100*(TP+TN)/(TP+TN+FP+FN)))

if __name__ == "__main__":
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\task6\watermelona3.txt"
    features = ["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"]
    X_train, X_test, y_train, y_test = load_split_dataset(dataset_path, features)
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_leaf_nodes = 4, 
                                                    random_state = 14), # 设计基学习器
                             n_estimators = 10,
                             learning_rate = 1,
                             random_state = None)  # 集成学习
    clf.fit(X_train, y_train["target"])
    predictions = clf.predict(X_test)
    test_result = pd.DataFrame({"target":y_test["target"], "predict":predictions})
    plot_confusion_matrix(test_result)  # 绘制混淆矩阵
    
    