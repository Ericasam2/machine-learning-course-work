# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:27:14 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    f_data = data[features]
    t_data = pd.DataFrame(data["target"])
    # t_data.loc[t_data["target"] > 10] = ">10"
    # t_data.loc[t_data["target"] != ">10"] = "<=10"
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=4)
    return X_train, X_test


def gauss(tci_data, test):
    '''
    # tci_data : DataFrame, 某一个标签的数据集
    # test : float64, 测试集某一个属性的值（连续值）
    ----------------------------------------------
    # result : float64, 高斯函数下的值
    function: 高斯函数，用来进行连续值的贝叶斯分类
    '''
    uc = tci_data.mean()  # 平均值
    std = tci_data.std()  # 方差
    return 1/(2**0.5 * std)*np.exp(-(test - uc) / (2 * std**2))


def naive_bayes_classifier(train_data, test_data):
    '''
    # train_data : DataFrame, 训练集
    # test_data : DataFrame, 测试集
    -------------------------------
    # result:
        test_data : DataFrame, 测试集+预测结果
    function : 通过训练集，来预测测试集上的结果
    '''
    predict = []  # 测试结果
    for i in range(len(test_data)):
        test = test_data.iloc[i]
        t_value = train_data["target"].unique()  # 所有标签
        pci_x = {}  # 记录预测结果中，不同标签的概率
        for ci in t_value:
            tci_data = train_data[train_data["target"] == ci]
            pci = (len(tci_data) + 1) / (len(train_data) + len(t_value))
            pci_x[ci] = pci
            for f in test.index.values:
                if (type(tci_data[f].iloc[0]) == np.float64):  # 连续值
                    fi_c = gauss(tci_data[f], test[f])
                else:
                    fi_c = (len(tci_data[tci_data[f] == test[f]]) + 1) \
                        / (len(tci_data[f]) + len(test.index.values))  # 离散值
                pci_x[ci] = pci_x[ci] * fi_c
        max = -9999
        for ci in pci_x:  # 找到最可能的预测结果
            if (pci_x[ci] > max):
                max = pci_x[ci]
                pos = ci
        predict.append(pos)
    test_data["predict"] = predict
    return test_data
    
def plot_confusion_matrix(data):
    '''
    # data : DataFrame, 测试集+预测结果
    ---------------------
    # result : None
    function : 绘制预测结果的混淆矩阵
    '''
    TP = 0
    TN = 0
    P = data[data["predict"] == "是"]
    len_P = len(P)
    for i in range(len_P):
        if (data["predict"].iloc[i] == data["target"].iloc[i]):
            TP = TP + 1
    FP = len_P - TP
    N = data[data["predict"] == "否"]
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
    train_data, test_data = load_split_dataset(dataset_path, features)
    result = naive_bayes_classifier(train_data, train_data)
    plot_confusion_matrix(result)
    # test.loc[test['asin']==item,'predict'] = predict
    
    # t_data = pd.DataFrame(data["target"])
    # f_data = data[{"色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"}]
    
          
        
        
    
    
    
    