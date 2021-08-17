# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:51:29 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
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


def U(X):
    '''
    # X : array shape=[feature, num], 数据集的特征向量
    -------------------------------------------------
    # 返回特征向量的平均值
    '''
    sum = np.zeros([1, X.shape[1]])
    n = X.shape[0]   
    for i in range(n):
        sum = sum + X[i, :]   
    return sum / n


def S_b(u_1, u_2):
    '''
    # u_1 : array shape=[1, feature], 某一类别向量的平均向量（正例\反例）
    # u_2 : array shape=[1, feature], 某一类别向量的平均向量（正例\反例）
    ------------------------------------------------------------------
    # 返回类间散度矩阵
    '''
    return np.dot((u_1-u_2).T, (u_1-u_2))


def S_w(X, u_1, u_2):
    '''
    # X : array shape=[feature, num], 数据集的特征向量
    # u_1 : array shape=[1, feature], 某一类别向量的平均向量（正例\反例）
    # u_2 : array shape=[1, feature], 某一类别向量的平均向量（正例\反例）
    ------------------------------------------------------------------
    # 返回类内散度矩阵
    '''
    sigma1 = 0
    sigma2 = 0
    for i in range(X.shape[0]):
        sigma1 = sigma1 + np.dot((X[i, :] - u_1).T, (X[i, :] - u_1))
        sigma2 = sigma2 + np.dot((X[i, :] - u_2).T, (X[i, :] - u_2))
    return sigma1 + sigma2


def J(w, S_b, S_w):
    '''
    # w : array shape=[feature, 1], 转置矩阵，用于降维参数
    # S_b : array shape=[feature, feature], 类间散度矩阵
    # S_w : array shape=[feature, feature], 类内散度矩阵
    ----------------------------------------------------
    # 输出广义瑞利商
    '''
    M = np.dot(np.dot(w.T, S_b), w)
    N = np.dot(np.dot(w.T, S_w), w)
    return M / N


def solution(S_w, u_1, u_2):
    '''
    # S_w : array shape=[feature, feature], 类内散度矩阵
    # u_1 : array shape=[1, feature], 某一类别向量的平均向量（正例\反例）
    # u_2 : array shape=[1, feature], 某一类别向量的平均向量（正例\反例）
    ------------------------------------------------------------------
    # 返回转置矩阵w的计算结果，用于降维投影
    '''
    return np.dot(np.linalg.inv(S_w), (u_1 - u_2).T)


def LDA(positive_data, negative_data):
    '''
    # positive_data : array shape=[num, feature], 正例的矩阵
    # negative_data : array shape=[num, feature], 反例的矩阵
    --------------------------------------------------------
    # 返回转置矩阵w的计算结果和对应瑞利商结果
    '''
    U_p = U(positive_data)
    U_n = U(negative_data)
    
    Sb = S_b(U_p, U_n)
    Sw = S_w(X, U_p, U_n)
    
    w = solution(Sw, U_p, U_n)
    res = J(w, Sb, Sw)
    
    return w, res[0][0]


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\samgao1999\Desktop\机器学习\task2\watermelona3.txt")
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = transform_to_int(data)
    
    P_data = data[data["target"] == 0]
    N_data = data[data["target"] == 1]
    
    t_data = data["target"]
    f_data = data[{"色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"}]
    X = f_data.values
    
    f_P_data = P_data[{"色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"}]
    f_N_data = N_data[{"色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"}]
    
    w, j = LDA(f_P_data.values, f_N_data.values)
    
    print("转置矩阵是：\n{}\n".format(w))
    print("J = {}".format(j))

    
    


