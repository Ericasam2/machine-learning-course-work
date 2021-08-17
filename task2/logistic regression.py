# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 21:03:51 2021

@author: samgao1999
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 对率回归

def sigmoid(z):
    '''
    # 符号函数
    # z : 常数
    '''
    return 1/(1+np.exp(-z))


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


def loss(beta, X_expect, Y):
    '''
    # 损失函数
    # beta : array, shape=[n,1], 函数解析式的参数[[w],[b]]
    # X_expect : array, shape=[n, 2], 预期的x的集合 
    # Y : array, shape=[n, 1], 预期的结果，二分类(0,1)
    -------------------------------------------------
    # 输出损失值的大小
    '''
    loss = 0  
    for i in range(X_expect.shape[0]):  # 遍历预期X和对应结果Y，以计算损失函数
        x_e = np.array([X_expect[i]]).T  # x_e : array, shape=[2,1] 将预期x的矩阵转置，便于运算
        # x_e = np.insert(x_e, x_e.shape[1], np.array([[1]]), axis=1).T
        y = Y[i]  # y : array, shape=[1,]
        loss = loss + (-y*np.dot(beta.T, x_e) 
                + np.log(np.exp(1), 1+np.exp(np.dot(beta.T, x_e)))) # 计算损失值 scaler*[1,2]*[2,1]
    return loss[0, 0]


def P(beta, x_e, y):
    '''
    # 概率预测函数
    # beta : array, shape=[2,1], 函数解析式的参数[[w],[b]]
    # X_expect : array, shape=[n, 2], 预期的x的集合 
    # Y : array, shape=[n, 1], 预期的结果，二分类(0,1)
    -------------------------------------------------
    # 输出预期x和结果y在sigmoid下的置信度
    '''
    tmp = np.exp(np.dot(beta.T, x_e)[0][0])  # tmp : float, 预期x下的函数值结果
    if (y == 0):
        return 1 / (1 + tmp)
    else:
        return tmp / (1 + tmp)


def newton_opt(beta, X_expect, Y):
    '''
    # 牛顿法进行参数优化
    # beta : array, shape=[2,1], 函数解析式的参数[[w],[b]]
    # X_expect : array, shape=[n, 2], 预期的x的集合 
    # Y : array, shape=[n, 1], 预期的结果，二分类(0,1)
    -------------------------------------------------
    # 对于参数beta进行优化并且输出
    '''
    d1 = 0
    d2 = 0
    for i in range(X_expect.shape[0]):
        x_e = np.array([X_expect[i]])  # x_e : array, shape=[1,2]
        # x_e = np.insert(x_e, x_e.shape[1], np.array([[1]]), axis=1)
        y = Y[i]  # y : array, shape=[1,]
        P1 = P(beta, x_e.T, y)  # P1 : float64
        d1 = d1 + x_e.T * (y - P1)  # d1 : array, shape=[2,1] 函数的一阶偏微分结果
        d2 = d2 + np.dot(x_e, x_e.T) * P1 * (1 - P1)  # d2 : float64 函数的二阶偏微分结果
        
    d1 = -d1
    d2 = d2[0][0]
    beta = beta - np.dot(1/d2, d1)  #array of int32, shape=[2,1]
    
    return beta


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\samgao1999\Desktop\机器学习\task2\watermelona3.txt")
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = transform_to_int(data)
    
    t_data = data["target"]
    f_data = data[{"色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"}]
    
    beta = np.ones([f_data.shape[1]+1, 1])
    
    Y = t_data.values
    f_data["bias"] = 1
    X_expect = f_data.values

    i = 0
    while (abs(loss(beta, X_expect, Y)) > 0.5):
        print(abs(loss(beta, X_expect, Y)))
        i = i+1
        beta = newton_opt(beta, X_expect, Y)
    print("after {} steps: opt complete".format(i))
    
    Y_fit = np.dot(X_expect, beta)
    
    for i in range(len(Y)):
        if (Y[i] == 0):
            plt.plot(f_data.index.values[i], Y_fit[i], ".", color="red")
        else:
            plt.plot(f_data.index.values[i], Y_fit[i], ".", color="blue")
        
        plt.xlabel("index")
        plt.ylabel("value")


    plt.show()

