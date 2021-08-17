# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:36:26 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def euclidean_distance(x_training, row):
    '''
    # x_traininig : DataFrame, 训练特征数据集
    # row : Series, 测试数据样本
    ----------------------------
    # result : 
        dis : Series, 测试样本与训练集样本的距离
    function : 基于欧式距离计算预测样本与训练集的距离
    '''
    dis = np.sum((row - x_training)**2, axis = 1)
    dis = np.sqrt(dis)
    return dis


def predict(dis, y_training, k):
    '''
    # dis : Series, 测试样本与训练集样本的距离
    # y_traininig : DataFrame, 训练特征数据集
    # k : int, 近邻个数
    ----------------------------
    # result : 
        y_predict : str, 预测结果
    function : 在距离空间的基础上，在k的最近邻中选择数量最多的作为预测结果
    '''
    statistics = {}  # 定义字典，用于统计k个数据点中各个类别的鸢尾花出现的次数
    for i in range(k):
        rank = dis.index[i]  # 距离表中的排名
        if y_training.loc[rank] in statistics:
            statistics[y_training.loc[rank]] = statistics[y_training.loc[rank]] + 1
        else:
            statistics[y_training.loc[rank]] = 1
    sort_statis = sorted(statistics.items(),
                         key=operator.itemgetter(1),
                         reverse = True)  # 对statistics字典按照value进行排序
    y_predict = sort_statis[0][0]
    return y_predict


def predict_all(X_train, X_test, y_train, y_test, k):
    '''
    # X_train : DataFrame, 训练特征集
    # X_test : DataFrame, 测试特征集
    # y_train : DataFrame, 训练标签集
    # y_test : DataFrame, 测试标签集
    # k : int, 近邻个数
    --------------------------------
    # result : 
        test_result : DataFrame, 预测结果
    function : 对于测试集中所有的数据样本进行预测
    '''
    prediction = []
    for i in range(len(X_test)):
        dis_list = euclidean_distance(X_train, X_test.iloc[i])
        dis_list = dis_list.sort_values(ascending=True)
        prediction.append(predict(dis_list, y_train["target"], k))
    test_result = pd.DataFrame({"target":y_test["target"], "predict":prediction})
    return test_result
    
    
def plot_confusion_matrix(predict_df, show=1):
    '''
    # predict_df : DataFrame, 预测结果集
    # show : int, 是否展示混淆矩阵图片
    --------------
    # function : 根据预测结果绘制混淆矩阵
    '''
    dict = {}
    correct = 0
    length = 0
    order = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    index = 0
    for i in order:
        if (i not in dict):
            dict[i] = {}
        for j in order:
            if (j not in dict[i]):
                dict[i][j] = 0
    for predict in predict_df["predict"]:
        if (predict not in dict):
            dict[predict] = {}
        target = predict_df["target"].iloc[index]
        if (target in dict[predict]):
            dict[predict][target] = dict[predict][target] + 1
        else:
            dict[predict][target] = 1
        index += 1
    if (show == 1):
        plt.figure()
        x_axis = [x for x in range(24)]
        y_axis = [y for y in range(24)]
        matrix = np.ones([24, 24])
        i = 0
        for x_batch in range(3):
            for y_batch in range(3):            
                matrix[x_batch*8:(x_batch+1)*8, y_batch*8:(y_batch+1)*8] *= 25*i
                i = i + 1
        for i in range(len(order)):
            plt.text(-8,8*i+4, str(order[i]), fontsize=10, color="red")
            plt.text(8*i,-2, str(order[i]), fontsize=10, color="red")
            for j in range(len(order)):
                plt.text(8*i+2,8*j+4, str(dict[order[i]][order[j]]), fontsize=15, color="red")
                if (order[i] == order[j]):
                    correct += dict[order[i]][order[j]]
                length += dict[order[i]][order[j]]
        plt.title("confusion matrix")
        plt.imshow(matrix)
    else:
        for i in range(len(order)):
            for j in range(len(order)):
                if (order[i] == order[j]):
                    correct += dict[order[i]][order[j]]
                length += dict[order[i]][order[j]]
    print("准确率为：{}%".format(100 * correct / length))
    return correct / length
    
    
def load_split_dataset(dataset_path, features):
    df = pd.read_csv(dataset_path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    f_data = data[features]
    t_data = pd.DataFrame(data["target"])
    X_train, X_test, y_train, y_test = train_test_split(f_data, t_data, test_size=0.2, random_state=4)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp1\iris.data"
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_train, X_test, y_train, y_test = load_split_dataset(dataset_path, features)
    precision = []
    steps = []
    for k in range(20):
        steps.append(k+1)
        predictions = predict_all(X_train, X_test, y_train, y_test, k+1)
        precision.append(100.*plot_confusion_matrix(predictions, show = 1))
    plt.figure()
    plt.plot(steps, precision)
    plt.ylabel("precision")
    plt.xlabel("k")
    plt.title("the relationship between precision and k")
    plt.show()
        