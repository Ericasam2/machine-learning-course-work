# -*- coding: utf-8 -*-
"""
Created on Wed May 12 00:05:20 2021

@author: samgao1999
"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np


def load_split_dataset(dataset_path, features):
    '''
    # dataset_path : str, 数据集的地址
    # features : list of str [feature1, feature2 ...], 要导入的数据集特征
    ------------------------------------------
    # result
    # X_train : DataFrame, 训练集的特征
    # X_test : DataFrame, 测试集的特征
    # y_train : DataFrame, 训练集的标签 
    # y_test : DataFrame, 测试集的标签
    导入数据集和数据集的某些特征，并且导出相应的训练集和测试集
    '''
    df = pd.read_csv(dataset_path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    f_data = data[features]
    t_data = pd.DataFrame(data["target"])
    X_train, X_test, y_train, y_test = train_test_split(f_data, t_data, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def factor_evaluation(X_train, y_train, X_test, y_test, factor = "MLN"):
    '''
    # X_train : DataFrame, 训练集的特征
    # X_test : DataFrame, 测试集的特征
    # y_train : DataFrame, 训练集的标签 
    # y_test : DataFrame, 测试集的标签
    factor : str, 待评估的参数名称
    -------------------------------------
    result
    coefficients : list of int, 待评估参数的取值
    acc_list : list of float, 各取值下的准确率
    根据训练集和测试集生成决策树模型，并且对某个参数进行评估
    '''
    acc_list = []
    coefficients = []
    if (factor == "MLN"):  # MLN : max_leaf_node
        for i in range(2, 20):
            dtc =  DecisionTreeClassifier(max_leaf_nodes = i, 
                                           random_state = 14)
            
            dtc.fit(X_train, y_train)
            y_predict = dtc.predict(X_test)
            accuracy = np.mean(y_predict == y_test["target"]) * 100
            coefficients.append(i)
            acc_list.append(accuracy)
            print("准确率为：{}".format(accuracy))
            
    elif (factor == "RS"):  # RS : random_state
        for i in range(1, 20):
            dtc =  DecisionTreeClassifier(max_leaf_nodes = 4, 
                                           random_state = i)
            
            dtc.fit(X_train, y_train)
            y_predict = dtc.predict(X_test)
            accuracy = np.mean(y_predict == y_test["target"]) * 100
            coefficients.append(i)
            acc_list.append(accuracy)
            print("准确率为：{}".format(accuracy))
    else:
        IOError("Cannot evaluate the factor")
    
    return coefficients, acc_list

    
if __name__ == "__main__":
    acc_list = []
    node_list = []
    random_list = []
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp2\iris.data"
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_train, X_test, y_train, y_test = load_split_dataset(dataset_path, features)
    
    node_list, acc_list = factor_evaluation(X_train, y_train, X_test, y_test, "MLN")
    plt.subplot(211)
    plt.plot(node_list, acc_list)
    plt.title("when random_state = 14, max_leaf_node as factor")
    plt.xlabel("max_leaf_node")
    plt.ylabel("accuracy")
    x_major_locator = MultipleLocator(1)
    ax = plt.gca() #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid()
    
    acc_list = []
    random_list, acc_list = factor_evaluation(X_train, y_train, X_test, y_test, "RS")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=2.5, wspace=None, hspace=None)
    plt.subplot(212)
    plt.plot(random_list, acc_list)
    plt.title("when max_leaf_node = 4, random_state as factor")
    plt.xlabel("random_state")
    plt.ylabel("accuracy")
    ax = plt.gca() #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    plt.grid()
    
    plt.figure(figsize=(24,16))
    dtc =  DecisionTreeClassifier(max_leaf_nodes = 4, 
                                  random_state = 14)
    dtc.fit(X_train, y_train)
    plot_tree(dtc, fontsize=30)
    
    
    
    
    

