# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:53:09 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import random
import copy

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
    train_data, test_data = train_test_split(f_data, test_size=0.2, random_state=4)
    return train_data, test_data


def generate_mini_dataset(dataset, feature_num, sample_num):
    '''
    # dataset : DataFrame, 原始数据集
    # feature_num : int, 特征个数
    # sample_num : int, 样例个数
    -----------------------
    # result : 
        mini_X : DataFrame, 根据样例个数和特征个数划分出的小特征集
        mini_y : DataFrame, 根据样例个数划分出的小标签集
    function : 从原始数据集中抽取某些特征和某些样例生成小数据集
    '''
    rand_sample = random.sample(range(0, len(train_data)-1), sample_num)  # 随机抽取样本
    rand_feature = random.sample(range(0, len(train_data.columns)-1), \
                                 feature_num)  # 随机抽取标签
    mini_feature = []
    for i in rand_feature:
        mini_feature.append(features[i])  
    mini_X = pd.DataFrame(train_data[mini_feature].iloc[rand_sample])
    mini_y = pd.DataFrame(train_data["target"].iloc[rand_sample])
    mini_X = mini_X.reset_index(drop=True)
    mini_y = mini_y.reset_index(drop=True)
    return mini_X, mini_y


def random_forest(dataset, max_tree_num = 30):
    '''
    # dataset : DataFrame, 原始数据集
    # max_tree_num : int(default=30), 最大树个数
    --------------------------------------------
    # result : 
        forest : list [clf1, clf2 ...], 随机森林列表，每一个元素代表一个决策树
        forest_features : list [[feature1, feature2..]...], \
                    随机森林的特征列表，每一个元素代表一个决策树的特征
    function : 通过数据集生成随机森林
    '''
    forest = []
    forest_features = []
    feature_num = 5
    sample_num = 5
    dtc = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 10)  # 定义决策树
    for tree in range(max_tree_num):  # 生成随机森林
        mini_X, mini_y = generate_mini_dataset(dataset, feature_num, sample_num)  # 划分mini数据集
        forest_features.append(list(mini_X.columns))
        dtc.fit(mini_X, mini_y)
        forest.append(copy.copy(dtc))
    return forest, forest_features


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
    features = ["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率","target"]
    train_data, test_data = load_split_dataset(dataset_path, features)
    test_y = test_data["target"]
    forest, forest_features = random_forest(train_data, max_tree_num = 30)
    plt.figure(figsize=(16,12))
    for j in range(9):  # 画决策树示意图
        plt.subplot("33"+str(j))
        plot_tree(forest[j])
    forest_predictions = []
    for i, dtc in enumerate(forest):
        # sample = pd.DataFrame(test_data[forest_features[i]].iloc[0])
        # sample = pd.DataFrame(sample.values.T, index=sample.columns, columns=sample.index)
        forest_predictions.append(dtc.predict(test_data[forest_features[i]]))
    predictions = []
    for i in range(len(test_data)):  # 对测试集进行预测
        forest_res = []
        for tree_res in forest_predictions:
            forest_res.append(tree_res[i])
        predictions.append(max(forest_res, key=forest_res.count))
    test_result = pd.DataFrame({"target":test_data["target"], "predict":predictions})
    plot_confusion_matrix(test_result)
        
    

        