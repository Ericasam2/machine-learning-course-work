# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:51:07 2021

@author: samgao1999
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pylab 
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置matplotlib的文字格式，使其显示中文
pylab.mpl.rcParams['axes.unicode_minus'] = False


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


def euclidean_distance(x_training, row):
    '''
    # x_traininig : DataFrame, 训练特征数据集
    # row : Series, 测试数据样本
    ----------------------------
    # result : 
        dis : Series, 测试样本与训练集样本的距离
    function : 基于欧式距离计算预测样本与训练集的距离
    '''
    dis = np.sum((row - x_training)**2)
    dis = np.sqrt(dis)
    return dis


def k_means(dataset, k, iteration):
    '''
    # dataset : array of float64, 数据集
    # k : int, 簇的个数
    # iteration : int, 迭代次数
    ------------------
    # result : 
        C : list [[ndarray, ndarrya...], []..], 分类后的结果列表
        vectors : list [[ndarray, ndarray...], []..], 簇中心列表
        labels : list [int, int...], 数据集向量分类
    function : 将数据集划分为k个簇
    '''
    # 初始化簇心向量
    index = random.sample(list(range(len(dataset))), k)
    vectors = []
    for i in index:
        vectors.append(list(dataset.iloc[i])[0:-1])
    # 初始化标签
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
    # 根据迭代次数重复k-means聚类过程
    while(iteration > 0):
        C = []
        for i in range(k):  # 初始化簇
            C.append([])
        for sample_i in dataset.index:
            classIndex = -1
            minDist = 9999
            for i, heart in enumerate(vectors):
                sample = list(dataset.iloc[sample_i][0:-1])
                dist = euclidean_distance(np.array(sample), np.array(heart))
                if (dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(sample_i)
            labels[sample_i] = classIndex
        for i, cluster in enumerate(C):  # 生成簇心
            clusterHeart = []
            for j in range(len(dataset.columns) - 1):
                clusterHeart.append(0)
            for sample_i in cluster:
                sample = list(dataset.iloc[sample_i][0:-1])
                clusterHeart += np.array(sample) / len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
    return C, vectors, labels

    
def load_dataset(dataset_path, features):
    '''
    # dataset_path : str, 数据集的地址
    # features : list of str [feature1, feature2 ...], 要导入的数据集特征
    ------------------------------------------
    # result : 
        data : DataFrame, 数据集
    function : 导入数据集地址和数据集的某些特征，并且导出相应的数据集
    '''
    df = pd.read_csv(dataset_path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = transform_to_int(data)
    data = data[features]
    return data
   
 
def plot_cluster(C, labels, hearts, dataset):
    '''
    # C : list [[index1, index2, ...],...], 簇的集合
    # labels : list [label1, label2, ...], 样本的标签
    # hearts : list [array1,, array2...], 簇心的坐标集
    # dataset : DataFrame, 数据集
    -------------------
    # result : None
    function : 根据簇绘制簇的图像
    '''
    color_list = ["red", "green", "blue", "yellow", "orange", "pink"]  # 不同的簇具有不同颜色
    color_index = random.sample(list(range(len(color_list))), len(C))
    plt.figure(figsize=(8,6))
    for index, cluster in enumerate(C):  # 遍历簇集合
        X0 = []; X1 = []
        Y0 = []; Y1 = []
        for sample_i in cluster:  # 遍历簇中的样本
            sample = list(dataset.iloc[sample_i])
            if (sample[2] == 0):
                X0.append(sample[0])  # 标签为0的样本坐标
                Y0.append(sample[1])
            else:
                X1.append(sample[0])  # 标签为1的样本坐标
                Y1.append(sample[1])
        plt.scatter(X0, Y0, c=color_list[color_index[index]], marker="*")  # 属于index簇且标签为0
        plt.scatter(list(hearts[index])[0], list(hearts[index])[1], \
                    c=color_list[color_index[index]], s=200, marker="o")  # 簇心的坐标
        plt.scatter(X1, Y1, c=color_list[color_index[index]], marker="x")  # 属于index簇且标签为1
    plt.xlabel("密度")
    plt.ylabel("含糖率")
    plt.grid()
    plt.show()
        

if __name__ == "__main__":
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\task8\watermelona3.txt"
    feature = ["密度","含糖率","target"]
    dataset = load_dataset(dataset_path, feature)
    C, hearts, labels = k_means(dataset, 6, 15)
    plot_cluster(C, labels, hearts, dataset)
    
    