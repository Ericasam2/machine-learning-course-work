# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 08:51:07 2021

@author: samgao1999
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
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
    while(1):
        index = random.sample(list(range(len(dataset))), k)
        tmp_label = []
        for i in index:
            tmp_label.append(dataset[i][-1])
        if (tmp_label[0]!=0 or tmp_label[1]!=1):
            index = random.sample(list(range(len(dataset))), k)
        else:
            break
    vectors = []
    for i in index:
        vectors.append(dataset[i])
    # 初始化标签
    labels = []
    for i in range(len(dataset)):
        labels.append(-1)
    # 根据迭代次数重复k-means聚类过程
    while(iteration > 0):
        C = []
        for i in range(k):  # 初始化簇
            C.append([])
        for labelIndex, item in enumerate(dataset):  # 为个样本寻找最近的簇心  
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = euclidean_distance(item, point)
                if(dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        for i, cluster in enumerate(C):  # 生成簇心
            clusterHeart = []
            dimension = len(dataset[0])
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate/len(cluster)
            vectors[i] = clusterHeart
        iteration -= 1
    return C, vectors, labels


def label_transform(target_data):
    '''
    # target_data : Dataframe, 数据集标签
    -----------------------------
    # result :
        trans_data : array of int, 实数化的数据集标签
    function : 输入数据集标签，返回实数化的数据集标签
    '''
    trans_data = np.zeros(target_data.shape)
    for i in range(len(target_data)):
        if target_data[i] == "spam":  # spam : 垃圾邮件
            trans_data[i] = 1
        else:
            trans_data[i] = 0
    return trans_data
    
    
def load_split_dataset(dataset_path):
    df = pd.read_csv(dataset_path, delimiter='\t')
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    vectorizer = TfidfVectorizer()
    f_data = vectorizer.fit_transform(data["text"]).toarray()
    t_data = label_transform(np.array(data["user"]))
    X_train, X_test, y_train, y_test = train_test_split(f_data, t_data,
                                                        test_size=0.2,
                                                        random_state=4)
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(predict_data):
    '''
    # data : DataFrame, 测试集+预测结果
    ---------------------
    # result : None
    function : 绘制预测结果的混淆矩阵
    '''
    TP = 0
    TN = 0
    P = predict_data[predict_data["predict"] == 0]
    len_P = len(P)
    for i in range(len_P):
        if (predict_data["predict"].iloc[i] == predict_data["target"].iloc[i]):
            TP = TP + 1
    FP = len_P - TP
    N = predict_data[predict_data["predict"] == 1]
    len_N = len(N)
    for i in range(len_N):
        if (predict_data["predict"].iloc[i] == predict_data["target"].iloc[i]):
            TN = TN + 1
    FN = len_N - TN
    plt.figure(0)
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
    plt.text(1,4, "TP:"+str(TP), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.text(1,12, "FP:"+str(FP), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.text(9,4, "TN:"+str(TN), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.text(9,12, "FN:"+str(FN), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.title("confusion matrix")
    plt.imshow(matrix)
    print("准确率为：{}%".format(100*(TP+TN)/(TP+TN+FP+FN))) 


def f(C, hearts):
    '''
    # C : list [[ndarray, ndarrya...], []..], 分类后的结果列表
    # hearts : list [ndarray, ndarrya...], 簇心
    ------------------
    # result : 
        x0 : list, 簇0距离簇心0的距离
        y0 : list, 簇0距离簇心1的距离
        x1 : list, 簇1距离簇心0的距离
        y1 : list, 簇1距离簇心1的距离
    function : 计算簇距离簇心的距离，方便之后进行可视化
    '''
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in C[0]:
        x0.append(euclidean_distance(i, hearts[0]))
        y0.append(euclidean_distance(i, hearts[1]))
    for i in C[1]:
        x1.append(euclidean_distance(i, hearts[0]))
        y1.append(euclidean_distance(i, hearts[1]))
    return x0, y0, x1, y1
    
    
if __name__ == "__main__":
    label_dict = {"spam" : 1, "ham" : 0}
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp6\SMSSpamCollection.txt"
    X_train, y_train, X_test, y_test = load_split_dataset(dataset_path)
    dataset = np.column_stack((X_train, y_train))
    C, hearts, predicts = k_means(dataset, 2, 1)
    test_result = pd.DataFrame({"target":y_train, "predict":predicts})
    plot_confusion_matrix(test_result)  # 绘制混淆矩阵
    x0, y0, x1, y1 = f(C, hearts)
    x3 = euclidean_distance(np.array(hearts[0]), hearts[0])
    y3 = euclidean_distance(np.array(hearts[0]), hearts[1])
    x4 = euclidean_distance(np.array(hearts[1]), hearts[0])
    y4 = euclidean_distance(np.array(hearts[1]), hearts[1])
    plt.figure(1)
    plt.scatter(x0, y0, color="red", marker=".")  # 绘制簇
    plt.scatter(x3, y3, color="red", marker="x")
    plt.scatter(x1, y1, color="blue", marker=".")
    plt.scatter(x4, y4, color="blue", marker="x")
    plt.show()
    
    