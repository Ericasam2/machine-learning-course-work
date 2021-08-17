# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:13:31 2021

@author: samgao1999
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def trainNB0(trainMatrix, trainCategory):
    '''
    # trainMatrix : array of float64, 训练集的标签
    # trainMatrix : array of float64, 训练集的特征
    -----------------------------
    # result 
        p0Vect : array of float64, 标签为0的词频对数
        p1Vect : array of float64, 标签为1的词频对数
        pAbusive : float64, 某一标签类型的比例
    function : 通过训练集，计算出词频的后验概率
    '''
    numTrainDocs = len(trainMatrix)  # numTrainDocs: int, 文本数量(词条个数)
    numWords = len(trainMatrix[0])  # numWords: int, 文本中单词总数
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)  # 标签为0的特征向量和（标签0单个单词出现次数）
    p1Num = np.ones(numWords)  # 标签为1的特征向量和
    p0Denom = 2  # 标签为0的稀疏矩阵总和（标签为0的总单词个数）
    p1Denom = 2  # 标签为1的稀疏矩阵总和
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 标签为1的每个词的词频的对数
    p0Vect = np.log(p0Num / p0Denom)  # 标签为0的每个词的词频的对数
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    # vec2Classify : array of float64, 测试数据
    # p0Vec : array of float64, 标签为0的词频对数
    # p1Vec : array of float64, 标签为1的词频对数
    # pClass1 : float64, 某一标签类型的比例
    -----------------------------
    # result : int, 0/1
    function : 输入测试数据，返回预测结果
    '''
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1 - pClass1)
    if (p1 > p0):
        return 1
    else:
        return 0


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
         

def predict(X_test, y_test, p0Vect, p1Vect, pAbusive):
    '''
    # X_test : array of float64, 测试特征集
    # y_test : array of float64, 测试标签集
    # p0Vec : array of float64, 标签为0的词频对数
    # p1Vec : array of float64, 标签为1的词频对数
    # pClass1 : float64, 某一标签类型的比例
    ----------------------------------------
    # result : 
        test_result : DataFrame, 测试标签集+预测值
    function : 预测训练集中的每条数据，并且得出预测结果
    '''
    predict = []
    for i in range(len(X_test)):
        predict.append(classifyNB(X_test[i], p0Vect, p1Vect, pAbusive))
    predict = np.array(predict)
    test_result = pd.DataFrame({"target":y_test, "predict":predict})
    return test_result


def plot_confusion_matrix(result):
    TP = 0
    TN = 0
    P = result[result["predict"] == 0]
    len_P = len(P)
    for i in range(len_P):
        if (result["predict"].iloc[i] == result["target"].iloc[i]):
            TP = TP + 1
    FP = len_P - TP
    N = result[result["predict"] == 1]
    len_N = len(N)
    for i in range(len_N):
        if (result["predict"].iloc[i] == result["target"].iloc[i]):
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
    plt.text(1,4, "TP:"+str(TP), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.text(1,12, "FP:"+str(FP), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.text(9,4, "TN:"+str(TN), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.text(9,12, "FN:"+str(FN), fontsize=20, color="red")  # 预测标签，红色，图片左下角
    plt.title("confusion matrix")
    plt.imshow(matrix)
    print("准确率为：{}%".format(100*(TP+TN)/(TP+TN+FP+FN)))

    
if __name__ == "__main__":
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp5\SMSSpamCollection.txt"
    X_train, y_train, X_test, y_test = load_split_dataset(dataset_path)
    p0Vect, p1Vect, pAbusive = trainNB0(X_train, y_train)
    test_result = predict(X_test, y_test, p0Vect, p1Vect, pAbusive)
    plot_confusion_matrix(test_result)
    
    
    
    
    