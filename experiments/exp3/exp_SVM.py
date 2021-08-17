# -*- coding: utf-8 -*-
"""
Created on Wed May 12 00:22:29 2021

@author: samgao1999
"""

from sklearn import svm
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np


def load_dataset(dataset_path):
    '''
    # dataset_path : str, 数据集的路径
    ---------------------
    # result :
        f_data : Dataframe, 特征数据集
        t_data : Dataframe, 标签数据集
    function : 输入数据集的地址，将数据集划分为特征数据集和标签数据集
    '''
    df = pd.read_csv(dataset_path, header = None)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = data.rename(columns = {64 : "target"})
    f_data = data[list(range(0, 64))]
    t_data = pd.DataFrame(data["target"])
    return f_data, t_data
    
    
def factor_evaluation(X_train, y_train, X_test, y_test, factor = "C"):
    '''
    # X_train : DataFrame, 训练集的特征
    # X_test : DataFrame, 测试集的特征
    # y_train : DataFrame, 训练集的标签 
    # y_test : DataFrame, 测试集的标签
    factor : str, 待评估的参数名称
    -------------------------------------
    result :
        coefficients : list of int, 待评估参数的取值
        acc_list : list of float, 各取值下的准确率
    function : 根据训练集和测试集生成决策树模型，并且对某个参数进行评估
    '''
    acc_list = []
    coefficients = []
    if (factor == "C"):  # MLN : max_leaf_node
        for i in range(1, 20):
            model = svm.SVC(C=i, kernel="rbf",gamma=0.001)
            model.fit(X_train, y_train["target"])
            
            y_predict = model.predict(X_test)
            accuracy = np.mean(y_predict == y_test["target"]) * 100
            coefficients.append(i)
            acc_list.append(accuracy)
            # print("准确率为：{}".format(accuracy))
            
    elif (factor == "gamma"):  # RS : random_state
        for i in range(1, 10, 1):
            model = svm.SVC(C=10, kernel="rbf",gamma=i/1000)
            model.fit(X_train, y_train["target"])
            
            y_predict = model.predict(X_test)
            accuracy = np.mean(y_predict == y_test["target"]) * 100
            coefficients.append(i/1000)
            acc_list.append(accuracy)
            # print("准确率为：{}".format(accuracy))
    else:
        IOError("Cannot evaluate the factor")
    
    return coefficients, acc_list


if __name__ == "__main__":
    test_dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp3\optdigits.tes"
    train_dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp3\optdigits.tra"
    X_train, y_train = load_dataset(train_dataset_path)
    X_test, y_test = load_dataset(test_dataset_path)
    
    C_list, accuracy = factor_evaluation(X_train, y_train, X_test, y_test, factor = "C")
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=2.5, wspace=None, hspace=None) # 设置子图的位置
    
    plt.subplot(211)  # 绘制子图
    plt.plot(C_list, accuracy)  
    plt.title("C" +" as factor")
    plt.xlabel("C")
    plt.ylabel("accuracy")
    x_major_locator = MultipleLocator(1)  # 设置横坐标的最小刻度间隔
    ax = plt.gca() #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  
    plt.grid()  # 绘制刻度线
    
    accuracy = []
    gamma_list, accuracy = factor_evaluation(X_train, y_train, X_test, y_test, factor = "gamma")
    plt.subplot(212)
    plt.plot(gamma_list, accuracy)
    plt.title("gamma" +" as factor")
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca() #ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.grid()
    plt.show()
    
    model = svm.SVC(C=2, kernel="rbf",gamma=0.002)
    print("Start Learning...")
    t0 = time.time()
    model.fit(X_train, y_train)
    t = time.time() - t0
    print("训练+CV耗时：{}分钟{}秒".format(int(t/60), t-60*int(t/60)))
    print("Learning is OK...")
    print("训练集准确率：{}".format(accuracy_score(y_train, model.predict(X_train))))
    print("测试集准确率：{}".format(accuracy_score(y_test, model.predict(X_test))))
    
    for i in range(9):  # 绘制数字图
        plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.5, wspace=None, hspace=None)  # 调整子图的位置
        index = np.random.random_integers(0, len(X_test))  # 随机选择测试集中的一张图片
        subplot = "33" + str(i+1)  # 设置子图位置
        plt.subplot(int(subplot))
        
        img = X_test.loc[index]  
        label = y_test.loc[[index]]["target"]
        predict = model.predict(X_test.iloc[[index]])[0]
        pic = np.ones([8,8])
        for x in range(8):  # 生成数字图
            for y in range(8):
                a = 8 * x + y
                b = pic[x, y]
                pic[x, y] = pic[x, y] * int(img.iloc[a])
        plt.text(0,7, str(predict), color="red")  # 预测标签，红色，图片左下角
        plt.text(6,7, str(label.iloc[0]), color="white")  # 实际标签，白色，图片右下角
        plt.imshow(pic)
        
                
                
        
        
        
    
