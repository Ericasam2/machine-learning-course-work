# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:36:21 2021

@author: samgao1999
"""
# LIBSVM

import sys
libsvm_path = r"C:\Users\samgao1999\Desktop\机器学习\task5\libsvm-3.25\python\libsvm"
sys.path.append(libsvm_path)

import svmutil
import copy
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
    original_dict = {}
    for i in list(data):
        status_dict[i] = data[i].unique().tolist()

    return status_dict


def get_transform_dict(original_dict, int_dict):
    transform_dict = {}
    for i in int_dict:
        tmp_dict = {}
        for j in int_dict[i]:
            tmp_dict[j] = original_dict[i][int_dict[i].index(j)]
            index = int_dict[i].index(j)
        transform_dict[i] = copy.copy(tmp_dict)
    return transform_dict
        

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


def load_split_dataset(dataset_path, f_features, t_feature="target"):
    df = pd.read_csv(dataset_path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = transform_to_int(data)
    f_data = pd.DataFrame(data[f_features])
    t_data = pd.DataFrame(data[t_feature])
    # t_data.loc[t_data["target"] > 10] = ">10"
    # t_data.loc[t_data["target"] != ">10"] = "<=10"
    X_train, X_test, y_train, y_test = train_test_split(f_data, t_data, test_size=0.001, random_state=4)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


def plot_predict(y_e, y_r):
    length = len(y_r)
    tp = []
    fp = []
    tp_index = []
    fp_index = []
    for i in range(length):
        if (y_e[i] == y_r[i]):
            tp_index.append(i)
            tp.append(y_e[i])
        else:
            fp_index.append(i)
            fp.append(y_e[i])
    plt.scatter(tp_index, tp, color="blue", label="正例")
    plt.scatter(fp_index, fp, color="red", label="反例")
    plt.xlabel("样本序号")
    plt.ylabel("预测值")
    plt.title("样本预测情况")
    plt.legend()
    plt.show()
            

def plot_SVM(y, x, model):
    X = np.array(np.linspace(0, 1000, 1000) / 1000).reshape(1000, 1)
    Y = svmutil.svm_predict([], X, model)
    plt.plot(X, Y[0], color="red", linestyle="-")
    plt.scatter(x, y, color="blue")
    plt.xlabel("密度")
    plt.ylabel("含糖率")
    plt.title("密度-含糖率图")
    plt.legend()
    plt.show()

    
def task_one():
    '''
    # 无输入
    ---------
    # 无返回值
    设定不同的SVM模型参数，训练SVM模型，并且得到预测结果
    '''
    # kernel_type = 0  # 线性核
    kernel_type = 2  # 高斯核
    SVM_type = 0  # one-class SVM
    # SVM_type = 3 # epsilon-SVR
    prama = "-s {} -t {}".format(SVM_type, kernel_type)
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\task5\watermelona3.txt"
    features = ["色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"]
    X_train, X_test, y_train, y_test = load_split_dataset(dataset_path, features)
    model = svmutil.svm_train(y_train[:,0], X_train, prama)
    predict = svmutil.svm_predict(y_test, X_test, model)
    plot_predict(predict[0], y_test)
    
    return predict
    

def task_two():
    '''
    # 无输入
    ---------
    # 无返回值
    设定不同的SVM模型参数，训练SVM模型，并且得到预测结果
    '''
    kernel_type = 0  # 线性核
    # kernel_type = 2  # 高斯核
    # SVM_type = 0  # one-class SVM
    SVM_type = 3 # epsilon-SVR
    prama = "-s {} -t {}".format(SVM_type, kernel_type)
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\task5\watermelona3.txt"
    f_features = "密度"
    t_feature = "含糖率"
    X_train, X_test, y_train, y_test = load_split_dataset(dataset_path, f_features, t_feature)
    model = svmutil.svm_train(y_train[:,0], X_train, prama)
    plot_SVM(y_train, X_train, model)
    
if __name__ == "__main__":
    task_two()
    