# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:31:35 2021

@author: samgao1999
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

max_column =  3000

def data_input(path):
    '''
    # path : str, 数据集的地址
    --------------------------
    # data : DataFrame, df格式的数据集
    用来读取数据集
    '''
    df = pd.read_csv(path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = data[:max_column]
    data = transform_to_int(data)    
    return data
    

def get_all_dict(data):
    '''
    # data : DataFrame
    -----------------------
    # status_dict : dict[str:array]  
    # 产生所有列名的一个字典
    # 列名->该列中所有的数据
    '''
    status_dict = {}
    for i in list(data):
        status_dict[i] = data[i].unique().tolist()
    return status_dict


def transform_to_int(data):
    '''
    # data : DataFrame
    ----------------------------------
    # data : DataFrame
    #将数据集中的字符型数据转化为数据型
    '''
    status_dict = get_all_dict(data)
    for i in status_dict:
        # print("{}:{}".format(status_dict[i],))
        if (type(status_dict[i][0]) == int or
            type(status_dict[i][0]) == float):
            continue
        data[i] = data[i].apply(lambda x : status_dict[i].index(x))  
    return data


def logistic_regression_evaluation(f_data, t_data, kfold):
    '''
    # f_data : DataFrame, 特征数据集
    # t_data : DataFrame, 目标数据集
    # kfold : str, 判断是10折交叉检验还是留一法检验
    --------------------------------
    # 返回错误率
    # 验证数据集在某种检验法下的错误率
    '''
    LR = linear_model.LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,  # 调用模型
                                  intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                  penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                  verbose=0, warm_start=False)
    
    if (kfold == "10FOLD"):  # 10折交叉检验
        score = cross_val_score(LR, f_data.values, t_data.values, scoring=None, cv=10)
        return 1-np.mean(score)  # 返回错误率
    elif (kfold == "LEAVEONEOUT"):  # 留一法检验
        LOO = LeaveOneOut()
        score = []
        for train_index, test_index in LOO.split(f_data.values, t_data.values):  # 对每一种划分方法的结构进行评估
            f_train, f_test = f_data.values[train_index], f_data.values[test_index]
            t_train, t_test = t_data.values[train_index], t_data.values[test_index]
            model = LR.fit(f_train, t_train)
            score.append(model.score(f_test, t_test))
        return 1-np.mean(score)  # 返回错误率
    else:
        print("not included in the function!")
        exit(1)
    

if __name__ == "__main__":
    path = r"C:\Users\samgao1999\Desktop\机器学习\task2\car.data"
    dataframe = data_input(path)
    t_data = dataframe["target"]
    f_data = dataframe[{"buying","maint","doors","persons","lug_boot","safety"}]
    eva_LEAVEONEOUT = logistic_regression_evaluation(f_data, t_data, "LEAVEONEOUT")
    print("对于car数据集，留一法的错误率为：{:3f}%".format(100*eva_LEAVEONEOUT))
    eva_10FOLD = logistic_regression_evaluation(f_data, t_data, "10FOLD")
    print("对于car数据集，10折交叉的错误率为：{:3f}%".format(100*eva_10FOLD))
    
    path = r"C:\Users\samgao1999\Desktop\机器学习\task2\adult.data"
    dataframe = data_input(path)
    t_data = dataframe["target"]
    f_data = dataframe[{"age","workclass","education","marital-status","occupation","relationship","race","sex","hours-per-week","native-country"}]
    eva_LEAVEONEOUT = logistic_regression_evaluation(f_data, t_data, "LEAVEONEOUT")
    print("对于adult数据集，留一法的错误率为：{:3f}%".format(100*eva_LEAVEONEOUT))
    eva_10FOLD = logistic_regression_evaluation(f_data, t_data, "10FOLD")
    print("对于adult数据集，10折交叉的错误率为：{:3f}%".format(100*eva_10FOLD))



