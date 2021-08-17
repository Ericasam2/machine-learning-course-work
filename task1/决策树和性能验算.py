# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:04:38 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import copy
from scipy.stats import ttest_rel

df = pd.read_csv(r"C:\Users\samgao1999\Desktop\机器学习\task1\adult.data")
data = df.replace(' ?', np.nan)
data = data.dropna()  #去除有缺失值的数据行


def get_all_dict(data):
    '''
    # data : DataFrame
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
    #将数据集中的字符型数据转化为数据型
    '''
    status_dict = get_all_dict(data)
    for i in status_dict:
        if (type(status_dict[i][0]) == int):
            continue
        data[i] = data[i].apply(lambda x : status_dict[i].index(x))  
    return data


def transform_to_original(data):
    '''
    # data : DataFrame
    #将数据集中的数据型数据转化为原类型
    '''
    status_dict = get_all_dict(data)
    for i in status_dict:
        if (type(status_dict[i][0]) == int):
            continue
        data[i] = data[i].apply(lambda x : status_dict[i])  
    return data

    
def show_relation(para_A, para_B):
    '''
    # para_A : str 某一个索引
    # para_B : str 某一个索引
    #用来显示任意两个属性的相关性
    #输出为图表形式
    '''
    a = data.groupby(para_A)[para_B].mean().to_frame(para_B).reset_index()
    pd.set_option('max_columns', None)
    a.sort_values(para_B).plot(
           x=para_A, 
           y=para_B,
           xticks=range(len(a.index)),
           yticks=range(len(a.index)),
           figsize=(7.2*2, 4.8),
           kind="line",
           title="relation of"+ para_A + " and " + para_B,
           rot=-20,
           grid=True)
    #show_relation("education", "education-num")
    

def devide_dataset(feature_data, target_data, n):
    '''
    #将原数据集十等分
    #返回的数据为列表，对应10等分的数据
    '''
    feature_dataset = []
    target_dataset = []
    length = len(data)
    size = length / n
    for i in range(0,n):
        if (i != n-1):
            feature_dataset.append(feature_data.loc[i*size : (i+1)*size])
            target_dataset.append(target_data.loc[i*size : (i+1)*size])
        else:
            feature_dataset.append(feature_data.loc[(i+1)*size : ])
            target_dataset.append(target_data.loc[(i+1)*size : ])
    return feature_dataset, target_dataset


def display_PR(model, result_list):
    '''
    # model : 分类器的模型
    # result_list : list[dic{list,list...},dic,...]结果列表，存储着n折检验的n次结果
    用来绘制PR曲线
    在result_list中，每一折的结果都使用dict，
    存放了P、R等参数的列表
    '''
    i = 1
    plt.figure(figsize=(7.2, 7.2))  #设置图纸大小
    plt.title(str(model)+"\'s P-R curve")  
    for result in result_list:  #遍历结果列表
        P = result["P"]
        R = result["R"]
        plt.xlabel("R")
        plt.ylabel("P")
        plt.plot(R, P, "--", label="N."+str(i))  #画图
        plt.legend()
        
    plt.grid(True)   #画栅格
    plt.show()
    
    return None


def cal_AUC(FPR, TPR):
    '''
    # FDR : float
    # TDR : float
    # 计算ROC曲线的面积
    '''
    AUC = 0
    for i in range(1, len(FPR)):
        AUC = AUC + (FPR[i] - FPR[i-1])*(TPR[i] + TPR[i-1])*0.5
    return AUC


def display_ROC(model, result_list):
    '''
    # model : 分类器的模型
    # result_list : 结果列表，存储着n折检验的n次结果
    # 用来画ROC曲线
    在result_list中，每一折的结果都使用dict，
    存放了P、R等参数的列表
    '''
    i = 1
    AUC = 0 
    plt.figure(figsize=(7.2, 7.2))
    plt.title(str(model)+"\'s ROC curve")  
    for result in result_list:
        FPR = result["FPR"]
        TPR = result["TPR"]
        AUC = cal_AUC(FPR, TPR)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot(FPR, TPR, "--", label="N."+str(i)+" AUC:{:.2f}".format(AUC))
        plt.legend()
        i = i + 1
        
    plt.grid(True)
    plt.show()
    
    return None


def display_matrix(model, result_list):
    '''
    # model : model 分类器模型
    # result_list : list[dic{list,list...},dic,...]结果列表，存储着n折检验的n次结果
    # 用于打印混淆矩阵
    '''
    length = len(result_list[1]["TP"])
    TP = result_list[1]['TP'][length//2]  #因为是设置了length个不同的阈值，所以length/2对应于0.5阈值
    FP = result_list[1]['FP'][length//2]
    TN = result_list[1]['TP'][length//2]
    FN = result_list[1]['FN'][length//2]
    matrix = [[TP, FP],[TN, FN]]
    matrix = pd.DataFrame(matrix, columns=['T','F'], index=['P','N'])  #用pd.dataframe来存储混淆矩阵
    # plt.title(str(model)+"\'s Matrix") 
    # plt.matshow(matrix)
    # plt.show()
    print(str(model) + "'s Matrix:")
    print(matrix)
    print("F1:{:.2F}".format(2*TP/(2*TP+FN)))
    print("\n")


def cross_validate(model, feature_data, target_data, n):  
    '''
    # model : model, 分类器模型
    # feature_data : dataframe, 数据集的特征索引
    # target_data : dataframe, 数据集的结果索引
    # n : int, 对数据集折n次
    # 进行n折交叉检验
    # 绘制PR曲线和ROG曲线
    # 通过t检验，给出模型的性能
    '''
    feature_dataset, target_dataset = devide_dataset(feature_data, target_data, n)  #将数据集n等分
    result_list = []
    # result = {"TP" : [], "FP" : [], "TN" : [], "FN" : [],
    #           "P" : [], "R" : [], 
    #           "TPR" : [], "FPR" : [],
    #           "precision" : []}
    
    for i in range(n):  #i代表第i折
        pvalue = []   #存储t检验结果的列表
        result = {"TP" : [], "FP" : [], "TN" : [], "FN" : [],
              "P" : [], "R" : [], 
              "TPR" : [], "FPR" : [],
              "precision" : []}
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        test_feature_dataset = feature_dataset[i] #选取验证集
        test_target_dataset = target_dataset[i]
        test_target_dataset.index = range(len(test_target_dataset)) 
        if (i == 0):
            k = 1
        else:
            k = 0
            
        train_feature_dataset = feature_dataset[k]  # 选取训练集
        train_target_dataset = target_dataset[k]
        for j in range(n):
            if (j == i or j == k):
                continue
            else:
                train_feature_dataset = train_feature_dataset.append(feature_dataset[j])  # 组合训练集
                train_target_dataset = train_target_dataset.append(target_dataset[j])
        
        model = model.fit(train_feature_dataset, train_target_dataset)  # 用训练集对模型进行训练
        # predict_result = model.predict(test_feature_dataset)
        # predict_result = pd.DataFrame(predict_result, columns=["predict_result"])
        
        predict_prob = model.predict_proba(test_feature_dataset)[: , 1]   # 估算模型分类结果的置信值
        predict_prob = pd.DataFrame(predict_prob, columns=["predict_prob"])  
        predict_result = copy.copy(predict_prob)
        
        for threshold in np.arange(1, 0, -0.01):  # 设置不同的阈值，方便绘制PR曲线，ROC曲线
            predict_result[predict_prob["predict_prob"] < threshold] = 0  # 通过阈值二分类预测结果
            predict_result[predict_prob["predict_prob"] >= threshold] = 1
        
            predict_result.columns = ["predict_result"]
            merged = pd.concat([predict_result, test_target_dataset], axis = 1)  # 组合Dataframe，便于操作
            merged["result"] = merged["predict_result"] - merged["attributes"] # 两列相减，便于得到FN, FT
        
            FN = len(merged[merged["result"] == -1])  # 统计各种性能参数
            FP = len(merged[merged["result"] == 1])
            TP = len(merged[merged["predict_result"] == 1]) - FP
            TN = len(merged[merged["predict_result"] == 0]) - FN
            result["TP"].append(TP)
            result["TN"].append(TN)
            result["FP"].append(FP)
            result["FN"].append(FN)
            result["P"].append(TP / (TP + FP + 0.0001))
            result["R"].append(TP / (TP + FN + 0.0001))
            result["TPR"].append(TP / (TP + FN + 0.0001))
            result["FPR"].append(FP / (TN + FP + 0.0001))
            result["precision"].append((TP + TN)/(TN + TP + FN + FP + 0.0001))
                
        result_list.append(result.copy())
        pvalue.append(ttest_rel(predict_result, test_target_dataset)[1])
        
    display_PR(model, result_list)  # 画PR曲线
    display_ROC(model, result_list)  # 画ROC曲线
    display_matrix(model, result_list)  # 画混淆矩阵
        # display_ROC(i, result)
        
    
    return np.mean(pvalue)



def generate_ft_data(data):
    '''
    #生成训练用参数的数据集和分类标签
    '''
    feature_data = data[["workclass","education","marital-status","occupation",
                  "relationship","race","sex","native-country"]]
    target_data = data[["attributes"]]
    
    return feature_data, target_data


if __name__ == "__main__":
    # dataset = generate_dataset(data)
    data = transform_to_int(data)
    f_data, t_data = generate_ft_data(data)  # 生成参数数据集和目标数据集
    model_A = tree.DecisionTreeClassifier(criterion = "gini", max_leaf_nodes=8)  # 设置决策树模型
    model_B = tree.DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes=8)

    
    # model_A = model_A.fit(f_data, t_data)
    # model_B = model_B.fit(f_data, t_data)
    # result_A = cross_val_score(
    #     model_A, 
    #     f_data, t_data,
    #     cv = 10)
    # result_B = cross_val_score(
    #     model_B, 
    #     f_data, t_data,
    #     cv = 10)
    
    result_A = cross_validate(model_A, f_data, t_data, 10)
    result_B = cross_validate(model_B, f_data, t_data, 10)
    
    if (result_A > result_B):   #判断两模型性能
        print("A is better")
    else:
        print("B is better")
    
    
    

