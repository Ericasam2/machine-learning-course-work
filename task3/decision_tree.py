# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:13:32 2021

@author: samgao1999
"""
# 决策树

import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
import pylab 
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置matplotlib的文字格式，使其显示中文
pylab.mpl.rcParams['axes.unicode_minus'] = False


class decisionnode:
    def __init__(self, feature = -1, property = None, value = None, isleaf = False, results = None):
        self.feature = feature   # col是待检验的判断条件所对应的列索引值
        self.property = property  # 节点具有的前向节点的属性
        self.value = value  # value对应于为了使结果为True，当前列必须匹配的值
        self.isleaf = isleaf  # 判断是否是叶子节点
        self.results = results  #保存的是针对当前分支的结果，它是一个字典
        
        
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


def entropy(f_data, feature):
    '''
    # f_data: dataframe, 数据集
    # feature: str, 特征
    ----------------------------------------
    # result: float, 特征的信息熵
    通过数据集和某一特征，计算该特征的信息熵
    '''
    f_list = f_data[feature].unique()
    f_length = len(f_data[feature])
    f_entropy = 0
    for i in f_list:
        i_length = len(f_data[f_data[feature] == i])
        p_i = i_length / f_length
        f_entropy = f_entropy + p_i * np.log(p_i)
    return -f_entropy

    
def conditional_entropy(f_data, t_data, feature):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # feature: str, 特征
    --------------------------------
    # result: 
    # cond_entropy: float, 数据集在某个特征已知的情况下的条件熵 
    # leave_points: dict, 当条件熵为0时，求得的叶子节点的字典
    通过数据集和特征，计算该特征的条件熵，并且输出可能的叶子节点的字典
    '''
    f_list = f_data[feature].unique()
    leave_points = {}
    cond_entropy = 0
    for i in f_list:
        tmp_t_data = t_data[f_data[feature] == i]  # tmp_t_data: series, 在feature特征下的类别i的数据表对于t_data的映射
        i_entropy = entropy(tmp_t_data, "target")
        p_i = len(f_data[f_data[feature] == i]) / len(f_data[feature])
        if (p_i * i_entropy == 0):
            leave_points[i] = t_data[f_data[feature] == i]["target"].unique()[0]
        cond_entropy = cond_entropy + p_i * i_entropy
        
    return cond_entropy, leave_points


def info_gain(f_data, t_data, feature):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # feature: str, 特征
    -------------------------------
    # result: float, 信息增益大小
    对于给定数据集，通过选择某个特征可以得到的信息增益大小
    '''
    return entropy(t_data, "target") - conditional_entropy(f_data, t_data, feature)[0]
    

def intrisic(f_data, t_data, feature):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # feature: str, 特征
    -------------------------------
    # result: float, 固有熵的大小
    给定数据集的某个特征下所具有的属性所带来的固有熵
    '''
    f_list = f_data[feature].unique()
    f_length = len(f_data[feature])
    intrisic = 0
    for i in f_list:
        p = len(f_data[f_data[feature] == i]) / f_length
        intrisic = intrisic + p * np.log(p)
    return -intrisic


def info_gain_ratio(f_data, t_data, feature):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # feature: str, 特征
    -------------------------------
    # result: float, 信息熵增益率
    给定数据集下，选择某个特征所带来的信息增益率
    '''
    return info_gain(f_data, t_data, feature) / intrisic(f_data, t_data, feature)


def gini(f_data, t_data, feature):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # feature: str, 特征
    -------------------------------
    # result: float, 基尼系数
    给定数据集下，选择某个特征的基尼系数
    '''
    f_list = f_data[feature].unique()
    f_length = len(f_data[feature])
    gini = 0
    for i in f_list:
        p = len(f_data[f_data[feature] == i]) / f_length
        gini = gini + p * (1 - p)
        
    return gini


def best_feature(f_data, t_data, features, method = None):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # features: list, 特征集合
    # method: str, 使用的划分选择方法
    -------------------------------
    # result: str, 最佳的特征
    按照某种划分法选择最佳的特征
    '''
    score = []
    if (method == "gain"):
        for feature in features:
            score.append(info_gain(f_data, t_data, feature))
        return features[score.index(max(score))]
    
    elif (method == "gain_ratio"):
        for feature in features:
            score.append(info_gain_ratio(f_data, t_data, feature))
        return features[score.index(max(score))]
    
    elif (method == "gini"):
        for feature in features:
            score.append(gini(f_data, t_data, feature))
        return features[score.index(max(score))]
    
    else:
        raise IOError("the method is not included in this function")


def pre_pruning(f_data, t_data, test_f_data, test_t_data, features = [], method = None):
    '''
    # f_data: dataframe, 训练特征数据集
    # t_data: dataframe, 训练结果数据集
    # test_f_data: dataframe, 验证特征数据集 
    # test_t_data: dataframe, 验证结果数据集
    # method: str, 使用的划分选择方法
    -------------------------------
    # result: decisionnode, 决策树模型
    通过训练数据集和验证数据集，通过预剪枝的方式得到决策树
    '''
    
    tree_node = decisionnode(feature = "feature", property = "root", value = None, results = {})  # 生成决策树节点
    if (method == "gain"):  # 划分方法为信息增益
        # features = list(f_data.columns[:])  # 获取数据集的全部特征
        
        ambiguous_value = list(t_data["target"].unique())
        num_list = []
        for i in list(t_data["target"].unique()):
            num_list.append(len(t_data[t_data["target"] == i]))
        tree_node.value = ambiguous_value[num_list.index(max(num_list))]
        
        if (entropy(t_data, "target") == 0):  # 划分后的结果数据集属性一致，判断为一个树节点
            tree_node.value = t_data.unique()[0]
            tree_node.isleaf = True
            return tree_node

        if (len(features) == 0):
            tree_node.isleaf = True
            return tree_node
        
        b_feature = best_feature(f_data, t_data, features, method)  # 选取最优的特征
        features.remove(b_feature)  
        cond_entropy, leaf_points = conditional_entropy(f_data, t_data, b_feature)  # 计算该特征的条件熵
        bf_list = list(f_data[b_feature].unique())  # 最佳特征所具有的属性
        tree_node.feature = b_feature  # 赋值树节点
        y = fit(tree_node, test_f_data, test_t_data)
        
        if (len(leaf_points) != 0):  # 生成部分叶子节点
            for leaf in leaf_points:
                bf_list.remove(leaf)
                value = leaf_points[leaf]
                tree_node.results[leaf] = decisionnode(feature = None,   # 生成叶子节点
                                                       property = leaf,
                                                       value = value, 
                                                       isleaf = True,
                                                       results = None)
        for i in bf_list:  # 先预判所有节点为叶子节点，给预判值
            num_list = []
            temp_t_data = t_data[f_data[b_feature] == i]
            ambiguous_value = list(temp_t_data["target"].unique())
            for j in list(t_data["target"].unique()):
                num_list.append(len(temp_t_data[temp_t_data["target"] == i]))
            tree_node.results[i] = decisionnode(feature = None, 
                                                property = i,  
                                                value = ambiguous_value[num_list.index(max(num_list))],  # 预判值
                                                results = None)    
        # print("devide: {}， nondevide:{}".format(fit(tree_node, test_f_data, test_t_data), y))
        if (fit(tree_node, test_f_data, test_t_data) > y):  # 比较节点进行进一步分化，准确率是否可以上升
            tree_node.results = None
            tree_node.isleaf = True  # 进行剪枝
        else:
            for i in bf_list:  #不剪枝并且继续分化
                tree_node.results[i] =  pre_pruning(f_data[f_data[b_feature] == i], 
                                                    t_data[f_data[b_feature] == i],
                                                    test_f_data,
                                                    test_t_data,
                                                    features = copy.copy(features),
                                                    method = method)
                tree_node.results[i].property = i
        return tree_node

    elif (method == "gain_ratio"):
        pass
    elif (method == "gini"):
        pass
    else:
        print("the method is not included in this function")
    
    pass


def post_pruning_one_layer(decision_tree, test_f_data, test_t_data, method = None):
    '''
    # decision_tree: decisionnode, 决策树
    # test_f_data: dataframe, 验证特征数据集 
    # test_t_data: dataframe, 验证结果数据集
    # method: str, 使用的划分选择方法
    -------------------------------
    # result: decisionnode, 决策树模型
    通过验证数据集和原生成树，通过后剪枝的方式得到决策树
    '''
    
    leaf_num = 0
    
    for i in decision_tree.results:
        if (decision_tree.results[i].isleaf == True):
            leaf_num = leaf_num + 1
            
    if (leaf_num == len(decision_tree.results) and len(decision_tree.results) != 0):  # 此时该节点为所有节点均为叶子节点，可以判断是否剪枝
        ori_y = fit(decision_tree, test_f_data, test_t_data)  # 得到剪枝前的验证集准确度
        buff =  copy.copy(decision_tree.results)  # 保存决策树的子树所有节点
        decision_tree.results = None  # 剪枝
        decision_tree.isleaf = True  
        new_y = fit(decision_tree, test_f_data, test_t_data)  # 得到剪枝后的验证集准确度
        if (new_y <= ori_y):  # 判断是否比原来准确
            decision_tree.results = buff  # 不剪枝
            decision_tree.isleaf = False
            return decision_tree
    else:
        for i in decision_tree.results:
            if (decision_tree.results[i].isleaf == False):  # 遍历寻找叶子节点
                decision_tree.results[i] = post_pruning_one_layer(decision_tree.results[i], test_f_data, test_t_data, method = method)
    return decision_tree


def post_pruning(decision_tree, test_f_data, test_t_data, layers = 5, method = None):
    for i in range(layers):
        decision_tree = post_pruning_one_layer(decision_tree, test_f_data, test_t_data, method = None)
    return decision_tree

        
def tree_generate(f_data, t_data, features = [], method = None):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 结果数据集
    # method: str, 使用的划分选择方法
    # pruning: str, 剪枝的方法
    -------------------------------
    # result: decisionnode, 决策树模型
    通过数据集和一定的数据划分、剪枝方法，生成决策树模型
    '''
    tree_node = decisionnode(feature = "feature", property="root", value = None, isleaf = False, results = {})  # 生成决策树节点
    if (method == "gain"):  # 划分方法为信息增益
        #features = list(f_data.columns[:])  # 获取数据集的全部特征
        
        ambiguous_value = list(t_data["target"].unique())  # 通过目标数据集的分类情况为节点定义value
        num_list = []
        for i in list(t_data["target"].unique()):
            num_list.append(len(t_data[t_data["target"] == i]))
        tree_node.value = ambiguous_value[num_list.index(max(num_list))]  # 服从多数
        
        if (entropy(t_data, "target") == 0):  # 划分后的结果数据集属性一致，判断为一个叶子节点
            tree_node.value = t_data.unique()[0]
            tree_node.isleaf == True
            return tree_node

        if (len(features) == 0):
            tree_node.isleaf == True
            return tree_node
        
        b_feature = best_feature(f_data, t_data, features, method)  # 选取最优的特征
        features.remove(b_feature)  
        cond_entropy, leaf_points = conditional_entropy(f_data, t_data, b_feature)  # 计算该特征的条件熵
        bf_list = list(f_data[b_feature].unique())  # 最佳特征所具有的属性
        tree_node.feature = b_feature  # 赋值树节点
        
        if (len(leaf_points) != 0):
            for leaf in leaf_points:
                bf_list.remove(leaf)
                a = leaf_points[leaf]
                value = leaf_points[leaf]
                tree_node.results[leaf] = decisionnode(feature = None,
                                                       property = leaf,
                                                       value = value, 
                                                       isleaf = True,
                                                       results = None)
        for i in bf_list:
            tree_node.results[i] = tree_generate(f_data[f_data[b_feature] == i], 
                                                 t_data[f_data[b_feature] == i], 
                                                 copy.copy(features),
                                                 method = method)
            tree_node.results[i].property = i
        return tree_node

    elif (method == "gain_ratio"):
        pass
    elif (method == "gini"):
        pass
    else:
        print("the method is not included in this function")  
        

def predict(decision_tree, test_data):
    '''
    # decision_tree: decisionnode, 决策树模型
    # test_data: dataframe, 测试用的数据
    -------------------------------------
    # result: str, 决策树对一个样本的判断结果
    通过输入一个样本和决策树模型，来预测这个样本的结果
    '''
    feature = decision_tree.feature
    if (decision_tree.results == None):
        return decision_tree.value
    if (len(decision_tree.results) == 0):
        return decision_tree.value
    if (not test_data[feature].any() in decision_tree.results):
        return None
    # print("\n")
    # print(feature)
    # print(test_data[feature])
    # print(decision_tree.results)
    # print(node_info(decision_tree.results[test_data[feature].any()]))
    if (decision_tree.results[test_data[feature].any()].isleaf == True):
        return decision_tree.results[test_data[feature].any()].value
    else:
        decision_tree = decision_tree.results[test_data[feature].all()]
        res = predict(decision_tree, test_data)  # 递归在决策树中进行遍历
        return res

    
def fit(decision_tree, test_f_data, test_t_data):
    '''
    # decision_tree: decisionnode, 决策树模型
    # test_f_data: dataframe, 测试用的特征数据集
    # test_t_data: dataframe, 测试用的结果数据集
    --------------------------------------------
    # result: float, 预测的准确率
    通过输入决策树模型和训练数据集，从而获得决策的准确率
    '''
    tp = 0
    for i in range(len(test_f_data)):
        f = test_f_data.iloc[[i]]
        t = test_t_data["target"].iloc[[i]]
        p = predict(copy.copy(decision_tree), f)
        if (p == None):
            continue
        if (t.all() == p):
            tp = tp + 1
    return tp / len(test_f_data) 


def node_info(node):
    info = "feature:{}.\nproperty:{}.\nvalue:{}.".format(node.feature, node.property, node.value)
    return info


def create_graph(G, node, pos = {}, x = 0, y = 0, layer = 1):
    pos[node_info(node)] = (x, y)
    index = 0
    for i in node.results:
        G.add_edge(node_info(node), node_info(node.results[i]))
        child_x, child_y = x + (index - len(node.results) / 2) * (len(node.results)) ** 2, y - 1
        child_layer = layer + 1
        index = index + 1
        pos[node_info(node.results[i])] = (child_x, child_y)
        if (node.results[i].isleaf == False):
            create_graph(G, node.results[i], x=child_x, y=child_y, pos=pos, layer=child_layer)
        
    return (G, pos)


def draw(node, title):   # 以某个节点为根画图
    graph = nx.DiGraph()
    graph, pos = create_graph(graph, node)
    fig, ax = plt.subplots(figsize=(48, 32))  # 比例可以根据树的深度适当调节
    plt.title("{}".format(str(title)))
    nx.draw_networkx(graph, pos, ax=ax, node_size=300)
    plt.show()


def load_split_dataset(dataset_path, features):
    df = pd.read_csv(dataset_path)
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    f_data = data[features]
    t_data = pd.DataFrame(data["target"])
    t_data.loc[t_data["target"] > 10] = ">10"
    t_data.loc[t_data["target"] != ">10"] = "<=10"
    X_train, X_test, y_train, y_test = train_test_split(f_data, t_data, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset_path = r"C:\Users\samgao1999\Desktop\机器学习\task3\dataset\student\student-mat.csv"
    features = ["school","sex","Pstatus","reason","studytime","failures","schoolsup","romantic","freetime","health"]
    X_train, X_test, y_train, y_test = load_split_dataset(dataset_path, features)
    decision_tree1 = tree_generate(X_train, y_train, list(X_train.columns[:]), method="gain")
    print("Decision tree generated! \nP={}".format(fit(decision_tree1, X_test, y_test)))
    draw(decision_tree1, "original decision_tree")
    decision_tree2 = pre_pruning(X_train, y_train, X_test, y_test, list(X_train.columns[:]), method="gain")
    print("Pre_Pruning ok! \nP={}".format(fit(decision_tree2, X_test, y_test)))
    draw(decision_tree2, "pre_pruning decision_tree")
    decision_tree3 = post_pruning(copy.copy(decision_tree1), X_test, y_test, 11)
    print("Post_Pruning ok! \nP={}".format(fit(decision_tree3, X_test, y_test)))
    draw(decision_tree3, "post_pruning decision_tree")

