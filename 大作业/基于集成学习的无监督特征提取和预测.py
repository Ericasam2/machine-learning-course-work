# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 21:55:57 2021

@author: samgao1999
"""
import pandas as pd
import gzip
import random
import copy
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree

def parse(path):
    '''
    # path : str, 文件路径
    ---------------------
    # result : None
    function : 打开zip文件
    '''
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    '''
    # path : str, 文件路径
    ------------------------
    # result : DataFrame, Music数据集
    function : 读取数据集
    '''
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


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
    features = list(dataset.columns)
    features.remove("asin")
    features.remove("overall")
    rand_sample = random.sample(range(0, len(dataset)-1), sample_num)  # 随机抽取样本
    rand_feature = random.sample(range(0, len(features)-1), \
                                 feature_num)  # 随机抽取标签
    mini_feature = []
    for i in rand_feature:
        mini_feature.append(features[i])  
    mini_X = pd.DataFrame(dataset[mini_feature].iloc[rand_sample])  # 迷你特征集
    mini_y = pd.DataFrame(dataset["overall"].iloc[rand_sample])  # 迷你标签集
    mini_X = mini_X.reset_index(drop=True)  # 重置index值
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
    feature_num = 6
    sample_num = 400
    dtc = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 10)  # 定义决策树
    for tree in range(max_tree_num):  # 生成随机森林
        mini_X, mini_y = generate_mini_dataset(dataset, feature_num, sample_num)  # 划分mini数据集
        forest_features.append(list(mini_X.columns))
        dtc.fit(mini_X, mini_y)
        forest.append(copy.copy(dtc))
    return forest, forest_features


def create_features(feature_num, dataset):
    '''
    # feature_num : int, 要生成的特征个数
    # dataset : DataFrame, 原始数据集
    ----------------------
    # result : 
        dataset : Dataframe, 加入新特征的数据集
    function : 通过聚类方法，为原始数据集增加新的特征
    '''
    cluster_num = random.sample(range(2, 2 * feature_num), feature_num)  # 随机生成簇的个数
    plt.figure(figsize=(18,12))
    plt.title("基于聚类的随机特征生成")
    for i in range(feature_num):  # 每次聚类为样本添加一个特征
        y_pred = KMeans(n_clusters = cluster_num[i], \
                        random_state = 0).fit_predict(dataset)  # 随机生成聚类并且预测标签
        if (i < 9):  # 绘制聚类图像
            plt.subplot("33{}".format(i+1))
            plt.scatter(list(dataset["asin"]), list(dataset["overall"]), c = y_pred)
        dataset["feature{}".format(i+1)] = y_pred  # 添加标签
    return dataset


def RMSE(predictions):
    '''
    # prediction : DataFrame, 预测结果
    ----------------------------
    # result : float, 均方根结果
    function : 对于预测结果进行均方根的评估
    '''
    res = 0
    for i in range(len(predictions)):
        res += pow(predictions["target"][i] - predictions["predict"][i], 2)
    return np.sqrt(res / len(predictions))

        
if __name__ == "__main__":
    df = getDF(r'C:\Users\samgao1999\Desktop\推荐算法\reviews_Digital_Music_5.json.gz')
    df = df[0:3000]  # 选取3000个数据
    data = transform_to_int(df[["asin", "overall"]])
    data = create_features(50, data)
    forest, forest_features = random_forest(data, max_tree_num = 50)
    plt.figure(figsize=(64,48))
    plt.title("随机森林")
    for j in range(9):  # 画决策树示意图
        plt.subplot("33"+str(j))
        plot_tree(forest[j])
    forest_predictions = []
    for i, dtc in enumerate(forest):
        forest_predictions.append(dtc.predict(data[forest_features[i]]))  # 随机森林进行预测
    predictions = []
    for i in range(len(data)):  # 对测试集进行预测
        forest_res = []
        for tree_res in forest_predictions:
            forest_res.append(tree_res[i])
        predictions.append(np.mean(forest_res))  # 使用均值作为预测值
    test_result = pd.DataFrame({"target":data["overall"], "predict":predictions})
    res = RMSE(test_result)  # 评估模型
    print("RMSE:{}".format(res))
    
    