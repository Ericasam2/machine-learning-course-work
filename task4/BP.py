# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:35:37 2021

@author: samgao1999
"""

import pandas as pd
import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
import pylab 
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置matplotlib的文字格式，使其显示中文
pylab.mpl.rcParams['axes.unicode_minus'] = False

class neuralnode:
    '''
    # neural_input: array shape=(n, 1), 神经元的输入量
    # weights: array shape=(1, n), 神经元的与之前节点连接的权重
    # sum: float, 输入矩阵与权重矩阵的乘积
    # threshold: float, 神经元的阈值
    # output: float, 神经元的输出
    ---------------------------------
    神经元的模型
    '''
    def __init__(self, neural_input, weights, threshold):
        '''
        神经元模型参数初始化
        '''
        self.input = neural_input  # array: shape=(n, 1)
        self.weights = weights  # array: shape=(1, n)
        self.sum = np.dot(self.weights, self.input)[0][0]  # float
        self.threshold = threshold
        self.output = self.sigmoid(self.sum - self.threshold)  # float
      
    def update(self):
        '''
        神经元模型参数更新
        '''
        self.input = self.input  # array: shape=(n, 1)
        self.weights = self.weights  # array: shape=(1, n)
        self.sum = np.dot(self.weights, self.input)[0][0]  # float
        self.threshold = self.threshold
        self.output = self.sigmoid(self.sum - self.threshold)  # float
    
    def sigmoid(self, num):
        '''
        # num: float, 数值
        -----------------
        # result: float, 对数值进行sigmoid
        sigmoid函数，作为激活函数
        '''
        return 1 / (1 + np.exp(-num))


class NeuralNetwork:
    '''
    # input_layer: dict {"feature":[node, node ...],...}, 神经网络的输入层
    # hiden_layer: dict {index:[node, node ...],...}, 神经网络的隐含层
    # output_layer: dict {"feature":[node, node ...],...}, 神经网络的输出层
    # error: float, 神经网路的损失值loss
    # learning_rate: float, 学习率n
    ----------------------------------
    构建神经网络
    '''
    step = 0
    error_list = []
    def __init__(self, input_layer, hiden_layer, output_layer, learning_rate):
        '''
        神经网络模型初始化
        '''
        self.input_layer = input_layer
        self.hiden_layer = hiden_layer
        self.output_layer = output_layer
        self.error = 0
        self.n = learning_rate
    
    def update(self):
        '''
        神经网络参数更新
        '''
        self.input_layer = self.input_layer
        self.hiden_layer = self.hiden_layer
        self.output_layer = self.output_layer
        self.error = self.error
        
    def get_layer_output(self, layer):
        '''
        # layer: dict {"feature":[node, node ...],...}, 神经网络的某一层 
        -------------------------------------------------------------
        # result: ndarray shape=[n,1], 神经网络某一层n个神经元的输出向量
        获取上一层网络的输出以作为下一层网络的输入
        '''
        n_input = np.zeros((1, len(layer)))
        index = 0
        for i in layer:
            node = layer[i]
            n_input[0][index] = node.output
            index = index + 1
        return n_input.T
    
    def update_all_layers(self, f_data):
        '''
        # f_data: dataframe, 输入的新样本
        -------------------------------
        # result: void
        根据新样本更新神经网路模型
        '''
        for i in list(f_data.columns[:]):
            input_node = self.input_layer[i]
            input_node.input = np.array([[f_data[i].iloc[0]]])
            input_node.update()      
        for i in self.hiden_layer:
            hiden_node = self.hiden_layer[i]
            hiden_node.input = self.get_layer_output(self.input_layer)
            hiden_node.update()      
        for i in self.output_layer:
            output_node = self.output_layer[i]
            output_node.input = self.get_layer_output(self.hiden_layer)
            output_node.update()
        self.update()
        
    def optimize_output_layer(self, y):
        '''
        # y: int, 标签
        ---------------
        # result: void
        用于更新网络输出层的权重和阈值
        '''
        n = self.n
        self.error = 0
        # print(y)
        for index in self.output_layer:
            if (index == 0):  # 针对二分类
                y_r = 1 - y
            else:
                y_r = y
            output_node = self.output_layer[index]
            y_e = output_node.output
            delta_w = np.zeros([1, len(self.hiden_layer)])
            for i in range(len(self.hiden_layer)):
                delta_w[0][i] = n * y_e * (1 - y_e) * (y_r - y_e) * self.hiden_layer[i].output
            # print("ye:{}, yr:{}".format(y_e, y_r))
            self.error = self.error + (y_r - y_e)**2
            delta_thre = -n * y_e * (1 - y_e) * (y_r - y_e)
            output_node.weights = output_node.weights + delta_w
            output_node.threshold = output_node.threshold + delta_thre
            output_node.update()
            self.update()
        
    def optimize_hiden_layer(self, y):
        '''
        # y: int, 标签
        ---------------
        # result: void
        用于更新网络隐含层的权重和阈值
        '''
        n = self.n
        output_w = np.zeros([1, len(self.output_layer)])
        output_y = np.zeros([len(self.output_layer), 1])
        for index in self.hiden_layer:
            hiden_node = self.hiden_layer[index]
            y_h = hiden_node.output
            delta_w = np.zeros(hiden_node.weights.shape) 
            index_output = 0
            for j in self.output_layer:
                if (j == 0):  # 针对二分类
                    y_r = 1 - y
                else:
                    y_r = y
                output_node = self.output_layer[j]
                y_e = output_node.output
                output_w[0][index_output] = self.output_layer[index_output].weights[0][index]
                output_y[index_output][0] = y_e * (1 - y_e) * (y_r - y_e)
                index_output = index_output + 1
            for i in range(delta_w.shape[1]):
                delta_w[0][i] = n * y_h * (1 - y_h) * np.dot(output_w, output_y)[0][0] * hiden_node.input[i][0]
            delta_thre = -n * y_h * (1 - y_h) * np.dot(output_w, output_y)[0][0]
            hiden_node.weights = hiden_node.weights + delta_w
            hiden_node.threshold = hiden_node.threshold + delta_thre 
            hiden_node.update()
            self.update()
        
    def optimize_all_layers(self, f_data, t_data):
        '''
        # f_data: dataframe, 特征数据集
        # t_data: dataframe, 目标数据集
        -------------------------------
        # result: void
        根据数据集优化神经网路中所有层
        '''
        while (1):
            #print(get_layer_output(self.input_layer))
            #print(get_layer_output(self.hiden_layer))
            
            for i in range(len(f_data)):
                self.update_all_layers(f_data[i:i+1])
                # p = t_data["target"].loc[i]
                self.optimize_output_layer(t_data["target"].loc[i])
                self.optimize_hiden_layer(t_data["target"].loc[i])
                self.update()
                # print(get_layer_output(self.output_layer))
                print(self.error)
                self.error_list.append(self.error)
                self.step = self.step + 1
            if (self.error < 0.01):
                self.update()
                break
    
    def predict(self, f_data):
        '''
        # f_data: dataframe, 样本特征
        -------------------------------
        # result: anytype, 根据样本预测的结果
        根据样本和目前的神经网络，对结果进行预测
        '''
        res = None
        res_value = 0
        self.update_all_layers(f_data)
        for i in self.output_layer:
            output_node = output_layer[i]
            if (output_node.output > res_value):
                res_value = output_node.output
                res = i
        return res
    
    def fit(self, test_f_data, test_t_data):
        '''
        # test_f_data: dataframe, 测试特征数据集
        # test_t_data: dataframe, 测试目标数据集
        -------------------------------
        # result: float,预测的准确率
        对于所有的样本进行预测，并且得到模型的准确率
        '''
        tp = 0
        for i in range(len(test_f_data)):
            f = test_f_data.iloc[[i]]
            t = test_t_data["target"].loc[[i]].iloc[0]
            p = self.predict(f)
            # print(p)
            if (t == p):
                tp = tp + 1
        return tp / len(test_f_data)
    
    def show_optimize(self):
        steps = list(range(self.step))
        error_list = self.error_list
        plt.figure(figsize = (48,32))
        plt.scatter(steps, error_list)
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.title("loss - steps")
        plt.show()
        

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


def generate_layers(f_data, t_data, num_hiden):
    '''
    # f_data: dataframe, 特征数据集
    # t_data: dataframe, 目标数据集
    # num_hiden: 隐含神经元的数量
    -------------------------------
    # result: input_layer, hiden_layer, output_layer
    初始化神经网络所有层
    '''
    input_layer = {}
    hiden_layer = {}
    output_layer = {}
    for i in list(f_data.columns[:]):
        input_layer[i] = neuralnode(np.array([[f_data[i].loc[0]]]), np.array([[1]]), 0)
        
    hiden_input = get_layer_output(input_layer) # 上一层是输入层，不是隐含层
    init_threshold = np.random.random((1, num_hiden))
    for i in range(num_hiden):
        init_weights = np.random.random((1, len(input_layer)))
        hiden_layer[i] = neuralnode(hiden_input, init_weights, init_threshold[0][i])
    
    output_input = get_layer_output(hiden_layer)
    init_threshold = np.random.random((1, len(t_data["target"].unique())))
    for i in list(t_data["target"].unique()):
        init_weights = np.random.random((1, num_hiden))
        output_layer[i] = neuralnode(output_input, init_weights, init_threshold[0][i])
    
    return input_layer, hiden_layer, output_layer


def get_layer_output(layer):
    '''
    # layer: dict {"feature":[node, node ...],...}, 神经网络的某一层 
    -------------------------------------------------------------
    # result: ndarray shape=[n,1], 神经网络某一层n个神经元的输出向量
    获取上一层网络的输出以作为下一层网络的输入
    '''
    n_input = np.zeros((1, len(layer)))
    index = 0
    for i in layer:
        node = layer[i]
        n_input[0][index] = node.output
        index = index + 1
    return n_input.T


def node_info(node):
    info = "input:{}.\noutput:{}.".format(node.input, node.output)
    return info


def create_graph(G, net, pos = {}, x = 0, y = 0, layer = 1):
    index_i = 0
    for i in net.input_layer:
        index_j = 0
        layer = 0
        input_node = net.input_layer[i]
        pos[node_info(input_node)] = (layer*5, index_i*2)
        index_i = index_i + 1
        for j in net.hiden_layer:
            index_k = 0
            layer = 1
            hiden_node = net.hiden_layer[j]
            pos[node_info(hiden_node)] = (layer*5, index_j*2)
            G.add_edge(node_info(input_node), node_info(hiden_node))
            index_j =index_j + 1
            for k in net.output_layer:
                layer = 2
                output_node = net.output_layer[k]
                pos[node_info(output_node)] = (layer*5, index_k*2)
                G.add_edge(node_info(hiden_node), node_info(output_node))
                index_k = index_k + 1
    return (G, pos)


def draw(net, title):   # 以某个节点为根画图
    graph = nx.DiGraph()
    graph, pos = create_graph(graph, net)
    fig, ax = plt.subplots(figsize=(48, 32))  # 比例可以根据树的深度适当调节
    plt.title("{}".format(str(title)))
    nx.draw_networkx(graph, pos, ax=ax, node_size=300)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\samgao1999\Desktop\机器学习\task4\watermelona3.txt")
    data = df.replace('?', np.nan)
    data = data.dropna()  #去除有缺失值的数据行
    data = transform_to_int(data)
    
    t_data = pd.DataFrame(data["target"])
    f_data = data[{"色泽","根蒂","敲声","纹理","脐部","触感","密度","含糖率"}]
    
    input_layer, hiden_layer, output_layer = generate_layers(f_data[0:1], t_data, 20)
    BP_model = NeuralNetwork(input_layer, hiden_layer, output_layer, 0.1)
    BP_model.optimize_all_layers(f_data, t_data)
    res = BP_model.fit(f_data, t_data)   
    print("the precision is: {}".format(res))
    BP_model.show_optimize()
    draw(BP_model, "BP")
    

    
    # model = (input_layer, hiden_layer, output_layer)
    # res = fit(model, f_data, t_data)

    
    