# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:48:43 2021

@author: samgao1999
"""
import torch
import torchvision
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
import time
n_epochs = 1  # 训练轮数
batch_size_train = 64  # 训练步长
batch_size_test = 1000  # 测试步长
learning_rate = 0.01  # 学习率
momentum = 0.5
log_interval = 10
random_seed = 1

torch.cuda.manual_seed(random_seed)  # 设置GPU随机种子
class Net(nn.Module):
    '''
    nn.Module : Module类, 继承自定义层类来设计网络层
    -------------------------------------------
    function : 设计神经网络 
    
    '''
    def __init__(self):
        super(Net, self).__init__()  # 继承父类（自定义层）
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)  # 设置卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)  # 设置卷积层
        self.conv2_drop = nn.Dropout2d()  # 随机闲置一部分节点
        self.fc1 = nn.Linear(320, 150)# 设置线性层
        self.fc2 = nn.Linear(150, 50)# 设置线性层
        self.fc3 = nn.Linear(50, 10)# 设置线性层
        
    
    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 最大值池化，激活函数
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # 将原数组变换成(m,n)的数组
        x = F.relu(self.fc1(x))  # 设fc1层的激活函数
        x = F.dropout(x, training = self.training)
        x = F.relu(self.fc2(x))  # 设fc2层的激活函数
        x = F.dropout(x, training = self.training)
        x = self.fc3(x)
        return F.log_softmax(x)
    
    
def download_MNIST():
    '''
    function : 下载数据集
    '''
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./data/", train=True, download=True,
                                   transform = torchvision.transforms.Compose(
                                       [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size = batch_size_train, shuffle = True)  # 训练集获取
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./data/", train=False, download=True,
                                   transform = torchvision.transforms.Compose(
                                       [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size = batch_size_test, shuffle = True)  # 测试集获取
    
def sample_plot():
    '''
    function : 绘制数据集中的样本图像
    '''
    dataset_path = r"C:/Users/samgao1999/Desktop/exp4/data/"
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./data/", train=False, download=True,
                                    transform = torchvision.transforms.Compose(
                                        [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size = batch_size_test, shuffle = True)
    
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)
    
    fig = plt.figure()
    for i in range(6):  # 绘制图形
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()  # 自动调整子图参数
        plt.imshow(example_data[i][0], cmap="gray", interpolation = "none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def train(epoch):
    '''
    # epoch : int, 训练轮数
    -----------------------
    # result
        train_losses : list of float, 训练集的损失值列表
        train_counter : list of int, 训练过程的轮数
    function : 在设定的神经网络上，使用训练集来训练优化网络参数
    '''
    train_losses = []
    train_counter = []
    network.train()  # 神经网络类
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 将梯度置零
        output = network(data)  
        loss = F.nll_loss(output, target)  # 求交叉熵
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重值
        
        if batch_idx % log_interval == 0:  # 迭代
            print("Train Epoch: {} [{}/{} ({:.0f})%] \ tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), "./model.pth")
            torch.save(optimizer.state_dict(), "./optimizer.pth")
    return train_losses, train_counter


def test():
    '''
    function : 在训练集上进行模型功能验证
    '''
    test_losses = []
    # test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average = False).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()  # 统计正确的预测
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print("\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses
    
if __name__ == "__main__":
    # download_MNIST()
    # sample_plot()
    dataset_path = r"C:/Users/samgao1999/Desktop/exp4/data/"
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./data/", train=True, download=True,
                                   transform = torchvision.transforms.Compose(
                                       [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size = batch_size_train, shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./data/", train=False, download=True,
                                    transform = torchvision.transforms.Compose(
                                        [torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size = batch_size_test, shuffle = True)
    
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation = "none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)
    
    train_losses = []  # 训练集上的损失值列表
    train_counter = []  # 统计训练轮数
    test_losses = []  # 测试集上的损失值列表
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    # train(n_epochs)
    test_losses_tmp = test()  # 训练前进行一次测试
    test_losses.extend(test_losses_tmp)
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        train_losses_tmp, train_counter_tmp = train(epoch)
        train_losses.extend(train_losses_tmp)
        train_counter.extend(train_counter_tmp)
        test_losses_tmp = test()
        test_losses.extend(test_losses_tmp)
    end = time.time()
    print("总共耗时：{} min {}s".format(int(end - start)//60, int(end-start)%60))
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc = "upper right")
    plt.xlabel("Number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.show()
    
    
    
   