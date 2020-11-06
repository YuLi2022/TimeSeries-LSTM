# -*- encoding: utf-8 -*-
'''
@File    :   LSTM时间序列预测.py
@Time    :   2020/10/18 
@Author  :   Yu Li 
@describe:   使用LSTM进行时间序列预测 
'''

from operator import mod
from pandas.core.algorithms import mode
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  


# 数据集合目标函数值赋值，其中dateset为数据，look_back为以几行数据为特征数目 
def createDataset(dataset, look_back):
    dataX = [] 
    dataY = [] 
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:i+look_back]) 
        dataY.append(dataset[i+look_back]) 
    dataX = torch.tensor(dataX) 
    dataX = dataX.reshape(-1, 1, look_back) 
    dataY = torch.tensor(dataY) 
    dataY = dataY.reshape(-1, 1, 1) 
    return dataX, dataY 

# 划分训练集和测试集 
# 由于是时间序列数据，不适合这样随机打乱 
def splitData(data,rate=0.7):
    # 默认训练集比例为0.7 
    dataX, dataY = data  
    nSample = dataX.shape[0] 
    nTrain = int(nSample*rate) 
    # shuffledIndices = torch.randperm(nSample) 
    trainData = (dataX[:nTrain], dataY[:nTrain]) 
    testData = (dataX[nTrain:], dataY[nTrain:]) 
    return trainData, testData 

# 定义模型 
class LstmModel(nn.Module):
    def __init__(self, inputSize=5, hiddenSize=6):
        super().__init__() 
        # LSTM层-> 两个LSTM单元叠加 
        self.lstm = nn.LSTM(input_size=inputSize, \
            hidden_size=hiddenSize, num_layers=2) 
        self.output = nn.Linear(6,1)  # 线性输出 
    
    def forward(self,x):
        # x: input->(time_step, batch, input_size) 
        x1, _ = self.lstm(x)  
        # x1: output->(time_step, batch, output_size) 
        a, b, c = x1.shape 
        out = self.output(x1.view(-1,c))  # 只有三维数据转化为二维才能作为输入 
        # 重新将结果转化为三维 
        out = out.view(a,b,-1) 
        return out 


# 训练函数 
def training_loop(nEpochs, model, optimizer, lossFn, trainData, testData=None): 
    trainX, trainY = trainData 
    if testData is not None:
        testX, testY = testData     
    for epoch in range(1, nEpochs+1): 
        optimizer.zero_grad()  # 梯度清0  
        trainP = model(trainX) 
        loss = lossFn(trainP, trainY) 
        loss.backward()  # 反向传播 
        optimizer.step() 
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}") 
    return model 


# 加载飞行数据 
flight_data = pd.read_csv('flights.csv') 
# 数据归一化 
maxPassenger = flight_data['passengers'].max() 
minPassenger = flight_data['passengers'].min() 
flight_data['passengers'] = (flight_data['passengers'] - minPassenger) \
     / (maxPassenger - minPassenger)

dataset = flight_data['passengers'].values.tolist() 
data = createDataset(dataset=dataset, look_back=3)   # 划分数据集
# 获取训练集和测试集，用80%的数据来训练拟合，20%的数据来预测
rate = 0.8 
trainData, testData = splitData(data, rate=rate)  

# 定义模型 
lstm = LstmModel(inputSize=3) # inputSize与look_back保持一致 
# 使用优化器Adam比SGD更好 
optimizer = optim.Adam(lstm.parameters(), lr=0.1) 
loss_func = nn.MSELoss() 

# 训练模型 
lstm = training_loop(
            nEpochs=1000,
            model= lstm, 
            optimizer=optimizer, 
            lossFn=loss_func,
            trainData=trainData)

dataX, dataY = data  # 原始数据 -> (time_step, batch, input_size) 
dataY = dataY.view(-1).data.numpy()  # 展开为1维 
dataY = dataY * (maxPassenger - minPassenger) + minPassenger 
dataP = lstm(dataX)  # 进行拟合 
dataP = dataP.view(-1).data.numpy()  # 展开为1维 
dataP = dataP * (maxPassenger - minPassenger) + minPassenger 

nTrain = int(dataX.shape[0] * rate)  # 拟合的数量 
nData = dataX.shape[0]  # 预测的数量

# 绘制对比图 
plt.rcParams['font.sans-serif'] = 'KaiTi'  # 正常显示中文
fig = plt.figure(dpi=400) 
ax = fig.add_subplot(111) 
ax.plot(dataY, color='blue', label="实际值") 
ax.plot(np.arange(nTrain), dataP[:nTrain], color='green',\
    linestyle='--', label = '拟合值') 
ax.plot(np.arange(nTrain, nData), dataP[nTrain:], \
     linestyle='--', color = 'red', label='预测值')
ax.legend()
fig.savefig('test.png', dpi=400) 