import random

import torchvision.datasets
from torchvision import transforms
from torch.utils import data
import torch
import matplotlib.pyplot as plt
import sys
import pandas as pd
# from ogb.nodeproppred import NodePropPredDataset
import numpy as np
import sys
import csv
import os


class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def R_CB_Xi_create2(H):
    return H.T

def euclideanDistance2(X,Y):
    A = torch.sum(X ** 2, dim=1).reshape(-1,1)
    B = torch.sum(Y ** 2, dim=2).reshape(-1,1,X.shape[0])
    C = torch.matmul(X,torch.transpose(Y,1,2))
    dist_matric = torch.sqrt(torch.abs(A + B - 2 * C))
    return dist_matric

def w_r_bc_S2(relative_closure_S):
    H = relative_closure_S
    O = H
    X = O
    # X = X.reshape(O.shape[0]*O.shape[0], 1, O.shape[0])
    H = H.reshape(O.shape[0], O.shape[0], 1)

    temp = H * O
    dist = euclideanDistance2(X,temp)
    num_zero = (torch.ones_like(torch.count_nonzero(dist, dim=1)) * O.shape[0] - torch.count_nonzero(
        torch.transpose(dist, 1, 2), dim=1)).T
    num_zero = torch.where(num_zero>1,1,num_zero)
    w_r_bc = 1 / O.shape[0] * torch.sum(num_zero / torch.sum(X, dim=1), dim=1)
    return w_r_bc

def w_single_node2(w_r_bc,relative_closure_S):
    w_r_bc = w_r_bc.to('cuda:0')
    w_node = torch.sum(w_r_bc.reshape(relative_closure_S.shape[0],-1)*relative_closure_S,dim=0)
    return w_node


#数据集
def load_data(data_name,batch_size,percent=0):

    def load_data_mnist(batch_size,resize=None,percent=percent):
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0,transforms.Resize(resize))
        trans =transforms.Compose(trans)
        mnist_train = torchvision.datasets.MNIST(
            root="dataset",train=True,transform=trans,download=True)
        if percent != 0:
            index = random.sample(range(mnist_train.shape[0]), int(mnist_train.shape[0] * percent))
            mnist_train[index, -1] = 1 - mnist_train[index, -1]
        return data.DataLoader(mnist_train,batch_size,shuffle=False,num_workers=0)

    def load_imdb(batch_size,percent):
        df = np.load('dataset/imdb.npy')
        # df = pd.read_csv(r'dataset/imdb.csv')
        # np.save('dataset/imdb',df.values[:43200,1:])
        if percent != 0:
            index = random.sample(range(df.shape[0]), int(df.shape[0] * percent))
            df[index, -1] = 1 - df[index, -1]

        return  data.DataLoader(df,batch_size,shuffle=False,num_workers=0)

    def load_dataset_FG2C2D(batch_size,percent):
        dataset = []
        with open(r'dataset/FG_2C_2D.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset.append([float(enum) for enum in line.strip().split(',')])

        dataset = torch.tensor(dataset).reshape(-1, 3)
        if percent != 0:
            index = random.sample(range(dataset.shape[0]), int(dataset.shape[0] * percent))
            dataset[index, -1] = 3 - dataset[index, -1]
        dataset[:,-1] = dataset[:,-1]-1
        return data.DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

    def load_dataset_GEARS2C2D(batch_size,percent):
        dataset = []
        with open('dataset/GEARS_2C_2D.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset.append([float(enum) for enum in line.strip().split(',')])

        dataset = torch.tensor(dataset).reshape(-1, 3)
        if percent != 0:
            index = random.sample(range(dataset.shape[0]), int(dataset.shape[0] * percent))
            dataset[index, -1] = 3 - dataset[index, -1]
        dataset[:, -1] = dataset[:, -1] - 1
        return data.DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

    def load_dataset_MG2C2D(batch_size,percent):
        dataset = []
        with open(r'dataset/MG_2C_2D.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset.append([float(enum) for enum in line.strip().split(',')])

        dataset = torch.tensor(dataset).reshape(-1, 3)
        print(dataset.shape,dataset.dtype)
        if percent != 0:
            index = random.sample(range(dataset.shape[0]), int(dataset.shape[0] * percent))
            dataset[index, -1] = 3 - dataset[index, -1]

        return data.DataLoader(dataset, batch_size, shuffle=False, num_workers=0)

    def load_dataset_CR4(batch_size,percent):
        dataset = []
        with open(r'dataset/4CR.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset.append([float(enum) for enum in line.strip().split(',')])

        dataset = torch.tensor(dataset).reshape(-1, 3)
        print(dataset.shape,dataset.dtype)
        if percent != 0:
            index = random.sample(range(dataset.shape[0]), int(dataset.shape[0] * percent))
            dataset[index, -1] = 5 - dataset[index, -1]
        dataset[:, -1] = dataset[:, -1] - 1
        return data.DataLoader(dataset, batch_size, shuffle=False  , num_workers=0)

    def load_electricity(batch_size,percent):
        dataset = []
        with open(r'dataset/multivariate-time-series-data-master/electricity/electricity.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset.append([float(enum) for enum in line.strip().split(',')])

        dataset = torch.tensor(dataset)
        dataset = dataset[:5000,:320]
        dataset = 1-(torch.max(dataset, dim=0)[0] - dataset) / (
                torch.max(dataset, dim=0)[0] - torch.min(dataset, dim=0)[0])

        data1 = []
        mean = []
        label = []
        # 样本
        window_size = 30
        for i in range(0,dataset.shape[0] - window_size + 1,window_size):   #移动时间窗，加入最新的，移除最旧的
            mean.append(torch.mean(dataset[i:window_size + i,0]))
            label.append((torch.mean(dataset[i:i+window_size,:],dim=0)>torch.mean(dataset[i+window_size:i+2*window_size,:])*1.0).numpy()) #下一个时刻下降的话标签为1，上升的话，标签为0
            data1.append(dataset[i:i+window_size,:].numpy())
        label = torch.tensor(np.array(label))*1.0
        label = label.reshape(label.shape[0],-1,320)
        data1  = torch.tensor(np.array(data1))

        data1 = torch.concat((data1,label),1)
        data1 = torch.transpose(data1,1,2).reshape(-1,window_size+1)

        # for i in range(5):
        #     plt.plot(range(20),dataset[i,:])
        # plt.show()
        if percent != 0:
            index = random.sample(range(data1.shape[0]), int(data1.shape[0] * percent))
            data1[index, -1] = 1 - data1[index, -1]
        return data.DataLoader(data1, batch_size, shuffle=False, num_workers=0)

    def load_traffic(batch_size,percent):
        dataset = []
        with open(
                r'dataset/multivariate-time-series-data-master/traffic/traffic.txt',
                'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset.append([float(enum) for enum in line.strip().split(',')])

        dataset = torch.tensor(dataset)[:5000, :400]
        dataset = 1 - (torch.max(dataset, dim=0)[0] - dataset) / (
                torch.max(dataset, dim=0)[0] - torch.min(dataset, dim=0)[0])

        data1 = []
        mean = []
        label = []
        # 样本
        window_size = 30
        for i in range(0, dataset.shape[0] - window_size + 1, window_size):
            mean.append(torch.mean(dataset[i:window_size + i, 0]))
            label.append((torch.mean(dataset[i:i + window_size, :], dim=0) > torch.mean(
                dataset[i + window_size:i + 2 * window_size, :]) * 1.0).numpy())  # 下一个时刻下降的话标签为1，上升的话，标签为0
            data1.append(dataset[i:i + window_size, :].numpy())
        label = torch.tensor(np.array(label)) * 1.0
        label = label.reshape(label.shape[0], -1, 400)
        data1 = torch.tensor(np.array(data1))

        data1 = torch.concat((data1, label), 1)
        data1 = torch.transpose(data1, 1, 2).reshape(-1, window_size + 1)
        if percent != 0:
            index = random.sample(range(data1.shape[0]), int(data1.shape[0] * percent))
            data1[index, -1] = 1 - data1[index, -1]
        return data.DataLoader(data1, batch_size, shuffle=False, num_workers=0)

    if data_name == 'mnist':
        load_data_mnist(batch_size=batch_size,percent=percent)
    elif data_name == 'imdb':
        load_imdb(batch_size=batch_size,percent=percent)
    elif data_name == 'CR4':
        load_dataset_CR4(batch_size,percent=percent)
    elif data_name == 'FG2C2D':
        load_dataset_FG2C2D(batch_size, percent=percent)
    elif data_name == 'MG2C2D':
        load_dataset_MG2C2D(batch_size, percent=percent)
    elif data_name == 'GEAR2C2D':
        load_dataset_GEARS2C2D(batch_size, percent=percent)
    elif data_name == 'electricity':
        load_electricity(batch_size, percent=percent)
    elif data_name == 'traffic':
        load_traffic(batch_size, percent=percent)

load_data("electricity",400)

# load_electricity(320)