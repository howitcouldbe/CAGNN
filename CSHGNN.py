import random

import numpy as np
from torch.nn.parameter import Parameter
from utils import R_CB_Xi_create2,  w_single_node2, w_r_bc_S2
import torch
import torch.nn as nn
import torch.nn.functional as F


class EGCN(nn.Module):
    def __init__(self, num_feature, batch_size, multi_head=16, device='cuda:0'):
        # 初始化权重矩阵
        super(EGCN, self).__init__()
        self.device = device  # 在什么设备上跑
        self.net_SingleHGCN = SingleHGCN(num_feature=num_feature)
        self.GRCU_layers = []
        self.Linear = nn.Linear(8 * 4, 32, device='cuda:0')
        self.Elu = nn.ELU()
        self.batch_size = batch_size
        self.tongji = 0

        for i in range(1, 3):  # 多少层，就多少次循环
            multi_head = multi_head / 4
            grcu_i = GRCU(multi_head=multi_head,num_feature=num_feature,batch_size=batch_size)
            # print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))  # 将每一层的RNN存起来备用
        for i, GRCU_layer in enumerate(self.GRCU_layers):
            self.add_module('col_layer_{}'.format(i), GRCU_layer)

    def forward(self, zip):
        Nodes_list, theta = zip
        flag1 = 0
        flag2 = 0
        X_list,E_list,H_list = [],[],[]
        for node_data in Nodes_list:
            X, E, H = self.net_SingleHGCN((node_data.reshape(self.batch_size, -1),theta))  # 单层超图卷积网络
            X_list.append(X)
            E_list.append(E)
            H_list.append(H)
        for i, unit in enumerate(self.GRCU_layers):
            # unit就是GRCU
            # if i == 0:
            X_list = unit(X_list,E_list,H_list, i + 1,flag1,flag2)  # ,nodes_mask_list)
            if i == 0:
                emb_list = []
                for X in X_list:
                    X = self.Linear(X)
                    X = self.Elu(X)
                    emb_list.append(X)
                X_list = emb_list

        out = X_list[-1]
        return out

class GRCU(nn.Module):
    # 这里就是竖直方向传递
    def __init__(self, multi_head,num_feature,batch_size):
        super(GRCU, self).__init__()
        self.multi_head = multi_head
        self.batch_size = batch_size
        self.evolve_weights_1 = mat_GRU_cell_1([16, 8])  # 多头注意力的参数
        self.evolve_weights_2 = mat_GRU_cell_2([16, 2])  # 单层注意力的参数

        # self.layer1 = nn.Linear(8,8)
        # self.layer2 = nn.Linear(2,2)

        self.hist_param_layer1 = [] #第一层历史参数
        self.hist_param_layer2 = [] #第二层历史参数
        self.time1 = 0   #记录批次，
        self.time2 = 0
        self.shgcn = 0
        self.GCN_weights =0
        # 多头注意力机制
        self.attentions = [DA_HGAN(in_features=32, multi_head=multi_head) for _ in
                           range(int(multi_head))]  # 多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, X_list,E_list,H_list, layer,flag1,flag2):
        emb_list = []
        for t,X in enumerate(X_list):
            E = E_list[t]
            H = H_list[t]
            # first,用初始权重参数去更新α_x和α_e
            if layer == 1:
                # if self.time1 == 0:
                #     self.time1+=1
                if flag1 == 0:
                    flag1=1
                    GCN_weights_x = torch.concat([att.alpha_x for att in self.attentions], dim=1)
                    GCN_weights_e = torch.concat([att.alpha_e for att in self.attentions], dim=1)
                    GCN_weights = torch.concat([GCN_weights_x, GCN_weights_e], dim=1).detach()
                # else:
                #     GCN_weights = self.hist_param_layer1.pop(0)
                # print(GCN_weights)
                GCN_weights = self.evolve_weights_1(GCN_weights)  # ,node_embs,mask_list[t]),这一步是获得RNN更新后的参数矩阵
                # GCN_weights = self.layer1(GCN_weights)
                # self.hist_param_layer1.append(GCN_weights)
            else:
                # if self.time2 == 0:
                #     self.time2+=1
                if flag2 == 0:
                    flag2 = 1
                    GCN_weights_x = torch.concat([att.alpha_x for att in self.attentions], dim=1)
                    GCN_weights_e = torch.concat([att.alpha_e for att in self.attentions], dim=1)
                    GCN_weights2 = torch.concat([GCN_weights_x, GCN_weights_e], dim=1).detach()
                # else:
                #     GCN_weights2 = self.hist_param_layer2.pop(0)
                GCN_weights2 = self.evolve_weights_2(GCN_weights2)
                # self.hist_param_layer2.append(GCN_weights2)
                # GCN_weights2 = self.layer2(GCN_weights2)

            #second,用更新后的α_x和α_e去更新矩阵
            node_embs = X  # 获得了Xi和Ek矩阵，准备扔进DAHGNN
            edge_embs = E
            node_embs = torch.cat([att(node_embs, edge_embs, H, GCN_weights[:, i].reshape(16, -1),
                                       GCN_weights[:, 4 + i].reshape(16, -1))
                                   if self.multi_head != 1
                                   else att(node_embs, edge_embs, H, GCN_weights2[:, i].reshape(16, -1),
                                            GCN_weights2[:, 1 + i].reshape(16, -1))
                                   for i, att in enumerate(self.attentions)], dim=1
                                  )  # 将每个head得到的表示进行拼接
            # 将需要更新的alpha系列参数拼在一起，这样只需要一个RNN
            emb_list.append(node_embs)
        return emb_list

class mat_GRU_cell_1(nn.Module):
    # 横向传递
    def __init__(self, GCN_weights):
        super(mat_GRU_cell_1, self).__init__()
        # self.activation = nn.Sigmoid()
        # the k here should be in_feats which is actually the rows

        self.update = mat_GRU_gate_1(GCN_weights[0],  # 16
                                     GCN_weights[1],  # 8
                                     torch.nn.Sigmoid())

        self.reset = mat_GRU_gate_1(GCN_weights[0],  #
                                    GCN_weights[1],
                                    torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate_1(GCN_weights[0],  #
                                     GCN_weights[1],
                                     torch.nn.Tanh())

    def forward(self, prev_Q):  # ,prev_Z,mask):
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q  # new_Q就是更新好后的参数，也就是我们要的a_x


class mat_GRU_cell_2(nn.Module):
    # 横向传递
    def __init__(self, GCN_weights):
        super(mat_GRU_cell_2, self).__init__()
        # self.activation = nn.Sigmoid()
        # the k here should be in_feats which is actually the rows

        self.update = mat_GRU_gate_2(GCN_weights[0],  # 16
                                     GCN_weights[1],  # 8
                                     torch.nn.Sigmoid())

        self.reset = mat_GRU_gate_2(GCN_weights[0],  #
                                    GCN_weights[1],
                                    torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate_2(GCN_weights[0],  #
                                     GCN_weights[1],
                                     torch.nn.Tanh())

    def forward(self, prev_Q):  # ,prev_Z,mask):
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q  # new_Q就是更新好后的参数，也就是我们要的a_x


class mat_GRU_gate_1(nn.Module):
    def __init__(self, rows, cols, activation):
        '''
        rows 162
        clos 100
        activation:sigmoid
        '''
        super(mat_GRU_gate_1, self).__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W1 = Parameter(torch.zeros(size=(rows, rows), requires_grad=True, device='cuda:0'))  # W.shape 16,16

        self.U = Parameter(torch.zeros(size=(rows, rows), requires_grad=True, device='cuda:0'))  # U.shape 16,16

        self.bias = Parameter(torch.zeros(rows, cols, requires_grad=True, device='cuda:0'))  # bias.shape,16,8

    def forward(self, x, hidden):
        out = self.activation(self.W1.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out


class mat_GRU_gate_2(nn.Module):
    def __init__(self, rows, cols, activation):
        '''
        rows 162
        clos 100
        activation:sigmoid
        '''
        super(mat_GRU_gate_2, self).__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W1 = Parameter(torch.zeros(size=(rows, rows), requires_grad=True, device='cuda:0'))  # W.shape 16,16

        self.U = Parameter(torch.zeros(size=(rows, rows), requires_grad=True, device='cuda:0'))  # U.shape 16,16

        self.bias = Parameter(torch.zeros(rows, cols, requires_grad=True, device='cuda:0'))  # bias.shape,16,8

    def forward(self, x, hidden):
        out = self.activation(self.W1.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out



class SingleHGCN(nn.Module):
    def __init__(self, num_feature):
        super(SingleHGCN, self).__init__()
        self.theta = Parameter(
            torch.normal(0, 0.01, size=(num_feature, 32), requires_grad=True, device=torch.device('cuda:0')))
        # self.layer1 = nn.Linear(30,32)
        # self.layer2 = nn.Linear(32,32)
        nn.init.xavier_uniform_(self.theta)

    def forward(self, zip):
        X, theta = zip
        H= self.euclideanDistance((X,theta))
        X, E = self.singleHGCN(X, H)
        # print(X.shape,E.shape)
        # X = self.layer1(X)
        # E = self.layer2(X)
        return X, E, H

    # 欧式距离计算出超边关系矩阵
    def euclideanDistance(self, zip):
        X,theta = zip
        A = torch.sum(X ** 2, dim=1).reshape(-1, 1)
        B = torch.sum(X ** 2, dim=1).reshape(1, -1)
        C = torch.mm(X, X.T)
        dist_matric = torch.sqrt(torch.abs(A + B - 2 * C) * (
                    torch.ones(size=(A.shape[0], A.shape[0]), device='cuda:0') - torch.eye(A.shape[0], device='cuda:0')))   #距离矩阵
        # radius = torch.mean(dist_matric,dim=0).reshape(-1,1)
        radius = torch.mean(dist_matric)
        H = torch.where(dist_matric < radius/theta, float(1), float(0))   #每个节点周围小于其半径的的节点，就算做在同一超边内
        return H.T

    # 单层卷积操作
    def singleHGCN(self, X, H):
        '''
        单层超图卷积网络，用于学习节点特征和超边特征的低纬嵌入
        :param X:初始节点的特征矩阵
        :param H:超图关联矩阵
        :param De:超边度的对角矩阵
        :param Dv:顶点度的对角矩阵
        实验过程中默认节点数n等于超边数m
        :return:
        '''
        Dv = torch.diag(torch.pow(torch.sum(H, dim=1), -1 / 2))
        De = torch.diag(torch.pow(torch.sum(H, dim=0), -1 / 2))
        X = torch.mm(torch.mm(torch.mm(torch.mm(De,H.T),Dv),X),self.theta)   #低维节点特征嵌入X
        E = torch.mm(torch.mm(torch.mm(Dv,H),De),X)           #超边特征嵌入E
        return X, E

class DA_HGAN(nn.Module):
    def __init__(self, in_features, alpha=0.2, multi_head=1):
        '''
        :param sigma: 相似度的阈值
        :param alpha: LeakyRelu的参数，默认0.2
        '''
        super(DA_HGAN, self).__init__()
        self.W = Parameter(
            torch.ones(size=(in_features, int(8)), requires_grad=True, device='cuda:0'))
        self.alpha_x = torch.zeros(size=(2 * int(8), 1), requires_grad=False, device=torch.device('cuda:0'))
        self.alpha_e = torch.zeros(size=(2 * int(8), 1), requires_grad=False, device=torch.device('cuda:0'))
        self.net_ELU = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, X, E, H, alpha_x, alpha_e):
        return self.DA_HGAN(X, E, H, alpha_x, alpha_e)

    def DA_HGAN(self, Xi, Ek, H, alpha_x, alph_e):
        '''
        :param X:低维节点特征嵌入 shape:nxd
        :param E:超边特征嵌入     shape:mxd
        :sigma 是预定义的阈值 超参数
        :return:

        密度感知超图注意网络主要由两部分组成：密度感知注意顶点聚合和密度感知注意超边聚合。
        '''
        # rho_xi = self.node_density(Xi,H,0.4)  #节点密度
        relative_closure_S= R_CB_Xi_create2(H)
        w_r_bc = w_r_bc_S2(relative_closure_S)
        rho_xi = w_single_node2(w_r_bc, relative_closure_S)

        rho_hyper_edge = self.hyper_edge_density(rho_xi, H)  # 超边密度
        E = self.attention(Xi, Ek, rho_xi, node=True, H=H, X=Xi, alpha_x=alpha_x,
                           alpha_e=alph_e)  # 节点的注意力值，node为true表示计算的是节点注意力值
        X = self.attention(Ek, Xi, rho_hyper_edge, node=False, H=H, X=E, alpha_x=alpha_x, alpha_e=alph_e)  # 超边的注意力值
        return X

    # 通过余弦相似度计算密度
    def node_density(self, X, H, sigma: float):
        neiji = torch.mm(X, X.T)  # 内积
        mochang = torch.sqrt(torch.sum(X * X, dim=1).reshape(1, -1) * (torch.sum(X * X, dim=1).reshape(-1, 1)))  # 模长
        cosim = neiji / mochang  # 余弦相似度矩阵

        # 矩阵元素小于sigma的全部置为0，对角线元素也置0，因为不需要自己和自己的相似度
        cosim = torch.where(cosim > sigma, cosim, torch.zeros_like(cosim)) \
                * (torch.ones_like(cosim, device='cuda:0') - torch.eye(cosim.shape[0], device='cuda:0'))
        # 节点和超边的关系矩阵H的内积的每一行，可以表示每个节点的所有相邻节点
        xx = torch.where(torch.mm(H, H.T) > 0, float(1), float(0)) \
             * (torch.ones_like(cosim, device='cuda:0') - torch.eye(cosim.shape[0], device='cuda:0'))
        # 将每个节点与相邻节点的相似度相加，就是该节点的密度
        rho = torch.sum(cosim * xx, dim=1).reshape(-1, 1)
        return rho

    # 计算节点与边的注意力值,然后计算标准化密度，最后得出注意力权重
    def attention(self, Xi: torch.Tensor, Ek: torch.Tensor, rho: torch.Tensor, node: bool, H, X, alpha_x, alpha_e):
        '''
        :param Xi:节点嵌入矩阵
        :param Ek: 超边嵌入矩阵
        :param rho:节点密度
        :return:注意力权重
        '''
        # 将WX和WE拼接

        a_input = self._prepare_attentional_mechanism_input(
            Xi, Ek, node)  # 实现论文中的特征拼接操作 Wh_i||Wh_j ，得到一个shape = (N ， N, 2 * out_features)的新特征矩阵
        if node:
            a_x = self.leakyrelu(torch.matmul(a_input, alpha_x).squeeze(2))
        else:
            a_x = self.leakyrelu(torch.matmul(a_input, alpha_e).squeeze(2))
        rho_tilde = torch.tensor(
            [(enum - torch.min(rho)) / (torch.max(rho) - torch.min(rho)) * torch.max(a_x) for enum in rho],
            device=torch.device('cuda:0')).reshape(-1, 1)
        # rho_tilde = rho*(torch.max(a_x)/torch.max(rho))
        a_x_tilde = a_x + rho_tilde
        # a_x_tilde = a_x

        zero_vec = -1e12 * torch.ones_like(a_x_tilde)  # 将没有连接的边置为负无穷
        if node:
            attention = torch.where(H > 0, a_x_tilde,
                                    zero_vec)  # [N, N]   #node为true，计算coe_x_e，a_x_tilde代表的是a_x_e,是节点-超边的形状
        else:
            attention = torch.where(H.T > 0, a_x_tilde, zero_vec)  ##node为false，计算coe_e_x，a_x_tilde其实是a_e_x,是超边-节点的形状
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        # attention = F.softmax(attention, dim=0)
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        # attention = F.dropout(attention, 0.2, training=self.training)  # dropout，防止过拟合
        if node:
            h_prime = self.net_ELU(torch.matmul(attention.T, torch.mm(X, self.W.to(
                'cuda:0'))))  # [N, N].[N, out_features] => [N, out_features]
        else:
            h_prime = self.net_ELU(
                torch.matmul(attention.T, X.to('cuda:0')))  # [N, N].[N, out_features] => [N, out_features]
        return h_prime

    def hyper_edge_density(self, rho_x: torch.Tensor, H: torch.Tensor):
        '''
        计算超边密度
        :param rho_x:  节点密度
        :param H:       关系矩阵
        :return:
        '''
        rho_hyper_edge = torch.sum(H * rho_x, dim=0)
        return rho_hyper_edge.reshape(-1, 1)

    def _prepare_attentional_mechanism_input(self, Xi, Ek, node: bool):
        WX = torch.mm(Xi, self.W.to('cuda:0'))  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        WE = torch.mm(Ek, self.W.to('cuda:0'))
        WXN = WX.size()[0]  # number of nodes
        WEN = WE.size()[0]
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        WX_repeated_in_chunks = WX.repeat_interleave(WEN, dim=0)
        # repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None)
        # 参数说明：
        # self: 传入的数据为tensor
        # repeats: 复制的份数
        # dim: 要复制的维度，可设定为0/1/2.....
        WE_repeated_alternating = WE.repeat(WXN, 1)
        # repeat方法可以对 Wh 张量中的单维度和非单维度进行复制操作，并且会真正的复制数据保存到内存中
        # repeat(N, 1)表示dim=0维度的数据复制N份，dim=1维度的数据保持不变

        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([WX_repeated_in_chunks, WE_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(WXN, WEN, 2 * self.W.shape[1])


