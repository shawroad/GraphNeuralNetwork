"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-19
"""
import torch
from torch import nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        '''
        :param input_dim: int 节点输入特征的维度
        :param output_dim: int 输出特征维度
        :param use_bias: bool 是否使用偏置
        '''
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        '''
        邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        :param adjacency: torch.sparse.FloatTensor 邻接矩阵
        :param input_feature: torch.Tensor 输入特征
        :return:
        '''
        support = torch.mm(input_feature, self.weight)   # 先对特征就行一个线性变化
        # print(support.size())   # torch.Size([2708, 16])
        output = torch.sparse.mm(adjacency, support)    #
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GCNNet(nn.Module):
    def __init__(self, input_dim=1433):
        # 之所以input_dim=1433 是因为节点的特征向量长度为1433
        super(GCNNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        # print(h.size())    # torch.Size([2708, 16])  (node_num, hidden_size)
        logits = self.gcn2(adjacency, h)
        return logits

