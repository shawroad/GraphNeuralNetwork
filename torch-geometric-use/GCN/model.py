"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-30
"""
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNModel(torch.nn.Module):
    def __init__(self, num_features=1433, num_classes=7):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True,)
        self.conv2 = GCNConv(16, num_classes, cached=True,)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

