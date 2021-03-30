"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-03-30
"""
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GATModel(torch.nn.Module):
    def __init__(self, num_features=1433, num_classes=7):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        # print(x.size())   # torch.Size([2708, 1433])
        # print(edge_index.size())   # torch.Size([2, 10556])
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
