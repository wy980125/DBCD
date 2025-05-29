import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch.nn import Linear

# 模型定义（关键修正！使用GCNConv）
class myGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_communities):
        super().__init__()

        #hidden_dim2 = 256
        self.conv1 = GCNConv(num_features, hidden_dim)

        #self.conv3 = GCNConv(512, 256)
        self.conv2 = GCNConv(hidden_dim,num_communities)



        #self.lp = Linear(hidden_dim2, num_communities)
    def forward(self, x, edge_index):
        # GCN生成软分配矩阵S
        h = F.relu(self.conv1(x, edge_index))
        #h = F.relu(self.conv3(h, edge_index))
        h = self.conv2(h, edge_index)

        #h = F.dropout(h, p=0.05)
        # h = h / h.sum()
        # h = torch.tanh(h) ** 2
        #
        # S = F.normalize(h)
        S = F.softmax(h, dim=1)
        # print("S shape:", S.shape)
        # print("s_adj_batch shape:", s_adj_batch.shape)
        # print("S.T shape:", S.T.shape)


        return S

class DGC(nn.Module):
    def __init__(self, in_dim, out_dim, base_model):
        super(DGC, self).__init__()
        if base_model == 'gcn':
            self.conv1 = GCNConv(in_dim, 512)
            self.conv2 = GCNConv(512, 256)
            # self.conv2 = GCNConv(256, out_dim)
            self.conv3 = GCNConv(256, out_dim)
            #self.conv4 = GCNConv(128, out_dim)

        elif base_model == 'gat':
            self.conv1 = GATConv(in_dim, 256)
            self.conv2 = GATConv(256, 128)
            self.conv3 = GATConv(128, out_dim)
        elif base_model == 'gin':
            self.conv1 = GINConv(nn.Linear(in_dim, 256))
            self.conv2 = GINConv(nn.Linear(256, 128))
            self.conv3 = GINConv(nn.Linear(128, out_dim))
        elif base_model == 'sage':
            self.conv1 = SAGEConv(in_dim, 256)
            self.conv2 = SAGEConv(256, 128)
            self.conv3 = SAGEConv(128, out_dim)

        self.mlp = Linear(out_dim, 32)
        self.mlp2 = Linear(in_dim, 32)

    def forward(self, x, edge_index):

        # x1 = x.clone()
        z = self.conv1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, training=self.training)

        z = self.conv2(z, edge_index)
        z = F.relu(z)
        z = F.dropout(z, training=self.training)

        z = self.conv3(z, edge_index)
        z = F.relu(z)
        #z = F.dropout(z, p=0.05)

        # z = self.conv4(z, edge_index)
        # z = F.relu(z)


        #z = self.mlp(z)
        # z = F.relu(z)
        # z = z / z.sum(dim=1, keepdim=True)
        z = z / z.sum()
        z = torch.tanh(z) ** 2
        # z = F.normalize(z, p=2, dim=1)
        z = F.normalize(z)
        # x1 = self.mlp2(x1)

        return z