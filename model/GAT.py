import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import dgl
from subNetwork import *
torch.set_default_tensor_type(torch.FloatTensor)


class SelfAttention(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(SelfAttention, self).__init__()
        self.wq = torch.nn.Parameter(torch.randn(in_dim,out_dim))
        self.wk = torch.nn.Parameter(torch.randn(in_dim,out_dim))
        self.wv = torch.nn.Parameter(torch.randn(in_dim,out_dim))

    def forward(self,input_feature):
        WQ = input_feature.mm(self.wq)
        WK = input_feature.mm(self.wk)
        WV = input_feature.mm(self.wv)
        QK = WQ.mm(WK.T)
        QK = F.softmax(QK,dim = 1)
        V = QK.mm(WV)
        return V


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = None #don't define the g before
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # gain = nn.init.calculate_gain('relu')
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  #aggregate
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h }

    def forward(self, g,h):
        self.g = g
        # equation (1) encoder polyline
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2) aggregate
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)  GNN
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

