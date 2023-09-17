
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import dgl
from subNetwork import *
from GAT import *
torch.set_default_tensor_type(torch.FloatTensor)

max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4
#特殊值处理
MAX_V = 100


class MTR(nn.Module):
    '''
    VectorNet + GNN (4 agents)
    '''
    def __init__(self,in_dim,hidden_size,out_dim):
        super(MTR,self).__init__()
        self.subMapNetwork = SubNetwork(in_dim,hidden_size,3)
        self.subAgentNetwork = SubNetwork(in_dim,hidden_size,3)
        self.subObjNetwork = SubNetwork(in_dim,hidden_size,3)
        self.GlobalNetwork = SelfAttention(hidden_size*2,hidden_size*2) #GATLayer(hidden_size*2,hidden_size*2)
        self.MLP = [nn.Sequential(
            nn.Linear(hidden_size*2,hidden_size*2),
            #nn.LayerNorm(hidden_size),
            nn.Tanh(), #ReLU?
            nn.Linear(hidden_size*2,out_dim)
        ) for i in range(num_agents)]

    def forward(self,agent_set,map_set,agent_feature,map_feature,map_mask, obj_set,obj_feature):
        MapOutputs = []

        agent_feature = torch.where(torch.isnan(agent_feature),torch.full_like(agent_feature, 0), agent_feature)

        Globalfeature = torch.max(self.subAgentNetwork(agent_set[0],agent_feature[0]), dim=1)[0].unsqueeze(0)  #agent gnn

        Globalfeature = torch.where(torch.isnan(Globalfeature),torch.full_like(Globalfeature, 0), Globalfeature)

        for i,graph in enumerate(agent_set):  #concate map attn
            if i>0:
                Globalfeature = torch.cat((Globalfeature,torch.max(self.subAgentNetwork(graph,agent_feature[i]), dim=1)[0].unsqueeze(0)),0)

        max_mask = torch.max(torch.sum(map_mask,dim = 1),dim = 0)[0].int()
        # nodeN = 1 + max_mask
        for i,graph in enumerate(map_set):  #concate map attn
            if i >= max_mask:
                break
            Globalfeature = torch.cat((Globalfeature,torch.max(self.subMapNetwork(graph,map_feature[i]), dim=1)[0].unsqueeze(0)),0)

        #新增obj
        for i,graph in enumerate(obj_set):  #concate map attn
            Globalfeature = torch.cat((Globalfeature,torch.max(self.subObjNetwork(graph,obj_feature[i]), dim=1)[0].unsqueeze(0)),0)

        v_feature = []
        for i in range(Globalfeature.shape[1]):
            v_feature.append(self.GlobalNetwork(Globalfeature[:,i])[0])

        #输出4辆agents特征
        outputs = [self.MLP[i](torch.stack(v_feature,0)) for i in range(num_agents)]
        print('MTR outputs 0-1: \n', outputs[0], '\n', outputs[1])
        # return self.MLP(torch.stack(v_feature,0))
        return torch.stack(outputs,dim = 0) #4*50

    def save(self,name=None):
        if name is None:
            prefix = 'checkpoints/' + 'MTR' + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name

    def load(self,path):
        self.load_state_dict(torch.load(path))



