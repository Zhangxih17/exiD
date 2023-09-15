import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import dgl
torch.set_default_tensor_type(torch.FloatTensor)


class MLP(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(MLP,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

    def forward(self,x):
        output = self.MLP(x)
        return output
    

import dgl
def gcn_reduce(nodes):
    #agg_feature = torch.max(nodes.mailbox['msg'], dim=1)
    return {'v_feature': torch.max(nodes.mailbox['msg'], dim=1)[0]} #rel agg

class GCNLayer(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(GCNLayer,self).__init__()

    def forward(self,g,inputs):
        gcn_message = dgl.function.copy_src('v_feature','msg') ###
        g.ndata['v_feature'] = inputs
        # g.send(g.edges(),gcn_message)
        # g.recv(g.nodes(),gcn_reduce)
        g.update_all(gcn_message,gcn_reduce)
        v_feature = g.ndata.pop('v_feature')
        return torch.cat([inputs,v_feature],dim = 1) #rel: concate


class SubNetwork(nn.Module):
    def __init__(self,in_feats,hidden_size,layernums):
        super(SubNetwork,self).__init__()
        self.encoder = []
        self.gcnlayer = []
        self.layernums = layernums
        input_size = in_feats
        for i in range(0,layernums):
            if i == 0:
                self.encoder.append(MLP(input_size, hidden_size))
            else:
                self.encoder.append(MLP(hidden_size*2, hidden_size))  #2
            self.gcnlayer.append(GCNLayer(hidden_size, hidden_size*2)) #2

    def forward(self,g,inputs):
        # print('##### subnetwork forward #####')
        g.ndata['v_feature'] = inputs
        g_batch = g
        for i in range(self.layernums):
            g_list = dgl.unbatch(g_batch)
            #进入MLP需要unbatch
            for subg in g_list:
                v_feature = self.encoder[i](subg.ndata['v_feature'])  #node encoder
                subg.ndata['v_feature'] = v_feature
            #进入gcn可以batch回来
            g_batch = dgl.batch(g_list)
            v_feature = self.gcnlayer[i](g_batch,g_batch.ndata['v_feature'])  #gcn
            # print('layer v_feature0: ', v_feature[0])
            g_batch.ndata['v_feature'] = v_feature
        g_list = dgl.unbatch(g_batch)
        v_feature = []
        for subg in g_list:
            v_feature.append(subg.ndata['v_feature'])
        # print('final v_feature0: ', v_feature[0])
        output = torch.stack(v_feature,0)
        # print('subnetwork output ', output.size())
        return output

