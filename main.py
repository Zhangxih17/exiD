import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./')

from data_loader.dataset import MTRDataset, collate, get_lane_centerlines, compose_graph
from config import DefaultConfig
import torch.nn as nn
import torch.nn.functional as F
import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model.MTR import *


max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4
#特殊值处理
MAX_V = 100


def train(**kwargs):
    '''
    训练
    '''
    opt.parse(kwargs)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    train_data = MTRDataset(opt.train_data_root,train = True)
    val_data = MTRDataset(opt.train_data_root,train = False)
    train_dataloader = DataLoader(train_data,opt.batch_size,shuffle = True,num_workers = opt.num_workers,collate_fn=collate)
    val_dataloader = DataLoader(val_data,opt.batch_size,shuffle = False,num_workers = opt.num_workers,collate_fn=collate)

    criterion = torch.nn.MSELoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = opt.weight_decay)
    TrainingLoss = []
    ValLoss = []
    epochs = []
    preloss = 1e100
    for epoch in range(opt.max_epoch):
        print('#################### epoch ', epoch, ' ####################')
        losses = 0
        num = 0
        for ii,(data,label) in enumerate(train_dataloader):
            print('############### ii ', ii, ' ###############')
            print('data agent', len(data['Agent']), '\n', data['Agent'][0])
            #input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                data['Agent'] = data['Agent'].to(torch.device('cuda:0'))
                for g in data['Map']:
                    g = g.to(torch.device('cuda:0'))
                data['Agentfeature'] = Variable(data['Agentfeature']).cuda()
                for feature in data['Mapfeature']:
                    feature = Variable(feature).cuda()
                target = target.cuda()
            if len(data['Map']) == 0:
                continue
            optimizer.zero_grad()
            # score = model(data['Agent'],data['Map'],data['Agentfeature'],data['Mapfeature'],data['Mapmask'])
            score = model(data['Agent'],data['Map'],data['Agentfeature'], \
                  data['Mapfeature'],data['Mapmask'], data['Obj'],data['Objfeature'])
            # loss = criterion(score.double().reshape(-1,60),target.double())
            print('score original size ', score.size())
            # score = torch.flatten(score.double().reshape(-1,150), end_dim=1)
            score = score.double().reshape(-1,150)
            print('score - target size', score.size(), target.size())

            loss = criterion(score,target.double())
            print(score[0], '\n', target[0])
            loss.backward()
            optimizer.step()
            losses += loss.data
            num += 1
            if num % 10 == 0:
                print(num)
        model.save()
        epochs.append(epoch)
        TrainingLoss.append(losses/num)
        print('#################### Training: ',losses/num, ' ####################')
        ValLoss.append(val(model,val_dataloader))
        if losses/num > preloss:
            lr = lr*opt.lr_decay

        preloss = losses/num

    plt.figure()
    plt.title('loss during training')  #标题
    plt.plot(epochs, TrainingLoss, label="TrainingLoss")
    plt.plot(epochs, ValLoss, label="ValLoss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('/content/gdrive/MyDrive/exitD/data/plt/' + 'MTR_' + '%s.jpg' \
                % datetime.datetime.now().strftime("%m%d_%H%M"))

    model.save('new.pth')

@torch.no_grad()
def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    '''
    model.eval()
    losses = 0
    num = 0
    criterion = torch.nn.MSELoss()
    for ii,(data,label) in enumerate(dataloader):
        #input = Variable(data)
        target = Variable(label)
        if opt.use_gpu:
            data['Agent'] = data['Agent'].to(torch.device('cuda:0'))
            for g in data['Map']:
                g = g.to(torch.device('cuda:0'))
            data['Agentfeature'] = data['Agentfeature'].cuda()
            for feature in data['Mapfeature']:
                feature = Variable(feature).cuda()
            target = target.cuda()
        if len(data['Map']) == 0:
            continue
        score = model(data['Agent'],data['Map'],data['Agentfeature'], \
                  data['Mapfeature'],data['Mapmask'],data['Obj'],data['Objfeature'])
        loss = criterion(score.double().reshape(-1,150),target.double())
        losses += loss.data
        num += 1
        if num % 10 == 0:
            print(num)
    model.train()
    print('#################### Eval:',losses/num, ' ####################')
    return losses/num   



if __name__=='__main__':
    opt = DefaultConfig()
    map_path = 'data/maps/'
    
    # 生成训练数据
    train_data = MTRDataset(opt.train_data_root, opt.map_root, train = True)

    #模型结构
    model = MTR(11,64,150)  #30s * 5

    #训练
    train()
