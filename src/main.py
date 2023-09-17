import numpy as np
import pandas as pd
import os
import sys

from data_loader.dataset import *
from config import DefaultConfig

max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4
#特殊值处理
MAX_V = 100



if __name__=='__main__':
    opt = DefaultConfig()
    map_path = 'data/maps/'
    
    # 生成训练数据
    train_data = MTRDataset(opt.train_data_root, opt.map_root, train = True)
