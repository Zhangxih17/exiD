import numpy as np
import pandas as pd
import os
import sys

from data_loader.data_process import *
from data_loader.config import DefaultConfig

max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4
#特殊值处理
MAX_V = 100

traj_map = {
    '0_cologne_butzweiler': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', \
                             '11', '12', '13', '14', '15', '16', '17', '18'],
    '1_cologne_deutz': []
}



if __name__=='__main__':
    opt = DefaultConfig()
    map_path = 'data/maps/'
    
    for file_name in os.listdir(map_path): #遍历地图
        file_name = file_name.split('.')[0]
        print(file_name)
        # gdf_nodes, gdf_edges = read_osm(file_name)

        # 提取轨迹片段
        agent_lane_dict = dict()
        for i in range(len(traj_map[file_name])): #遍历地图对应的多个tracks
            file_id = traj_map[file_name][i]
            agent_lane_dict_i = get_traj(id, i*300+1, 300, opt)
            agent_lane_dict.update(agent_lane_dict_i)

        lane_centerlines_dict, left_id_dict, right_id_dict = get_lane_dict(file_name)

        # 生成训练数据

