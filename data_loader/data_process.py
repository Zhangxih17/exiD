import numpy as np
import pandas as pd
import os
import sys
# from config import DefaultConfig
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import utm
import xml.etree.ElementTree as ET
import torch

max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4


def read_osm(file_name):
  '''解析osm并导出csv
  '''
  file_path = 'data/maps/' + file_name + '.osm'
  G = ox.graph_from_xml(file_path, simplify=False, retain_all=True)
  gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
  gdf_nodes.to_csv('data/gdf_nodes_' + file_name.split('.')[0] + '.csv',encoding='utf-8')
  gdf_edges.to_csv('data/gdf_edges_' + file_name.split('.')[0] + '.csv', encoding='utf-8')

  return gdf_nodes, gdf_edges


def get_traj(file_id, map_id,  nums):
  '''解析csv并导出轨迹片段 并解析车-车道对应
  args:
    file_id: recording
    map_id: osm地图编号
    start_id: 起始序号
    nums: 截取片段数量
  return:
    agent_lane_dict, key: track_id, value: lane_id
  '''
  tracks_path = 'data/tracks/' + file_id + '_tracks.csv'
  trackmeta_path = 'data/tracks/' + file_id + '_tracksMeta.csv'
  total_data = pd.read_csv(tracks_path)
  data=pd.DataFrame(total_data[['frame', 'trackId', 'laneletId', 'xCenter', 'yCenter', 'xVelocity', 'yVelocity',\
              'width', 'length', 'heading','leadId', 'rearId', \
                    'leftLeadId', 'rightLeadId', 'leftRearId', 'rightRearId']])
  del total_data
  type_data = pd.read_csv(trackmeta_path)
  type_data = type_data[['recordingId', 'trackId', 'class']]
  type_data = type_data[type_data['recordingId']==int(file_id)]
  data = pd.merge(data, type_data, on='trackId', how='left')

  # print(agent_lane_dict[1])

  i = 1
  row = 0
  while i <= nums:
    #每50帧截取一个csv
    df = data[(data['frame'] >= row) & (data['frame'] < row+50)].reset_index()
    ids_50 = list(set(list(df[df['frame'] == row]['trackId'].unique()))\
                  & set(list(df[df['frame'] == (row+49)]['trackId'].unique())))
    print(ids_50)  #50帧内一直存在的车辆
    if len(ids_50) == 0: break
    if len(ids_50) > num_agents:
      mid = int(len(ids_50) / 2)
      # print(mid-(num_agents/2), mid+(num_agents/2))
      agent_id = set(ids_50[int(mid-(num_agents/2)):int(mid+(num_agents/2))])  #选预测目标
      print('agent: ', agent_id)

      for j in range(len(df)):
        if df['trackId'][j] in agent_id:
          df['class'][j] = 'agent'

      # print(track_df.head(2))
      df.sort_values(by=['frame', 'trackId'],ascending=True)
      df.to_csv('data/train/' + str(map_id) + '/' + str(file_id) + '/' + str(i) + '.csv', index=False)
      i += 1

    row += 100


if __name__=='__main__':
  # for i in range(3):
  num_tracks = 200  #自定义划分数据集数
  get_traj('01', '0', num_tracks)


  