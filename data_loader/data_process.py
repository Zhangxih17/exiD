import numpy as np
import pandas as pd
import os
import sys

import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import utm
import xml.etree.ElementTree as ET


def read_osm(file_name):
  '''解析osm并导出csv
  '''
  file_path = 'data/maps/' + file_name + '.osm'
  G = ox.graph_from_xml(file_path, simplify=False, retain_all=True)
  gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
  gdf_nodes.to_csv('/content/gdrive/MyDrive/exitD/data/gdf_nodes_' + file_name.split('.')[0] + '.csv',encoding='utf-8')
  gdf_edges.to_csv('/content/gdrive/MyDrive/exitD/data/gdf_edges_' + file_name.split('.')[0] + '.csv', encoding='utf-8')

  return gdf_nodes, gdf_edges


def get_traj(file_id, start_id, nums, opt):
  '''解析csv并导出轨迹片段 并解析车-车道对应
  args:
    file_id: recording
    nums: 截取片段数量
  return:
    agent_lane_dict, key: track_id, value: lane_id
  '''
  tracks_path = '/content/gdrive/MyDrive/exitD/data/' + file_id + '_tracks.csv'
  trackmeta_path = '/content/gdrive/MyDrive/exitD/data/' + file_id + '_tracksMeta.csv'
  total_data = pd.read_csv(tracks_path)
  # data=pd.DataFrame(total_data[['frame', 'trackId', 'laneletId', 'xCenter', 'yCenter', 'lonVelocity', 'latVelocity',\
  #             'width', 'length', 'heading','leadId', 'rearId', \
  #                   'leftLeadId', 'rightLeadId', 'leftRearId', 'rightRearId']])
  data=pd.DataFrame(total_data[['frame', 'trackId', 'laneletId', 'xCenter', 'yCenter', 'xVelocity', 'yVelocity',\
              'width', 'length', 'heading','leadId', 'rearId', \
                    'leftLeadId', 'rightLeadId', 'leftRearId', 'rightRearId']])
  del total_data
  type_data = pd.read_csv(trackmeta_path)
  type_data = type_data[['recordingId', 'trackId', 'class']]
  type_data = type_data[type_data['recordingId']==int(file_id)]
  data = pd.merge(data, type_data, on='trackId', how='left')

  agent_lane_dict = dict()
  for id in list(data['trackId'].unique()):
    lanes = list(data[data['trackId']==1]['laneletId'].unique())
    agent_lane_dict[id] = list()
    for lane in lanes:
      if len(lane) > 5:
        agent_lane_dict[id].append(lane.split(';')[0])
        agent_lane_dict[id].append(lane.split(';')[1])
      else:
        agent_lane_dict[id].append(lane)

  print(agent_lane_dict[1])

  # print(data.head(3))

  i = 1
  row = 0
  while i <= nums:
    #每50帧截取一个csv
    df = data[(data['frame'] >= row) & (data['frame'] < row+50)].reset_index()
    ids_50 = list(set(list(df[df['frame'] == row]['trackId'].unique()))\
                  & set(list(df[df['frame'] == (row+49)]['trackId'].unique())))
    print(ids_50)  #50帧内一直存在的车辆
    if len(ids_50) > opt.num_agents:
      mid = int(len(ids_50) / 2)
      # print(mid-(opt.num_agents/2), mid+(opt.num_agents/2))
      agent_id = set(ids_50[int(mid-(opt.num_agents/2)):int(mid+(opt.num_agents/2))])  #选预测目标
      print('agent: ', agent_id)

      for j in range(len(df)):
        if df['trackId'][j] in agent_id:
          df['class'][j] = 'agent'

      # print(track_df.head(2))
      df.sort_values(by=['frame', 'trackId'],ascending=True)
      df.to_csv('/content/gdrive/MyDrive/exitD/data/train_mtr/' + str(start_id+i) + '.csv', index=False)
      i += 1

    row += 100

  return agent_lane_dict


def get_lane_dict(file_name):
  file_path = '/content/gdrive/MyDrive/exitD/lanelet2/' + file_name
  G = ox.graph_from_xml(file_path, simplify=False, retain_all=True)
  gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

  # 车道对应的中心线向量
  lane_centerlines_dict = dict()  #key(lane_id), value(lane_props)

  i = 0
  id_list = []
  for id in gdf_edges['osmid']:
    id_list.append(id)
    if id not in lane_centerlines_dict:
      lane_centerlines_dict[id] = []
    i += 1

  i = 0
  reverse_list = []
  for r in gdf_edges['reversed']:
    reverse_list.append(r)
    i += 1

  i = 0
  for line in gdf_edges['geometry']:
    line = str(line)
    # print(str(line).split(' (')[2])
    x0 = float(line.split('(')[1].split(' ')[0])
    y0 = float(line.split(' ')[2].split(',')[0])
    # long, lat -> UTM
    xy = utm.from_latlon(latitude=y0, longitude=x0)
    x0 = xy[0] - 352146.6
    y0 = xy[1] - 5651141.9

    id = id_list[i]
    reverse = reverse_list[i]
    if not reverse:
      lane_centerlines_dict[id].append([x0, y0])
    i += 1

  print(lane_centerlines_dict[1500])

  # 车道对应左右车道线
  tree = ET.parse(file_path)
  # 获取根节点
  root = tree.getroot()
  ref_df = []  #id, left, right

  for r in root.iter('relation'):
    # print(r.attrib)
    relation = [r.attrib['id'], "", ""]
    for mem in r[0:2]:
      # print(mem.attrib)
      # print(len(mem.attrib))
      if mem.attrib['role'] == 'left':
        relation[1] = mem.attrib['ref']
      elif mem.attrib['role'] == 'right':
        relation[2] = mem.attrib['ref']
    ref_df.append(relation)

  #建立 lanlet_id 与左右车道映射
  left_id_dict = dict()
  right_id_dict = dict()
  for i in range(len(ref_df)):
    left_id_dict[ref_df[i][0]] = ref_df[i][1]
    right_id_dict[ref_df[i][0]] = ref_df[i][2]

  return lane_centerlines_dict, left_id_dict, right_id_dict


def collate(samples):
    datas,labels = map(list,zip(*samples))
    AgentGraph = [data['Agent'] for data in datas]
    Center = [data['centerAgent'] for data in datas]
    Mask = [data['Mapmask'] for data in datas]
    # batched_graph = dgl.batch(AgentGraph)

    map_set = []
    feature_set = []
    for index in range(max_lanes):
      map_batch = []
      for data in datas:
        map_batch.append(data['Map'][index])
      mgraph = dgl.batch(map_batch)
      map_set.append(mgraph)
      feature_set.append(mgraph.ndata['v_feature'])

    agent_set = []
    agent_feature_set = []
    for index in range(4): #agents num
      agent_batch = []
      for data in datas:
        agent_batch.append(data['Agent'][index])
      agraph = dgl.batch(agent_batch)
      agent_set.append(agraph)
      agent_feature_set.append(agraph.ndata['v_feature'])

    obj_set = []
    obj_feature_set = []
    for index in range(max_objects): #20
      obj_batch = []
      for data in datas:
        obj_batch.append(data['Obj'][index])
      ograph = dgl.batch(obj_batch)
      obj_set.append(ograph)
      obj_feature_set.append(ograph.ndata['v_feature'])

    new_data = {}
    new_data['Map'] = map_set
    new_data['Mapfeature'] = torch.stack(feature_set,dim = 0)
    # new_data['Agent'] = batched_graph
    # new_data['Agentfeature'] = batched_graph.ndata['v_feature']
    new_data['Agent'] = agent_set
    new_data['Agentfeature'] = torch.stack(agent_feature_set,dim = 0)
    new_data['Obj'] = obj_set
    new_data['Objfeature'] = torch.stack(obj_feature_set,dim = 0)
    new_data['centerAgent'] = Center
    new_data['Mapmask'] = torch.stack(Mask,dim = 0)
    #new_label = []
    #for l in labels:
        #new_label += list(l.flatten())
    # return new_data,torch.stack(labels,dim = 0).reshape(-1,60)
    return new_data,torch.stack(labels,dim = 0).reshape(-1,150) #30*5






