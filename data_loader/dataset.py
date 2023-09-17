
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence, Union
from data_process import *
import torch
import torch.utils.data as torch_data
from torch.utils.data import DataLoader,Dataset
import dgl
from exiD_loader import *

max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4


def get_lane_centerlines(argoverse_data, lane_centerlines_dict, agent_lane_dict, left_id_dict, right_id_dict):
  '''
  根据车辆位置信息和地图，获取周围的车道信息
  Args:
    argoverse_data: edl, 获取车辆轨迹, 包含车道id, traj_xy
  return:
    lane_centerlines: list [[]]
  '''
  agent_obs_traj = argoverse_data.agent_traj
  track_id = int(argoverse_data.seq_df[argoverse_data.seq_df["class"] == "agent"]["trackId"].unique()[0])
  lane_centerlines_id = agent_lane_dict[track_id]

  lane_centerlines = []
  for id in lane_centerlines_id:
    if id in left_id_dict:
      left_id = left_id_dict[id]
      lane_centerlines.append(lane_centerlines_dict[int(left_id)])
    if id in right_id_dict:
      right_id = right_id_dict[id]
      lane_centerlines.append(lane_centerlines_dict[int(right_id)])

  return lane_centerlines


def compose_graph(lane,label):
    '''
    输入的是单条车道向量
    把车道组织成图,并返回结点特征((x1,y1,vx1,vy1,h1),(x2,y2,vx1,vy1,h1),label)
    Args:
    lane: list, 车道中心线向量
    label: int, 车辆周围车道编号
    '''
    nodeN = lane.shape[0]-1
    lane_dim = len(lane[0])
    # print('nodeN max_nodes ', nodeN, max_nodes)
    features = torch.zeros(max_nodes, 11) #统一变量维度
    graph = dgl.DGLGraph()
    # graph = dgl.graph()
    graph.add_nodes(max_nodes)
    mask = []
    for i in range(max_nodes):
      if i < nodeN:
        for j in range(lane_dim):
          features[i][j] = lane[i][j]
          features[i][j+lane_dim] = lane[i+1][j]

        features[i][2*lane_dim] = label  #torch.tensor(list(lane[i])+list(lane[i+1])+[label])
        mask.append(1)
      else:
        mask.append(0)

    # print('compose_graph ', features.size())
    src = []
    dst = []
    for i in range(nodeN):
      for j in range(nodeN):
        if i != j:
          src.append(i)
          dst.append(j)
    graph.add_edges(src,dst)
    # print('graph ', graph.num_nodes(), features.size())
    graph.ndata['v_feature'] = features
    return graph,features


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


class MTRDataset(torch_data.Dataset):
    '''
    数据集
    '''
    def __init__(self, root, map_root, train = True,test = False):
        '''
        根据路径获得数据，并根据训练、验证、测试划分数据
        train_data 和 test_data路径分开
        '''
        self.test = test
        self.train = train
        self.edl = exiDLoader(root, map_root)
        # self.avm = ArgoverseMap()
        root_dir = Path(root)
        r = [(root_dir / x).absolute() for x in os.listdir(root_dir)]
        n = len(r)

        if self.test == True:
            self.start = 0
            self.end = n
        elif self.train:
            self.start = 0
            self.end = int(0.7*n)
        else:
            self.start = int(0.7*n)+1
            self.end = n

    def __getitem__(self,index):
        '''
        从csv创建图输入模型
        '''
        data = {}
        agent_lane_dict = self.edl[self.start+index].agent_lane_dict
        lane_centerlines_dict = self.edl[self.start+index].lane_centerlines_dict
        left_id_dict, right_id_dict = self.edl[self.start+index].left_right_dict
        lane_centerlines = get_lane_centerlines(self.edl[self.start+index], lane_centerlines_dict, agent_lane_dict, left_id_dict, right_id_dict)
        # print('lane dim ', len(lane_centerlines[0]), len(lane_centerlines[0][0]))
        agent_obs_traj = self.edl[index].agent_traj # num_agents: 4
        center_obs_info = self.edl[index].agent_info #4,250 [x,y,vx,vy,h]
        # print(center_obs_info.shape)
        # print('center_obs_info ', center_obs_info[:,0])
        obj_obs_info = self.edl[index].obj_info
        # print('obj_obs_info ', obj_obs_info.shape)
        data['centerAgent'] = [center_obs_info[1,19], center_obs_info[1,max_frames+19], \
                      center_obs_info[1,2*max_frames+19], center_obs_info[1,3*max_frames+19], \
                        center_obs_info[1,4*max_frames+19]]
        # print(data['centerAgent'])

        #把车道组织成向量和图
        map_set = []
        map_feature = []
        map_mask = []
        #进行norm
        x_min = min(min(center_obs_info[0,0:max_frames]), min(center_obs_info[3,0:max_frames]))
        x_max = max(max(center_obs_info[0,0:max_frames]), max(center_obs_info[3,0:max_frames]))
        y_min = min(min(center_obs_info[0,max_frames:2*max_frames]), min(center_obs_info[3,max_frames:2*max_frames]))
        y_max = max(max(center_obs_info[0,max_frames:2*max_frames]), max(center_obs_info[3,max_frames:2*max_frames]))
        vx_min = min(min(center_obs_info[0,2*max_frames:3*max_frames]), min(center_obs_info[3,2*max_frames:3*max_frames])) #近似
        vx_max = max(max(center_obs_info[0,2*max_frames:3*max_frames]), max(center_obs_info[3,2*max_frames:3*max_frames]))
        vy_min = min(min(center_obs_info[0,3*max_frames:4*max_frames]), min(center_obs_info[3,3*max_frames:4*max_frames]))
        vy_max = max(max(center_obs_info[0,3*max_frames:4*max_frames]), max(center_obs_info[3,3*max_frames:4*max_frames]))
        h_min = min(min(center_obs_info[0,4*max_frames:5*max_frames]), min(center_obs_info[3,4*max_frames:5*max_frames])) #近似
        h_max = max(max(center_obs_info[0,4*max_frames:5*max_frames]), max(center_obs_info[3,4*max_frames:5*max_frames]))
        # print('norm scale ', x_min, x_max, y_min, y_max)

        for lane in lane_centerlines:
          lane = np.array([[(lane[i][0]-data['centerAgent'][0])/(x_max-x_min), \
                    (lane[i][1]-data['centerAgent'][1])/(y_max-y_min)] for i in range(len(lane))]) #归一化
          graph,features = compose_graph(lane,len(map_set))
          map_set.append(graph)
          map_feature.append(features)
          map_mask.append(1)

        # print('map set ', len(map_set))
        if len(map_set) < max_lanes:
          while(len(map_set) < max_lanes):
            #形成空的图，保证大小一致
            lane = np.array([[0,0]])
            graph,features = compose_graph(lane,0)
            map_set.append(graph)
            map_feature.append(features)
            map_mask.append(0)
        else:
          raise Exception("the max lanes is not enough:",len(map_set))

        # print('agent set ', len(center_obs_info))
        center_graphs, center_features, center_traj_norm = [], [], []
        for traj in center_obs_info: #4,250
          # print('center_shape ', traj.shape)
          traj_norm = np.array([[(traj[i]-data['centerAgent'][0])/(x_max-x_min), \
                      (traj[i+max_frames]-data['centerAgent'][1])/(y_max-y_min) ,\
                      (traj[i+2*max_frames]-data['centerAgent'][2])/(vx_max-vx_min+0.1) ,\
                       (traj[i+3*max_frames]-data['centerAgent'][3])/(vy_max-vy_min+0.1) ,\
                      (traj[i+4*max_frames]-data['centerAgent'][4])/(h_max-h_min+0.1)] for i in range(max_frames)]) #归一化
          # print('traj_norm ', traj_norm.shape)
          center_traj_norm.append(traj_norm)
          graph, features = compose_graph(traj_norm[:20],len(center_graphs))
          center_graphs.append(graph)
          center_features.append(features)

        # print('obj set ', len(center_obs_info))
        obj_graphs, obj_features, obj_traj_norm = [], [], []
        #截取固定长度
        if obj_obs_info.shape[0] >= max_objects:
          mid = int(obj_obs_info.shape[0] / 2)
          for i in range(int(mid - max_objects/2), int(mid + max_objects/2)):
            traj = obj_obs_info[i]
            traj_norm = np.array([[(traj[i]-data['centerAgent'][0])/(x_max-x_min), \
                      (traj[i+max_frames]-data['centerAgent'][1])/(y_max-y_min) ,\
                      (traj[i+2*max_frames]-data['centerAgent'][2])/(vx_max-vx_min+0.1) ,\
                      (traj[i+3*max_frames]-data['centerAgent'][3])/(vy_max-vy_min+0.1) ,\
                      (traj[i+4*max_frames]-data['centerAgent'][4])/(h_max-h_min+0.1)] for i in range(max_frames)])
            # traj_norm = (traj - data['centerAgent'])/np.array([x_max-x_min,y_max-y_min,vx_max-vx_min,vy_max-vy_min,h_max-h_min]) #归一化
            obj_traj_norm.append(traj_norm)
            graph, features = compose_graph(traj_norm[:20],len(obj_graphs))
            obj_graphs.append(graph)
            obj_features.append(features)

        else: #填补空车
          while(len(obj_graphs) < max_objects):
            obj = np.array([[0,0,0,0,0]])
            graph,features = compose_graph(obj,0)
            obj_graphs.append(graph)
            obj_features.append(features)

        data['Map'] = map_set # 2dim
        data['Mapfeature'] = map_feature # 2dim
        data['Agent'] = center_graphs  # 2dim 4,250
        data['Agentfeature'] = center_features  # 2dim
        data['Obj'] = obj_graphs  # 2dim 20,250
        data['Objfeature'] = obj_features
        data['Mapmask'] = torch.Tensor(map_mask)
        label = []

        # print(len(center_traj_norm))
        for j in range(len(center_traj_norm)):
          # temp_label = []
          # print(len(center_traj_norm[j]))
          for i in range(19,49): #30*5
            # temp_label.append(torch.Tensor(center_traj_norm[j][i+1]-center_traj_norm[j][i]))
            label.append(torch.Tensor(center_traj_norm[j][i+1]-center_traj_norm[j][i]))
          # label.append(temp_label)

        label = torch.stack(label,dim = 0)
        # print('Agent shape ', len(data['Agent']), data['Agent'][0].num_nodes())
        return data,label.flatten()

    def __len__(self):
        return self.end - self.start


#test
train_data_root = 'data/train/0/00'
map_root = 'data/maps/0_cologne_butzweiler.osm'
train_data = MTRDataset(train_data_root, map_root, train = True)
print(train_data[0].agent_lane_dict)
