
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence, Union
from data_loader.data_process import *
import torch
from torch.utils.data import DataLoader,Dataset

max_nodes = 20
max_lanes = 8
max_frames = 50
max_objects = 10
num_agents = 4

@lru_cache(128)
def _read_csv(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """A caching CSV reader

    Args:
        path: Path to the csv file
        *args, **kwargs: optional arguments to be used while data loading

    Returns:
        pandas DataFrame containing the loaded csv
    """
    return pd.read_csv(path, *args, **kwargs)


class ArgoverseForecastingLoader:
    def __init__(self, root_dir: Union[str, Path]):
        """Initialization function for the class.

        Args:
            root_dir: Path to the folder having sequence csv files
        """
        self.counter: int = 0

        root_dir = Path(root_dir)
        self.seq_list: Sequence[Path] = [(root_dir / x).absolute() for x in os.listdir(root_dir)]

        self.current_seq: Path = self.seq_list[self.counter]

    @property
    def track_id_list(self) -> List[int]:
        """Get the track ids in the current sequence.

        Returns:
            list of track ids in the current sequence
        """
        _track_id_list: List[int] = np.unique(self.seq_df["trackId"].values).tolist()
        return _track_id_list

    @property
    def num_tracks(self) -> int:
        """Get the number of tracks in the current sequence.

        Returns:
            number of tracks in the current sequence
        """
        return len(self.track_id_list)

    @property
    def seq_df(self) -> pd.DataFrame:
        """Get the dataframe for the current sequence.

        Returns:
            pandas DataFrame for the current sequence
        """
        return _read_csv(self.current_seq)

    @property
    def agent_traj(self) -> np.ndarray:
        """Get the trajectory for the track of type 'AGENT' in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the agent trajectory
        """
        #test
        # print(self.seq_df.columns.values)
        agent_id = self.seq_df[self.seq_df["class"] == "agent"]["trackId"].unique()
        agent_x, agent_y = [], []
        for id in list(agent_id):
          agent_x.append(self.seq_df[(self.seq_df["class"] == "agent") \
                        & (self.seq_df["trackId"] == int(id))]["xCenter"])
          agent_y.append(self.seq_df[(self.seq_df["class"] == "agent") \
                        & (self.seq_df["trackId"] == int(id))]["yCenter"])
        # print('agent_x: ', len(agent_x), len(agent_x[0]))
        agent_traj = np.column_stack((agent_x, agent_y))
        # print('agent_traj: ', agent_traj.shape)
        return agent_traj # [4,50,50]

    @property
    def agent_info(self) -> np.ndarray:
        """Get the trajectory for the track of type 'AGENT' in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the agent trajectory
        """
        agent_id = self.seq_df[self.seq_df["class"] == "agent"]["trackId"].unique()
        agent_x, agent_y = [], []
        vel_x, vel_y = [], []
        heading = []
        for id in list(agent_id):
          temp_df = self.seq_df[(self.seq_df["class"] == "agent") \
                        & (self.seq_df["trackId"] == int(id))]
          # print('agent columns ', temp_df.columns.values)
          agent_x.append(temp_df["xCenter"])
          agent_y.append(temp_df["yCenter"])
          vel_x.append(temp_df["xVelocity"])
          vel_y.append(temp_df["yVelocity"])
          heading.append(temp_df["heading"])

        agent_info = np.column_stack((agent_x, agent_y, vel_x, vel_y, heading))
        # print('agent_info: ', agent_info.shape, agent_info[0,:])
        return agent_info # [4,50,5] agents, frames, features

    @property
    def obj_info(self) -> np.ndarray:
        """Get the trajectory for the track of type 'OTHER' in the current sequence.

        Returns:
            numpy array of shape (seq_len x 2) for the agent trajectory
        """
        obj_id = self.seq_df[self.seq_df["class"] != "agent"]["trackId"].unique()
        agent_x, agent_y = [], []
        vel_x, vel_y = [], []
        heading = []
        start_frame = min(self.seq_df["frame"])
        end_frame = max(self.seq_df["frame"])
        # print('start - end ', start_frame, end_frame)
        for id in list(obj_id):
          temp_df = self.seq_df[(self.seq_df["class"] != "agent") \
                        & (self.seq_df["trackId"] == int(id))].reset_index()
          # print(temp_df["xCenter"][0:5])
          # 序列缺失值填补
          df_start = max(0, min(temp_df["frame"]) - start_frame)
          df_end = min(max(temp_df["frame"]) - start_frame + 1, max_frames)
          # print('df ', df_start, df_end, len(temp_df["xCenter"]))
          temp_x, temp_y, temp_vx, temp_vy, temp_h = [0]*max_frames,[0]*max_frames,[0]*max_frames,[0]*max_frames,[0]*max_frames
          # print( temp_df["xCenter"][0])
          for i in range(df_start, df_end):
            temp_x[i] = temp_df["xCenter"][i-df_start]
            temp_y[i] = temp_df["yCenter"][i-df_start]
            temp_vx[i] = temp_df["xVelocity"][i-df_start]
            temp_vy[i] = temp_df["yVelocity"][i-df_start]
            temp_h[i] = temp_df["heading"][i-df_start]
          # print('temp x ', temp_x)
          agent_x.append(temp_x)
          agent_y.append(temp_y)
          vel_x.append(temp_vx)
          vel_y.append(temp_vy)
          heading.append(temp_h)

        obj_info = np.column_stack((agent_x, agent_y, vel_x, vel_y, heading))
        # print('obj_info ', obj_info.shape)
        return obj_info # [4,50,5] objs, frames, features

    def __iter__(self) -> "ArgoverseForecastingLoader":
        """Iterator for enumerating over sequences in the root_dir specified.

        Returns:
            Data Loader object for the first sequence in the data
        """
        self.counter = 0
        return self

    def __next__(self) -> "ArgoverseForecastingLoader":
        """Get the Data Loader object for the next sequence in the data.

        Returns:
            Data Loader object for the next sequence in the data
        """
        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_seq = self.seq_list[self.counter]
            self.counter += 1
            return self

    def __len__(self) -> int:
        """Get the number of sequences in the data

        Returns:
            Number of sequences in the data
        """
        return len(self.seq_list)

    def __str__(self) -> str:
        """Decorator that returns a string storing some stats of the current sequence

        Returns:
            A string storing some stats of the current sequence
        """
        return f"""Seq : {self.current_seq}
        ----------------------
        || City: {self.city}
        || # Tracks: {len(self.track_id_list)}
        ----------------------"""

    def __getitem__(self, key: int) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the sequence corresponding to the given index.

        Args:
            key: index of the element

        Returns:
            Data Loader object for the given index
        """

        self.counter = key
        self.current_seq = self.seq_list[self.counter]
        return self

    def get(self, seq_id: Union[Path, str]) -> "ArgoverseForecastingLoader":
        """Get the DataLoader object for the given sequence path.

        Args:
            seq_id: Fully qualified path to the sequence

        Returns:
            Data Loader object for the given sequence path
        """
        self.current_seq = Path(seq_id).absolute()
        return self


def get_lane_centerlines(argoverse_data, lane_centerlines_dict, agent_lane_dict, left_id_dict, right_id_dict):
  '''
  根据车辆位置信息和地图，获取周围的车道信息
  Args:
    argoverse_data: afl, 获取车辆轨迹, 包含车道id, traj_xy
  return:
    lane_centerlines: list [[]]
  '''
  agent_obs_traj = argoverse_data.agent_traj
  track_id = int(argoverse_data.seq_df[argoverse_data.seq_df["class"] == "agent"]["trackId"].unique()[0])
  lane_centerlines_id = agent_lane_dict[track_id]
  # for id in data['laneletId']:
  #   if len(id) < 6:
  #     lane_centerlines_id.add(id)
  #   else:
  #     ids = id.split(';')
  #     lane_centerlines_id.add(id[0])
  #     lane_centerlines_id.add(id[1])

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


class MTRDataset(torch_data.Dataset):
    '''
    数据集
    '''
    def __init__(self, root, train = True,test = False):
        '''
        根据路径获得数据，并根据训练、验证、测试划分数据
        train_data 和 test_data路径分开
        '''
        self.test = test
        self.train = train
        self.afl = ArgoverseForecastingLoader(root)
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
        # agent_data = pd.read_csv(self.root_dir)
        lane_centerlines = get_lane_centerlines(self.afl[self.start+index], lane_centerlines_dict)
        # print('lane dim ', len(lane_centerlines[0]), len(lane_centerlines[0][0]))
        agent_obs_traj = self.afl[index].agent_traj # num_agents: 4
        center_obs_info = self.afl[index].agent_info #4,250 [x,y,vx,vy,h]
        # print(center_obs_info.shape)
        # print('center_obs_info ', center_obs_info[:,0])
        obj_obs_info = self.afl[index].obj_info
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





