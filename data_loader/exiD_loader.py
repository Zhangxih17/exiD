
import pandas as pd
import numpy as np
import utm
import xml.etree.ElementTree as ET
import osmnx as ox
import geopandas as gpd

from functools import lru_cache
from pathlib import Path
from typing import Any, List, Sequence, Union
from data_loader.data_process import *
import torch
import torch.utils.data as torch_data
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


class exiDLoader:
    def __init__(self, root_dir: Union[str, Path], map_path: Union[str, Path]):
        """Initialization function for the class.

        Args:
            root_dir: Path to the folder having sequence csv files
        """
        self.counter: int = 0

        root_dir = Path(root_dir)
        self.seq_list: Sequence[Path] = [(root_dir / x).absolute() for x in os.listdir(root_dir)]
        self.map_path = map_path
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
    def agent_lane_dict(self) -> dict():
      agent_lane_dict = dict()
      for id in list(self.seq_df['trackId'].unique()):
        lanes = list(self.seq_df[self.seq_df['trackId']==id]['laneletId'].unique())
        agent_lane_dict[id] = list()
        for lane in lanes:
          if len(lane) > 5:
            agent_lane_dict[id].append(lane.split(';')[0])
            agent_lane_dict[id].append(lane.split(';')[1])
          else:
            agent_lane_dict[id].append(lane)
      return agent_lane_dict
    
    @property
    def lane_centerlines_dict(self) -> dict:
        """Get the lane-centerline id dict for the current map.

        Returns:
            dict
        """
        G = ox.graph_from_xml(self.map_path, simplify=False, retain_all=True)
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

        # print(lane_centerlines_dict[1500])
        return lane_centerlines_dict

    @property
    def left_right_dict(self) -> dict:
        """Get the left-right lane id dict for the current map.
        """
        # 车道对应左右车道线
        tree = ET.parse(self.map_path)
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

        return left_id_dict, right_id_dict

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


    def __iter__(self) -> "exiDLoader":
        """Iterator for enumerating over sequences in the root_dir specified.

        Returns:
            Data Loader object for the first sequence in the data
        """
        self.counter = 0
        return self

    def __next__(self) -> "exiDLoader":
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

    def __getitem__(self, key: int) -> "exiDLoader":
        """Get the DataLoader object for the sequence corresponding to the given index.

        Args:
            key: index of the element

        Returns:
            Data Loader object for the given index
        """

        self.counter = key
        self.current_seq = self.seq_list[self.counter]
        return self

    def get(self, seq_id: Union[Path, str]) -> "exiDLoader":
        """Get the DataLoader object for the given sequence path.

        Args:
            seq_id: Fully qualified path to the sequence

        Returns:
            Data Loader object for the given sequence path
        """
        self.current_seq = Path(seq_id).absolute()
        return self

