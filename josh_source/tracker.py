

import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from .geometry import *
from gui_source.pose_data import Camera, FrameGroup, Instance


@dataclass
class ProjCache:
    '''
    Stores projection matrix P and Fundamental matricies F
    for a tracking session
    '''
    F: dict[str, np.ndarray]
    P: dict[str, np.ndarray]


    def __init__(self):
        self.F = {}
        self.P = {}


    def getF(self, cam1: Camera, cam2: Camera) -> np.ndarray:
        '''
        Returns fundamental matrix F such that (x1.T @ F @ x2) = 0
        '''
        F = self.F.get((cam1.name, cam2.name), None)


        if F is None:
            FT = self.F.get((cam2.name, cam1.name), None)
            if FT: return F.T # previously calculated the other direction


            F = calc_fundamental_matrix(cam1, cam2)

            self.F[(cam1.name, cam2.name)] = F
            self.F[(cam2.name, cam1.name)] = F.T

        return F
    

    def getP(self, cam: Camera) -> np.ndarray:
        '''
        Returns projection matrix P
        '''
        P = self.P.get(cam.name, None)
        if P is None:
            R = rodrigues(cam.rvec)
            P = cam.matrix @ np.hstack([R, np.asarray(cam.tvec).reshape(-1, 1)])
            self.P[cam.name] = P
        return P


@dataclass
class Group:
    # points: dict[str: np.ndarray]
    # score: float
    # valid: bool
    id: int = None
    points_3d: np.ndarray = None


    def __init__(self, group, instance_list, id_, cam_map, proj_cache):

        # default to a correct group with max 1 instance per cam
        self.valid = True

        
        self.cams_by_count = defaultdict(int)
        
        points_by_cam = {}
        cam_track: list[tuple[str, int]] = []
        for instance_idx in group:
            track_idx, cam, pts = instance_list[instance_idx]


            # store cam information
            self.cams_by_count[cam] += 1
            cam_track.append((cam, track_idx))

            # add points to dictionary
            if points_by_cam.get(cam, []):
                points_by_cam[cam].append(pts)
                self.valid = False # multiple instances in the same view
            else:
                points_by_cam[cam] = [pts]


        self.points_by_cam = points_by_cam
        self.cam_track = cam_track
        self.id = id_
        self.points_3d, reprojs = self.triangulate(cam_map, proj_cache)
        self.reproj_score = self._calc_reprojs(reprojs)


    def triangulate(self, cam_map, cache):
        group = {cam: self.points_by_cam[cam][0] for cam in self.points_by_cam}
        points_3d, reprojs = triangulate_group(group, cam_map, cache, get_reprojections=True)
        # print(f'points_3D for group {self.id} {points_3d}')

        return points_3d, reprojs


    def _calc_reprojs(self, reprojs) -> float:

        reproj_score = {}
        for cam in self.cams_by_count:
            reproj_score[cam] = instance_pixel_distance(reprojs[cam], self.points_by_cam[cam][0])

        
        return reproj_score
    

class SingleFrameTrack:

    def __init__(
            self,
            fg: FrameGroup,
            cameras: Camera,
            proj_cache: ProjCache,
            skel_weight: np.ndarray
        ) -> None:
        '''
        Params:
        - fg: Framegroup of the current frame
        - skel_weight: np array of the weights of each node in the skeleton, rannges from 0-1
                        nodes with 0 weight are effectively ignored in all calculations
        '''

        self.fg = fg
        self.proj_cache = proj_cache
        self.skel_weight = None # this is updated in get_instance_dict() to remove zeros

        instance_by_cam, inst_list = self.get_instance_dict(fg, skel_weight)

        self.instance_by_cam: dict[tuple[int, np.ndarray]] = instance_by_cam
        # key: cam name
        # value: (instance_count:int, points: np.ndarray)


        self.instance_list: list[tuple[int, str, np.ndarray]] = inst_list
        # (track idx, cam name, points: np.ndarray)


        self.cam_count_dict: dict[str, int] = {cam: len(insts) for cam, insts in fg.instances.items()}
        self.cameras = [c for c in cameras if self.cam_count_dict.get(c.name, 0)]
        self.cam_map = {c.name: c for c in self.cameras}

        self.n_insts = len(inst_list)
        self.adjacency_matrix = np.full((self.n_insts, self.n_insts), np.inf)
        # self._calc_edge_weights()
        # self._get_groups()

    
    def get_instance_dict(self, fg, skel_weight):

        bool_mask = skel_weight != 0
        self.skel_weight = skel_weight[bool_mask]

        instance_by_cam = {}
        instance_list = []
        counter = 0
        for cam_name, instances in fg.instances.items():
            cam_points = []
            for track_idx, inst in enumerate(instances):

                pt = np.array([p if p is not None else (np.nan, np.nan) for p in inst.points])
                pt = pt[bool_mask]

                instance_list.append((track_idx, cam_name, pt))
                cam_points.append((counter, pt))
                
                counter += 1
            
            instance_by_cam[cam_name] = cam_points

        return instance_by_cam, instance_list
    


    def _calc_edge_weights(self):
        camera_pairs = self._get_camera_pairs()
        
        for (cam1, cam2) in camera_pairs:
            edges:np.ndarray = self._calc_pairwise_edges(self.cam_map[cam1], self.cam_map[cam2])
            rows, cols = linear_sum_assignment(edges)
            # print(f'calculated edges with cams {cam1}, {cam2}')

            # add row and column assignments from edge hungarian to the 
            for r, c in zip(rows, cols):
                inst1_idx = self.instance_by_cam[cam1][r][0]
                inst2_idx = self.instance_by_cam[cam2][c][0]

                self.adjacency_matrix[inst1_idx, inst2_idx] = edges[r, c]


    def _calc_pairwise_edges(self, cam1: Camera, cam2: Camera) -> np.ndarray:

        
        cam1_points = self.instance_by_cam[cam1.name]
        cam2_points = self.instance_by_cam[cam2.name]


        edges = np.zeros((len(cam1_points), len(cam2_points)), dtype=np.float64)
        

        for idx1, (_, cam1_pt) in enumerate(cam1_points):
            for idx2, (_, cam2_pt) in enumerate(cam2_points):

                F = self.proj_cache.getF(cam1, cam2)
                edges[idx1, idx2] = calc_epipolar_score(cam1_pt, cam2_pt, F)

        return edges



    def _run_bfs(self, matrix):
        
        visited = np.zeros(self.n_insts, dtype=bool)

        groups = []
        while not np.all(visited):

            non_visisted = np.where(visited == np.False_)[0][0]
            explored = [non_visisted]
            visited[non_visisted] = True
            
            group = []

            while explored:
                vertex = explored.pop(0)
                group.append(int(vertex))

                for neighbor in np.where(matrix[vertex] != np.inf)[0]:
                    if not visited[neighbor]:    
                        # print(f'vertex {vertex} has neighbors {}')
                        visited[neighbor] = True
                        explored.append(neighbor)

            groups.append(Group(group, self.instance_list, len(groups), self.cam_map, self.proj_cache))
            # groups.append(group)
        return groups




    def _get_groups(self):
        raise NotImplementedError
    

    def _get_camera_pairs(self):
        cam_names = [c.name for c in self.cameras]

        cam_pairs = []
        for i in range(len(cam_names)):
            for j in range(i+1, len(cam_names)):
                pair = (cam_names[i], cam_names[j])
                cam_pairs.append(pair)

        return cam_pairs

    
    
    def __repr__(self):
        return f'{self.instance_dict}'
        




def track_frame(framegroup):

    pass


def track_all(session):
    '''
    tracks all the frames.
    
    nowhere near done completion
    '''

    for frame in (session.frame_indices):
        fg = session.frame_group(frame)
        track_frame(fg)

        raise NotImplementedError
    
