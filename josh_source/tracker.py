

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
from PySide6.QtGui import QImage

from .geometry import *
from gui_source.pose_data import Camera, FrameGroup, Instance

def qimg_to_np(qimg: QImage):
    img = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes())
    arr = arr.reshape(h, img.bytesPerLine())[:, : w * 3]                         
    return arr.reshape(h, w, 3).copy() 


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
            if FT is not None: return FT.T # previously calculated the other direction


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



class Group:

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
            if cam in points_by_cam:
                points_by_cam[cam].append(pts)
                self.valid = False # multiple instances in the same view
            else:
                points_by_cam[cam] = [pts]


        self.points_by_cam = points_by_cam
        self.cam_track = cam_track
        self.id = id_

        if self.valid:
            points_3d, reproj_score = self.triangulate(cam_map, proj_cache)
            self.points_3d = points_3d
            self.reproj_score = reproj_score
        else:
            self.points_3d = None
            self.reproj_scores = {}



    def triangulate(self, cam_map, cache):
        group = {cam: self.points_by_cam[cam][0] for cam in self.points_by_cam}
        points_3d, reprojs = triangulate_group(group, cam_map, cache, get_reprojections=True)
        # print(f'points_3D for group {self.id} {points_3d}')
        reproj_scores = self._calc_reprojs(reprojs)

        return points_3d, reproj_scores


    def _calc_reprojs(self, reprojs) -> dict[str, float]:

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
            node_weights: np.ndarray
        ) -> None:
        '''
        Params:
        - fg: Framegroup of the current frame
        - skel_weight: np array of the weights of each node in the skeleton, rannges from 0-1
                        nodes with 0 weight are effectively ignored in all calculations
        '''

        self.fg = fg
        self.frame_idx = fg.frame_idx
        self.proj_cache = proj_cache
        self.node_weights = np.array(list(node_weights.values()))


        # attributes to be updated a future time
        self._camera_pairs = None
        self.adjacency_matrix = None
        self.edges = None
        self.groups = None
        self.bad_instances = []
        self.match_conflicts = []


        self.cam_count_dict: dict[str, int] = {cam: len(insts) for cam, insts in fg.instances.items()}
        self.cameras = [c for c in cameras if self.cam_count_dict.get(c.name, 0)]
        self.cam_map = {c.name: c for c in self.cameras}

        # hyper parameters
        self.EPIPOLE_THRESHOLD = 10
        self.MIN_NODES = 3


        # save instances to dict and list
        instance_by_cam, inst_list = self.get_instances(fg)
        self.n_insts = len(inst_list)
        self.instance_by_cam: dict[tuple[int, np.ndarray]] = instance_by_cam
        # key: cam name
        # value: (instance_count:int, points: np.ndarray)

        self.instance_list: list[tuple[int, str, np.ndarray]] = inst_list
        # (track idx, cam name, points: np.ndarray)


        # run the tracking algorithm    
        self._runSingleFrameTracker()

    
    def get_instances(self, fg):

        
        # make a mask to remove zeros in the node_weights - reduces computation
        bool_mask = self.node_weights != 0
        self.node_weights = self.node_weights[bool_mask]

        instance_by_cam = {}
        instance_list = []
        counter = 0
        for cam_name, instances in fg.instances.items():
            cam_points = []
            for inst in instances:

                pts = np.array([p if p is not None else (np.nan, np.nan) for p in inst.points])
                K = np.array(self.cam_map[cam_name].matrix)
                pts = cv2.undistortPoints(pts, K, np.array(self.cam_map[cam_name].dist), P=K).squeeze(1)
                pts = pts[bool_mask]

                # check number of valid nodes
                num_valid_nodes = np.sum(~np.isnan(pts).any(axis=1))

                if num_valid_nodes < self.MIN_NODES:
                    self.bad_instances.append((inst.track_idx, cam_name, pts))
                    continue

                instance_list.append((inst.track_idx, cam_name, pts))
                cam_points.append((counter, pts))
                
                counter += 1
            
            instance_by_cam[cam_name] = cam_points

        return instance_by_cam, instance_list
    

    def _runSingleFrameTracker(self):
                               
        self.adjacency_matrix = np.full((self.n_insts, self.n_insts), np.inf)
        self.edges = self._calc_edge_weights()
        self.groups = self._run_bfs()


    def smart_hungarian(self, edges):
        mask = edges < self.EPIPOLE_THRESHOLD
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)

        match_conflicts = []

        # row collision
        for r in np.where(row_counts > 1)[0]:
            c_indices = np.where(mask[r, :])[0]
            match_conflicts.append(np.column_stack((np.full(len(c_indices), r), c_indices)))

        # columns collisions
        for c in np.where(col_counts > 1)[0]:
            r_indices = np.where(mask[:, c])[0]
            match_conflicts.append(np.column_stack((r_indices, np.full(len(r_indices), c))))

        matches = mask & (row_counts == 1) & (col_counts == 1)
        matches = np.column_stack(np.where(matches))

        return matches, match_conflicts


    def _calc_edge_weights(self):

        
        edge_dict = {}
        for (cam1, cam2) in self.camera_pairs:
            edges:np.ndarray = self._calc_edge(self.cam_map[cam1], self.cam_map[cam2])
            edge_dict[(cam1, cam2)] = edges

            # remove columns whose values are all above threshold
            valid_cols = np.min(edges, axis=0) < self.EPIPOLE_THRESHOLD
            edges = edges[:, valid_cols]
            orig_col_idx =  np.where(valid_cols)[0]

            # save edges

            # run hungarian
            # rows, cols = linear_sum_assignment(edges)


            # run 'smart' hungarian
            matches, match_conflicts = self.smart_hungarian(edges)
            self.match_conflicts.append(match_conflicts)


            # print(f'calculated edges with cams {cam1}, {cam2}')

            # add row and column assignments from edge hungarian to the adjacency matrix
            for r, c in zip(rows, cols):
                

                inst1_idx = self.instance_by_cam[cam1][r][0]
                inst2_idx = self.instance_by_cam[cam2][orig_col_idx[c]][0]

                try:
                    self.adjacency_matrix[inst1_idx, inst2_idx] = edges[r, c]
                    self.adjacency_matrix[inst2_idx, inst1_idx] = edges[r, c]
                except:
                    print(f'c is {c}')
                    print(self._calc_edge(self.cam_map[cam1], self.cam_map[cam2]))
                    print(orig_col_idx)
                    print(edges)

                    raise AssertionError

        return edge_dict


    def _calc_edge(self, cam1: Camera, cam2: Camera) -> np.ndarray:

        
        cam1_points = self.instance_by_cam[cam1.name]
        cam2_points = self.instance_by_cam[cam2.name]


        edges = np.zeros((len(cam1_points), len(cam2_points)), dtype=np.float64)
        

        for idx1, (_, cam1_pt) in enumerate(cam1_points):
            for idx2, (_, cam2_pt) in enumerate(cam2_points):

                F = self.proj_cache.getF(cam1, cam2)
                edges[idx1, idx2] = calc_epipolar_score(cam1_pt, cam2_pt, F, self.node_weights)

        return edges



    def _run_bfs(self):
        
        visited = np.zeros(self.n_insts, dtype=bool)

        groups = []
        while not np.all(visited):

            unvisited = np.where(~visited)[0][0]
            # unvisited = np.where(visited == np.False_)[0][0]
            explored = [unvisited]
            visited[unvisited] = True
            
            group = []

            while explored:
                vertex = explored.pop(0)
                group.append(int(vertex))

                for neighbor in np.where(self.adjacency_matrix[vertex] != np.inf)[0]:
                    if not visited[neighbor]:    
                        # print(f'vertex {vertex} has neighbors {}')
                        visited[neighbor] = True
                        explored.append(neighbor)

            groups.append(Group(group, self.instance_list, len(groups), self.cam_map, self.proj_cache))
            # groups.append(group)
        return groups



    def get_edges(self, cam1_name, cam2_name):
        
        key = (cam1_name, cam2_name)
        edges = self.edges.get(key)

        if edges is None:
            key = (cam2_name, cam1_name)
            return self.edges.get(key), key
        
        return edges, key
    


    def visualize_epipolar_pair(self, window, cam1_name, cam2_name, track_idx: tuple):

        '''
        Draws a 1x2 grid, where each row has the two cameras and
        a point is drawn on both views and projected epipolar lines are displayed

        sample use:
        '''
        if len(track_idx) == 1:
            track1_idx, track2_idx = track_idx[0], track_idx[0]
        elif len(track_idx) == 2:
            track1_idx, track2_idx = track_idx
        else:
            raise AssertionError

        F2 = self.proj_cache.getF(self.cam_map[cam1_name], self.cam_map[cam2_name])
        F1 = F2.T

        pts1 = self.instance_by_cam[cam1_name][track1_idx][-1]
        pts2 = self.instance_by_cam[cam2_name][track2_idx][-1]
        points = [pts1, pts2]
        
        lines1 = calc_epipolar_lines(F1, homogenize(pts2))
        lines2 = calc_epipolar_lines(F2, homogenize(pts1))
        errors = calc_epipolar_score(pts1, pts2, F2, self.node_weights)
        lines = [lines1, lines2]        


        colors = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", 
            "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000"
            ]
        

        fig, axes = plt.subplots(1, 2, figsize=(14, 12))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        # print(f'frame {self.fg.frame_idx}')

        # axes = np.array(axes).ravel()
        for cam, ax, pts in zip([cam1_name, cam2_name], axes, points):
            
            # get frame video data
            qimg = window._video_panels[cam]._decoder.get_frame(self.fg.frame_idx)
            frame = qimg_to_np(qimg)

            height, width = frame.shape[:2]
            camera = self.cam_map[cam]
            K = np.asarray(camera.matrix)
            dist = np.asarray(camera.dist)
            mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (width, height), cv2.CV_32FC1)
            undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)


            # display frame
            ax.imshow(undistorted_frame)
            # ax.imshow(frame)
            ax.set_title(cam)
            ax.axis('off')

            # display points
            ax.scatter(pts[:, 0], pts[:, 1], s=10, c=colors[:pts.shape[0]])
            # ax.scatter(pts[2, 0], pts[2, 1], s=10, c=colors[:1])

            # print(f'points: {pts}')`
            # print(f'shape of points`: {pts.shape}')
            # print(f'shape of line: {lines[1].shape}')


            # display the line in the second camera
            idx = [cam1_name, cam2_name].index(cam)
            if idx == -1: raise AssertionError
            for node_idx in range(pts.shape[0]):
            # for node_idx in [2]:
                height, width = frame.shape[:2]
                line = lines[idx][:, node_idx]
                if np.isnan(line).any(): continue
                a, b, c = line
                if abs(b) > abs(a):
                    x_s = [0, width]
                    y_s = [-c/b, -(a * width + c) / b]
                else:
                    x_s = [-c/a, -(b * height + c) / a]
                    y_s = [0, height]
                ax.plot(x_s, y_s, lw=0.5, c=colors[node_idx])
    
        return errors


    @property
    def camera_pairs(self):
        if self._camera_pairs: return self._camera_pairs

        cam_names = [c.name for c in self.cameras]

        cam_pairs = []
        for i in range(len(cam_names)):
            for j in range(i+1, len(cam_names)):
                pair = (cam_names[i], cam_names[j])
                cam_pairs.append(pair)

        self._camera_pairs = cam_pairs
        return self._camera_pairs


    @property
    def valid(self):
        return all([group.valid for group in self.groups])

    
    def __repr__(self):
        return f'{self.cam_count_dict}'
        




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
    


