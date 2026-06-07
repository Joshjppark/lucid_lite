

import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union


from scipy.optimize import linear_sum_assignment
from PySide6.QtGui import QImage

from .geometry import *
from gui_source.pose_data import Camera, FrameGroup, Instance


logging.basicConfig(level=logging.INFO)


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

    # def __init__(self, group, instance_list, id_, cam_map, proj_cache):
    def __init__(self, group, instance_list, cam_map, proj_cache):

        self.id = None
        self.valid, self.cams_by_count, self.points_by_cam, self.cam_track = self.is_valid(
            group=group,
            instance_list=instance_list,
            return_dict=True
        )
        
        if self.valid:
            points3d, reproj_score = self.triangulate(cam_map, proj_cache)
            self.points3d = points3d
            self.reproj_score = reproj_score
        else:
            self.points3d = None
            self.reproj_scores = {}


    def triangulate(self, cam_map, cache):
        group = {cam: self.points_by_cam[cam][0] for cam in self.points_by_cam}
        points3d, reprojs = triangulate_group(group, cam_map, cache, get_reprojections=True)
        # print(f'points3d for group {self.id} {points3d}')
        reproj_scores = self._calc_reprojs(reprojs)

        return points3d, reproj_scores


    def _calc_reprojs(self, reprojs) -> dict[str, float]:

        reproj_score = {}
        for cam in self.cams_by_count:
            reproj_score[cam] = instance_pixel_distance(reprojs[cam], self.points_by_cam[cam][0])

        
        return reproj_score
    

    @staticmethod
    def is_valid(group, instance_list, return_dict=False) -> Union[bool, Optional[dict]]:
        '''
        checks validity of group
        '''

        cams_by_count = defaultdict(int)
        cam_track = []
        points_by_cam = {}
        isvalid = True
        for instance_idx in group:
            track_idx, cam, pts = instance_list[instance_idx]

            # store cam information
            cams_by_count[cam] += 1
            cam_track.append((cam, track_idx)) 

            # add points to dictionary
            if cam in points_by_cam:
                points_by_cam[cam].append(pts)
                isvalid = False # multiple instances in the same view
            else:
                points_by_cam[cam] = [pts]

        if return_dict:
            return isvalid, cams_by_count, points_by_cam, cam_track
        else:
            return isvalid

        
    
@dataclass
class TrackedIdentity:
    id: int
    group: Optional[Group] = None               # current frame's detection, None if occluded
    frames_hidden: int = 0                      # consecutive frames with no detection
    last_points3d: Optional[np.ndarray] = None

    def __post_init__(self):
        
        if self.group is not None:
            self.group.id = self.id

    @property
    def visible(self) -> bool:
        return self.group is not None and self.group.valid



class SingleFrameTrack:

    def __init__(
            self,
            fg: FrameGroup,
            cameras: Camera,
            proj_cache: ProjCache,
            node_weights: np.ndarray,
            prev_trackIds: dict[int, TrackedIdentity] = None,
            max_ids: int = None,
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
        self.prev_trackIds = prev_trackIds
        self.max_ids = max_ids


        # attributes to be updated a future time
        self._camera_pairs = None
        self.adjacency_matrix = None
        self.edges = None
        self.groups = None
        self.trackIds = None
        self.visible_ids = []
        self.invalid_instances = []
        self.nonmatch_instances = []
        self.nonmatch_groups = []
        self.match_conflicts: dict[tuple(str, str): tuple(int, int)] = []


        self.cam_count_dict: dict[str, int] = {cam: len(insts) for cam, insts in fg.instances.items()}
        self.cameras = [c for c in cameras if self.cam_count_dict.get(c.name, 0)]
        self.cam_map = {c.name: c for c in self.cameras}

        # hyper parameters
        self.EPIPOLE_THRESHOLD = 9
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
                    self.invalid_instances.append((inst.track_idx, cam_name, pts))
                    continue

                instance_list.append((inst.track_idx, cam_name, pts))
                cam_points.append((counter, pts))
                
                counter += 1
            
            instance_by_cam[cam_name] = cam_points

        return instance_by_cam, instance_list
    

    def _runSingleFrameTracker(self):
                               
        self.adjacency_matrix = np.full((self.n_insts, self.n_insts), np.inf)
        self.edges = self._calc_edge_weights()
        # self.groups = self._run_bfs()
        self.groups = self._run_union_find()

        if self.prev_trackIds is None:
            # initial frame - assign identities to groups
            self.trackIds = self._init_identities() 
        else:
            # subsequent frames - assign identites to match prev frame

            # TEMP SOLN
            if not all([g.valid for g in self.groups]):
                self.trackIds = None
                return


            self.trackIds = self._match_prev_groups()


    def smart_hungarian(self, edges):
        valid = edges < self.EPIPOLE_THRESHOLD
        row_sum = valid.sum(axis=1)
        col_sum = valid.sum(axis=0)

        safe_mask = valid & (row_sum[:, None] == 1) & (col_sum[None, :] == 1) 
        matches = list(zip(*np.where(safe_mask)))

        # look for duplicates along the columns
        conflicts = []
        for c in np.where(col_sum > 1)[0]:
            row_dups = np.where(valid[:, c])[0]
            conflicts += [(int(r), int(c)) for r in row_dups]


        # look for duplicates along the rows
        for r in np.where(row_sum > 1)[0]:
            col_dups = np.where(valid[r, :])[0]
            conflicts += [(int(r), int(c)) for c in col_dups]

        return matches, conflicts


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
            matches, conflicts = self.smart_hungarian(edges)
            # print(conflicts, (cam1, cam2))
            for conflict in conflicts:

                # print((inst1_idx, inst2_idx))
                r, c = conflict
                inst1_idx = self.instance_by_cam[cam1][r][0]
                inst2_idx = self.instance_by_cam[cam2][orig_col_idx[c]][0]
                self.match_conflicts.append((inst1_idx, inst2_idx))

            # add row and column assignments from edge hungarian to the adjacency matrix
            for (r, c) in matches:
                inst1_idx = self.instance_by_cam[cam1][r][0]
                inst2_idx = self.instance_by_cam[cam2][orig_col_idx[c]][0]

                self.adjacency_matrix[inst1_idx, inst2_idx] = edges[r, c]
                self.adjacency_matrix[inst2_idx, inst1_idx] = edges[r, c]

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
            explored = [unvisited]
            visited[unvisited] = True
            
            group = []

            # run bfs on subgraphs
            while explored:
                vertex = explored.pop(0)
                group.append(int(vertex))

                for neighbor in np.where(self.adjacency_matrix[vertex] != np.inf)[0]:
                    if not visited[neighbor]:    
                        # print(f'vertex {vertex} has neighbors {}')
                        visited[neighbor] = True
                        explored.append(neighbor)

            if len(group) < 2:
                for v in group: self.nonmatch_instances.extend(self.instance_list[v])
                continue

            # valid = Group.is_valid(group=group, instance_list=self.instance_list)

            # if valid or :
            groups.append(Group(group, self.instance_list, len(groups), self.cam_map, self.proj_cache))
            


        return groups


    def _run_union_find(self):

            n_nodes = self.adjacency_matrix.shape[0]
            parent = {n: n for n in range(n_nodes)}
            cam_to_id = {c.name: i for i, c in enumerate(self.cameras)}
            camera_ids = [cam_to_id[tup[1]] for tup in self.instance_list]

            cameras_in_component = {n: {camera_ids[n]} for n in range(n_nodes)}


            def find(i):
                if parent[i] == i: return i
                parent[i] = find(parent[i])
                return parent[i]

            def union(i, j):
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_i] = root_j
                    cameras_in_component[root_j].update(cameras_in_component[root_i])

            # TODO: put in the 'good' bad_edges back into the graph
            edge_conflicts = {frozenset(edge) for edge in self.match_conflicts}

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if self.adjacency_matrix[i][j] != np.inf and frozenset({i, j}) not in edge_conflicts:
                        union(i, j)

            # add in valid edges back in
            for u, v in edge_conflicts:
                root_u = find(u)
                root_v = find(v)

                if root_u == root_v:
                    continue

                if cameras_in_component[root_u].isdisjoint(cameras_in_component[root_v]):
                    # merge the two if valid subgraph
                    parent[root_u] = root_v
                    cameras_in_component[root_v].update(cameras_in_component[root_u])


            # find all groups
            distinct_groups = defaultdict(list)
            for node in range(n_nodes):
                root = find(node)
                distinct_groups[root].append(node)


            # check validity of groups
            groups = []
            for group in list(distinct_groups.values()):
                if len(group) < 2:
                    for v in group: self.nonmatch_instances.extend(self.instance_list[v])    
                elif Group.is_valid(group, self.instance_list):
                    groups.append(Group(group, self.instance_list, self.cam_map, self.proj_cache))
                else:
                    # group itself is 
                    fixed_groups = self._fix_group(group)
                    for fixed_group in fixed_groups:
                        if len(fixed_group) < 2:
                            for v in group: self.nonmatch_instances.extend(self.instance_list[v])
                        else:
                            groups.append(Group(fixed_group, self.instance_list, self.cam_map, self.proj_cache))



            return groups


    def _fix_group(self, group):
    
        parent = {n: n for n in group}
        cam_to_id = {cam.name: i for i, cam in enumerate(self.cameras)}
        camera_ids = [cam_to_id[t[1]]for t in self.instance_list]
        cameras_in_component = {n: {camera_ids[n]} for n in group}

        def find(i):
            if parent[i] == i: return i
            parent[i] =  find(parent[i])
            return parent[i]

        # extract all the possible edges
        edges = []
        n_nodes = len(group)
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                u, v = group[i], group[j]
                weight = self.adjacency_matrix[u, v]
                if weight != np.inf:
                    edges.append((u, v, weight))

        edges.sort(key=lambda x: x[2])


        for u, v, _ in edges:
            root_u = find(u)
            root_v = find(v)

            # same parent --> do nothing; nodes already in same group
            if root_u == root_v:
                continue

            if cameras_in_component[root_v].isdisjoint(cameras_in_component[root_u]):
                # disjoint cameras --> valid to joint nodes
                parent[root_v] = root_u
                cameras_in_component[root_u].update(cameras_in_component[root_v])
            else:
                # non disjoint camera sets --> do not join nodes
                # don't connect the groups
                continue

        distinct_groups = defaultdict(list)
        for node in group:
            root = find(node)
            distinct_groups[root].append(node)
        
        return list(distinct_groups.values())


    def _init_identities(self):
        
        if not self.max_ids:
            self.max_ids = max(self.cam_count_dict.values())

        trackIds = {}
        # assign identities with visible tracks
        for i, group in enumerate(self.groups):
            trackIds[i] = TrackedIdentity(
                id = i,
                group = group,
                last_points3d=group.points3d,
            )
            self.visible_ids.append(i)

        # assign leftover identities to reach max_ids
        for i in range(len(self.groups), self.max_ids):
            trackIds[i] = TrackedIdentity(
                id = i,
                group = None,
            )

        return trackIds


    def _match_prev_groups(self):

        '''
        Matches current group assignment to prev frame groups
        so that identities can be consist across consecutive frames
        '''

        curr_groups_num = len(self.groups)
        prev_groups_num = len(self.prev_trackIds)

        # CLAUDE FIX: `matchable_ids[col]` is the *real* identity_id
        # that column `col` of the cost matrix corresponds to. Once
        # any prev identity has `last_points3d is None` and gets
        # filtered out here, the columns of `costs` no longer line up
        # 1-to-1 with identity_ids — every downstream use of `col` must
        # go through `matchable_ids[col]` to recover the id.
        matchable_ids =  [i for i, p in self.prev_trackIds.items() if p.last_points3d is not None]

        # CLAUDE FIX (guard): if no prev identity has a usable
        # last_points3d (e.g. every id is hidden after a long occlusion
        # gap) or there are no current groups, the cost matrix has
        # shape (N, 0) / (0, M) and the argmin calls below would raise.
        # Skip straight to the open-id assignment path with an empty
        # final_matches.
        if len(matchable_ids) == 0 or curr_groups_num == 0:
            costs = np.empty((curr_groups_num, len(matchable_ids)))
            final_matches = np.empty((0, 2), dtype=int)
        else:
            # populate the cost matrix
            costs = np.full((curr_groups_num, len(matchable_ids)), np.inf)
            for col, id_ in enumerate(matchable_ids):
                for row in range(curr_groups_num):

                    # print(self.groups[row].points3d)
                    assert self.groups[row].points3d is not None
                    last = self.prev_trackIds[id_].last_points3d

                    costs[row, col] = np.nanmean(
                        np.linalg.norm(
                            self.groups[row].points3d-last,
                            axis=1,
                        )
                    )

            # first: assign best matches - matches that are minumum along cost row and columns
            row_mins = np.argmin(costs, axis=1)
            col_mins = np.argmin(costs, axis=0)
            best_match_rows = np.where(col_mins[row_mins] == np.arange(costs.shape[0]))[0]
            best_match_cols = row_mins[best_match_rows]

            mutual_matches = np.column_stack((best_match_rows, best_match_cols))

            # second: assign remaining 'non' mutual matches with hungarian
            remaining_r = np.setdiff1d(np.arange(costs.shape[0]), best_match_rows)
            remaining_c = np.setdiff1d(np.arange(costs.shape[1]), best_match_cols)
            submatrix = costs[np.ix_(remaining_r, remaining_c)]

            # CLAUDE FIX (guard): linear_sum_assignment on a zero-row or
            # zero-col submatrix returns empty arrays; handle the branch
            # explicitly so the column_stack always produces shape (k, 2).
            if submatrix.size == 0:
                hungarian_matches = np.empty((0, 2), dtype=int)
            else:
                local_r, local_c = linear_sum_assignment(submatrix)
                global_r = remaining_r[local_r]
                global_c = remaining_c[local_c]
                hungarian_matches = np.column_stack((global_r, global_c))

            final_matches = np.vstack((mutual_matches, hungarian_matches)).astype(int)

            # assert number of matches is number of groups in curr frame
            if len(np.unique(final_matches[:, 0])) != curr_groups_num:
                print(f'{self.frame_idx} has {len(np.unique(final_matches[:, 0]))} matches with {len(self.groups)} groups ')
            # assert len(np.unique(final_matches[:, 0])) == curr_groups_num

        # initialize and poopulate trackIds dict
        trackIds = {}

        # assign matches to tracks that exist in prev frame
        # CLAUDE FIX (root cause of increased swaps): `col` is a column
        # index into the cost matrix, which after the matchable_ids
        # filter is a *position* in that list — NOT the identity_id.
        # The old `id_ = int(col)` would silently rebind groups to the
        # wrong prev identity any time `matchable_ids` had gaps (i.e.
        # any time even one prev id was hidden), because col=k pointed
        # at id=k while matchable_ids[k] was a different id. The
        # correct id is `matchable_ids[col]`.
        for (row, col) in final_matches:
            id_ = int(matchable_ids[int(col)])
            trackIds[id_] = TrackedIdentity(
                id = id_,
                group = self.groups[row],
                last_points3d = self.groups[row].points3d,
            )
            # self.groups[row].id = id_
            self.visible_ids.append(id_)


        # assign matches to id's with uninitialized points
        open_ids = [i for i, p in self.prev_trackIds.items() if p.last_points3d is None]
        # CLAUDE FIX (consistency): materialize matched rows as a set
        # so the membership check is unambiguous even when
        # final_matches is the empty (0, 2) array from the early-exit
        # branch above.
        matched_rows = set(int(r) for r in final_matches[:, 0]) if final_matches.size else set()
        for row in (r for r in range(curr_groups_num) if r not in matched_rows):
            if not open_ids:
                # all ids are used up, remaining groups are not matched
                self.nonmatch_groups.append(self.groups[row])
                continue
            id_ = open_ids.pop(0)


            assert id_ not in trackIds

            trackIds[id_] = TrackedIdentity(
                id = id_,
                group = self.groups[row],
                last_points3d = self.groups[row].points3d,
            )
            self.groups[row].id = id_
            self.visible_ids.append(id_)

        # Propagate id's that are hidden
        for id_, prev_track in self.prev_trackIds.items():
            if id_ not in trackIds:
                trackIds[id_] = TrackedIdentity(
                    id = id_,
                    group = None,
                    last_points3d = prev_track.last_points3d,
                    frames_hidden = prev_track.frames_hidden + 1,
                )
                # self.groups[row].id = id_

        # save groups that were not assigned to an ID
        for group in self.groups:
            if group.id is None:
                print(f'{self.frame_idx} Group {group} not assigned ')


        # debugging
        self._costs = costs
        self._final_matches = final_matches

        return trackIds


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
            qimg = window._video_panels[cam].get_frame_sync(self.fg.frame_idx)
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
        return f'frame: {self.frame_idx}; {self.cam_count_dict}'
        


class MultiFrameTrack:

    def __init__(
        self,
        window,
        start=0,
        end=None,
        node_weights=None,
        max_ids = None,
    ):
        self.window = window
        self.session = window.session
        self.proj_cache = ProjCache()
        self.start = start
        self.end = end if end else window.session.max_frame
        self.max_ids = max_ids

        if node_weights is None:
            self.node_weights = {node: 1 for node in self.session.skeleton.nodes}
        else:
            self.node_weights = node_weights

        # attributes to be updated later
        self.frames = []
        self.trackIds = []
        self.invalid_instances = []
        self.nonmatch_instances = []
        self.visible_ids = []


    def track(self):

        prev_trackIds = None
        if not self.max_ids:
            max_ = 0
            for i in range(self.session.max_frame):
                lst = list(self.session.frame_group(i).instances.values())
                num = max((len(sublist) for sublist in lst), default=0)
                max_ = max(max_, num)
            self.max_ids = max_

        for frame_idx in tqdm(range(self.start, self.end)):
            fg = self.session.frame_group(frame_idx)

            sft = SingleFrameTrack(
                fg, self.session.cameras,
                node_weights=self.node_weights, proj_cache=self.proj_cache,
                prev_trackIds=prev_trackIds, max_ids=self.max_ids,
            )
            prev_trackIds = sft.trackIds
            self.frames.append(sft)


            # update attributes
            self.trackIds.append(sft.trackIds)
            self.visible_ids.append(tuple(sft.visible_ids))
            self.invalid_instances.extend(sft.invalid_instances)
            self.nonmatch_instances.extend(sft.nonmatch_instances)
            

    


