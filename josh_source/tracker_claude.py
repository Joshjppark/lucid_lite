
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, DefaultDict, List, Tuple


from scipy.optimize import linear_sum_assignment
from PySide6.QtGui import QImage

from .geometry import *
from gui_source.pose_data import Camera, FrameGroup, Identity
from gui_source.colors import next_palette_color


logging.basicConfig(level=logging.INFO)


def qimg_to_np(qimg: QImage):
    img = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes())
    arr = arr.reshape(h, img.bytesPerLine())[:, : w * 3]                         
    return arr.reshape(h, w, 3).copy() 


def get_matches(costs: np.ndarray, THRESH_LOW: float, THRESH_HIGH: float,
                prev_track_id, frames_hidden=None, alpha: float = 0.0) -> tuple:
    '''
    Two-level probabilistic temporal matcher.

    Returns the SAME 5-bucket schema get_matches has always returned
    (valid_match, ambig_curr, ambig_prev, unmatched_curr, unmatched_prev), so
    every downstream consumer (SpatialMatches.add_camera_match and the
    resolvers) is unchanged -- only the boundaries between buckets are now
    probability-derived.

    Params:
      costs: (n_curr, n_prev) scale-normalized reproj distances (unitless,
             np.inf where unset). Lower = better match.
      THRESH_LOW:  high-confidence accept gate. A curr whose best candidate is
                   < LOW and is the unique high-confidence candidate in its row
                   and column is accepted outright (valid_match).
      THRESH_HIGH: reject-for-match gate. Candidates with cost < HIGH are
                   "plausible"; >= HIGH is a spawn proposal (unmatched_curr).
                   LOW <= cost < HIGH is "medium" -> deferred to the cross-cam
                   resolver via ambig_curr.
      frames_hidden: (n_prev,) consecutive-unseen counts in prev_track_id
                   column order. Used to widen the gates for long-hidden ids.
      alpha: per-hidden-frame gate inflation. Effective gate for prev id j is
             base * (1 + alpha * frames_hidden[j]) -- equivalently a Gaussian
             match-likelihood exp(-cost^2 / 2 sigma_j^2) with sigma_j growing
             linearly in frames_hidden (sigma0 folded into the gate values).
    '''
    n_curr, n_prev = costs.shape

    if n_curr == 0 or n_prev == 0:
        # No currs (nothing to assign) or no prev ids (everything is new):
        # every curr is a spawn proposal with no available prev id; every prev
        # id is unmatched in this cam.
        prev_ids = np.asarray(prev_track_id, dtype=int)
        return (
            np.empty((0, 2), dtype=int),                  # valid_match
            [],                                           # ambig_curr
            [],                                           # ambig_prev
            (np.arange(n_curr), prev_ids),                # unmatched_curr (2-tuple!)
            prev_ids,                                     # unmatched_prev
        )

    # Per-prev-id gate inflation from the sigma model.
    if frames_hidden is None:
        fh = np.zeros(n_prev)
    else:
        fh = np.asarray(frames_hidden, dtype=float)
    inflate = 1.0 + alpha * fh                            # (n_prev,)
    low_row = (THRESH_LOW * inflate)[np.newaxis, :]
    high_row = (THRESH_HIGH * inflate)[np.newaxis, :]

    accept_mask = costs <= low_row                        # high confidence
    cand_mask = costs <= high_row                         # plausible (match-or-defer)

    # valid: a high-confidence pair that is the UNIQUE high-confidence candidate
    # in both its row and column (a medium runner-up does not spoil it).
    a_row = accept_mask.sum(axis=1)
    a_col = accept_mask.sum(axis=0)
    valid_indices = accept_mask & (a_row[:, None] == 1) & (a_col[None, :] == 1)
    vr, vc = np.where(valid_indices)                      # col INDEX space (pre-remap)

    valid_match = np.column_stack((vr, prev_track_id[vc])) if vr.size \
        else np.empty((0, 2), dtype=int)

    # Plausible pairs that aren't clean valid matches need resolving. Suppress
    # rows already validly matched (curr is taken) and columns already validly
    # claimed (one instance per cam per id).
    ambig_indices = cand_mask & ~valid_indices
    ambig_indices[vr, :] = False
    ambig_indices[:, vc] = False

    # unmatched currs: rows with no plausible candidate -> spawn proposal.
    # available prev ids = ids no curr plausibly matched (empty column).
    cand_col = cand_mask.sum(axis=0)
    unmatched_curr = (
        np.where(cand_mask.sum(axis=1) == 0)[0],
        prev_track_id[np.where(cand_col < 1)[0]],
    )
    unmatched_prev = prev_track_id[np.where(cand_col == 0)[0]]

    ambig_curr = []
    ambig_prev = []

    # column contention (one prev id plausibly pulled by multiple currs)
    for c in np.where(ambig_indices.sum(axis=0) > 1)[0]:
        ambig_prev.append([(int(r), int(prev_track_id[c])) for r in np.where(ambig_indices[:, c])[0]])

    # row contention (one curr plausibly matching multiple prev ids)
    for r in np.where(ambig_indices.sum(axis=1) > 1)[0]:
        ambig_curr.append([(int(r), int(prev_track_id[c])) for c in np.where(ambig_indices[r, :])[0]])

    # medium-but-unique rows: a single plausible candidate that is not high
    # confidence. Not "ambiguous" by count, but must still be verified by the
    # cross-cam resolver rather than accepted outright -> route to ambig_curr.
    for r in np.where(ambig_indices.sum(axis=1) == 1)[0]:
        c = int(np.where(ambig_indices[r, :])[0][0])
        ambig_curr.append([(int(r), int(prev_track_id[c]))])

    return (valid_match, ambig_curr, ambig_prev, unmatched_curr, unmatched_prev)


@dataclass
class SpatialMatches:
    """
    Aggregates spatial matches across all cameras, mapping current frame 
    instances to previous frame track IDs.
    """
    valid_matches: DefaultDict[int, List[Tuple[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))
    
    ambig_currs: List[Tuple[str, List[Tuple[int, int]]]] = field(default_factory=list)
    ambig_prevs: List[Tuple[str, List[Tuple[int, int]]]] = field(default_factory=list)
    
    # (camera_name, current_idx, list_of_available_prev_indices)
    unmatched_currs: List[Tuple[str, int, List[int]]] = field(default_factory=list)
    
    unmatched_prevs: List[Tuple[str, int]] = field(default_factory=list)

    def add_camera_match(self, cam: str, cam_instances: list, match_tuple: tuple):
        """
        Unpacks the tuple from get_matches() and appends the camera name 
        to aggregate the global tracking state.
        """
        valid_match, ambig_curr, ambig_prev, unmatched_curr, unmatched_prev = match_tuple

        for curr_idx, id_ in valid_match:
            inst = cam_instances[curr_idx][1]
            self.valid_matches[int(id_)].append((cam, inst))

        for ambig_curr_pair in ambig_curr:
            self.ambig_currs.append((cam, ambig_curr_pair))

        for ambig_prev_pair in ambig_prev:
            self.ambig_prevs.append((cam, ambig_prev_pair))

        # --- NEW UNMATCHED CURR LOGIC ---
        # Unpack the new 2-part tuple
        unmatched_c_indices, available_p_indices = unmatched_curr
        
        # Convert numpy array to standard Python list of ints for clean storage
        available_p_list = [int(p) for p in available_p_indices]

        for c in unmatched_c_indices:
            self.unmatched_currs.append((cam, int(c), available_p_list))
        # --------------------------------

        for p in unmatched_prev:
            self.unmatched_prevs.append((cam, int(p)))


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

    def __init__(self, group, cam_map, proj_cache):

        # invalid group of more than one instance from the same camera
        if len(group) != len(set([g[0] for g in group])):
            raise AssertionError
        

        self.id = None
        self.points_by_cam = {cam: pts for cam, pts in group}
        self.valid = len(group) > 1
        
        if self.valid:
            points3d, reproj_scores = self.triangulate(cam_map, proj_cache)
            self.points3d = points3d
            self.reproj_scores = reproj_scores
        else:
            self.points3d = None
            self.reproj_scores = {}


    def triangulate(self, cam_map, cache):
        group = {cam: self.points_by_cam[cam] for cam in self.points_by_cam}
        points3d, reprojs = triangulate_group(group, cam_map, cache, get_reprojections=True)
        reproj_scores = self._calc_reprojs(reprojs)

        return points3d, reproj_scores


    def _calc_reprojs(self, reprojs) -> dict[str, float]:

        reproj_score = {}
        for cam in self.points_by_cam:
            # Normalize by the group's reprojected apparent scale in this cam so
            # the score is a unitless fraction of body length (depth-invariant).
            ref = apparent_scale(dehomogenize(reprojs[cam]))
            reproj_score[cam] = instance_pixel_distance(
                reprojs[cam], self.points_by_cam[cam], ref_scale=ref)


        return reproj_score
    
        
    
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
            cameras: list[Camera],
            proj_cache: ProjCache,
            node_weights: np.ndarray,
            next_avail_id: int,
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
        self.next_avail_id = next_avail_id


        # attributes to be updated a future time
        self._camera_pairs = None
        self.adjacency_matrix = None
        self.edges = None
        self.groups: list[Group] | None = None
        self.trackIds = None
        self.visible_ids = []
        self.invalid_instances = []
        self.nonmatch_instances = []
        # self.nonmatch_groups = []
        self.match_conflicts: dict[tuple(str, str): tuple(int, int)] = []

        # camera attributes
        self.cam_count_dict: dict[str, int] = {cam: len(insts) for cam, insts in fg.instances.items()}
        self.cameras = [c for c in cameras if self.cam_count_dict.get(c.name, 0)]
        self.cam_map = {c.name: c for c in self.cameras}

        # hyperparameters
        # All distance thresholds are now UNITLESS fractions of body length
        # (pixel error / apparent instance scale), not raw pixels. The old
        # pixel values are noted for reference; converted by S_rep ~= 30px.
        # Measured instance RMS spread on this set is ~82 px (not 30), so the
        # px->unitless divisor is ~82: EPIPOLE 9/82~=0.11, TEMPORAL 90/82~=1.1,
        # MAX_POTENTIAL 20/82~=0.24. Cost histograms are cleanly bimodal:
        # epipolar true matches <0.06, noise >1.0; best temporal match p50=0.52.
        self.EPIPOLE_THRESHOLD = 0.15            # was 9 px
        self.MAX_POTENTIAL_REPROJ_THRESHOLD = 0.25   # was 20 px
        # Keep this SHORT. Hidden ids reproject from a STALE 3D prior (no
        # velocity model yet, review §4.3); coasting them longer lets them
        # intercept current instances into the ambiguous bucket and starve
        # leftover-spawn -> tracking collapses. 10 lets dead ids clear quickly.
        self.FRAMES_MISSING_THRESHOLD = 10       # frames
        self.MIN_NODES = 3

        # Two-level probabilistic temporal gates (replace TEMPORAL_THRESHOLD=90px).
        #   cost < THRESH_LOW  & unambiguous -> accept (valid match)
        #   THRESH_LOW <= cost < THRESH_HIGH -> defer to cross-cam resolver
        #   cost >= THRESH_HIGH              -> spawn proposal (unmatched curr)
        # HIDDEN_INFLATION_ALPHA widens both gates for long-hidden ids:
        # effective gate = base * (1 + alpha * frames_hidden), the Gaussian
        # match-likelihood exp(-cost^2 / 2 sigma^2) with sigma ~ (1+alpha*fh).
        # THRESH_LOW must be TIGHT: a loose accept gate makes many prev ids
        # fall below it per curr, so uniqueness (the seed for valid groups)
        # fails and everything becomes ambiguous. Best-match p25~=0.17, p50~=0.52
        # -> 0.5 keeps the clear matches unique while deferring the rest.
        self.THRESH_LOW = 0.5                    # high-confidence accept
        self.THRESH_HIGH = 1.1                   # plausible (was TEMPORAL=90 px)
        self.HIDDEN_INFLATION_ALPHA = 0.1

        # save instances to dict and list
        instance_by_cam, inst_list = self.get_instances(fg)

        self.n_insts = len(inst_list)
        self.instance_by_cam: dict[str, list[tuple[int, np.ndarray]]] = instance_by_cam
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
                               
        '''
        Runs tracking algorithm for the frame, then matches group assignments
        with previous matches, if provided
        '''

        # if there are no instances in the current frame, then set trackIds to None
        if not self.instance_list:
            self.trackIds = {}
            return
        
        self.adjacency_matrix = np.full((self.n_insts, self.n_insts), np.inf)

        if self.prev_trackIds is None:
            # initial frame - assign identities to groups
            self.edges = self._calc_edge_weights()
            self.groups = self._run_union_find()
            self.trackIds = self._init_identities() 
        else:
            # subsequent frames - assign identites to match prev frame
            self.groups, self.trackIds = self._match_prev_groups()


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


    def _calc_edge_weights(self, target_nodes: list[int] = None):

        
        edge_dict = {}

        target_set = set(target_nodes) if target_nodes is not None else None
        # TODO: make this more efficient; currently the calculations are ignored if 
        # running on subset of nodes

        for (cam1, cam2) in self.camera_pairs:
            edges:np.ndarray = self._calc_edge(self.cam_map[cam1], self.cam_map[cam2])
            edge_dict[(cam1, cam2)] = edges

            if edges.size == 0:
                continue

            # remove columns whose values are all above threshold
            valid_cols = np.min(edges, axis=0) < self.EPIPOLE_THRESHOLD
            edges = edges[:, valid_cols]
            orig_col_idx =  np.where(valid_cols)[0]


            # run 'smart' hungarian
            matches, conflicts = self.smart_hungarian(edges)
            # print(conflicts, (cam1, cam2))
            for conflict in conflicts:

                # print((inst1_idx, inst2_idx))
                r, c = conflict
                inst1_idx = self.instance_by_cam[cam1][r][0]
                inst2_idx = self.instance_by_cam[cam2][orig_col_idx[c]][0]
                
                if target_set is None or (inst1_idx in target_set and inst2_idx in target_set):
                    self.match_conflicts.append((inst1_idx, inst2_idx))

            # add row and column assignments from edge hungarian to the adjacency matrix
            for (r, c) in matches:
                inst1_idx = self.instance_by_cam[cam1][r][0]
                inst2_idx = self.instance_by_cam[cam2][orig_col_idx[c]][0]

                if target_set is None or (inst1_idx in target_set and inst2_idx in target_set):
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
                # Symmetric scale ref (geometric mean of the pair) keeps the
                # normalized score symmetric under getF direction flips.
                ref = np.sqrt(apparent_scale(cam1_pt) * apparent_scale(cam2_pt))
                edges[idx1, idx2] = calc_epipolar_score(
                    cam1_pt, cam2_pt, F, self.node_weights, ref_scale=ref)

        return edges


    def _run_union_find(self) -> list[Group]:

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
            groups: list[Group] = []
            for group in list(distinct_groups.values()):

                if len(group) < 2:
                    for v in group: self.nonmatch_instances.append(self.instance_list[v])    
                    continue

                elif len(group) == len(set([self.instance_list[v][1] for v in group])):
                    # valid group

                    g = [(self.instance_list[v][1], self.instance_list[v][2]) for v in group]
                    groups.append(Group(g, self.cam_map, self.proj_cache))
                else:
                    # group is invalid (multiple instances from the same camera)
                    fixed_groups = self._fix_group(group)
                    for fixed_group in fixed_groups:
                        if len(fixed_group) < 2:
                            for v in group: self.nonmatch_instances.append(self.instance_list[v])
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


    def calc_reproj_distance(self, x, y, ref_scale=1.0):
        '''
            Returns scale-normalized mean L2 between two instances (unitless).
            ref_scale divides out apparent instance size; default 1.0 keeps
            the original raw-pixel behavior.
        '''


        assert x.shape == y.shape

        error = np.nanmean(
            np.linalg.norm(x - y,axis=1,)
        )

        return error / ref_scale


    def _init_identities(self):
        
        if not self.max_ids:
            self.max_ids = max(self.cam_count_dict.values())

        trackIds = {}
        # assign identities with visible tracks
        assert type(self.groups) == list
        for group in self.groups:
            id_ = self.next_avail_id
            trackIds[id_] = TrackedIdentity(
                id = id_,
                group = group,
                last_points3d=group.points3d,
            )
            self.visible_ids.append(id_)
            self.next_avail_id += 1

        self._costs = None

        return trackIds


    def _match_prev_groups(self):
        '''
        temporal -> spatial
            1. match each instance in the view with one of the ID's in the previous frame
                If the error > threshold, then make a new ID
            2. If there is only 
        '''

        # temporal matching
        self.temporal_cost_by_cam: dict[str, np.ndarray] = {}
        sm = SpatialMatches()
        self.sm = sm

        for cam in self.instance_by_cam:
            temporal_cost, temporal_matches = self._temporal_matching(cam)
            self.temporal_cost_by_cam[cam] = temporal_cost

            cam_instances = self.instance_by_cam[cam]
            sm.add_camera_match(cam, cam_instances, temporal_matches)
            
        groups, trackIds = self._spatial_matching(sm)

        self.groups = groups
        self.sm = sm
        
        return self.groups, trackIds
    

    def _temporal_matching(self, cam: str):
        '''
        Matches ID's of current `cam` instances to previous IDs
        '''
        
        num_prev_tracks = len(self.prev_trackIds.values())

        # get camera instances and P matrix
        cam_instances = self.instance_by_cam[cam]
        P = self.proj_cache.getP(self.cam_map[cam])

        # define temporal cost matrix
        temporal_cost = np.full((len(cam_instances), num_prev_tracks), np.inf)

        # converts row index -> ID number for prev tracks
        prev_track_id = []
        # consecutive-unseen counts, aligned to the columns of temporal_cost
        frames_hidden = []

        # loop over all the prev tracks
        for j, prev_track in enumerate(self.prev_trackIds.values()):
            # save id of the prev track
            prev_track_id.append(prev_track.id)
            frames_hidden.append(prev_track.frames_hidden)

            # get reproj of prev track
            prev_track_reproj = dehomogenize(reproject_points(prev_track.last_points3d, P))

            # normalize each curr-vs-prev distance by the (stable) reprojected
            # prev instance's apparent scale -> unitless, depth-invariant cost.
            ref = apparent_scale(prev_track_reproj)
            for i in range(len(cam_instances)):
                temporal_cost[i, j] = self.calc_reproj_distance(
                    cam_instances[i][1], prev_track_reproj, ref_scale=ref)

        prev_track_id = np.array(prev_track_id, dtype=int)
        frames_hidden = np.array(frames_hidden, dtype=float)
        matches = get_matches(
            temporal_cost, self.THRESH_LOW, self.THRESH_HIGH, prev_track_id,
            frames_hidden=frames_hidden, alpha=self.HIDDEN_INFLATION_ALPHA,
        )

        return temporal_cost, matches

        
    def _spatial_matching(self, sm: SpatialMatches):
        '''
        Matches instances across space (different views)
        '''
        # assign visible groups
        self.groups_by_id = {}
        hidden_ids = []
        for id_, cam_points in sm.valid_matches.items():

            if len(cam_points) < 2:
                hidden_ids.append(id_)

            group = Group(
                group = cam_points,
                cam_map=self.cam_map,
                proj_cache=self.proj_cache
            )
            self.groups_by_id[id_] = group

        # resolve ambiguous current instances and update unmatched_prevs
        new_unmatched_prevs = self._resolve_ambiguous_currs(sm.ambig_currs)
        sm.unmatched_prevs.extend(new_unmatched_prevs)


        # edge case: check if resolving any ambiguous currents resolved ambiguous prevs
        # TODO


        # resolve ambiguous prev intances
        new_unmatched_currs = self._resolve_ambiguous_prevs(sm.ambig_prevs)
        # unmatched_currs.extend(new_unmatched_currs)

         
        # resolve unmatched curr:
        new_groups: list[Group] = self._resolve_unmatched_currs(sm.unmatched_currs)


        # resolve unmatched previous
        # 1. unmatched prevs are hidden if no view accepts them
        #.    (in the case that one view accepts them, then a singleton Group should be made)
        num_cams = len(self.cam_map.keys())
        hidden_cams_by_id = defaultdict(set)
        for cam, id_ in sm.unmatched_prevs:
            hidden_cams_by_id[id_].add(cam)
        hidden_ids.extend([id_ for id_, cams in hidden_cams_by_id.items() if len(cams) == num_cams])


        # now make the TrackedIdentity() objects
        trackIds = {}

        # hidden IDs (§3.4 fix: key on hidden_id, not stale id_/group from the
        # valid_matches loop above; carry a fresh group=None hidden identity).
        for hidden_id in hidden_ids:
            prev_track = self.prev_trackIds[hidden_id]
            if prev_track.frames_hidden >= self.FRAMES_MISSING_THRESHOLD:
                # track has been 'hidden' for too long; stop propagating it
                if prev_track.group is not None:
                    cam_points = list(prev_track.group.points_by_cam.items())
                    self.nonmatch_instances.extend(cam_points)
                continue

            # make 'hidden' TrackedIdentity (no detection this frame -> group None)
            trackIds[hidden_id] = TrackedIdentity(
                id = hidden_id,
                group=None,
                last_points3d=prev_track.last_points3d,
                frames_hidden=prev_track.frames_hidden + 1,
            )

        # iterate through the matched ones; singleton groups are considered 'hidden'
        for id_, group in self.groups_by_id.items():
            if group.valid:
                trackIds[id_] = TrackedIdentity(
                    id = id_,
                    group = group,
                    last_points3d=group.points3d,
                )
                continue

            # Hidden TrackedIdentites
            prev_track = self.prev_trackIds[id_]
            if prev_track.frames_hidden >= self.FRAMES_MISSING_THRESHOLD:
                # track has been 'hidden' for too long; stop propagating it
                cam_points = list(group.points_by_cam.items())
                self.nonmatch_instances.extend(cam_points)
                continue

            # make 'hidden' TrackedIdentity
            trackIds[id_] = TrackedIdentity(
                id = id_,
                group=group,
                last_points3d=prev_track.last_points3d,
                frames_hidden=prev_track.frames_hidden + 1,
            )

        # make new TrackedIdentity()
        for group in new_groups:
            assert group.valid
            id_ = self.next_avail_id
            self.groups_by_id[id_] = group
            trackIds[id_] = TrackedIdentity(
                    id = id_,
                    group = group,
                    last_points3d=group.points3d,
            )
            self.next_avail_id += 1


        # debugging
        self.valid_matches = sm.valid_matches
        self.ambig_currs = sm.ambig_currs
        self.ambig_prevs = sm.ambig_prevs
        self.unmatched_currs = sm.unmatched_currs
        self.unmatched_prevs = sm.unmatched_prevs
        self.hidden_ids = hidden_ids

        groups = list(self.groups_by_id.values())

        return groups, trackIds
    

    def _resolve_ambiguous_currs(self, ambig_currs) -> list[tuple[str, int]]:
        '''
        Resolve ambiguous currs

        This is done by inserting the instance to each possible group of associated IDs
        and then taking the group with the highest change reprojection score.

        Edits the self.groups_by_id: dict[int, Group] attribute

        Returns:
            new_unmatched_prevs: list --> List of IDs of previous ID's not assigned to a group

        This assumes that one of the possible assignemnts is the true ID
        TODO: Implement a user-defined threshold for no assignments to pass

        Pitfalls:
            This function can handle multiple ambig_curr groupings, but assumes there is one. If there
            are multiple, the order of which they are handled can produce different group outcomes.
            The number of different possible matches grow exponentially as more ambiguous pairs are made
        '''

        new_unmatched_prevs = []
        # entry in ambig_currs is (cam_name, [(track, id), (track, id) ...] )
        for cam, points in ambig_currs:
            track = points[0][0]
            assert all(p[0]==track for p in points)

            reproj_diff = {}
            for track_idx, id_ in points:
                # §3.1.1: a candidate id with no group this frame can't be
                # evaluated; and skip ids whose group already has this cam
                # (would add a second instance from the same view).
                if id_ not in self.groups_by_id:
                    continue
                if cam in self.groups_by_id[id_].points_by_cam:
                    continue
                # extract the group and its original reprojection score
                pot_group, pot_group_reproj_score, og_reproj_score = self._eval_potential_group(cam, track_idx, id_)
                diff = pot_group_reproj_score - og_reproj_score
                # §3.1.2: NaN never compares True, so it would make min()
                # order-dependent -> drop non-finite candidates.
                if not np.isfinite(diff):
                    continue
                reproj_diff[id_] = (pot_group, diff)

            # If nothing is assignable, every candidate id for this curr is
            # released back to the hidden/unmatched-prev pool.
            if not reproj_diff:
                new_unmatched_prevs.extend([(cam, p[1]) for p in points])
                continue

            # the new group with the lowest reproj_diff (smallest change) gets
            # the new instance; the other instances are pooled into unmatched_prev
            update_id = min(reproj_diff, key=lambda k: reproj_diff[k][1])
            self.groups_by_id[update_id] = reproj_diff[update_id][0] # new group at index 0

            new_unmatched_prevs.extend([(cam, p[1]) for p in points if p[1] != update_id])

        return new_unmatched_prevs

        
    def _resolve_ambiguous_prevs(self, ambig_prevs) -> list[tuple[str, int]]:
        pass


    def _resolve_unmatched_currs(self, unmatched_currs) -> list[Group]:
        '''
        # 1. check to see if it can be matched with other tracks: low reproj score --> join group
        # 2. leftover unmatched current instances; make new groups
        '''
        leftovers: list[tuple[str, int]] = []

        for cam, track_idx, potential_ids in unmatched_currs:
            if not potential_ids: 
                leftovers.append((cam, track_idx))
                continue
            
            # iterate through the potential_ids:
            
            reproj_diff = {}
            for pot_id in potential_ids:

                # make sure the id exists
                if pot_id not in self.groups_by_id:
                    continue

                # §3.2.1 (mirror of §3.1.1): can't add a second instance from
                # this same cam to a group that already has it.
                if cam in self.groups_by_id[pot_id].points_by_cam:
                    continue

                pot_group, pot_score, og_score = self._eval_potential_group(cam, track_idx, pot_id)

                diff = pot_score - og_score
                # §3.2.5: NaN comparison is False (silently dropped); make the
                # finite-only guard explicit so the polarity can't be flipped.
                if np.isfinite(diff) and diff < self.MAX_POTENTIAL_REPROJ_THRESHOLD:
                    reproj_diff[pot_id] = (pot_group, diff)

            if reproj_diff:
                update_id = min(reproj_diff, key=lambda k: reproj_diff[k][1])
                self.groups_by_id[update_id] = reproj_diff[update_id][0]
                # TODO: remove `update_id` from any potential other unmatched_currs
            else:
                # put the instance in the leftovers
                leftovers.append((cam, track_idx))

        # make new groups from leftovers
        groups = []
        if len(leftovers) < 2:
            self.nonmatch_instances.extend(leftovers)
            return groups
            
        
        target_nodes = [self.instance_by_cam[cam][track_idx][0] for cam, track_idx in leftovers]
        self.edges = self._calc_edge_weights(target_nodes=target_nodes)
        new_groups = self._run_union_find()

        return new_groups                
            
        # for cam, points in ambig_currs:
        #     track = points[0][0]
        #     assert all(p[0]==track for p in points)

        #     reproj_diff = {}
        #     for track_idx, id_ in points:
        #         # extract the group and its original reprojection score
        #         pot_group, pot_group_reproj_score, og_reproj_score = self._eval_potential_group(cam, track_idx, id_)    
        #         reproj_diff[id_] = (pot_group, pot_group_reproj_score - og_reproj_score)


    def _eval_potential_group(self, cam, track_idx, id_):
        group = self.groups_by_id[id_]

        if cam in group.points_by_cam:
            raise AssertionError
            return None, float('inf'), 0

        og_reproj_score = np.mean(list(group.reproj_scores.values()))

        # make a new group with the addition of the new ambiguous instance
        point = self.instance_by_cam[cam][track_idx][-1]
        potential_group = Group(
            group = [(c, pts) for c, pts in group.points_by_cam.items()] + [(cam, point)],
            cam_map=self.cam_map,
            proj_cache=self.proj_cache
        )


        if potential_group.reproj_scores is None:
            potential_group_reproj_score = 0
        else:    
            potential_group_reproj_score = np.mean(list(potential_group.reproj_scores.values()))

        return potential_group, potential_group_reproj_score, og_reproj_score


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

    
    def __repr__(self):
        return f'frame: {self.frame_idx}; {self.cam_count_dict}'
        


class MultiFrameTrack:

    def __init__(
        self,
        all_labels,
        start: int = 0,
        end: int = None,
        palette: dict = None,
        node_weights: dict =None,
        max_ids: int = None,
    ):
        """Track multi-view detections without any GUI dependency.

        Params:
          all_labels: a MultiviewLabels (josh_source.multiview_labels). The
                      tracker reads .cameras / .skeleton / .frame_group(i) /
                      .max_frame and never touches a Qt object. To render
                      assignments back into a live GUI, call
                      `push_assignments_to_gui(window)` with the LucidLiteWindow.
        """
        self.all_labels = all_labels
        self.proj_cache = ProjCache()
        self.start = start
        self.end = end if end else all_labels.max_frame
        self.palette = palette if palette else {}
        self.max_ids = max_ids
        self.next_avail_id = 0

        if node_weights is None:
            self.node_weights = {node: 1 for node in all_labels.skeleton.nodes}
        else:
            self.node_weights = node_weights



        # attributes to be updated later
        self.frames = []
        self.trackIds = []
        self.invalid_instances = []
        self.nonmatch_instances = []
        self.nonmatch_groups = []
        self.visible_ids = []


    def track(self, verbose=False):

        prev_trackIds = None
        if not self.max_ids:
            max_ = 0
            for i in range(self.end):
                fg = self.all_labels.frame_group(i)
                if fg is None:
                    continue
                lst = list(fg.instances.values())
                num = max((len(sublist) for sublist in lst), default=0)
                max_ = max(max_, num)
            self.max_ids = max_

        for frame_idx in tqdm(range(self.start, self.end)):
            fg = self.all_labels.frame_group(frame_idx)
            if fg is None: continue

            sft = SingleFrameTrack(
                fg, self.all_labels.cameras,
                node_weights=self.node_weights, proj_cache=self.proj_cache,
                prev_trackIds=prev_trackIds, next_avail_id=self.next_avail_id,
                max_ids=self.max_ids,
            )
            self.next_avail_id = sft.next_avail_id
            if sft.trackIds:
                prev_trackIds = sft.trackIds
            self.frames.append(sft)


            # update attributes
            self.trackIds.append(sft.trackIds)
            self.visible_ids.append(tuple(sft.visible_ids))
            self.invalid_instances.extend(sft.invalid_instances)
            self.nonmatch_instances.extend(sft.nonmatch_instances)
            # self.nonmatch_groups.extend(sft.nonmatch_groups)

        if verbose: self.print_results()


    def push_assignments_to_gui(self, window):
        '''
        Renders identity assignments to a LUC3D-LITE GUI window.

        Params:
          window: a LucidLiteWindow whose `session` will receive the
                  assignments. Required — the tracker is now GUI-free; this
                  method is the only adapter that talks to a Session.
        '''
        session = window.session

        session.set_color_mode("identity")

        with session.batch_updates():
            for sft in self.frames:
                # Group no longer stores cam_track. The point arrays held in
                # group.points_by_cam are the same objects as in instance_list,
                # so map each array back to its (cam, track_idx) by identity.
                pts_to_track: dict[int, tuple[str, int]] = {
                    id(pts): (cam_name, track_idx)
                    for track_idx, cam_name, pts in sft.instance_list
                }

                assignments: dict[tuple[str, int], int] = {}
                if sft.trackIds is not None:
                    for ident_id, ti in sft.trackIds.items():
                        if ti.group is None:
                            continue
                        # group.points_by_cam is now {cam -> single pts ndarray},
                        # not {cam -> list[pts]}. Iterating .values() yields each
                        # pts array directly; id() on those matches pts_to_track.
                        for pts in ti.group.points_by_cam.values():
                            key = pts_to_track.get(id(pts))
                            if key is not None:
                                assignments[key] = ident_id

                # 3. Snapshot every visible (cam, track) at this frame so we know which
                #    instances need the -1 sentinel (didn't land in any group).
                visible_pairs: list[tuple[str, int]] = []
                for cam in session.camera_names():
                    seen: set[int] = set()
                    fg = session.frame_group(sft.frame_idx)
                    for inst in fg.get_instances(cam):
                        if inst.track_idx is None or inst.track_idx in seen:
                            continue
                        seen.add(inst.track_idx)
                        visible_pairs.append((cam, inst.track_idx))


                # 4. Create any identity_id the tracker produced that doesn't exist.
                needed_ids = set(assignments.values())
                existing_ids = {i.id for i in session.identities}
                added = needed_ids - existing_ids
                for ident_id in sorted(added):
                    if  ident_id in self.palette:
                        name, color = self.palette[ident_id]
                    else:
                        name = f"id_{ident_id}"
                        color = next_palette_color(len(session.identities))
                    session.identities.append(Identity(id=ident_id, name=name, color=color))
                    session._identity_counter = max(session._identity_counter, ident_id + 1)
                if added:
                    session._emit("identities_changed")

                # 5. Reset prior per-frame overrides at this frame.
                prefix = f"{sft.frame_idx}:"
                for key in [k for k in session.frame_identity_map if k.startswith(prefix)]:
                    del session.frame_identity_map[key]

                # 6. -1 sentinel for visible pairs not covered by any group.
                assigned = set(assignments.keys())
                for (cam, track) in visible_pairs:
                    if (cam, track) not in assigned:
                        session.frame_identity_map[f"{sft.frame_idx}:{cam}:{track}"] = -1

                # 7. Write the tracker-derived assignments verbatim.
                for (cam, track), ident_id in assignments.items():
                    session.frame_identity_map[f"{sft.frame_idx}:{cam}:{track}"] = ident_id

                session._emit("identity_map_changed")

                # 8. Store the full bundle so graph_window + triangulation_3d_window
                #    can draw edges/skeletons. Both subscribe to
                #    `session.frame_tracker_groups[frame_idx]` and the
                #    `groups_changed` signal; without this they stay empty.
                session.set_groups_for_frame(
                    sft.frame_idx,
                    sft.groups,
                    adjacency_matrix=sft.adjacency_matrix,
                    instance_list=sft.instance_list,
                    trackIds=sft.trackIds,
                )

                    

    def print_results(self):
            label_width = 28
            num_width = 12

            print(f'{"Num invalid instances":<{label_width}} {len(self.invalid_instances):>{num_width}}')
            print(f'{"Num nonmatch instances":<{label_width}} {len(self.nonmatch_instances):>{num_width}}')
            print(f'{"Num nonmatch groups":<{label_width}} {len(self.nonmatch_groups):>{num_width}}')
            print()

            total_insts = sum(sft.n_insts for sft in self.frames)
            print(f'{"Total Instances":<{label_width}} {total_insts:>{num_width}}')

            print(f'{"Fraction Valid":<{label_width}} {(total_insts - len(self.invalid_instances)) / total_insts:>{num_width}.5f}')
            print(f'{"Fraction Matched Single":<{label_width}} {(total_insts - len(self.nonmatch_instances)) / total_insts:>{num_width}.5f}')
            # print(f'{"Fraction Matched With Group":<{label_width}} {(total_insts - len(self.nonmatch_instances) - nonmatch_group_instances) / total_insts:>{num_width}.5f}')
