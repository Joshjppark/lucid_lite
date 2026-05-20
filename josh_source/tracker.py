
import logging
import numpy as np
from numpy import ndarray

from dataclasses import dataclass

from .geometry import triangulate_group, reproject_points, instance_pixel_distance
from .geometry import calc_epipolar_score, calc_reprojection_score, calc_fundamental_matrix
from .geometry import *

from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)


@dataclass
class TrackerCache:
    F: dict
    P: dict


    def __init__(self):
        self.F = {}
        self.P = {}

    
    def getF(self, cam2, cam1) -> np.ndarray:
        key = (cam2.name, cam1.name)
        if key in self.F:
            return self.F[key]
        
        F = calc_fundamental_matrix(cam2, cam1)
        self.F[key] = F

        return F
    
    def getP(self, cam) -> np.ndarray:

        P = self.P.get(cam.name, None)
        if P is None:
            R = rodrigues(cam.rvec)
            P = cam.matrix @ np.hstack([R, np.asarray(cam.tvec).reshape(-1, 1)])
            self.P[cam.name] = P
        return P



def get_instances(frame_group, cameras):
    cam_instances: dict[str, list] = {}
    cam_map: dict[str, object] = {}
    active_cams: list[str] = []

    for cam in cameras:
        instances: list = []
        cam_map[cam.name] = cam

        for inst in frame_group.get_instances(cam.name):
            instances.append(inst)

        for inst in frame_group.unlinked_instances.get(cam.name, []):
            instances.append(inst)
        
        if len(instances) > 0:
            active_cams.append(cam.name)
            cam_instances[cam.name] = instances

    return cam_instances, cam_map, active_cams


def hungarian(cost):

    # cost = np.where(np.isfinite(cost), cost, 1e18) # set inf values to 1e18
    rows, cols = linear_sum_assignment(cost)

    out = np.full(cost.shape[0], -1, dtype=int)
    for r, c in zip(rows, cols):
        out[r] = c
    return out


def cross_view_score(insts1, cam1, insts2, cam2, caches) -> float:
    """0.4 * epipolar + 0.6 * reprojection."""
    epipolar_score = calc_epipolar_score(insts1, cam1, insts2, cam2, caches)
    reproj_score = calc_reprojection_score(insts1, cam1, insts2, cam2, caches)
    return 0.4 * epipolar_score + 0.6 * reproj_score


def match_pairwise(frame_group, session, cache, num_animals=0):


    cam_instances, cam_map, active_cams = get_instances(frame_group, session.cameras)
    cam_by_count = sorted(active_cams, key=lambda x: -len(cam_instances[x]))

    most1, most2 = cam_by_count[0], cam_by_count[1]
    cam1, cam2 = cam_map[most1], cam_map[most2]
    insts1 = cam_instances[cam1.name]
    insts2 = cam_instances[cam2.name]
    ordered_cams = [most1, most2] + [c for c in active_cams if c != most1 and c != most2]

    logging.info(f'cam1: {cam1.name}, cam2: {cam2.name}')

    # add epipolar error into scoring matrix
    n_a, n_b = len(insts1), len(insts2)
    score_matrix = np.zeros((n_a, n_b))
    for a in range(n_a):
        for b in range(n_b):
            # get epipolar score
            score = calc_epipolar_score(insts1[a], cam1, insts2[b], cam2, cache)
            score_matrix[a, b] = score


    # get assignsments into matches
    assignment = hungarian(-score_matrix)
    matches: list = []
    for r in range(n_a):
        c = assignment[r]
        score = score_matrix[r, c]

        
        matches.append({
            'a': int(r),
            'b': int(c),
            'score': score
        })
    matches.sort(key=lambda m: -m['score'])


    # refine the matching score with reprojection score
    for m_i in range(len(matches)):
        m = matches[m_i]
        new_score = cross_view_score(insts1[m['a']], cam1, insts2[m['b']], cam2, cache)
        m['score'] = new_score
    matches.sort(key=lambda m: -m["score"])

    if num_animals:
        matches = matches[:num_animals]
    else:
        matches = [m for m in matches if m['score'] > 0.05]

    # make Groups list of identity groups
    groups = []
    matched1, matched2 = set(), set()
    for m in matches:
        groups.append({
            most1: insts1[m['a']],
            most2: insts2[m['b']]
        })
        matched1.add(m['a'])
        matched2.add(m['b'])

    
    # pad out if num_animals is defined
    if num_animals and len(groups) < num_animals:
        for a in range(n_a):
            if len(groups) >= num_animals:
                break
            if a not in matched1:
                groups.append({most1: insts1[m['a']]})
        for b in range(n_b):
            if len(groups) >= num_animals:
                break
            if b not in matched2:
                groups.append({most2: insts2[m['b']]})

    
    if not num_animals:
        raise NotImplementedError
    
    
    # add remaining cameras via reprojection-distance Hungarian
    for cam_name in ordered_cams[2:]:
        cam3 = cam_map[cam_name]
        insts3 = cam_instances.get(cam3.name, [])
        if not insts3:
            continue
        
        costs3 = np.full((len(groups), len(insts3)), np.inf)
        P3 = cache.getP(cam3)
        
        for group_i, group in enumerate(groups):

            X = triangulate_group(group, cam_map, cache)

            reprojs = reproject_points(X, P3)

            for inst_i in range(len(insts3)):
                reproj_error = instance_pixel_distance(reprojs, insts3[inst_i])

                costs3[group_i, inst_i] = reproj_error
        assignment3 = hungarian(costs3)

        # print(f'{cam3.name},\n {insts3}')
        # return costs3
        # print(f'assign3 is {assignment3}')
        # print(f'length of groups is {len(groups)}')

        for group_i in range(len(groups)):
            inst_i = assignment3[group_i]

            if inst_i < 0 or inst_i >= n_b: continue

            if inst_i < len(insts3) and costs3[group_i, inst_i] < 100:
                # add this instance to the group
                groups[group_i][cam_name] = insts3[inst_i]

        if not num_animals:
            raise NotImplementedError
        


    return groups



def match_frame_instances(frame_group, cameras, session):
    pass