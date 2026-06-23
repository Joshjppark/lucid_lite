import cv2
import logging

import numpy as np


def rodrigues(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3)

    u = rvec / theta
    ux, uy, uz = u
    u_skew = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0],
    ])
    return np.eye(3) + np.sin(theta) * u_skew + (1-np.cos(theta)) * (u_skew@u_skew)

def skew(v):
    v = v.reshape(-1,)
    x, y, z = v
    v_skew = np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])
    return v_skew


def homogenize(arr):
    if arr.ndim == 1:
        return np.concatenate([arr, [1]]).reshape(-1, 1)
    
    if arr.ndim == 2 and arr.shape[-1] in (2, 3):
        return np.hstack([arr, np.ones((arr.shape[0], 1))])

    raise AssertionError


def dehomogenize(arr: np.ndarray):

    if arr.shape[1] == 3:
        return arr[:, :2] / arr[:, -1].reshape(-1, 1)

    if arr.shape[1] == 4:
        return arr[:, :3] / arr[:, -1].reshape(-1, 1)


def calc_fundamental_matrix(cam1, cam2):
    K1 = np.asarray(cam1.matrix, dtype=np.float64)
    K2 = np.asarray(cam2.matrix, dtype=np.float64)
    t1 = np.asarray(cam1.tvec, dtype=np.float64).reshape(-1, 1)
    t2 = np.asarray(cam2.tvec, dtype=np.float64).reshape(-1, 1)
    R1 = rodrigues(cam1.rvec)
    R2 = rodrigues(cam2.rvec)    

    # get coords in cam 1 w.r.t cam 2
    E_R = R2 @ R1.T
    E_T = -E_R @ t1 + t2

    E = skew(E_T) @ E_R

    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


def calc_epipolar_score(pt1, pt2, F, node_weights=None, ref_scale=1.0):


    pt1, pt2 = homogenize(pt1), homogenize(pt2)

    logging.debug(f'inst1 {pt1.shape}; inst2 {pt2.shape}')
    lines1 = calc_epipolar_lines(F.T, pt2) # shape (3, N__nodes
    lines2 = calc_epipolar_lines(F, pt1) # shape (3, N__nodes

    # print(f'node_weights is {node_weights}')
    if node_weights is None:
       node_weights = np.ones(pt1.shape[0])

    error1 = np.abs(node_weights * np.sum(pt1 * lines1.T, axis=1)) / (np.linalg.norm(lines1.T[:, :2], axis=1))
    error2 = np.abs(node_weights * np.sum(pt2 * lines2.T, axis=1)) / (np.linalg.norm(lines2.T[:, :2], axis=1))
    assert error1.shape == error2.shape == (pt1.shape[0],)
    err = [error1, error2]

    # ref_scale normalizes the pixel error by the instances' apparent size so
    # the score is a unitless fraction of body length (depth-invariant).
    # Default 1.0 preserves the original raw-pixel behavior.
    return np.nanmean(err) / ref_scale
    # print(f'total, {total}')
    # neg_ave = -np.nanmean(error)
    # return np.exp(neg_ave / 10)


def calc_epipolar_lines(F, pt):
    assert pt.shape[1] == 3
    return F @ pt.T




def triangulate_dlt(points, Ps):
    '''
    Triangulate 3D points using Direct Linear Transformation (DLT)
    points: 
    '''
    A_s = []
    for pt, P in zip(points, Ps):
        x1, y1 = pt

        A = np.array([
            x1 * P[2, :] - P[0, :],
            y1 * P[2, :] - P[1, :],
        ])
        A_s.append(A)

    _, _, Vt =  np.linalg.svd(np.vstack(A_s))
    
    X = Vt[-1, :]
    return X


def triangulate_group(group: dict, cam_map, cache, get_reprojections=False) -> np.ndarray:
    '''group = dict[cam_name, Instance]. Returns homogenized np.array of 3D points'''

    assert len(group) > 1

    cam_names = list(group.keys())

    cams = [cam_map[cam] for cam in cam_names]
    points = [group[cam] for cam in cam_names]
    Ps = [cache.getP(cam) for cam in cams]

    n_nodes = len(points[0])
    points3D = []

    undistorted_pts = []
    for cam, pt in zip(cams, points):
        K = np.array(cam.matrix)
        # pts = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst.points])
        undistorted_pts.append(cv2.undistortPoints(pt, K, np.array(cam.dist), P=K).squeeze(1))

    for n in range(n_nodes):
        points2D = []
        valid_Ps = []
        for cam, pt_undist, P in zip(cams, undistorted_pts, Ps):

            pt = pt_undist[n]
            if np.any(np.isnan(pt)):
                continue
            
            valid_Ps.append(P)
            points2D.append(pt)

        if len(points2D) < 2:
            points3D.append(np.full(4, np.nan))
        else:
            points2D = np.vstack(points2D)
            points3D.append(triangulate_dlt(points2D, valid_Ps))
    
    points3D = np.vstack(points3D)


    if get_reprojections:
        reprojs = {}
        for cam, P in zip(cams, Ps):
            reprojs[cam.name] = reproject_points(points3D, P)
        
        return dehomogenize(points3D), reprojs

    
    # return points2D, Ps
    return dehomogenize(points3D)


def reproject_points(X, P):
    if X.shape[1] == 3:
        X = homogenize(X)

    assert P.shape == (3, 4)
    assert X.shape[1] == 4
    reprojs = P @ X.T # shape 3, n_nodes

    return reprojs.T


def instance_pixel_distance(reproj, pts, ref_scale=1.0) -> float:

    assert reproj.shape[1] == 3
    reproj = dehomogenize(reproj)

    # pts = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst.points])
    norms = np.linalg.norm(reproj - pts, axis=1)

    # print(f'norms is {norms}, {reproj.shape}, {pts.shape}')
    # ref_scale normalizes by apparent instance size (unitless fraction of body
    # length); default 1.0 preserves the original raw-pixel behavior.
    return np.nanmean(norms) / ref_scale


def apparent_scale(pts: np.ndarray, floor: float = 1.0) -> float:
    '''
    Apparent pixel scale of an instance from its valid (non-NaN) nodes.

    pts: (n_nodes, 2) undistorted pixel coords, NaN for missing nodes.
    Returns the RMS spread of the valid nodes about their centroid, floored.
    Dividing a pixel reprojection error by this value cancels the ~f/Z depth
    dependence, yielding a unitless "fraction of a body length" error.

    RMS (rather than bbox diagonal) is robust to a single outlier node and
    degrades gracefully as nodes drop out. The floor prevents a near-zero
    denominator (degenerate / near-coincident nodes) from exploding the score.
    '''
    valid = pts[~np.isnan(pts).any(axis=1)]
    if valid.shape[0] < 2:
        return floor
    centroid = valid.mean(axis=0)
    rms = np.sqrt(np.mean(np.sum((valid - centroid) ** 2, axis=1)))
    return max(float(rms), floor)
