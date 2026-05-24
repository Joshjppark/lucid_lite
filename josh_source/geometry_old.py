
import cv2
import logging

import numpy as np
from numpy import ndarray



logging.basicConfig(level=logging.INFO)

def rodrigues(rvec):
    theta = np.linalg.norm(rvec)
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
    
    if arr.ndim == 2 and arr.shape[-1] == 2:
        return np.hstack([arr, np.ones((arr.shape[0], 1))])

    raise AssertionError


def dehomogenize(arr: ndarray):

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


def calc_epipolar_score(inst1, cam1, inst2, cam2, cache):
    
    non_homog_inst1 = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst1.points])
    non_homog_inst2 = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst2.points])

    inst1, inst2 = homogenize(non_homog_inst1), homogenize(non_homog_inst2)


    logging.debug(f'inst1 {inst1.shape}; inst2 {inst2.shape}')

    F = cache.getF(cam2, cam1)

    total_error = 0
    lines = F @ inst1.T  # shape (3, n_nodes)
    error = np.abs(np.sum(inst2.T * lines, axis=0)) / np.linalg.norm(lines[:2, :], axis=0)


    # print(f'total, {total}')
    neg_ave = -np.nanmean(error)
    return np.exp(neg_ave / 10)


def triangulate_dlt(points, Ps):

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


def triangulate_group(group: dict, cam_map, cache):
    '''group = dict[cam_name, Instance]. Returns homogenized np.array of 3D points'''
    cam_names = list(group.keys())

    cams = [cam_map[cam] for cam in cam_names]
    insts = [group[cam] for cam in cam_names]
    Ps = [cache.getP(cam) for cam in cams]

    n_nodes = len(insts[0].points)
    points3D = []

    undistorted_pts = []
    for cam, inst in zip(cams, insts):
        K = np.array(cam.matrix)
        pts = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst.points])
        undistorted_pts.append(cv2.undistortPoints(pts, K, np.array(cam.dist), P=K).squeeze(1))

    for n in range(n_nodes):
        points2D = []
        for cam, pt_undist in zip(cams, undistorted_pts):

            pt = pt_undist[n]
            if np.any(np.isnan(pt)):
                continue
            
            points2D.append(pt)

        if len(points2D) < 2:
            points3D.append(np.full(4, np.nan))
        else:
            points2D = np.vstack(points2D)
            points3D.append(triangulate_dlt(points2D, Ps))
    
    # return points2D, Ps
    return np.vstack(points3D)


def reproject_points(X, P):
    assert P.shape == (3, 4)
    assert X.shape[1] == 4
    reprojs = P @ X.T # shape 3, n_nodes

    return reprojs.T


def instance_pixel_distance(reproj, inst) -> float:

    assert reproj.shape[1] == 3
    reproj = dehomogenize(reproj)
    
    pts = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst.points])
    norms = np.linalg.norm(reproj - pts, axis=1)
    return np.nanmean(norms)



def calc_reprojection_score(inst1, cam1, inst2, cam2, cache):
    
    '''
    make a 3D model of the given points, reproject back to the cameras,
    then sum the L2 distances from the reprojections to the instances

    '''

    pts1 = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst1.points])
    pts2 = np.asarray([p if p is not None else (np.nan, np.nan) for p in inst2.points])

    # get undistorted points
    P1, P2 = cache.getP(cam1), cache.getP(cam2)
    K1, K2 = np.array(cam1.matrix), np.array(cam2.matrix)
    pts1_undistort = cv2.undistortPoints(pts1, K1, np.array(cam1.dist), P=K1).squeeze(1)
    pts2_undistort = cv2.undistortPoints(pts2, K2, np.array(cam2.dist), P=K2).squeeze(1)

    
    # get 3D image
    n_nodes = pts1.shape[0]

    points_3D = []

    for n in range(n_nodes):
        x1, y1 = pts1_undistort[n, :]
        x2, y2 = pts2_undistort[n, :]


        if np.any(np.isnan([x1, y1, x2, y2])):
            points_3D.append([np.nan] * 4)
            continue

        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :],
        ])

        _, _, Vt =  np.linalg.svd(A)
        
        X = Vt[-1, :]
        points_3D.append(X)

    # return points_3D
    points_3D = np.vstack(points_3D)

    # reprojections
    pts1_reproject = P1 @ points_3D.T # shape (3, n_nodes)
    pts1_reproject = dehomogenize(pts1_reproject.T)
    pts2_reproject = P2 @ points_3D.T # shape (3, n_nodes)
    pts2_reproject = dehomogenize(pts2_reproject.T)

    reproj_image1_error = np.linalg.norm(pts1 - pts1_reproject, axis=1)
    reproj_image2_error = np.linalg.norm(pts2 - pts2_reproject, axis=1)

    # Object-Keypoint Similarity Gaussian
    sigma = 20 # pixels
    gauss_sum = 0.5 * (np.exp(-reproj_image1_error**2/ (2 * sigma**2)) + 
                       np.exp(-reproj_image2_error**2/ (2 * sigma**2))) 


    return np.nanmean(gauss_sum)
