import numpy as np


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


def calc_fundamental_matrix(cam2, cam1):
    K1 = np.asarray(cam1.matrix, dtype=np.float64)
    K2 = np.asarray(cam2.matrix, dtype=np.float64)
    t1 = np.asarray(cam1.tvec, dtype=np.float64).reshape(-1, 1)
    t2 = np.asarray(cam2.tvec, dtype=np.float64).reshape(-1, 1)
    R1 = rodrigues(cam1.rvec)
    R2 = rodrigues(cam2.rvec)    

    # get coords in cam 1 w.r.t cam 2
    E_R = R2 @ R1.T
    # print(E_R.shape)
    # print(t1.shape)
    # return
    E_T = -E_R @ t1 + t2

    E = skew(E_T) @ E_R

    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F
