import numpy as np

def calculate_epipolar_distance(pt1, pt2, cam1, cam2, cameras):
    """
    Calculates the mean epipolar distance between two sets of 2D keypoints.
    
    pt1, pt2: numpy arrays of shape (N_keypoints, 2) representing inhomogeneous coordinates (x).
    cam1, cam2: identifiers to fetch the correct Fundamental Matrix.
    """
    # 1. Fetch the 3x3 Fundamental Matrix for this camera pair
    # (You will need to implement this fetch based on your specific calibration file structure)
    F = get_fundamental_matrix(cam1, cam2, cameras)


    # 2. Filter out missing keypoints (NaNs)
    # SLEAP instances often have NaNs for occluded nodes. We only want to calculate
    # distance for nodes that are visible in BOTH cameras.
    valid_mask = ~np.isnan(pt1[:, 0]) & ~np.isnan(pt2[:, 0])
    
    if not np.any(valid_mask):
        # If they share absolutely no valid keypoints, return infinity
        return np.inf 
        
    x1_valid = pt1[valid_mask] # Inhomogeneous coordinates (x, y)
    x2_valid = pt2[valid_mask]
    
    # 3. Convert x1 to homogeneous coordinates (X1)
    N = x1_valid.shape[0]
    X1 = np.ones((N, 3))
    X1[:, :2] = x1_valid
    
    # 4. Calculate the epipolar lines in camera 2
    # Equation: l = F * X1
    # We transpose X1 to multiply, then transpose back so lines_in_cam2 is shape (N, 3)
    lines_in_cam2 = (F @ X1.T).T
    
    # 5. Calculate the perpendicular distance from x2 to the epipolar lines
    # A line is represented as [a, b, c], where ax + by + c = 0
    a = lines_in_cam2[:, 0]
    b = lines_in_cam2[:, 1]
    c = lines_in_cam2[:, 2]
    
    u2 = x2_valid[:, 0]
    v2 = x2_valid[:, 1]
    
    # Distance Formula: D = |a*u + b*v + c| / sqrt(a^2 + b^2)
    numerators = np.abs(a * u2 + b * v2 + c)
    denominators = np.sqrt(a**2 + b**2)
    
    # Prevent division by zero just in case of a degenerate line
    denominators[denominators == 0] = 1e-6 
    
    distances = numerators / denominators
    
    # 6. Return the aggregate distance
    return np.mean(distances)



import numpy as np

def get_fundamental_matrix(cam1, cam2, cameras):
    """Compute F such that x2.T @ F @ x1 = 0 for corresponding points
    in (undistorted) pixel coordinates of cam1 and cam2.
    """
    by_name = {c.name: c for c in cameras}
    c1, c2 = by_name[cam1], by_name[cam2]

    # World -> camera extrinsics
    R1, t1 = c1.extrinsic_matrix[:3, :3], c1.extrinsic_matrix[:3, 3]
    R2, t2 = c2.extrinsic_matrix[:3, :3], c2.extrinsic_matrix[:3, 3]

    # Relative pose: cam1 frame -> cam2 frame
    R = R2 @ R1.T
    t = t2 - R @ t1

    # Skew-symmetric [t]_x
    tx = np.array([
        [    0, -t[2],  t[1]],
        [ t[2],     0, -t[0]],
        [-t[1],  t[0],     0],
    ])

    # Essential then fundamental
    E = tx @ R
    F = np.linalg.inv(c2.matrix).T @ E @ np.linalg.inv(c1.matrix)
    return F