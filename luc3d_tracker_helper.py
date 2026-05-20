"""Python port of luc3d/pose/tracker.js — cross-view instance matching + identity assignment.

Drop-in companion to lucid-lite. Operates on a live `pose_data.Session`
(see gui_source/pose_data.py); mutates `session.identities`,
`session.track_identity_map`, and `session.frame_identity_map` exactly the
way the JS app does. Visualization happens in luc3d_tracker.ipynb.

What's ported
-------------
- `compute_fundamental_matrix(cam_a, cam_b)` — F such that x_b^T F x_a = 0
  for *undistorted* pixel coords.
- `epipolar_score`, `reprojection_score`, `cross_view_score` — exact ports
  of the JS scoring functions (σ = 20 px, 0.4/0.6 mix).
- `match_pairwise` — sort cameras by candidate count, Hungarian on the
  best pair, refine top matches with reprojection-OKS, graft remaining
  cameras via reprojection distance.
- `reorder_groups_by_prev` — 4-signal Hungarian to keep identities stable
  across frames (reprojection / 3D-distance / OKS / track continuity ×2).
- `match_frame_instances` — single-frame entry point.
- `track_all` — full sweep mirroring `trackAll()`; **wipes identities** on
  entry like the JS does.

Bug parity
----------
By default `match_frame_instances` and `track_all` run with
`enforce_uniqueness=False` so the Python sweep reproduces the JS web app's
duplicate-identity bug exactly. The bug surfaces wherever
`reorder_groups_by_prev` silently drops a group (because the padded
Hungarian rows claimed it), since the dropped group's (cam, track) keys
never get per-frame overrides written and the overlay then falls back to
whatever `track_identity_map` carried forward from earlier frames. On
10072022145420_small with `num_animals=None`, this manifests as duplicate
id_3 at frame 148 midL across 33 frames in a 1800-frame sweep.

Pass `enforce_uniqueness=True` to either function to enable the per-frame
`-1` sentinel guard at the end of `match_frame_instances`, which suppresses
the stale-global fallback and eliminates the visible duplicates without
changing the matcher's identity count.

Caveats
-------
- Hungarian is a direct Python port of `triangulation.js:hungarianAlgorithm`
  (Jonker-Volgenant). We use a port rather than `scipy.optimize.linear_sum_assignment`
  because the JS solver tie-breaks differently on equal-cost assignments and
  pads rectangular matrices to square with zeros — both of which materially
  change the bootstrap / graft / reorder outputs on real data.
- Distortion: uses `cv2.undistortPoints` if OpenCV is installed; otherwise
  raw (already-undistorted) pixels. With analysis.h5 SLEAP exports this
  matters only if `Camera.dist` is non-zero.
- F-matrix cache: identical to JS (per-run, never invalidated — cameras
  don't move).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _cv2 = None
    _HAS_CV2 = False


# ===========================================================================
# Geometry primitives
# ===========================================================================

def rodrigues(rvec) -> np.ndarray:
    """Rotation vector (3,) -> 3x3 rotation matrix. Matches cv2.Rodrigues."""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2],  0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def projection_matrix(cam) -> np.ndarray:
    """P = K [R | t], shape (3, 4). World -> ideal pixel."""
    K = np.asarray(cam.matrix, dtype=float)
    R = rodrigues(cam.rvec)
    t = np.asarray(cam.tvec, dtype=float).reshape(3, 1)
    return K @ np.hstack([R, t])


def compute_fundamental_matrix(cam_a, cam_b) -> np.ndarray:
    """F (3x3): x_b^T F x_a = 0 in undistorted-pixel coordinates."""
    K1 = np.asarray(cam_a.matrix, dtype=float)
    K2 = np.asarray(cam_b.matrix, dtype=float)
    R1 = rodrigues(cam_a.rvec)
    R2 = rodrigues(cam_b.rvec)
    t1 = np.asarray(cam_a.tvec, dtype=float).reshape(3)
    t2 = np.asarray(cam_b.tvec, dtype=float).reshape(3)
    R = R2 @ R1.T
    t = t2 - R @ t1
    tx = np.array([[0.0, -t[2], t[1]],
                   [t[2], 0.0, -t[0]],
                   [-t[1], t[0], 0.0]])
    E = tx @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def undistort_points(uv: np.ndarray, cam) -> np.ndarray:
    """Raw observed pixels (N,2) -> ideal pixels (N,2). Identity when cv2 absent or dist ~ 0."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    K = np.asarray(cam.matrix, dtype=float)
    dist = np.asarray(cam.dist, dtype=float).ravel()
    if not _HAS_CV2 or np.allclose(dist, 0):
        return uv
    out = _cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist, P=K)
    return out.reshape(-1, 2)


def triangulate_dlt(obs, Ps) -> Optional[np.ndarray]:
    """obs: list of (2,) per view; Ps: list of (3,4) projection. -> (3,) or None."""
    rows = []
    for (u, v), P in zip(obs, Ps):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    if len(rows) < 4:
        return None
    A = np.vstack(rows)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]


def triangulate_dlt_batch(
    obs_batch: np.ndarray,    # (N, C, 2)
    P_batch: np.ndarray,      # (C, 3, 4)
    valid_mask: np.ndarray,   # (N, C) bool
) -> np.ndarray:
    """Vectorized DLT for N keypoints sharing the same C cameras.

    Each (kp, cam) row pair contributes two equations to the 4-unknown
    homogeneous solve. Padding a row with zeros leaves AᵀA — and therefore
    the right singular vectors of A — unchanged, so we can keep a fixed
    (2C, 4) shape per keypoint and dispatch one batched np.linalg.svd
    over the leading axis. Returns (N, 3) with NaN rows where <2 views
    were available or the homogeneous denominator collapsed.

    This matches the per-node `triangulate_dlt` numerically; the speedup
    comes from amortizing SVD's per-call dispatch overhead.
    """
    N, C, _ = obs_batch.shape
    if N == 0 or C < 2:
        return np.full((N, 3), np.nan)

    u = obs_batch[..., 0]  # (N, C)
    v = obs_batch[..., 1]  # (N, C)
    P0 = P_batch[:, 0, :]  # (C, 4)
    P1 = P_batch[:, 1, :]  # (C, 4)
    P2 = P_batch[:, 2, :]  # (C, 4)

    # Broadcast: (N, C, 4) for each of the two equations.
    row_u = u[..., None] * P2[None, :, :] - P0[None, :, :]
    row_v = v[..., None] * P2[None, :, :] - P1[None, :, :]

    # Zero out invalid observations so they contribute nothing to AᵀA.
    valid3 = valid_mask[..., None]
    row_u = np.where(valid3, row_u, 0.0)
    row_v = np.where(valid3, row_v, 0.0)

    # Interleave per-camera (u, v) row-pairs into A of shape (N, 2C, 4).
    A = np.empty((N, 2 * C, 4), dtype=row_u.dtype)
    A[:, 0::2, :] = row_u
    A[:, 1::2, :] = row_v

    _, _, Vh = np.linalg.svd(A, full_matrices=False)  # Vh: (N, 4, 4)
    Xh = Vh[:, -1, :]
    w = Xh[:, 3]
    safe = np.abs(w) > 1e-12
    n_valid = valid_mask.sum(axis=1)
    insufficient = (n_valid < 2) | (~safe)
    # Avoid divide-by-zero in the unsafe slots — they get NaN'd below.
    w_safe = np.where(safe, w, 1.0)
    X = Xh[:, :3] / w_safe[:, None]
    X[insufficient] = np.nan
    return X



def reproject_point(X, P) -> Optional[np.ndarray]:
    """3D world -> (u,v) ideal pixel. None if behind the camera plane."""
    Xh = np.array([X[0], X[1], X[2], 1.0])
    p = P @ Xh
    if abs(p[2]) < 1e-12:
        return None
    return p[:2] / p[2]


def reproject_points(Xs, P):
    """list of 3D points (may include None) -> list of (u,v) or None."""
    out = []
    for X in Xs:
        if X is None:
            out.append(None)
        else:
            out.append(reproject_point(X, P))
    return out


# ===========================================================================
# Per-run caches  (mirrors _fMatrixCache / _undistortCache in tracker.js)
# ===========================================================================

@dataclass
class TrackerCaches:
    F: dict
    P: dict
    undist: dict

    def __init__(self):
        self.F = {}
        self.P = {}
        self.undist = {}

    def get_F(self, cam_a, cam_b) -> np.ndarray:
        key = (cam_a.name, cam_b.name)
        F = self.F.get(key)
        if F is None:
            F = compute_fundamental_matrix(cam_a, cam_b)
            self.F[key] = F
        return F

    def get_P(self, cam) -> np.ndarray:
        P = self.P.get(cam.name)
        if P is None:
            P = projection_matrix(cam)
            self.P[cam.name] = P
        return P

    def get_undist_kps(self, inst, cam):
        """Per-keypoint undistorted pixels for one Instance in one camera.

        Cached by `id(inst)`. Returns list aligned to inst.points, with None
        wherever the original keypoint is missing.
        """
        key = id(inst)
        cached = self.undist.get(key)
        if cached is not None:
            return cached
        out = []
        K = np.asarray(cam.matrix, dtype=float)
        dist = np.asarray(cam.dist, dtype=float).ravel()
        identity = (not _HAS_CV2) or np.allclose(dist, 0)
        for p in inst.points:
            if p is None:
                out.append(None)
            elif identity:
                out.append((float(p[0]), float(p[1])))
            else:
                uv = _cv2.undistortPoints(
                    np.array([[[p[0], p[1]]]], dtype=float), K, dist, P=K
                ).reshape(2)
                out.append((float(uv[0]), float(uv[1])))
        self.undist[key] = out
        return out


# ===========================================================================
# Scoring
# ===========================================================================

def epipolar_score(inst_a, cam_a, inst_b, cam_b, caches: TrackerCaches) -> float:
    """[0..1]. exp(-mean_line_dist / 10). Higher = better."""
    F = caches.get_F(cam_a, cam_b)
    pts_a, pts_b = inst_a.points, inst_b.points
    n = min(len(pts_a), len(pts_b))
    total = 0.0
    n_valid = 0
    for k in range(n):
        if pts_a[k] is None or pts_b[k] is None:
            continue
        x1, y1 = pts_a[k]
        x2, y2 = pts_b[k]
        l0 = F[0, 0] * x1 + F[0, 1] * y1 + F[0, 2]
        l1 = F[1, 0] * x1 + F[1, 1] * y1 + F[1, 2]
        l2 = F[2, 0] * x1 + F[2, 1] * y1 + F[2, 2]
        num = abs(x2 * l0 + y2 * l1 + l2)
        denom = math.sqrt(l0 * l0 + l1 * l1)
        if denom > 1e-12:
            total += num / denom
            n_valid += 1
    if n_valid == 0:
        return 0.0
    
    neg_ave = -total / n_valid
    # print(f'total, {total}')
    # print(f'neg_ave, {neg_ave}')
    return math.exp(neg_ave / 10.0)


def reprojection_score(inst_a, cam_a, inst_b, cam_b, caches: TrackerCaches) -> float:
    """[0..1]. OKS-style Gaussian on reprojection error; σ = 20 px."""
    u_a = caches.get_undist_kps(inst_a, cam_a)
    u_b = caches.get_undist_kps(inst_b, cam_b)
    P_a = caches.get_P(cam_a)
    P_b = caches.get_P(cam_b)
    n = min(len(u_a), len(u_b))
    sigma2x2 = 2.0 * 20.0 * 20.0
    total = 0.0
    n_valid = 0
    
    total
    X_s = []
    for k in range(n):
        if u_a[k] is None or u_b[k] is None:
            continue
        X = triangulate_dlt([u_a[k], u_b[k]], [P_a, P_b])
        X_s.append(X)
        if X is None:
            continue
        rep_a = reproject_point(X, P_a)
        rep_b = reproject_point(X, P_b)
        if rep_a is None or rep_b is None:
            continue        
        # JS measures error against the RAW (distorted) observations.
        ax, ay = inst_a.points[k]
        bx, by = inst_b.points[k]
        dxA, dyA = ax - rep_a[0], ay - rep_a[1]
        dxB, dyB = bx - rep_b[0], by - rep_b[1]
        # total += np.linalg.norm([dxA, dyA])
        # n_valid += 1
    # return total/n_valid, np.vstack(X_s)

        total += (math.exp(-(dxA * dxA + dyA * dyA) / sigma2x2) +
                  math.exp(-(dxB * dxB + dyB * dyB) / sigma2x2)) / 2.0
        n_valid += 1
    if n_valid == 0:
        return 0.0
    return total / n_valid


def cross_view_score(inst_a, cam_a, inst_b, cam_b, caches: TrackerCaches) -> float:
    """0.4 * epipolar + 0.6 * reprojection."""
    return (0.4 * epipolar_score(inst_a, cam_a, inst_b, cam_b, caches)
            + 0.6 * reprojection_score(inst_a, cam_a, inst_b, cam_b, caches))


# ===========================================================================
# Geometry helpers
# ===========================================================================

def instance_pixel_distance(reproj, points) -> float:
    """Mean Euclidean distance over visible keypoint pairs (inf if none)."""
    n = min(len(reproj), len(points))
    total = 0.0
    n_valid = 0
    for k in range(n):
        if reproj[k] is None or points[k] is None:
            continue
        dx = reproj[k][0] - points[k][0]
        dy = reproj[k][1] - points[k][1]
        total += math.sqrt(dx * dx + dy * dy)
        n_valid += 1
    return total / n_valid if n_valid > 0 else math.inf


def triangulate_group(group, cam_map, caches: TrackerCaches):
    """group = dict[cam_name, Instance]. Returns list of 3D points (or None) per keypoint.

    Numerically equivalent to the per-keypoint DLT loop, but builds one
    (n_kp, 2*C, 4) tensor and dispatches a single batched np.linalg.svd
    (see `triangulate_dlt_batch`).
    """
    cam_names = list(group.keys())
    if len(cam_names) < 2:
        return None
    cams = [cam_map[c] for c in cam_names]
    insts = [group[c] for c in cam_names]
    C = len(cams)
    n_kp = len(insts[0].points)
    if n_kp == 0:
        return []

    # Per-camera projection matrices (cheap — cached).
    P_batch = np.stack([caches.get_P(cam) for cam in cams], axis=0)  # (C, 3, 4)

    obs_batch = np.zeros((n_kp, C, 2), dtype=np.float64)
    valid_mask = np.zeros((n_kp, C), dtype=bool)
    for i, (cam, inst) in enumerate(zip(cams, insts)):
        undist = caches.get_undist_kps(inst, cam)
        for k in range(n_kp):
            uv = undist[k]
            if uv is None:
                continue
            obs_batch[k, i, 0] = uv[0]
            obs_batch[k, i, 1] = uv[1]
            valid_mask[k, i] = True

    X = triangulate_dlt_batch(obs_batch, P_batch, valid_mask)  # (n_kp, 3)
    # `out` matches the per-keypoint API: a list with None entries where the
    # solve was undefined.
    nan_rows = np.isnan(X[:, 0])
    out = [None if nan_rows[k] else X[k] for k in range(n_kp)]
    return out


def hungarian(cost: np.ndarray) -> np.ndarray:
    """Cost (n_rows, n_cols) -> assignment[r] = column or -1 if unmatched.

    Faithful Python port of triangulation.js:599 (JS `hungarianAlgorithm`):
    clamps non-finite entries to a SENTINEL=1e15, transposes if n > m so the
    inner solve has n <= m, pads to square with zeros, runs Jonker-Volgenant
    with the same iteration order, and extracts the rectangular result.

    scipy's `linear_sum_assignment` also solves to optimality, but its
    tie-breaking and rectangular-matrix handling differ from the JS JV
    implementation. When multiple equally-optimal assignments exist (common
    in this dataset's pairwise epipolar costs), the chosen subset can put
    instances into different groups, which downstream causes singletons to
    appear in places where JS produces a clean match. Mirroring the JS
    solver exactly removes that source of divergence — verified on the
    10072022145420_small dataset: identity count goes from 8 → 4 with
    num_animals=None, matching the JS web app.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.zeros(cost.shape[0], dtype=int)

    orig_n, orig_m = cost.shape

    SENTINEL = 1e15
    BIG_INF = float("inf")

    # If everything is non-finite, no real assignment is possible.
    if not np.any(np.isfinite(cost)):
        return np.full(orig_n, -1, dtype=int)

    # Ensure n <= m by transposing.
    transposed = orig_n > orig_m
    if transposed:
        C = np.where(np.isfinite(cost), cost, SENTINEL).T.astype(float)
    else:
        C = np.where(np.isfinite(cost), cost, SENTINEL).astype(float)
    n, m = C.shape  # n <= m now.

    # Pad to square with zeros.
    sz = max(n, m)  # == m since n <= m, but keep symmetric.
    sq = np.zeros((sz, sz), dtype=float)
    sq[:n, :m] = C

    # Jonker-Volgenant. Indices are 1-based to mirror the JS port exactly
    # (translation of triangulation.js:660–705).
    u = [0.0] * (sz + 1)
    v = [0.0] * (sz + 1)
    p = [0] * (sz + 1)     # p[j] = row assigned to col j
    way = [0] * (sz + 1)   # way[j] = previous col in augmenting path

    for i1 in range(1, sz + 1):
        p[0] = i1
        j0 = 0
        minv = [BIG_INF] * (sz + 1)
        used = [False] * (sz + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = BIG_INF
            j1 = -1

            for j in range(1, sz + 1):
                if used[j]:
                    continue
                cur = sq[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j2 in range(0, sz + 1):
                if used[j2]:
                    u[p[j2]] += delta
                    v[j2] -= delta
                else:
                    minv[j2] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Reverse augmenting path.
        while j0:
            j3 = way[j0]
            p[j0] = p[j3]
            j0 = j3

    # Extract assignment, mapping back through the transpose if needed.
    if not transposed:
        out = np.full(orig_n, -1, dtype=int)
        for j4 in range(1, sz + 1):
            if 0 < p[j4] <= n and j4 <= m:
                out[p[j4] - 1] = j4 - 1
    else:
        out = np.full(orig_n, -1, dtype=int)
        for j5 in range(1, sz + 1):
            if 0 < p[j5] <= n and j5 <= m:
                orig_row = j5 - 1
                orig_col = p[j5] - 1
                if orig_row < orig_n and orig_col < orig_m:
                    out[orig_row] = orig_col

    # Final pass: reject any "match" that was actually the SENTINEL (i.e.,
    # the caller's original cost was Infinity/NaN). The caller's <100 threshold
    # would filter most of these anyway, but be explicit so callers reading
    # `out` directly never see a "matched" pair whose true cost was infeasible.
    for r in range(orig_n):
        c = out[r]
        if c >= 0 and not np.isfinite(cost[r, c]):
            out[r] = -1

    return out


# ===========================================================================
# Collect instances per camera
# ===========================================================================

def collect_instances(frame_group, cameras):
    """Same shape as collectInstances() in tracker.js."""
    cam_instances: dict[str, list] = {}
    cam_map: dict[str, object] = {}
    active: list[str] = []
    for cam in cameras:
        cam_map[cam.name] = cam
        flat: list = []
        for inst in frame_group.instances.get(cam.name, []):
            flat.append(inst)
        for ul in frame_group.unlinked_instances.get(cam.name, []):
            flat.append(ul.instance)
        if flat:
            cam_instances[cam.name] = flat
            active.append(cam.name)
    return cam_instances, cam_map, active


# ===========================================================================
# matchPairwise — bootstrap on the densest 2-cam pair, graft the rest
# ===========================================================================

# def match_pairwise(cam_instances, cam_map, active_cams, num_animals,
#                    prev_assignments, caches: TrackerCaches):
#     # Sort by candidate count descending. Pair is [most, second].

def match_pairwise(frame_group, session, caches, num_animals=0, prev_assignments=None):


    cam_instances, cam_map, active_cams = collect_instances(frame_group, session.cameras)

    cams_by_count = sorted(active_cams, key=lambda c: -len(cam_instances[c]))
    best1, best2 = cams_by_count[0], cams_by_count[1]
    cam1, cam2 = cam_map[best1], cam_map[best2]
    insts1, insts2 = cam_instances[best1], cam_instances[best2]
    remaining = [c for c in active_cams if c != best1 and c != best2]
    ordered_cams = [best1, best2] + remaining


    n_a, n_b = len(insts1), len(insts2)
    if n_a == 0 or n_b == 0:
        return []

    # Fast epipolar score matrix
    score_matrix = np.zeros((n_a, n_b))
    for a in range(n_a):
        for b in range(n_b):
            s = epipolar_score(insts1[a], cam1, insts2[b], cam2, caches)
            if prev_assignments is not None:
                pid_a = prev_assignments.get(f"{best1}:{insts1[a].track_idx}")
                pid_b = prev_assignments.get(f"{best2}:{insts2[b].track_idx}")
                if pid_a is not None and pid_b is not None and pid_a == pid_b:
                    s += 0.3
            score_matrix[a, b] = s

    assignment = hungarian(-score_matrix)
    

    matches = []
    for a in range(n_a):
        b = assignment[a]
        if 0 <= b < n_b:
            # print(f'appending {a} and {b}')
            matches.append({
                "a": a,
                "b": int(b),
                "score": float(score_matrix[a, b])
            })
    matches.sort(key=lambda m: -m["score"])


    # Refine top-N with full reprojection score
    top_n = min(num_animals * 2, len(matches)) if num_animals else len(matches)
    for mi in range(top_n):
        m = matches[mi]
        full = cross_view_score(insts1[m["a"]], cam1, insts2[m["b"]], cam2, caches)
        if prev_assignments is not None:
            pid_a = prev_assignments.get(f"{best1}:{insts1[m['a']].track_idx}")
            pid_b = prev_assignments.get(f"{best2}:{insts2[m['b']].track_idx}")
            if pid_a is not None and pid_b is not None and pid_a == pid_b:
                full += 0.3
        m["score"] = full
    matches.sort(key=lambda m: -m["score"])


    # Filter
    if num_animals:
        matches = matches[:num_animals]
    else:
        matches = [m for m in matches if m["score"] > 0.05]


    # EXPLAIN FROM HERE

    # Build groups from surviving matches
    groups = []
    matched1, matched2 = set(), set()
    for m in matches:
        groups.append({best1: insts1[m["a"]], best2: insts2[m["b"]]})
        matched1.add(m["a"])
        matched2.add(m["b"])


    # Pad to numAnimals with singletons from each cam's unmatched
    if num_animals and len(groups) < num_animals:
        for a in range(n_a):
            if len(groups) >= num_animals:
                break
            if a not in matched1:
                groups.append({best1: insts1[a]})
        for b in range(n_b):
            if len(groups) >= num_animals:
                break
            if b not in matched2:
                groups.append({best2: insts2[b]})

    # Solo groups for the unconstrained case
    if not num_animals:
        for a in range(n_a):
            if a not in matched1:
                groups.append({best1: insts1[a]})
        for b in range(n_b):
            if b not in matched2:
                groups.append({best2: insts2[b]})


    # Graft remaining cameras via reprojection-distance Hungarian
    for cam_name in ordered_cams[2:]:
        cam3 = cam_map[cam_name]
        insts3 = cam_instances.get(cam_name, [])
        if not insts3:
            continue
        cost3 = np.full((len(groups), len(insts3)), np.inf)
        P3 = caches.get_P(cam3)
        for gi, g in enumerate(groups):
            pts3d = triangulate_group(g, cam_map, caches)
            if pts3d is None:
                continue
            reproj = reproject_points(pts3d, P3)
            # print(pts3d, P3, reproj)
            X = np.asarray([p if p is not None else (np.nan, np.nan, np.nan) for p in pts3d])
            # print(g)
            # print(X)
            # return


            for ii in range(len(insts3)):
                d = instance_pixel_distance(reproj, insts3[ii].points)

                cost3[gi, ii] = d
        if cost3.size == 0:
            continue
        assign3 = hungarian(cost3)
        # print('cams by count', cams_by_count)
        # print('ordered_cams', ordered_cams)
        # print(f'{cam3.name},\n {insts3}')
        # return cost3
        matched3 = set()
        for gi in range(len(groups)):

            
            ii = assign3[gi]
            if 0 <= ii < len(insts3) and cost3[gi, ii] < 100.0:
                groups[gi][cam_name] = insts3[ii]
                matched3.add(int(ii))
        if not num_animals:
            for ii in range(len(insts3)):
                if ii not in matched3:
                    groups.append({cam_name: insts3[ii]})

    return groups


# ===========================================================================
# reorderGroupsByPrevTargets — keep identity labels stable over time
# ===========================================================================

def reorder_groups_by_prev(groups, prev_targets3d, cam_map,
                           prev_assignments, caches: TrackerCaches):
    n_t = len(prev_targets3d)
    n_g = len(groups)
    n = max(n_t, n_g)
    if n_g == 0 or n_t == 0:
        return groups

    group_pts3d = [triangulate_group(g, cam_map, caches) for g in groups]

    cost = np.full((n, n), 1000.0)
    for ti in range(n_t):
        for gi in range(n_g):
            prev_pts = prev_targets3d[ti].get("points3d")
            curr_pts = group_pts3d[gi]
            score = 0.0
            score_count = 0

            # Signal 1: reprojection of prev 3D into curr instances
            if prev_pts is not None:
                reproj_total, reproj_count = 0.0, 0
                for cn, inst in groups[gi].items():
                    cam = cam_map.get(cn)
                    if cam is None:
                        continue
                    reproj = reproject_points(prev_pts, caches.get_P(cam))
                    d = instance_pixel_distance(reproj, inst.points)
                    if math.isfinite(d):
                        reproj_total += d
                        reproj_count += 1
                if reproj_count > 0:
                    score += math.exp(-(reproj_total / reproj_count) / 50.0)
                    score_count += 1

            # Signal 2: 3D distance prev <-> curr keypoints
            if prev_pts is not None and curr_pts is not None:
                total_d, count = 0.0, 0
                n_kp = min(len(prev_pts), len(curr_pts))
                for k in range(n_kp):
                    if prev_pts[k] is not None and curr_pts[k] is not None:
                        d = float(np.linalg.norm(np.asarray(prev_pts[k]) - np.asarray(curr_pts[k])))
                        total_d += d
                        count += 1
                if count > 0:
                    score += math.exp(-(total_d / count) / 30.0)
                    score_count += 1

            # Signal 3: cross-view consistency vs prev instances
            prev_insts = prev_targets3d[ti].get("prevInstances")
            if curr_pts is not None and prev_insts:
                oks_total, oks_count = 0.0, 0
                for cn, prev_inst in prev_insts.items():
                    cam = cam_map.get(cn)
                    if cam is None or prev_inst is None:
                        continue
                    reproj = reproject_points(curr_pts, caches.get_P(cam))
                    d = instance_pixel_distance(reproj, prev_inst.points)
                    if math.isfinite(d):
                        oks_total += math.exp(-d / 50.0)
                        oks_count += 1
                if oks_count > 0:
                    score += oks_total / oks_count
                    score_count += 1

            # Signal 4: track-identity continuity (2x weight)
            prev_id = prev_targets3d[ti].get("identityId")
            if prev_assignments is not None and prev_id is not None:
                matching, total = 0, 0
                for cn, inst in groups[gi].items():
                    total += 1
                    pid = prev_assignments.get(f"{cn}:{inst.track_idx}")
                    if pid is not None and pid == prev_id:
                        matching += 1
                if total > 0:
                    score += 2.0 * (matching / total)
                    score_count += 1

            cost[ti, gi] = -(score / score_count) if score_count > 0 else 0.0

    assignment = hungarian(cost)

    # Faithful port of JS reorderGroupsByPrevTargets (tracker.js:357–364):
    #
    #   var usedGroups = new Set(assignment.filter(g => g >= 0 && g < nGroups));
    #   for (var gi3 = 0; gi3 < nGroups; gi3++) {
    #     if (!usedGroups.has(gi3)) reordered.push(groups[gi3]);
    #   }
    #
    # JS consults ALL of `assignment` (length n = max(n_t, n_g)). When n_g > n_t
    # the Hungarian solves over the padded square and padded targets
    # (ti >= n_t) optimally consume extra real groups via their cost=1000 rows.
    # JS treats those padded-claim groups as "used", so the leftover-append
    # loop SKIPS them → those groups get silently dropped from the result.
    #
    # We faithfully reproduce that drop here because it's the actual JS
    # behavior that produces the user-visible duplicate-identity bug at
    # frame 148 midL on the 10072022145420_small dataset. The per-frame
    # uniqueness guard in match_frame_instances (below) is the *fix* for
    # that bug — it ensures the viewer never falls back to a stale global
    # for tracks whose group got dropped here.
    n = len(assignment)
    used = {int(assignment[i]) for i in range(n)
            if 0 <= int(assignment[i]) < n_g}

    reordered = []
    for ti in range(n_t):
        gi = assignment[ti]
        if 0 <= gi < n_g:
            reordered.append(groups[gi])
        else:
            reordered.append({})  # empty slot — matches JS's `new Map()`
    for gi in range(n_g):
        if gi not in used:
            reordered.append(groups[gi])
    return reordered


# ===========================================================================
# matchFrameInstances — single-frame entry point
# ===========================================================================

def match_frame_instances(frame_group, cameras, session,
                          num_animals: Optional[int] = None,
                          prev_assignments: Optional[dict] = None,
                          prev_targets3d: Optional[list] = None,
                          per_frame: bool = True,
                          caches: Optional[TrackerCaches] = None,
                          enforce_uniqueness: bool = False):
    """Single-frame cross-view match + identity assignment.

    Returns dict with:
      groups          list[dict[cam_name, Instance]]
      num_identities  int
      assignments     dict[str, int]   "cam:track_idx" -> identity_id
      targets3d       list[dict]       {points3d, groupIdx, prevInstances, identityId?}

    Side-effects on `session`:
      - Adds default "id_N" Identities if num_animals > len(session.identities).
      - Writes to session.track_identity_map (global) and, if per_frame=True,
        session.frame_identity_map (per-frame override).

    `enforce_uniqueness` defaults to **False** to mirror the JS implementation
    exactly — including its duplicate-identity bug at frames where
    `reorder_groups_by_prev` silently drops groups. Pass `True` to enable the
    per-frame `-1` sentinel guard that suppresses the bug (see the
    "THE FIX" block at the end of this function).
    """
    if caches is None:
        caches = TrackerCaches()

    cam_instances, cam_map, active_cams = collect_instances(frame_group, cameras)
    if len(active_cams) < 2:
        return {"groups": [], "num_identities": 0,
                "assignments": {}, "targets3d": []}

    # match_pairwise re-collects from frame_group internally (see its body);
    # we still hold cam_map for the reorder / triangulate steps below.
    groups = match_pairwise(frame_group, session, caches,
                            num_animals=num_animals or 0,
                            prev_assignments=prev_assignments)

    if prev_targets3d and len(prev_targets3d) > 0 and len(groups) > 0:
        groups = reorder_groups_by_prev(groups, prev_targets3d, cam_map,
                                        prev_assignments, caches)

    # Triangulate each surviving group
    targets3d = []
    for gi, g in enumerate(groups):
        pts3d = triangulate_group(g, cam_map, caches) if g else None
        targets3d.append({"points3d": pts3d, "groupIdx": gi, "prevInstances": g})

    # Top-up identity pool to num_animals
    if num_animals:
        while len(session.identities) < num_animals:
            session.add_identity(f"id_{len(session.identities)}")

    assignments: dict[str, int] = {}
    used_ids: set[int] = set()

    for g_i, group in enumerate(groups):
        if not group:
            continue

        identity = None

        # Vote phase — only meaningful when prior assignments exist
        if prev_assignments is not None:
            votes: dict[int, int] = {}
            for cn, inst in group.items():
                pid = prev_assignments.get(f"{cn}:{inst.track_idx}")
                if pid is not None:
                    votes[pid] = votes.get(pid, 0) + 1
            best_id, best_vote = None, -1
            for vid, cnt in votes.items():
                if cnt > best_vote and vid not in used_ids:
                    best_vote = cnt
                    best_id = vid
            if best_id is not None:
                identity = session.get_identity(best_id)

        # Fallback: first unused identity in the first N
        if identity is None:
            max_id = num_animals if num_animals else len(session.identities)
            for ei in range(min(max_id, len(session.identities))):
                if session.identities[ei].id not in used_ids:
                    identity = session.identities[ei]
                    break

        # Unconstrained: spawn a fresh one
        if identity is None and not num_animals:
            identity = session.add_identity(f"id_{len(session.identities)}")

        if identity is None:
            continue

        used_ids.add(identity.id)
        targets3d[g_i]["identityId"] = identity.id

        for cn, inst in group.items():
            if inst.track_idx is None:
                continue
            session.track_identity_map[f"{cn}:{inst.track_idx}"] = identity.id
            if per_frame:
                session.set_frame_identity(frame_group.frame_idx, cn,
                                           inst.track_idx, identity.id)
            assignments[f"{cn}:{inst.track_idx}"] = identity.id

    # ------------------------------------------------------------------
    # Per-frame uniqueness guard (THE FIX) — OPT-IN
    # ------------------------------------------------------------------
    # DEFAULT BEHAVIOR (enforce_uniqueness=False): this guard is skipped,
    # which faithfully reproduces the JS web app's duplicate-identity bug.
    # When `reorder_groups_by_prev` drops a group claimed by a padded
    # Hungarian row, the (cam, track) entries in that dropped group never
    # get per-frame overrides written above. The viewer's identity-color
    # path then falls back to `track_identity_map` (global), which still
    # holds stale values from earlier frames. If that stale global value
    # happens to coincide with an identity already assigned to another
    # track in the same camera at this frame, two skeletons render in
    # the same color — the "duplicate id_3 at frame 148 midL" bug on
    # 10072022145420_small with num_animals=None.
    #
    # PASS enforce_uniqueness=True to enable the fix: for every visible
    # (cam, track) at this frame that didn't get a per-frame override
    # above, write an explicit sentinel (-1). The viewer's
    # `get_identity_for_track` and `get_identity_id_for_track` paths
    # short-circuit on this value (returns no Identity), so the track
    # renders in track-color rather than colliding with another track's
    # identity color. The underlying matcher still drops groups exactly
    # the same way; only the symptom is suppressed.
    if per_frame and enforce_uniqueness:
        fi = frame_group.frame_idx
        written = set(assignments.keys())  # "cam:track" strings
        for cam, insts in frame_group.instances.items():
            for inst in insts:
                if inst.track_idx is None:
                    continue
                key = f"{cam}:{inst.track_idx}"
                if key in written:
                    continue
                # Set sentinel only if the global would otherwise show a
                # value (so we suppress the fallback). If global has no
                # value either, leave it alone — the viewer falls through
                # to "no identity" naturally.
                if key in session.track_identity_map:
                    session.set_frame_identity(fi, cam, inst.track_idx, -1)
        # Also cover unlinked-only instances (rare in SLP imports but
        # the JS viewer renders them too).
        for cam, ul_list in frame_group.unlinked_instances.items():
            for ul in ul_list:
                if ul.instance.track_idx is None:
                    continue
                key = f"{cam}:{ul.instance.track_idx}"
                if key in written:
                    continue
                if key in session.track_identity_map:
                    session.set_frame_identity(fi, cam, ul.instance.track_idx, -1)

    return {"groups": groups,
            "num_identities": sum(1 for g in groups if g),
            "assignments": assignments,
            "targets3d": targets3d}


# ===========================================================================
# trackAll — full-session sweep
# ===========================================================================

def prune_orphan_identities(session, min_global: int = 1) -> list:
    """Optional utility — NOT called by `track_all` by default. Remove identities
    that have fewer than `min_global` entries in `session.track_identity_map`.

    The matcher can transiently create "leftover" identities for short
    singleton groups that the next frame overwrites. The JS web app keeps
    those entries in `session.identities` too — they're not removed
    automatically. Call this manually only when you want a curated view.

    Returns the list of removed Identity objects.
    """
    from collections import Counter
    keep_counts = Counter(session.track_identity_map.values())
    removed = [
        ident for ident in session.identities
        if keep_counts.get(ident.id, 0) < min_global
    ]
    removed_ids = {ident.id for ident in removed}
    if not removed:
        return []

    # Wipe per-frame overrides that still point at pruned identities so the
    # GUI's colored-by-identity timeline doesn't carry phantom segments.
    stale_keys = [k for k, v in session.frame_identity_map.items() if v in removed_ids]
    for k in stale_keys:
        del session.frame_identity_map[k]

    # Remove from session.identities. Preserve the order of survivors.
    session.identities = [
        ident for ident in session.identities if ident.id not in removed_ids
    ]
    # Trigger views to refresh so the assignment panel + timeline drop them.
    session._emit("identities_changed")
    session._emit("identity_map_changed")
    return removed


def track_all(session, num_animals: Optional[int] = None,
              per_frame: bool = True, on_progress=None,
              clear_existing: bool = True,
              enforce_uniqueness: bool = False):
    """Sweep every frame, propagating prev_assignments / prev_targets3d forward.

    Mirrors trackAll() in tracker.js exactly. By default WIPES existing
    identities (set clear_existing=False to keep them — JS behavior matches
    the default).

    `enforce_uniqueness` defaults to **False** so that the sweep reproduces
    the JS bug verbatim — identity counts match JS, and visible (cam, track)
    entries whose group was dropped by `reorder_groups_by_prev` fall back to
    stale global mappings, causing per-camera identity duplicates (eg.
    "duplicate id_3 at frame 148 midL" on the 10072022145420_small dataset).

    Set `enforce_uniqueness=True` to route through the per-frame `-1` sentinel
    guard in `match_frame_instances` (the fix). With the guard enabled, the
    underlying algorithm in `reorder_groups_by_prev` still drops groups
    exactly the same way, but the per-frame override blocks the stale-global
    fallback so no visible duplicates appear.

    Returns list[dict] — one match_frame_instances result per processed frame,
    plus a "frame" key for indexing.
    """
    if not session.cameras or len(session.cameras) < 2:
        raise ValueError("Need ≥2 cameras to track")
    if not session.frame_groups:
        raise ValueError("Session has no frames")

    if clear_existing:
        session.identities = []
        session.track_identity_map = {}
        session.frame_identity_map = {}
        session._identity_counter = 0
        session.identities_changed.emit()
        session.identity_map_changed.emit()

    caches = TrackerCaches()
    prev_assignments = None
    prev_targets3d = None
    history = []

    # Mirror JS trackAll(): the JS version does plain `Map.set` calls and only
    # redraws (drawAllOverlays + timeline.refreshTracks) AFTER the loop. The
    # PySide version emits identity_map_changed / identities_changed per write,
    # each of which causes the assignment panel to rebuild 22 QComboBoxes —
    # before this batch context, that accounted for ~89% of track_all wall
    # time (profile: 30 frames in 5.67s, 5.05s of which was assignment_panel
    # rebuild). The `batch_updates()` context queues those signals and emits
    # each unique one exactly once on exit.
    #
    # The progress callback (which may call e.g. QProgressDialog.setValue and
    # implicitly pump the event loop) runs INSIDE the batch — that's fine,
    # since the only views that subscribe to these signals are doing data
    # repaints and don't need per-frame freshness during the sweep.
    with session.batch_updates():
        for k, fi in enumerate(session.frame_indices):
            fg = session.frame_group(fi)
            if fg is None:
                continue
            result = match_frame_instances(
                fg, session.cameras, session,
                num_animals=num_animals,
                prev_assignments=prev_assignments,
                prev_targets3d=prev_targets3d,
                per_frame=per_frame,
                caches=caches,
                enforce_uniqueness=enforce_uniqueness,
            )
            history.append({"frame": fi, **result})
            if result["assignments"]:
                prev_assignments = result["assignments"]
            if result["targets3d"]:
                prev_targets3d = result["targets3d"]
            if on_progress is not None:
                on_progress(k, fi, result)

    return history
