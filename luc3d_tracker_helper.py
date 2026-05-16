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

Caveats
-------
- Hungarian is `scipy.optimize.linear_sum_assignment` (added as a dep).
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
from scipy.optimize import linear_sum_assignment

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
    return math.exp(-total / n_valid / 10.0)


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
    for k in range(n):
        if u_a[k] is None or u_b[k] is None:
            continue
        X = triangulate_dlt([u_a[k], u_b[k]], [P_a, P_b])
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
    """group = dict[cam_name, Instance]. Returns list of 3D points (or None) per keypoint."""
    cam_names = list(group.keys())
    if len(cam_names) < 2:
        return None
    cams = [cam_map[c] for c in cam_names]
    insts = [group[c] for c in cam_names]
    n_kp = len(insts[0].points)
    out = []
    for k in range(n_kp):
        obs, Ps = [], []
        for cam, inst in zip(cams, insts):
            p = inst.points[k]
            if p is None:
                continue
            u = caches.get_undist_kps(inst, cam)[k]
            if u is None:
                continue
            obs.append(u)
            Ps.append(caches.get_P(cam))
        if len(obs) < 2:
            out.append(None)
        else:
            out.append(triangulate_dlt(obs, Ps))
    return out


def hungarian(cost: np.ndarray) -> np.ndarray:
    """Cost (n_rows, n_cols) -> assignment[r] = column or -1 if unmatched."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.zeros(cost.shape[0], dtype=int)
    # scipy handles rectangular cost — solves with smaller dim.
    big = 1e18
    finite = np.where(np.isfinite(cost), cost, big)
    row_ind, col_ind = linear_sum_assignment(finite)
    out = np.full(cost.shape[0], -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= big - 1:  # was infinite
            continue
        out[r] = int(c)
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

def match_pairwise(cam_instances, cam_map, active_cams, num_animals,
                   prev_assignments, caches: TrackerCaches):
    # Sort by candidate count descending. Pair is [most, second].
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
            matches.append({"a": a, "b": int(b),
                            "score": float(score_matrix[a, b])})
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
            for ii in range(len(insts3)):
                d = instance_pixel_distance(reproj, insts3[ii].points)
                cost3[gi, ii] = d
        if cost3.size == 0:
            continue
        assign3 = hungarian(cost3)
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

    reordered = []
    used = set()
    for ti in range(n_t):
        gi = assignment[ti]
        if 0 <= gi < n_g:
            reordered.append(groups[gi])
            used.add(int(gi))
        else:
            reordered.append({})  # empty slot — same as JS behavior
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
                          caches: Optional[TrackerCaches] = None):
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
    """
    if caches is None:
        caches = TrackerCaches()

    cam_instances, cam_map, active_cams = collect_instances(frame_group, cameras)
    if len(active_cams) < 2:
        return {"groups": [], "num_identities": 0,
                "assignments": {}, "targets3d": []}

    groups = match_pairwise(cam_instances, cam_map, active_cams,
                            num_animals, prev_assignments, caches)

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

    return {"groups": groups,
            "num_identities": sum(1 for g in groups if g),
            "assignments": assignments,
            "targets3d": targets3d}


# ===========================================================================
# trackAll — full-session sweep
# ===========================================================================

def track_all(session, num_animals: Optional[int] = None,
              per_frame: bool = True, on_progress=None,
              clear_existing: bool = True):
    """Sweep every frame, propagating prev_assignments / prev_targets3d forward.

    Mirrors trackAll() in tracker.js. By default WIPES existing identities
    (set clear_existing=False to keep them — JS behavior matches the default).

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
        )
        history.append({"frame": fi, **result})
        if result["assignments"]:
            prev_assignments = result["assignments"]
        if result["targets3d"]:
            prev_targets3d = result["targets3d"]
        if on_progress is not None:
            on_progress(k, fi, result)

    return history
