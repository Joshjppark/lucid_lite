"""Adapter from SLEAP analysis-HDF5 -> LUCID-Lite Instances / FrameGroups.

The SLEAP *analysis.h5* format is a tracked, post-proofread export with
predicted-only instances. Schema (per-camera file):

    tracks            (n_tracks, 2 [x,y], n_nodes, n_frames)    float64, NaN=missing
    track_names       (n_tracks,)                                bytes
    node_names        (n_nodes,)                                 bytes
    edge_inds         (n_edges, 2)                               int32, node-index pairs
    edge_names        (n_edges, 2)                               bytes pairs (debug only)
    track_occupancy   (n_frames, n_tracks)                       uint8 0/1
    instance_scores   (n_tracks, n_frames)                       float64
    point_scores      (n_tracks, n_nodes, n_frames)              float64
    tracking_scores   (n_tracks, n_frames)                       float64
    video_path / labels_path / provenance / video_ind            scalars

We map this onto the same shape `slp_reader` uses:

    one Instance per (frame, track) where occupancy is set OR any point is non-NaN
    track_idx indexes Session.tracks (union-by-name across cameras)
    type = "predicted"  (analysis.h5 holds predictions only)
    score = instance_scores[track, frame]
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from pose_data import FrameGroup, Instance, Session, Skeleton


def merge_analysis_h5_into_session(
    session: Session,
    h5_path: Path,
    camera_name: str,
) -> None:
    """Load one camera's analysis.h5 and fill session.frame_groups[idx].instances[cam]."""
    with h5py.File(str(h5_path), "r") as f:
        tracks = f["tracks"][...]              # (n_tracks, 2, n_nodes, n_frames)
        node_names = [_b2s(b) for b in f["node_names"][...]]
        track_names = [_b2s(b) for b in f["track_names"][...]]
        edge_inds = f["edge_inds"][...] if "edge_inds" in f else np.zeros((0, 2), dtype=int)
        occupancy = (
            f["track_occupancy"][...]
            if "track_occupancy" in f
            else np.ones(tracks.shape[::1][:0], dtype=np.uint8)  # placeholder
        )
        instance_scores = (
            f["instance_scores"][...]
            if "instance_scores" in f
            else np.zeros((tracks.shape[0], tracks.shape[3]), dtype=float)
        )

    n_tracks, _, n_nodes, n_frames = tracks.shape

    # Defensive occupancy fallback if shape was bogus.
    if occupancy.shape != (n_frames, n_tracks):
        occupancy = (~np.isnan(tracks).all(axis=(1, 2))).T.astype(np.uint8)
        # ^ (n_tracks, n_frames) -> transposed to (n_frames, n_tracks)

    # Seed skeleton on first reader if not already set.
    if session.skeleton is None:
        edges = [(int(a), int(b)) for a, b in edge_inds]
        session.skeleton = Skeleton(name="skeleton", nodes=node_names, edges=edges)
    else:
        # Re-check node-count compatibility — if the existing skeleton has a
        # different node count we still produce instances at that size, but
        # warn so the user knows.
        if len(session.skeleton.nodes) != n_nodes:
            print(
                f"[analysis_h5_reader] WARN: {h5_path.name} has {n_nodes} nodes "
                f"but session.skeleton has {len(session.skeleton.nodes)}; "
                f"keeping the first skeleton seen."
            )

    # Union tracks by name (same policy as slp_reader).
    track_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(session.tracks)}
    local_track_to_session_idx: list[int] = []
    for tname in track_names:
        if tname not in track_name_to_idx:
            track_name_to_idx[tname] = len(session.tracks)
            session.tracks.append(tname)
        local_track_to_session_idx.append(track_name_to_idx[tname])

    # Build instances frame-by-frame.
    for frame_idx in range(n_frames):
        fg = session.frame_groups.get(frame_idx)

        # Decide which tracks contribute at this frame.
        for t in range(n_tracks):
            occ = bool(occupancy[frame_idx, t]) if occupancy.size else True
            xy = tracks[t, :, :, frame_idx]    # (2, n_nodes)
            any_pt = bool(np.isfinite(xy).any())
            if not occ and not any_pt:
                continue

            points: list[tuple[float, float] | None] = []
            for n in range(n_nodes):
                x, y = float(xy[0, n]), float(xy[1, n])
                if math.isnan(x) or math.isnan(y):
                    points.append(None)
                else:
                    points.append((x, y))

            score = float(instance_scores[t, frame_idx]) if instance_scores.size else 0.0
            if math.isnan(score):
                score = 0.0

            inst = Instance(
                points=points,
                track_idx=local_track_to_session_idx[t],
                type="predicted",
                score=score,
            )

            if fg is None:
                fg = FrameGroup(frame_idx=frame_idx)
                session.frame_groups[frame_idx] = fg
            fg.add_instance(camera_name, inst)


def _b2s(b) -> str:
    """Decode an HDF5 bytes scalar (or already-str) into a Python str."""
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    if isinstance(b, np.bytes_):
        return bytes(b).decode("utf-8", errors="replace")
    return str(b)


# Naming convention used by SLEAP to identify analysis exports.
ANALYSIS_H5_SUFFIX = ".analysis.h5"


def looks_like_analysis_h5(path: Path) -> bool:
    """Cheap filename check; the loader does an open-and-probe before using it."""
    name = path.name.lower()
    return name.endswith(ANALYSIS_H5_SUFFIX) or name.endswith(".h5")


def is_analysis_h5(path: Path) -> bool:
    """Structural check — does the file have the SLEAP analysis-h5 datasets?"""
    try:
        with h5py.File(str(path), "r") as f:
            return "tracks" in f and "node_names" in f and "track_names" in f
    except Exception:
        return False
