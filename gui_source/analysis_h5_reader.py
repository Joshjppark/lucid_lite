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

    _merge_normalized_tracks(
        session, camera_name, tracks, track_names, occupancy, instance_scores
    )


def _merge_normalized_tracks(
    session: Session,
    camera_name: str,
    tracks: np.ndarray,          # (n_tracks, 2, n_nodes, n_frames), NaN = missing
    track_names: list[str],
    occupancy: np.ndarray,       # (n_frames, n_tracks) or empty
    instance_scores: np.ndarray, # (n_tracks, n_frames) or empty
) -> None:
    """Fill session.frame_groups from a normalized `tracks` array.

    Shared by every pose source (proofread analysis.h5, raw per-cam analysis.h5,
    and the aggregated predictions H5s) so they all map onto Instances the same
    way. `track_idx` indexes Session.tracks (union-by-name across cameras): pass
    identity-stable names (e.g. ``global_0``) for proofread data, or per-cam slot
    names (``track_0`` …) for raw detections that the tracker still has to link.
    """
    n_tracks, _, n_nodes, n_frames = tracks.shape

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


def merge_aggregated_h5_into_session(
    session: Session,
    h5_path: Path,
    camera_name: str,
    session_idx: int,
    *,
    node_names: list[str] | None = None,
    n_frames: int | None = None,
) -> None:
    """Load one camera from an *aggregated* predictions H5 into the session.

    These are the bench's `predictions_h5s/<cam>_predictions.h5` (raw SLEAP) and
    `sleap_nn_predictions_h5s/<cam>_predictions.h5` (filtered) files. Unlike a
    per-camera analysis.h5, the `tracks` dataset stacks every session on axis 0
    and uses a different axis order:

        tracks   (n_sessions, n_frames, n_tracks, n_nodes, 2)   NaN = missing

    so we slice `[session_idx]` and transpose to the normalized
    `(n_tracks, 2, n_nodes, n_frames)` layout the shared builder expects.

    These files carry ONLY `tracks` (no node_names / occupancy / scores), so:
      * `node_names` must come from the caller (read them from a reference
        analysis.h5 so the skeleton order matches) — else generic names are
        synthesized and a warning is printed;
      * occupancy is derived from finite points;
      * track names are per-cam slots (``track_0`` …) because raw detections are
        NOT identity-resolved across cameras — the tracker links them.
    """
    with h5py.File(str(h5_path), "r") as f:
        arr = f["tracks"]
        if arr.ndim != 5:
            raise ValueError(
                f"{h5_path.name}: expected aggregated tracks of ndim 5 "
                f"(n_sessions, n_frames, n_tracks, n_nodes, 2), got shape {arr.shape}"
            )
        if not (0 <= session_idx < arr.shape[0]):
            raise IndexError(
                f"{h5_path.name}: session_idx {session_idx} out of range "
                f"[0, {arr.shape[0]})"
            )
        stop = arr.shape[1] if n_frames is None else min(n_frames, arr.shape[1])
        sess = arr[session_idx, :stop]         # (n_frames, n_tracks, n_nodes, 2)

    # (n_frames, n_tracks, n_nodes, xy) -> (n_tracks, xy, n_nodes, n_frames)
    tracks = np.transpose(sess, (1, 3, 2, 0)).astype(np.float64)
    n_tracks, _, n_nodes, n_frames_eff = tracks.shape

    if node_names is None:
        if session.skeleton is not None and len(session.skeleton.nodes) == n_nodes:
            node_names = list(session.skeleton.nodes)
        else:
            node_names = [f"node_{i}" for i in range(n_nodes)]
            print(
                f"[analysis_h5_reader] WARN: {h5_path.name} has no node_names and "
                f"none were supplied; synthesizing {n_nodes} generic names. Node "
                f"order / node_weights keyed by real names will NOT line up."
            )

    # Seed skeleton (no edges available from aggregated files).
    if session.skeleton is None:
        session.skeleton = Skeleton(name="skeleton", nodes=list(node_names), edges=[])
    elif len(session.skeleton.nodes) != n_nodes:
        print(
            f"[analysis_h5_reader] WARN: {h5_path.name} has {n_nodes} nodes but "
            f"session.skeleton has {len(session.skeleton.nodes)}; keeping the first."
        )

    occupancy = (~np.isnan(tracks).all(axis=(1, 2))).T.astype(np.uint8)  # (n_frames, n_tracks)
    instance_scores = np.zeros((n_tracks, n_frames_eff), dtype=float)
    track_names = [f"track_{i}" for i in range(n_tracks)]  # per-cam slots, not identities

    _merge_normalized_tracks(
        session, camera_name, tracks, track_names, occupancy, instance_scores
    )


def read_node_names(h5_path: Path) -> list[str] | None:
    """Read `node_names` from a SLEAP analysis.h5 (for seeding aggregated sources)."""
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "node_names" not in f:
                return None
            return [_b2s(b) for b in f["node_names"][...]]
    except (OSError, KeyError):
        return None


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
