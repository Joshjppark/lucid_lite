"""Adapter from sleap-io Labels -> LUCID-Lite Instances / FrameGroups.

Reuses sleap-io for HDF5 parse (mirrors slp-import-worker.js role) — we do not
re-implement columnar frame/instance decoding here.
"""
from __future__ import annotations

import math
from pathlib import Path

import sleap_io as sio

from pose_data import FrameGroup, Instance, Session, Skeleton


def skeleton_from_sio(skel: sio.Skeleton) -> Skeleton:
    nodes = list(skel.node_names)
    name_to_idx = {n: i for i, n in enumerate(nodes)}
    edges: list[tuple[int, int]] = []
    for edge in skel.edges:
        src = name_to_idx.get(edge.source.name)
        dst = name_to_idx.get(edge.destination.name)
        if src is not None and dst is not None:
            edges.append((src, dst))
    return Skeleton(name=getattr(skel, "name", "skeleton") or "skeleton",
                    nodes=nodes, edges=edges)


def merge_slp_into_session(
    session: Session,
    slp_path: Path,
    camera_name: str,
) -> None:
    """Load one camera's .slp and fill session.frame_groups[idx].instances[cam_name]."""
    labels = sio.load_slp(str(slp_path))

    # Seed skeleton + tracks from the first SLP we load.
    if session.skeleton is None and labels.skeletons:
        session.skeleton = skeleton_from_sio(labels.skeletons[0])

    # Tracks are shared across cameras in LUCID. Union by name (mirrors JS behavior
    # where tracks from each SLP are merged into Session.tracks).
    track_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(session.tracks)}
    for t in labels.tracks:
        if t.name not in track_name_to_idx:
            track_name_to_idx[t.name] = len(session.tracks)
            session.tracks.append(t.name)

    for lf in labels.labeled_frames:
        frame_idx = int(lf.frame_idx)
        fg = session.frame_groups.get(frame_idx)
        if fg is None:
            fg = FrameGroup(frame_idx=frame_idx)
            session.frame_groups[frame_idx] = fg

        for sio_inst in lf.instances:
            inst = _sio_instance_to_lite(sio_inst, track_name_to_idx)
            fg.add_instance(camera_name, inst)


def _sio_instance_to_lite(sio_inst, track_name_to_idx: dict[str, int]) -> Instance:
    pts_arr = sio_inst.numpy()  # (n_nodes, 2) with NaN for missing
    points: list[tuple[float, float] | None] = []
    for i in range(pts_arr.shape[0]):
        x, y = float(pts_arr[i, 0]), float(pts_arr[i, 1])
        if math.isnan(x) or math.isnan(y):
            points.append(None)
        else:
            points.append((x, y))

    if sio_inst.track is not None:
        track_idx: int | None = track_name_to_idx.get(sio_inst.track.name)
    else:
        track_idx = None

    is_predicted = type(sio_inst).__name__ == "PredictedInstance"
    score = float(getattr(sio_inst, "score", 0.0) or 0.0)

    return Instance(
        points=points,
        track_idx=track_idx,
        type="predicted" if is_predicted else "user",
        score=score,
    )
