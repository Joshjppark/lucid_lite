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


def merge_project_slp_into_session(
    session: Session,
    slp_path: Path,
    cam_key_fn=None,
) -> dict[str, int]:
    """Multi-video SLP loader — parses a root-level `project.slp` once and
    dispatches each `LabeledFrame` to the correct camera by filename match.

    The JS LUCID web app reads this single project file. Our per-camera
    `.analysis.h5` fallback loses any camera that doesn't have its own
    per-cam analysis file (eg. `side/` and `sideL/` in the
    `10072022145420_small` test dataset), which silently dropped two of the
    eight calibrated views. The tracker then ran with two fewer cross-view
    constraints and produced ~2× more transient identities.

    Returns `{camera_name: n_instances_loaded}` for caller diagnostics.

    `cam_key_fn(name) -> str` should be the same normalizer the loader uses
    to bind calibration entries to subfolder names (eg. lowercase, strip
    `cam_` prefix); we use it to resolve videos → calibrated camera names.
    """
    if cam_key_fn is None:
        from session_loader import _cam_key as cam_key_fn  # type: ignore

    labels = sio.load_slp(str(slp_path))

    if session.skeleton is None and labels.skeletons:
        session.skeleton = skeleton_from_sio(labels.skeletons[0])

    # Build the {video → camera_name} map by filename matching against the
    # calibration cameras. The video filenames are typically
    # `<camname>-<timestamp>-...mp4`; we strip the leading non-`-` segment
    # and normalize via cam_key_fn for tolerant matching.
    calibrated_keys = {cam_key_fn(c.name): c.name for c in session.cameras}
    video_to_cam: dict[int, str | None] = {}
    for vid in labels.videos:
        fn = Path(vid.filename).name
        # Take everything before the first '-' as the candidate cam name,
        # then fall back to scanning calibrated names embedded in the path.
        first_seg = fn.split("-", 1)[0]
        key = cam_key_fn(first_seg)
        cam_name = calibrated_keys.get(key)
        if cam_name is None:
            # As a fallback, look for any calibrated camera key appearing
            # as a token in the filename — handles oddball naming.
            low = fn.lower()
            for k, cn in calibrated_keys.items():
                if f"{k}-" in low or f"{k}_" in low or low.startswith(f"{k}-"):
                    cam_name = cn
                    break
        video_to_cam[id(vid)] = cam_name

    # Seed Session.tracks from the SLP's track list (union by name).
    track_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(session.tracks)}
    for t in labels.tracks:
        if t.name not in track_name_to_idx:
            track_name_to_idx[t.name] = len(session.tracks)
            session.tracks.append(t.name)

    counts: dict[str, int] = {}
    for lf in labels.labeled_frames:
        cam_name = video_to_cam.get(id(lf.video))
        if cam_name is None:
            continue  # video not matched to a calibrated camera — skip silently.
        frame_idx = int(lf.frame_idx)
        fg = session.frame_groups.get(frame_idx)
        if fg is None:
            fg = FrameGroup(frame_idx=frame_idx)
            session.frame_groups[frame_idx] = fg
        for sio_inst in lf.instances:
            try:
                inst = _sio_instance_to_lite(sio_inst, track_name_to_idx)
            except Exception:
                # sleap-io occasionally fails to numpy() an instance that
                # carries inhomogeneous custom metadata; skip individual bad
                # instances rather than aborting the whole load.
                continue
            fg.add_instance(cam_name, inst)
            counts[cam_name] = counts.get(cam_name, 0) + 1
    return counts


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
