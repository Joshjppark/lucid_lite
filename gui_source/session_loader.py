"""Scan a per-camera session folder and build a Session.

Expected layout (index.html:12524–12576):

    session_folder/
    ├── calibration.toml          (or calibration.json, or calib*.toml/.json)
    ├── skeleton.json             (optional)
    ├── <cam_name>/
    │   ├── *.mp4
    │   └── *.slp
    └── ...
"""
from __future__ import annotations

from pathlib import Path

from analysis_h5_reader import is_analysis_h5, merge_analysis_h5_into_session
from calibration import find_calibration, parse_calibration_json, parse_calibration_toml
from pose_data import InstanceGroup, Session
from slp_reader import merge_project_slp_into_session, merge_slp_into_session


def load_session_from_folder(folder: Path) -> Session:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    session = Session()
    session.folder = folder

    # Calibration
    calib_path = find_calibration(folder)
    if calib_path is not None:
        if calib_path.suffix.lower() == ".toml":
            session.cameras = parse_calibration_toml(calib_path)
        else:
            session.cameras = parse_calibration_json(calib_path)
        session.has_calibration = True

    # Discover per-camera subfolders
    camera_folders = sorted([p for p in folder.iterdir() if p.is_dir()])
    if not camera_folders:
        raise ValueError(f"No camera subfolders found in {folder}")

    # Match subfolders to calibration cameras (case-insensitive, strip 'cam_' prefix)
    calib_by_key = {_cam_key(c.name): c for c in session.cameras}
    resolved: list[tuple[str, Path]] = []  # (camera_name, subfolder)
    for sub in camera_folders:
        key = _cam_key(sub.name)
        if key in calib_by_key:
            resolved.append((calib_by_key[key].name, sub))
        else:
            # No calibration match — use folder name directly.
            resolved.append((sub.name, sub))

    if not session.has_calibration:
        # Synthesize camera records from folder names so downstream code has them.
        from pose_data import Camera
        session.cameras = [
            Camera(name=name, matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   dist=[0, 0, 0, 0, 0], rvec=[0, 0, 0], tvec=[0, 0, 0],
                   size=(0, 0))
            for name, _ in resolved
        ]

    # Video paths (mp4/avi) per camera — needed regardless of label source.
    for cam_name, sub in resolved:
        video = _first_with_suffix(sub, ".mp4") or _first_with_suffix(sub, ".avi")
        if video is not None:
            session.video_paths[cam_name] = video

    # Label source preference (matches the JS LUCID web app):
    #
    #   1. Root-level `project.slp` (or any *.slp directly under `folder`) —
    #      the canonical multi-video SLEAP project file. This is what the JS
    #      app loads, and it covers ALL calibrated cameras in one shot,
    #      including ones that don't have their own per-cam .analysis.h5
    #      (eg. `side/` and `sideL/` in test fixtures).
    #   2. Per-camera `.slp` inside each `<cam>/` subfolder.
    #   3. Per-camera `.analysis.h5` fallback (proofread predictions).
    #
    # Earlier versions skipped step 1 and silently dropped cameras with no
    # per-cam label file — which caused the tracker to run on fewer
    # cross-view constraints than the JS app and produce 2× more transient
    # identities (lucid-lite#track-frames-discrepancy).
    root_slp = _first_with_suffix(folder, ".slp")
    if root_slp is not None:
        merge_project_slp_into_session(session, root_slp, cam_key_fn=_cam_key)
    else:
        for cam_name, sub in resolved:
            slp = _first_with_suffix(sub, ".slp")
            h5 = _first_analysis_h5(sub)
            if slp is not None:
                merge_slp_into_session(session, slp, cam_name)
            elif h5 is not None:
                merge_analysis_h5_into_session(session, h5, cam_name)

    # Build InstanceGroups (mirrors slp-merge.js:151–180)
    rebuild_instance_groups(session)
    return session


def rebuild_instance_groups(session: Session) -> None:
    session.instance_groups.clear()
    for frame_idx, fg in session.frame_groups.items():
        # bucket instances by track_idx across cameras
        by_track: dict[int, list[tuple[str, object]]] = {}
        for cam_name, insts in fg.instances.items():
            for inst in insts:
                t = inst.track_idx if inst.track_idx is not None else -1
                by_track.setdefault(t, []).append((cam_name, inst))

        groups: list[InstanceGroup] = []
        for track_idx, entries in by_track.items():
            if track_idx < 0:
                # Untracked — leave out of InstanceGroups (they live only in fg.instances)
                continue
            group = InstanceGroup(id=session.next_instance_group_id(), identity_id=None)
            for cam_name, inst in entries:
                group.add_instance(cam_name, inst)
            groups.append(group)
        session.instance_groups[frame_idx] = groups


def _cam_key(name: str) -> str:
    k = name.lower().strip()
    for prefix in ("camera_", "cam_", "camera", "cam"):
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    return k


def _first_with_suffix(folder: Path, suffix: str) -> Path | None:
    hits = sorted(folder.glob(f"*{suffix}"))
    return hits[0] if hits else None


def _first_analysis_h5(folder: Path) -> Path | None:
    """Return the first SLEAP analysis-style .h5 in `folder`, structurally verified.

    We prefer files matching the explicit ``.analysis.h5`` suffix; otherwise
    fall back to any ``.h5`` whose root has the expected datasets. This skips
    unrelated h5 files (e.g. project-level ``points3d.h5`` if one ever leaks
    into a camera subfolder).
    """
    # Prefer the canonical suffix.
    for hit in sorted(folder.glob("*.analysis.h5")):
        if is_analysis_h5(hit):
            return hit
    for hit in sorted(folder.glob("*.h5")):
        if hit.name.endswith(".analysis.h5"):
            continue  # already tried
        if is_analysis_h5(hit):
            return hit
    return None
