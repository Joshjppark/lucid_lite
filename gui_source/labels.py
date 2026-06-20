"""Labels = detections, decoupled from the GUI, represented as ``sleap_io.Labels``.

The canonical "labels" object in lucid_lite is a **sleap_io.Labels** (a standard
SLEAP multi-video project: one Video per camera, LabeledFrames keyed by
(video, frame_idx)). This makes detections interoperable with the SLEAP
ecosystem — you can `sio.save_slp(labels, ...)`, hand them to sleap-io tooling,
or load any SLEAP project as input.

A `sio.Labels` has no calibration, so a session is built from BOTH a folder
(calibration + videos) and a `sio.Labels` (detections):

    import labels as L, sleap_io as sio

    # auto-discover detections in the folder, as a sio.Labels:
    lbls = L.from_folder("/path/to/session")                  # -> sio.Labels

    # or point at raw SLEAP / sleap-nn aggregated H5s:
    lbls = L.from_aggregated_h5("/path/to/session",
                                h5_dir=".../predictions_h5s", session_idx=70)

    session = L.build_session("/path/to/session", lbls)       # calib from folder
    window  = LucidLiteWindow(session)

    # main.main also accepts these directly:
    app, window = main.main("/path/to/session", labels=lbls)

Headless pose arrays:  L.to_arrays("/path/to/session", lbls)
Session -> sio.Labels: L.session_to_sio_labels(session)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from pose_data import Session

# Default cameras the bench scores; used when a caller doesn't pass `cams`.
DEFAULT_BENCH_CAMS = ["back", "backL", "mid", "midL", "top", "topL"]


# ---------------------------------------------------------------------------
# Builders — produce a single multi-video sio.Labels for a session
# ---------------------------------------------------------------------------
def from_folder(session_dir: Path | str) -> "object":
    """Auto-discover a session folder's detections as one multi-video sio.Labels.

    Reuses the standard loader (root project.slp -> per-cam .slp -> per-cam
    .analysis.h5) and converts the result, so discovery rules stay in one place.
    """
    from session_loader import load_session_from_folder
    return session_to_sio_labels(load_session_from_folder(Path(session_dir)))


def from_aggregated_h5(
    session_dir: Path | str,
    h5_dir: Path | str,
    session_idx: int,
    *,
    cams: list[str] | None = None,
    suffix: str = "_predictions.h5",
    node_names: list[str] | None = None,
    node_names_from: Path | str | None = None,
    n_frames: int | None = None,
    min_nodes: int = 1,
) -> "object":
    """Build one multi-video sio.Labels from the bench's aggregated detection H5s.

    Each `<h5_dir>/<cam><suffix>` is a `(n_sessions, n_frames, n_tracks, n_nodes,
    2)` stack (`predictions_h5s` = raw SLEAP, `sleap_nn_predictions_h5s` =
    filtered); row `session_idx` is sliced per camera. `node_names` are read from
    a proofread analysis.h5 in `session_dir` unless supplied. Track names are
    per-camera slots (``track_0`` …) since raw detections aren't identity-linked
    across cameras.
    """
    import h5py

    session_dir = Path(session_dir).expanduser().resolve()
    h5_dir = Path(h5_dir).expanduser().resolve()

    if cams is None:
        cams = _discover_cams(session_dir, h5_dir, suffix)
    if node_names is None:
        ref = Path(node_names_from) if node_names_from else _find_any_analysis_h5(session_dir)
        node_names = _read_node_names(ref) if ref else None
    if n_frames is None:
        n_frames = _session_n_frames(session_dir)

    per_cam: dict[str, tuple[Path | None, list]] = {}
    for cam in cams:
        h5 = h5_dir / f"{cam}{suffix}"
        if not h5.exists():
            print(f"[labels] WARN: no aggregated H5 for {cam!r} at {h5}; skipping.")
            continue
        with h5py.File(h5, "r") as f:
            slab = f["tracks"][session_idx, :n_frames]      # (F, T, N, 2)
        frames = []
        F, T = slab.shape[0], slab.shape[1]
        for fi in range(F):
            insts = []
            for t in range(T):
                pts = slab[fi, t]                            # (N, 2)
                if int(np.isfinite(pts[:, 0]).sum()) < min_nodes:
                    continue
                insts.append((np.asarray(pts, dtype="float32"), f"track_{t}", 1.0))
            if insts:
                frames.append((fi, insts))
        per_cam[cam] = (_cam_video(session_dir, cam), frames)

    n_nodes = slab.shape[2] if per_cam else 0
    if node_names is None:
        node_names = [f"node_{i}" for i in range(n_nodes)]
    return _build_multivideo_labels(node_names, [], per_cam)


# ---------------------------------------------------------------------------
# Session <-> sio.Labels
# ---------------------------------------------------------------------------
def build_session(session_dir: Path | str, labels: "object | None" = None) -> Session:
    """Build a lucid Session: calibration + videos from `session_dir`, detections
    from `labels` (a multi-video sio.Labels). If `labels` is None, detections are
    auto-discovered from the folder.
    """
    from session_loader import load_session_from_folder, load_session_structure, \
        rebuild_instance_groups
    from slp_reader import merge_sio_labels_into_session

    if labels is None:
        return load_session_from_folder(Path(session_dir))

    session, _ = load_session_structure(Path(session_dir))
    merge_sio_labels_into_session(session, labels)
    rebuild_instance_groups(session)
    return session


def session_to_sio_labels(session: Session) -> "object":
    """Convert a built Session into one multi-video sio.Labels (inverse of
    `build_session`): one Video per camera, LabeledFrames keyed by (video,
    frame_idx). `Instance.track_idx` becomes the sio Track."""
    nodes = list(session.skeleton.nodes) if session.skeleton else []
    edges = list(session.skeleton.edges) if session.skeleton else []

    per_cam: dict[str, tuple[Path | None, list]] = {}
    for cam in session.camera_names():
        frames = []
        for fi in sorted(session.frame_groups):
            insts = []
            for inst in session.frame_groups[fi].get_instances(cam):
                if inst.track_idx is None:
                    continue
                pts = np.array(
                    [p if p is not None else (np.nan, np.nan) for p in inst.points],
                    dtype="float32",
                )
                tname = (session.tracks[inst.track_idx]
                         if inst.track_idx < len(session.tracks)
                         else f"track_{inst.track_idx}")
                insts.append((pts, tname, float(inst.score or 0.0)))
            if insts:
                frames.append((fi, insts))
        per_cam[cam] = (session.video_paths.get(cam), frames)

    return _build_multivideo_labels(nodes, edges, per_cam)


def to_arrays(
    session_dir: Path | str,
    labels: "object | None" = None,
    cams: list[str] | None = None,
    frames: range | list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Headless one-shot: build the session and return pose arrays
    ``{cam: (n_frames, n_tracks, n_nodes, 2)}`` (NaN = missing)."""
    return extract_poses(build_session(session_dir, labels), cams=cams, frames=frames)


def extract_poses(
    session: Session,
    cams: list[str] | None = None,
    frames: range | list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Pull a Session's poses into dense per-camera arrays (headless-friendly).

    Returns ``{cam: array (n_frames, n_tracks, n_nodes, 2)}``, NaN where a
    keypoint / track / frame is missing. The 2nd axis is indexed by
    ``Instance.track_idx`` (per-camera slot for raw detections, identity for
    proofread data), so it round-trips with the aggregated H5 layout.
    """
    cams = cams or session.camera_names()
    frames = list(frames if frames is not None else range(session.max_frame + 1))
    n_nodes = len(session.skeleton.nodes) if session.skeleton else 0
    n_tracks = max(
        (i.track_idx
         for fg in session.frame_groups.values()
         for insts in fg.instances.values()
         for i in insts
         if i.track_idx is not None),
        default=-1,
    ) + 1

    fidx = {f: k for k, f in enumerate(frames)}
    out = {c: np.full((len(frames), n_tracks, n_nodes, 2), np.nan) for c in cams}
    for f in frames:
        fg = session.frame_groups.get(f)
        if fg is None:
            continue
        for cam in cams:
            for inst in fg.get_instances(cam):
                if inst.track_idx is None or inst.track_idx >= n_tracks:
                    continue
                pts = np.array(
                    [p if p is not None else (np.nan, np.nan) for p in inst.points],
                    dtype=float,
                )
                out[cam][fidx[f], inst.track_idx] = pts
    return out


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _build_multivideo_labels(node_names, edges, per_cam) -> "object":
    """per_cam: {cam: (video_path|None, [(frame_idx, [(pts(N,2), track_name, score), ...]), ...])}
    -> one sio.Labels with a shared skeleton, one Video per camera."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=list(node_names), edges=list(edges or []), name="skeleton")
    videos, lfs = [], []
    tracks_by_name: dict[str, object] = {}

    def track(name):
        if name not in tracks_by_name:
            tracks_by_name[name] = sio.Track(name=name)
        return tracks_by_name[name]

    for cam, (vpath, frames) in per_cam.items():
        # video filename must let merge map it back to this camera: keep the
        # real video name (``back-...mp4``) or fall back to ``<cam>-video.mp4``.
        fname = str(vpath) if vpath else f"{cam}-video.mp4"
        vid = sio.Video(filename=fname, open_backend=False)
        videos.append(vid)
        for fi, insts in frames:
            sio_insts = [
                sio.PredictedInstance.from_numpy(
                    np.asarray(pts, dtype="float32"), skeleton=skel,
                    track=track(tname), score=float(sc))
                for pts, tname, sc in insts
            ]
            if sio_insts:
                lfs.append(sio.LabeledFrame(video=vid, frame_idx=int(fi), instances=sio_insts))

    return sio.Labels(labeled_frames=lfs, videos=videos, skeletons=[skel],
                      tracks=list(tracks_by_name.values()))


def _discover_cams(session_dir: Path, h5_dir: Path, suffix: str) -> list[str]:
    subdirs = sorted(p.name for p in session_dir.iterdir() if p.is_dir())
    matched = [c for c in subdirs if (h5_dir / f"{c}{suffix}").exists()]
    return matched or [c for c in DEFAULT_BENCH_CAMS if (h5_dir / f"{c}{suffix}").exists()]


def _cam_video(session_dir: Path, cam: str) -> Path | None:
    for ext in ("*.mp4", "*.avi"):
        hits = sorted((session_dir / cam).glob(ext))
        if hits:
            return hits[0]
    return None


def _find_any_analysis_h5(session_dir: Path) -> Path | None:
    hits = sorted(session_dir.glob("*/*.analysis.h5"))
    return hits[0] if hits else None


def _read_node_names(h5_path: Path) -> list[str] | None:
    import h5py
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "node_names" not in f:
                return None
            return [b.decode() if isinstance(b, (bytes, np.bytes_)) else str(b)
                    for b in f["node_names"][...]]
    except (OSError, KeyError):
        return None


def _session_n_frames(session_dir: Path) -> int | None:
    """Frame count from any proofread analysis.h5 (bounds aggregated-H5 reads)."""
    import h5py
    ref = _find_any_analysis_h5(session_dir)
    if ref is None:
        return None
    try:
        with h5py.File(str(ref), "r") as f:
            if "track_occupancy" in f:
                return int(f["track_occupancy"].shape[0])
            if "tracks" in f:
                return int(f["tracks"].shape[-1])
    except (OSError, KeyError):
        return None
    return None
