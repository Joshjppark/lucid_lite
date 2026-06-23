"""SLEAP-io-native multi-camera label container.

Wraps a list of per-camera labels (loaded from either proofread analysis H5s
or raw `.predictions.slp` files) together with camera calibration so the
tracker can run without ever touching the lucid-lite GUI / Session / Qt.

The tracker consumes:
    cameras                                list[Camera]
    skeleton                               Skeleton
    camera_names()                         list[str]
    max_frame                              int
    frame_group(frame_idx)                 FrameGroup or None

A GUI can still be built on top later via main.make_window(labels=...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gui_source.analysis_h5_reader import merge_analysis_h5_into_session
from gui_source.pose_data import Camera, FrameGroup, Session, Skeleton
from gui_source.session_loader import (
    load_session_structure,
    rebuild_instance_groups,
)
from gui_source.slp_reader import merge_slp_into_session


class MultiviewLabels:
    """Per-camera labels + calibration, decoupled from the GUI."""

    def __init__(
        self,
        session: Session,
        source_paths: dict[str, Path],
        source_kind: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ):
        self._session = session
        self.source_paths = source_paths
        self.source_kind = source_kind                  # 'proofread' | 'predictions' | 'mixed'
        self.start_frame = start_frame                  # inclusive; None = unbounded
        self.end_frame = end_frame                      # exclusive; None = unbounded

    @classmethod
    def from_folder(
        cls,
        folder,
        *,
        prefer: str = "proofread",
        start: Optional[int] = None,
        end: Optional[int] = None,
        videos_dir: "str | Path | None" = None,
        verbose: bool = True,
    ) -> "MultiviewLabels":
        """Discover per-cam labels under `folder` and load them.

        Search order per camera subdir:
          prefer='proofread' (default): *.predictions.proofread.slp.analysis.h5
                                        -> fallback to *.predictions.slp
          prefer='predictions':         *.predictions.slp
                                        -> fallback to *.predictions.proofread.slp.analysis.h5

        Frame subset:
          start (inclusive) / end (exclusive) trim session.frame_groups
          to that range. Both default to None = unbounded. The GUI's
          playback / timeline / navigation all derive their bounds from
          session.min_frame / session.max_frame, so they naturally
          restrict to the kept range. The underlying video files are
          NOT modified — the on-demand decoder still seeks any frame in
          the full mp4 — only the GUI's reachable range is trimmed.

        Video lookup:
          The lucid GUI's video panels need actual mp4/avi files. By default
          they're discovered next to the labels (one mp4 per cam subdir of
          `folder`). When labels live in a "predictions-only" mirror like
          slap_GT/<date>/<sid>/<cam>/ (no videos), pass `videos_dir` to
          point at the sibling folder that DOES carry mp4s — e.g.
          videos_dir=f'{LUCID_FOLDERS}/{SESSION_ID}_small'. Each cam subdir
          under `videos_dir` is searched for `*.mp4` then `*.avi`.

        Prints what it found at each cam (file + kind) so the state of
        Finding+using the predictions is visible from the start.
        """
        if prefer not in ("proofread", "predictions"):
            raise ValueError(f"prefer must be 'proofread' or 'predictions', got {prefer!r}")
        if start is not None and end is not None and start >= end:
            raise ValueError(f"start ({start}) must be < end ({end})")

        folder = Path(folder).expanduser().resolve()
        session, resolved = load_session_structure(folder)

        if verbose:
            print(f"[multiview-labels] folder: {folder}")
            print(f"[multiview-labels] prefer: {prefer!r}")
            if start is not None or end is not None:
                print(f"[multiview-labels] frame subset: [{start}, {end})")
            print(f"[multiview-labels] calibration cameras: {len(session.cameras)}")

        source_paths: dict[str, Path] = {}
        kinds_found: set[str] = set()

        for cam_name, sub in resolved:
            file_path, kind = _pick_file(sub, prefer)
            if file_path is None:
                if verbose:
                    print(f"[multiview-labels]   {cam_name:>6} -> (no labels found)")
                continue
            source_paths[cam_name] = file_path
            kinds_found.add(kind)
            if verbose:
                print(f"[multiview-labels]   {cam_name:>6} -> {file_path.name}  ({kind})")

            if kind == "proofread":
                merge_analysis_h5_into_session(session, file_path, cam_name)
            else:
                merge_slp_into_session(session, file_path, cam_name)

        if kinds_found == {"proofread"}:
            source_kind = "proofread"
        elif kinds_found == {"predictions"}:
            source_kind = "predictions"
        elif kinds_found:
            source_kind = "mixed"
        else:
            source_kind = "empty"

        # Trim frame_groups to [start, end) BEFORE rebuilding instance_groups
        # so the rebuild doesn't waste work on frames we're about to drop.
        if start is not None or end is not None:
            total_before = len(session.frame_groups)
            for fi in list(session.frame_groups):
                if start is not None and fi < start:
                    del session.frame_groups[fi]
                elif end is not None and fi >= end:
                    del session.frame_groups[fi]
            if verbose:
                print(
                    f"[multiview-labels] trimmed {total_before - len(session.frame_groups)} "
                    f"frames outside [{start}, {end})"
                )

        rebuild_instance_groups(session)

        # Override video paths if the labels folder doesn't carry them.
        # session.video_paths was already populated by load_session_structure
        # from `folder/<cam>/*.mp4`; this lets the user redirect to a sibling
        # folder that actually has the videos.
        if videos_dir is not None:
            videos_root = Path(videos_dir).expanduser().resolve()
            if not videos_root.is_dir():
                raise ValueError(f"videos_dir not a directory: {videos_root}")
            found_videos: dict[str, Path] = {}
            for cam_name, _ in resolved:
                cam_video_dir = videos_root / cam_name
                if not cam_video_dir.is_dir():
                    if verbose:
                        print(f"[multiview-labels] videos   {cam_name:>6} -> (no cam dir under {videos_root.name})")
                    continue
                hits = sorted(cam_video_dir.glob("*.mp4")) or sorted(cam_video_dir.glob("*.avi"))
                if not hits:
                    if verbose:
                        print(f"[multiview-labels] videos   {cam_name:>6} -> (no mp4/avi in {cam_video_dir.name})")
                    continue
                found_videos[cam_name] = hits[0]
                if verbose:
                    print(f"[multiview-labels] videos   {cam_name:>6} -> {hits[0].name}")
            session.video_paths = found_videos
        else:
            if verbose:
                missing = [c for c, _ in resolved if c not in session.video_paths]
                if missing:
                    print(f"[multiview-labels] WARNING: no mp4/avi found for cams {missing} "
                          f"(GUI panels will be blank). Pass videos_dir=... to redirect.")

        if verbose:
            print(f"[multiview-labels] source kind: {source_kind}")
            print(f"[multiview-labels] frames loaded: {len(session.frame_groups)}")

        return cls(session, source_paths, source_kind, start_frame=start, end_frame=end)

    # ------------- tracker-facing API ------------- #

    @property
    def cameras(self) -> list[Camera]:
        return self._session.cameras

    @property
    def skeleton(self) -> Skeleton:
        return self._session.skeleton

    def camera_names(self) -> list[str]:
        return [c.name for c in self._session.cameras]

    @property
    def max_frame(self) -> int:
        if not self._session.frame_groups:
            return 0
        return max(self._session.frame_groups) + 1

    def frame_group(self, frame_idx: int) -> Optional[FrameGroup]:
        return self._session.frame_groups.get(frame_idx)

    # ------------- GUI hook ------------- #

    @property
    def session(self) -> Session:
        """Underlying Session — for building a window via main.make_window."""
        return self._session


def _pick_file(cam_dir: Path, prefer: str) -> tuple[Optional[Path], Optional[str]]:
    proofread = sorted(cam_dir.glob("*.predictions.proofread.slp.analysis.h5"))
    predictions = sorted(
        p for p in cam_dir.glob("*.predictions.slp") if "proofread" not in p.name
    )

    if prefer == "proofread":
        if proofread:
            return proofread[0], "proofread"
        if predictions:
            return predictions[0], "predictions"
    else:  # prefer == 'predictions'
        if predictions:
            return predictions[0], "predictions"
        if proofread:
            return proofread[0], "proofread"
    return None, None
