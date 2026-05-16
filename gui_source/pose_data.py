"""LUCID-Lite data model. Mirrors pose-data.js, skipping editor-only state.

Session is a QObject that emits signals on mutation. Everything below the
Session level is a plain dataclass / mutable container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

import colors


@dataclass
class Skeleton:
    name: str
    nodes: list[str]
    edges: list[tuple[int, int]]


@dataclass
class Camera:
    name: str
    matrix: list[list[float]]
    dist: list[float]
    rvec: list[float]
    tvec: list[float]
    size: tuple[int, int]


@dataclass
class Instance:
    """Per-view 2D pose. Geometry is read-only in the Lite app."""
    points: list[Optional[tuple[float, float]]]  # one (x, y) per skeleton node or None
    track_idx: Optional[int]
    type: str = "predicted"  # "user" or "predicted"
    score: float = 0.0


@dataclass
class UnlinkedInstance:
    instance: Instance
    camera_name: str
    id: int


@dataclass
class Identity:
    id: int
    name: str
    color: str


@dataclass
class FrameGroup:
    frame_idx: int
    instances: dict[str, list[Instance]] = field(default_factory=dict)
    unlinked_instances: dict[str, list[UnlinkedInstance]] = field(default_factory=dict)

    def add_instance(self, camera_name: str, instance: Instance) -> None:
        self.instances.setdefault(camera_name, []).append(instance)

    def get_instances(self, camera_name: str) -> list[Instance]:
        return self.instances.get(camera_name, [])


@dataclass
class InstanceGroup:
    id: int
    identity_id: Optional[int] = None
    instances: dict[str, Instance] = field(default_factory=dict)  # SINGULAR per cam

    def add_instance(self, camera_name: str, instance: Instance) -> None:
        self.instances[camera_name] = instance


class Session(QObject):
    """Container for a full per-camera multi-view session.

    Emits Qt signals so views (video panels, timeline, assignment dock) can
    repaint themselves on mutation without direct coupling.
    """

    # Emitted after bulk load or large structural change.
    frame_groups_changed = Signal()
    # Emitted when set_frame_identity flips an assignment.
    identity_map_changed = Signal()
    # Emitted when Session.identities gains/loses/renames an entry.
    identities_changed = Signal()
    # Emitted when a track name is added/renamed.
    tracks_changed = Signal()
    # Emitted when the instance color mode flips ("track" <-> "identity").
    color_mode_changed = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.skeleton: Skeleton | None = None
        self.cameras: list[Camera] = []
        self.has_calibration: bool = False
        self.tracks: list[str] = []
        self.identities: list[Identity] = []
        self.color_mode: str = "track"  # "track" | "identity"
        # frame_idx -> FrameGroup
        self.frame_groups: dict[int, FrameGroup] = {}
        # frame_idx -> list[InstanceGroup]
        self.instance_groups: dict[int, list[InstanceGroup]] = {}
        # "camName:trackIdx" -> identity_id
        self.track_identity_map: dict[str, int] = {}
        # "frameIdx:camName:trackIdx" -> identity_id
        self.frame_identity_map: dict[str, int] = {}
        # camera_name -> path to mp4
        self.video_paths: dict[str, Path] = {}
        # internal counters
        self._identity_counter: int = 0
        self._instance_group_counter: int = 0

    # ---- queries ------------------------------------------------------

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.frame_groups.keys())

    @property
    def min_frame(self) -> int:
        return min(self.frame_groups) if self.frame_groups else 0

    @property
    def max_frame(self) -> int:
        return max(self.frame_groups) if self.frame_groups else 0

    def frame_group(self, idx: int) -> FrameGroup | None:
        return self.frame_groups.get(idx)

    def instance_groups_for(self, idx: int) -> list[InstanceGroup]:
        return self.instance_groups.get(idx, [])

    def camera_names(self) -> list[str]:
        return [c.name for c in self.cameras]

    # ---- identity / track lookup (mirror pose-data.js:631–676) --------

    def get_identity_id_for_track(
        self, frame_idx: int, camera_name: str, track_idx: int | None
    ) -> int | None:
        if track_idx is None:
            return None
        per_frame_key = f"{frame_idx}:{camera_name}:{track_idx}"
        if per_frame_key in self.frame_identity_map:
            return self.frame_identity_map[per_frame_key]
        global_key = f"{camera_name}:{track_idx}"
        return self.track_identity_map.get(global_key)

    def get_identity(self, identity_id: int | None) -> Identity | None:
        if identity_id is None:
            return None
        for ident in self.identities:
            if ident.id == identity_id:
                return ident
        return None

    # ---- mutations ----------------------------------------------------

    def set_frame_identity(
        self,
        frame_idx: int,
        camera_name: str,
        track_idx: int,
        identity_id: int | None,
    ) -> None:
        """Set per-frame identity for (camera, track). identity_id=None clears it."""
        key = f"{frame_idx}:{camera_name}:{track_idx}"
        if identity_id is None:
            self.frame_identity_map.pop(key, None)
        else:
            self.frame_identity_map[key] = identity_id
        self.identity_map_changed.emit()

    def set_global_track_identity(
        self, camera_name: str, track_idx: int, identity_id: int | None
    ) -> None:
        key = f"{camera_name}:{track_idx}"
        if identity_id is None:
            self.track_identity_map.pop(key, None)
        else:
            self.track_identity_map[key] = identity_id
        self.identity_map_changed.emit()

    def add_identity(self, name: str, color: str | None = None) -> Identity:
        if color is None:
            color = colors.next_palette_color(len(self.identities))
        ident = Identity(id=self._identity_counter, name=name, color=color)
        self._identity_counter += 1
        self.identities.append(ident)
        self.identities_changed.emit()
        return ident

    def add_track(self, name: str) -> int:
        if not name:
            name = f"track_{len(self.tracks)}"
        self.tracks.append(name)
        self.tracks_changed.emit()
        return len(self.tracks) - 1

    def set_color_mode(self, mode: str) -> None:
        if mode not in ("track", "identity"):
            raise ValueError(f"color_mode must be 'track' or 'identity', got {mode!r}")
        if mode == self.color_mode:
            return
        self.color_mode = mode
        self.color_mode_changed.emit(mode)

    def next_instance_group_id(self) -> int:
        self._instance_group_counter += 1
        return self._instance_group_counter

    # ---- loader entry point ------------------------------------------

    @classmethod
    def load_from_folder(cls, path: Path | str) -> "Session":
        from session_loader import load_session_from_folder
        return load_session_from_folder(Path(path))
