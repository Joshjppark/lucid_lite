"""LUCID-Lite data model. Mirrors pose-data.js, skipping editor-only state.

Session is a QObject that emits signals on mutation. Everything below the
Session level is a plain dataclass / mutable container.
"""
from __future__ import annotations

from contextlib import contextmanager
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
    # Emitted when overlay appearance (node radius, edge width) is tweaked.
    appearance_changed = Signal()
    # Emitted when frame_tracker_groups is mutated for some frame (the graph
    # window listens to know when to redraw). Payload-free: consumers re-read
    # state via Session.frame_tracker_groups.get(frame_idx).
    groups_changed = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.skeleton: Skeleton | None = None
        self.cameras: list[Camera] = []
        self.has_calibration: bool = False
        self.tracks: list[str] = []
        self.identities: list[Identity] = []
        self.color_mode: str = "track"  # "track" | "identity"
        # Overlay appearance (mutated by the Track/ID tab sliders).
        self.node_size: float = 4.0   # node circle radius in video pixels
        self.edge_width: float = 2.0  # skeleton edge stroke width in video pixels
        # Toggle for the per-instance identity-name text label drawn next to
        # the skeleton. Track-index labels ("t0", "t1") are unconditionally
        # suppressed in overlay_renderer._instance_label.
        self.show_identity_labels: bool = True
        # frame_idx -> FrameGroup
        self.frame_groups: dict[int, FrameGroup] = {}
        # frame_idx -> list[InstanceGroup]
        self.instance_groups: dict[int, list[InstanceGroup]] = {}
        # "camName:trackIdx" -> identity_id
        self.track_identity_map: dict[str, int] = {}
        # "frameIdx:camName:trackIdx" -> identity_id
        self.frame_identity_map: dict[str, int] = {}
        # frame_idx -> opaque bundle of tracker output. Populated only by
        # explicit user calls (see notebooks/tracking_test.ipynb
        # push_frame_assignments). Read by graph_window. Stored opaquely so
        # this module does not import from josh_source/tracker.py (which is
        # actively churning). Bundle shape:
        #   {
        #     "groups":           list[tracker.Group],          # from sft._run_bfs
        #     "adjacency_matrix": np.ndarray (n, n) | None,     # sft.adjacency_matrix
        #     "instance_list":    list[(track_idx, cam_name, pts)] | None,
        #   }
        # Graph rendering depends on Group attributes (.cam_track, .valid)
        # plus the matrix/instance_list pair for edge weights and layout.
        self.frame_tracker_groups: dict[int, dict] = {}
        # camera_name -> path to mp4
        self.video_paths: dict[str, Path] = {}
        # internal counters
        self._identity_counter: int = 0
        self._instance_group_counter: int = 0
        # Signal batching state. While `_batch_depth > 0`, mutation-emit
        # helpers route into `_pending_signals` instead of firing immediately.
        # On context exit each pending signal is emitted exactly once. This
        # mirrors the JS tracker, which never fires per-mutation observers —
        # `trackAll` writes plain Map entries and only redraws after the
        # whole loop completes.
        self._batch_depth: int = 0
        self._pending_signals: set[str] = set()

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

    # ---- signal batching ---------------------------------------------

    @contextmanager
    def batch_updates(self):
        """Suppress per-mutation signal emissions for the duration of the
        block, then emit each unique signal exactly once on exit.

        Use this around bulk operations (eg. `track_all`) so views don't
        rebuild themselves once per identity write. Nests safely.
        """
        self._batch_depth += 1
        try:
            yield self
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._pending_signals:
                pending = self._pending_signals
                self._pending_signals = set()
                # Order is intentional: identities first (so views that bind
                # to it can size containers), then global maps, then track
                # legend, then frame_groups, then appearance / color_mode.
                for name in (
                    "identities_changed", "tracks_changed",
                    "identity_map_changed", "frame_groups_changed",
                    "color_mode_changed", "appearance_changed",
                    "groups_changed",
                ):
                    if name in pending:
                        sig = getattr(self, name)
                        # color_mode_changed carries a payload — emit with
                        # the current mode rather than a stale arg.
                        if name == "color_mode_changed":
                            sig.emit(self.color_mode)
                        else:
                            sig.emit()

    def _emit(self, signal_name: str, *args) -> None:
        """Emit a signal — or, if batching, queue it for flush at end-of-batch."""
        if self._batch_depth > 0:
            self._pending_signals.add(signal_name)
            return
        sig = getattr(self, signal_name)
        sig.emit(*args)

    # ---- identity / track lookup (mirror pose-data.js:631–676) --------

    def get_identity_id_for_track(
        self, frame_idx: int, camera_name: str, track_idx: int | None
    ) -> int | None:
        """Strict lookup: per-frame override → global (cam, track) → None.

        Mirrors JS `getIdentityIdForTrack` (pose-data.js:790). No
        cross-camera fallback — if (cam, track) isn't mapped, returns None.

        A per-frame override of `-1` is the *uniqueness sentinel* written
        by `match_frame_instances` to suppress fallback to a stale global.
        It means "no identity at this frame for this (cam, track)".
        """
        if track_idx is None:
            return None
        per_frame_key = f"{frame_idx}:{camera_name}:{track_idx}"
        if per_frame_key in self.frame_identity_map:
            v = self.frame_identity_map[per_frame_key]
            return None if v is None or v < 0 else v
        global_key = f"{camera_name}:{track_idx}"
        return self.track_identity_map.get(global_key)

    def get_identity_for_track(
        self,
        track_idx: int | None,
        camera_name: str | None = None,
        frame_idx: int | None = None,
    ) -> "Identity | None":
        """Permissive lookup used by the OVERLAY renderer — mirrors JS
        `getIdentityForTrack` (pose-data.js:753), **including its
        cross-camera suffix-scan fallback**.

        Resolution order:
          1. per-frame override at (frame_idx, camera_name, track_idx)
          2. global direct hit at (camera_name, track_idx)
          3. **suffix scan**: first entry in `track_identity_map` whose key
             ends in `:trackIdx` — regardless of camera.

        Step (3) is the source of the duplicate-identity bug the user
        reported in the JS viewer: when (midL, 0) has no direct global
        mapping, the suffix scan returns whatever camera's `:0` is
        iterated first (insertion order). That returned identity may
        coincide with another track's identity already shown in the same
        camera at the same frame, producing the visual duplicate even
        though `get_identity_id_for_track` would have returned None.
        """
        if track_idx is None:
            return None

        # 1. Per-frame override. A value of `-1` is the uniqueness sentinel
        # written by match_frame_instances — it means "explicitly no
        # identity at this frame, do NOT fall back to global or suffix-
        # scan." Return None and stop.
        if frame_idx is not None and camera_name is not None:
            frame_key = f"{frame_idx}:{camera_name}:{track_idx}"
            if frame_key in self.frame_identity_map:
                v = self.frame_identity_map[frame_key]
                if v is None or v < 0:
                    return None
                return self.get_identity(v)

        # 1b. Per-frame without camera — match any cam at this frame.
        if frame_idx is not None and camera_name is None:
            frame_prefix = f"{frame_idx}:"
            track_suffix = f":{track_idx}"
            for k, v in self.frame_identity_map.items():
                if k.startswith(frame_prefix) and k.endswith(track_suffix):
                    return self.get_identity(v)

        # 2. Direct global lookup.
        if camera_name is not None:
            v = self.track_identity_map.get(f"{camera_name}:{track_idx}")
            if v is not None:
                return self.get_identity(v)

        # 3. Suffix scan — JS's overly-permissive fallback.
        suffix = f":{track_idx}"
        for key, val in self.track_identity_map.items():
            if key.endswith(suffix):
                return self.get_identity(val)

        return None

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
        self._emit("identity_map_changed")

    def set_global_track_identity(
        self, camera_name: str, track_idx: int, identity_id: int | None
    ) -> None:
        key = f"{camera_name}:{track_idx}"
        if identity_id is None:
            self.track_identity_map.pop(key, None)
        else:
            self.track_identity_map[key] = identity_id
        self._emit("identity_map_changed")

    def add_identity(self, name: str, color: str | None = None) -> Identity:
        if color is None:
            color = colors.next_palette_color(len(self.identities))
        ident = Identity(id=self._identity_counter, name=name, color=color)
        self._identity_counter += 1
        self.identities.append(ident)
        self._emit("identities_changed")
        return ident

    def add_track(self, name: str) -> int:
        if not name:
            name = f"track_{len(self.tracks)}"
        self.tracks.append(name)
        self._emit("tracks_changed")
        return len(self.tracks) - 1

    def set_color_mode(self, mode: str) -> None:
        if mode not in ("track", "identity"):
            raise ValueError(f"color_mode must be 'track' or 'identity', got {mode!r}")
        if mode == self.color_mode:
            return
        self.color_mode = mode
        self._emit("color_mode_changed", mode)

    def set_node_size(self, value: float) -> None:
        v = max(0.5, float(value))
        if abs(v - self.node_size) < 1e-6:
            return
        self.node_size = v
        self._emit("appearance_changed")

    def set_edge_width(self, value: float) -> None:
        v = max(0.5, float(value))
        if abs(v - self.edge_width) < 1e-6:
            return
        self.edge_width = v
        self._emit("appearance_changed")

    def set_show_identity_labels(self, value: bool) -> None:
        v = bool(value)
        if v == self.show_identity_labels:
            return
        self.show_identity_labels = v
        self._emit("appearance_changed")

    def set_groups_for_frame(
        self,
        frame_idx: int,
        groups,
        adjacency_matrix=None,
        instance_list=None,
        trackIds=None,
    ) -> None:
        """Store the tracker-derived groups for a single frame.

        `groups` is a list (or tuple) of tracker.Group objects from
        SingleFrameTrack._run_bfs. `adjacency_matrix` is the (n, n) ndarray
        with np.inf marking missing edges; `instance_list` is the parallel
        list of `(track_idx, cam_name, points)` tuples that maps matrix
        index → instance. Both are stored opaquely so the graph viewer
        can render edge weights without re-running the tracker.

        `trackIds` is the `sft.trackIds` dict (identity_id ->
        TrackedIdentity) at this frame, or None for an invalid frame.
        The graph window uses it to map each Group back to the stable
        identity_id assigned by the tracker — so node colors and
        toggle-row labels reflect the actual identity, not the
        positional index in `groups`.

        Pass `groups=None` or an empty sequence to clear the entry. The
        graph window listens on `groups_changed` to know to redraw.
        """
        if groups:
            self.frame_tracker_groups[frame_idx] = {
                "groups": list(groups),
                "adjacency_matrix": adjacency_matrix,
                "instance_list": (
                    list(instance_list) if instance_list is not None else None
                ),
                "trackIds": trackIds,
            }
        else:
            self.frame_tracker_groups.pop(frame_idx, None)
        self._emit("groups_changed")

    def get_groups_for_frame(self, frame_idx: int):
        """Returns the bundle dict (with keys 'groups', 'adjacency_matrix',
        'instance_list') or None if no entry exists at this frame.
        """
        return self.frame_tracker_groups.get(frame_idx)

    def next_instance_group_id(self) -> int:
        self._instance_group_counter += 1
        return self._instance_group_counter

    # ---- loader entry point ------------------------------------------

    @classmethod
    def load_from_folder(cls, path: Path | str) -> "Session":
        from session_loader import load_session_from_folder
        return load_session_from_folder(Path(path))
