"""3D viewport of the triangulated skeleton from sft.trackIds.

Mirrors `luc3d/ui/viewport3d.js` (Three.js + OrbitControls) using
pyqtgraph's `GLViewWidget` so it embeds cleanly in a Qt dialog. Reads the
per-frame Group bundle that `track_push.push_frame_assignments` writes via
`session.set_groups_for_frame` and draws one skeleton per visible group:

  * scatter points  — every triangulated node (`group.points3d`)
  * line segments   — the skeleton edges connecting those nodes
  * faint XY grid   — a ground reference so orbit doesn't lose its frame

Mouse:
    left-drag        orbit
    middle-drag      pan (also: Shift + left-drag)
    wheel            dolly / zoom

Coloring follows `session.color_mode`:
    "track"     palette(track_idx of group.cam_track[0])
    "identity"  identity color via session.get_identity(id) — falls back to
                the track-mode color when the identity isn't registered.

Hooks `LucidLiteWindow.currentFrameChanged` so the view follows the
playhead automatically. Listens to `color_mode_changed`,
`identities_changed`, `identity_map_changed`, `frame_groups_changed` so a
swatch flip or fresh push refreshes the scene with no manual reload.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout

from colors import get_track_color


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    qc = QColor(hex_color)
    return (qc.redF(), qc.greenF(), qc.blueF(), alpha)


def _valid3(p) -> bool:
    if p is None:
        return False
    arr = np.asarray(p, dtype=float)
    return arr.shape == (3,) and bool(np.all(np.isfinite(arr)))


class Triangulation3DWindow(QDialog):
    def __init__(self, session, frame_idx: int, parent=None):
        super().__init__(parent)
        self.session = session
        self._frame_idx = int(frame_idx)
        self._items: list = []
        # First-frame autocenter so we don't blow the view away every refresh.
        self._centered = False

        self.setWindowTitle(f"3D Triangulation — Frame {self._frame_idx}")
        self.resize(960, 720)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # Header row: frame label on the left, color-mode indicator on the right.
        header = QHBoxLayout()
        self._frame_label = QLabel(f"Frame {self._frame_idx}")
        self._frame_label.setStyleSheet("font-weight: bold;")
        self._mode_label = QLabel(f"Color mode: {self.session.color_mode}")
        self._mode_label.setStyleSheet("color: #888;")
        header.addWidget(self._frame_label)
        header.addStretch()
        header.addWidget(self._mode_label)
        root.addLayout(header)

        # GLViewWidget — orbit/pan/zoom built in. Black background to mimic
        # the 2D video panels, so colored skeletons read with the same
        # contrast as on the videos.
        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor("#181818")
        self.gl.setCameraPosition(distance=900, elevation=20, azimuth=-60)
        root.addWidget(self.gl, stretch=1)

        # Faint grid as ground reference (XY plane at z=0).
        grid = gl.GLGridItem()
        grid.setSize(2000, 2000)
        grid.setSpacing(200, 200)
        grid.setColor((80, 80, 80, 180))
        self.gl.addItem(grid)
        # Stash grid so it isn't cleared on refresh.
        self._grid = grid

        # Wire up signals so the view re-renders when relevant state changes.
        self.session.frame_groups_changed.connect(self._on_frame_groups_changed)
        self.session.color_mode_changed.connect(self._on_color_mode_changed)
        self.session.identities_changed.connect(self._refresh)
        self.session.identity_map_changed.connect(self._refresh)
        # Follow the main window's playhead automatically — saves the user
        # from re-opening the window every time they seek.
        top = self.window().parent() if isinstance(self.window(), QDialog) else self.parent()
        # `parent` passed by caller is the LucidLiteWindow itself.
        main_window = parent if parent is not None else top
        if main_window is not None and hasattr(main_window, "currentFrameChanged"):
            main_window.currentFrameChanged.connect(self.set_frame)

        self._refresh()

    # ---- external API ----------------------------------------------------

    def set_frame(self, frame_idx: int) -> None:
        self._frame_idx = int(frame_idx)
        self._frame_label.setText(f"Frame {self._frame_idx}")
        self.setWindowTitle(f"3D Triangulation — Frame {self._frame_idx}")
        self._refresh()

    # ---- signal handlers -------------------------------------------------

    def _on_frame_groups_changed(self, frame_idx: int) -> None:
        if int(frame_idx) == self._frame_idx:
            self._refresh()

    def _on_color_mode_changed(self, mode: str) -> None:
        self._mode_label.setText(f"Color mode: {mode}")
        self._refresh()

    # ---- rendering -------------------------------------------------------

    def _clear_items(self) -> None:
        for it in self._items:
            self.gl.removeItem(it)
        self._items.clear()

    def _refresh(self) -> None:
        self._clear_items()
        bundle = self.session.frame_tracker_groups.get(self._frame_idx)
        if not bundle:
            return
        groups = bundle.get("groups", [])
        edges = list(self.session.skeleton.edges) if self.session.skeleton else []

        all_pts: list[np.ndarray] = []   # for autocenter
        for grp_idx, group in enumerate(groups):
            if not getattr(group, "valid", False):
                continue
            pts3d = getattr(group, "points3d", None)
            if pts3d is None:
                continue

            color_rgba = _hex_to_rgba(self._group_color(group, grp_idx))

            # Scatter: every node with finite (x, y, z).
            node_xyz = [np.asarray(p, dtype=float) for p in pts3d if _valid3(p)]
            if node_xyz:
                pos = np.stack(node_xyz, axis=0)
                scatter = gl.GLScatterPlotItem(
                    pos=pos, color=color_rgba, size=10, pxMode=True,
                )
                self.gl.addItem(scatter)
                self._items.append(scatter)
                all_pts.extend(node_xyz)

            # Skeleton edges — only the ones whose endpoints are both valid.
            edge_pts: list[np.ndarray] = []
            for src, dst in edges:
                if src >= len(pts3d) or dst >= len(pts3d):
                    continue
                p1, p2 = pts3d[src], pts3d[dst]
                if not (_valid3(p1) and _valid3(p2)):
                    continue
                edge_pts.append(np.asarray(p1, dtype=float))
                edge_pts.append(np.asarray(p2, dtype=float))
            if edge_pts:
                lines = gl.GLLinePlotItem(
                    pos=np.stack(edge_pts, axis=0),
                    color=color_rgba, width=2.0, mode="lines", antialias=True,
                )
                self.gl.addItem(lines)
                self._items.append(lines)

        # Center the camera on the first frame's centroid so the orbit
        # rig isn't pointing at the origin while the skeleton is ~1.2 m down
        # the +z axis (SLAP-2M calibration scale).
        if all_pts and not self._centered:
            centroid = np.mean(np.stack(all_pts, axis=0), axis=0)
            self.gl.opts["center"] = pg.Vector(*centroid.tolist())
            self.gl.update()
            self._centered = True

    def _group_color(self, group, grp_idx: int) -> str:
        """Resolve a hex color for a tracked group, honoring session.color_mode."""
        if self.session.color_mode == "identity":
            # Pick the first member of cam_track and look up its per-frame
            # identity override — that's the same path the 2D overlay uses
            # via Session.get_identity_id_for_track.
            for cam, ti in getattr(group, "cam_track", []):
                ident_id = self.session.get_identity_id_for_track(
                    self._frame_idx, cam, ti
                )
                if ident_id is None or ident_id < 0:
                    continue
                ident = self.session.get_identity(ident_id)
                if ident is not None:
                    return ident.color
            # Fall back to track-mode coloring when no identity is resolved.
        cam_track = getattr(group, "cam_track", None) or []
        if cam_track:
            _, first_track_idx = cam_track[0]
            return get_track_color(first_track_idx)
        return get_track_color(grp_idx)
