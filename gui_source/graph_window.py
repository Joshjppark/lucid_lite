"""Group-assignment graph viewer for LUCID-Lite.

Self-contained renderer that draws the k-partite group graph for a single
frame. Opened by the "Show Group Assignment Graphs" button in the
assignment panel's Groups tab. The data itself is populated by the notebook
side via `push_frame_assignments` — this window only reads from
`Session.frame_tracker_groups`.

Data source:
    The stored bundle has three pieces:
      * "groups"           — list[tracker.Group] from SingleFrameTrack._run_bfs
      * "adjacency_matrix" — np.ndarray (n, n); np.inf = no edge
      * "instance_list"    — list[(track_idx, cam_name, pts)] aligned with
                             matrix indices

    Edges are read directly from the adjacency matrix: every (i, j) with
    a finite weight becomes one edge labeled with that weight. Groups
    determine node coloring; the matrix determines connectivity. (BFS in
    the tracker uses the same matrix to build the groups, so every finite
    edge lies inside exactly one group — but we render whatever the matrix
    actually says, so the graph stays honest if the tracker semantics
    change later.)

Layout strategy:
    * One partition per camera (= corner of a regular k-sided polygon).
      Cameras are inferred from the visible instances each render — toggling
      a group off can shrink the polygon if that group was the sole tenant
      of some camera.
    * Within a camera, instances spread along the tangent at the vertex.
    * Nodes are colored by their group's identity color (matches the video
      overlay); fall back to a deterministic track palette color when no
      Identity is registered yet.

Display panel (right side):
    * "Weights" toggle — show / hide edge-weight numeric labels.
    * "Id" section — one toggle per identity group at the rendered frame.
      Turning a group off removes its nodes AND any edges that involve
      those nodes (and therefore their weight labels).
    * Toggle state is reset when the frame changes; preserved on same-
      frame redraws (including ones triggered by groups_changed).

Tracker coupling:
    * This file does NOT import from josh_source.tracker — Group objects
      are read by attribute access only (`.cam_track`, `.valid`,
      `.edge_weights` hook). If tracker.py renames those, the only
      failure surface is here.

Dependencies: PySide6 + matplotlib (both already required). No networkx —
layout is direct math.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from colors import CAMERA_COLORS as _CAMERA_COLORS, get_camera_color, get_track_color
from pose_data import Session


# Display-panel sizing.
_DISPLAY_PANEL_MIN_W = 170
_DISPLAY_PANEL_MAX_W = 230


class _DoubleClickLabel(QLabel):
    """QLabel that fires a callback on mouse double-click."""

    def __init__(self, on_double_click, parent=None):
        super().__init__(parent)
        self._on_double_click = on_double_click

    def mouseDoubleClickEvent(self, event):  # noqa: N802 (Qt naming)
        self._on_double_click()
        super().mouseDoubleClickEvent(event)


class GroupGraphWindow(QDialog):
    """Non-modal dialog showing the group graph for one frame.

    Listens on `Session.groups_changed` so a push_frame_assignments call in
    the notebook triggers a live redraw if the window is open.
    """

    def __init__(self, session: Session, initial_frame: int, parent=None):
        super().__init__(parent)
        self.session = session
        self._frame_idx: int = int(initial_frame)

        # Display-panel state — preserved across same-frame redraws.
        self._show_weights: bool = True
        # False = color nodes by identity (group_idx → identity color),
        # edges colored to match. True = color nodes by camera, edges
        # rendered neutral gray (cameras don't have inter-camera identity
        # by definition, so a coloured edge would be meaningless).
        self._color_by_camera: bool = False
        self._visible_groups: set[int] = set()
        # Identity rows whose per-view reprojection-score breakdown is
        # currently expanded. Reset on frame change / group cardinality
        # change (see `_refresh`); preserved across toggle redraws.
        self._expanded_groups: set[int] = set()
        # Tracks last-rendered frame + group count so we know when to reset
        # `_visible_groups` (new frame OR cardinality changed → reset to all).
        self._last_rendered_frame: Optional[int] = None
        self._last_group_count: int = -1

        # ---- node-drag state ------------------------------------------
        # Per-frame user-stashed positions in data coordinates. Survives
        # redraws (color-mode flips, toggle changes, re-pushes) and frame
        # navigations. Keyed by (frame_idx, (cam, track)). Empty until the
        # user drags. Reset per-frame via the "Reset Layout" button.
        self._node_positions: dict[int, dict[tuple[str, int], np.ndarray]] = {}
        # Set during _draw so the mouse handlers can hit-test against the
        # actual rendered positions (which may be polygon defaults or user
        # overrides). Keyed by (cam, track).
        self._current_node_pos: dict[tuple[str, int], np.ndarray] = {}
        # Drag bookkeeping.
        self._drag_node: Optional[tuple[str, int]] = None
        self._drag_offset: Optional[np.ndarray] = None
        self._hovered_node: Optional[tuple[str, int]] = None

        self.setWindowTitle("Group Assignment Graph")
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(900, 640)

        root = QVBoxLayout(self)

        # ---- Header row ------------------------------------------------
        header = QHBoxLayout()
        header.addWidget(QLabel("Group Assignment Graph"))
        header.addStretch(1)
        header.addWidget(QLabel("Frame:"))

        self._frame_label = _DoubleClickLabel(on_double_click=self._begin_frame_edit)
        self._frame_label.setText(str(self._frame_idx))
        self._frame_label.setStyleSheet(
            "font-weight: bold; padding: 2px 8px; "
            "border: 1px solid #888; border-radius: 3px; min-width: 48px;"
        )
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setToolTip("Double-click to edit")
        header.addWidget(self._frame_label)

        self._frame_edit = QLineEdit()
        self._frame_edit.setValidator(QIntValidator())
        self._frame_edit.setMaximumWidth(80)
        self._frame_edit.setAlignment(Qt.AlignCenter)
        self._frame_edit.editingFinished.connect(self._commit_frame_edit)
        self._frame_edit.hide()
        header.addWidget(self._frame_edit)

        root.addLayout(header)

        # ---- Body: canvas (left) + display panel (right) ---------------
        body = QHBoxLayout()
        body.setSpacing(8)

        # Canvas.
        self._canvas = FigureCanvas(Figure(figsize=(7, 5), tight_layout=True))
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Matplotlib event hooks for click-drag node repositioning. The
        # canvas keeps the connection ids around for its own lifetime; we
        # do not need to disconnect — they die with the widget.
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        body.addWidget(self._canvas, stretch=1)

        # "No data" placeholder occupies the same body slot — shown via swap.
        self._no_data_label = QLabel("No Grouping Data")
        self._no_data_label.setAlignment(Qt.AlignCenter)
        self._no_data_label.setStyleSheet(
            "color: #999; font-size: 18pt; font-style: italic;"
        )
        self._no_data_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._no_data_label.hide()
        body.addWidget(self._no_data_label, stretch=1)

        # Display panel: a scrollable container so many id toggles fit.
        self._display_panel = self._build_display_panel_scaffold()
        body.addWidget(self._display_panel, stretch=0)

        root.addLayout(body, stretch=1)

        # Live updates: groups_changed fires after any push_frame_assignments.
        session.groups_changed.connect(self._refresh)

        self._refresh()

    # ------------------------------------------------------------------
    # Frame-index editor swap
    # ------------------------------------------------------------------

    def _begin_frame_edit(self) -> None:
        self._frame_edit.setText(str(self._frame_idx))
        self._frame_label.hide()
        self._frame_edit.show()
        self._frame_edit.setFocus()
        self._frame_edit.selectAll()

    def _commit_frame_edit(self) -> None:
        text = self._frame_edit.text().strip()
        try:
            new_frame = int(text)
        except ValueError:
            new_frame = self._frame_idx
        self._frame_idx = new_frame
        self._frame_label.setText(str(self._frame_idx))
        self._frame_edit.hide()
        self._frame_label.show()
        self._refresh()

    def set_frame(self, frame_idx: int) -> None:
        """Switch the displayed frame without opening the editor."""
        self._frame_idx = int(frame_idx)
        self._frame_label.setText(str(self._frame_idx))
        self._refresh()

    # ------------------------------------------------------------------
    # Display panel
    # ------------------------------------------------------------------

    def _build_display_panel_scaffold(self) -> QFrame:
        """Build the right-side panel chrome (title + scrollable body).

        Per-frame toggles are added to `self._display_body_layout` in
        `_rebuild_display_panel`.
        """
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        panel.setMinimumWidth(_DISPLAY_PANEL_MIN_W)
        panel.setMaximumWidth(_DISPLAY_PANEL_MAX_W)

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 8, 8, 8)

        title = QLabel("Display")
        title.setStyleSheet("font-weight: bold;")
        outer.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(sep)

        # Scrollable container for the toggle rows.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        self._display_body_layout = QVBoxLayout(body)
        self._display_body_layout.setContentsMargins(0, 0, 0, 0)
        self._display_body_layout.setSpacing(4)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

        return panel

    def _rebuild_display_panel(self, groups: list) -> None:
        """Repopulate the toggle rows based on the groups list. Restores
        toggle state from `self._show_weights` / `self._visible_groups`.
        """
        # Clear existing rows. QLayout.takeAt(0) + deleteLater is the safe
        # idiom for fully clearing a parented layout.
        while self._display_body_layout.count():
            item = self._display_body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        # ---- Layout reset --------------------------------------------
        # Drag-state is kept per-frame; Reset Layout clears overrides
        # for the currently-rendered frame only.
        reset_btn = QPushButton("Reset Layout")
        reset_btn.setToolTip(
            "Restore default polygon layout for this frame (clears any "
            "node drags you've performed)."
        )
        reset_btn.clicked.connect(self._on_reset_layout)
        self._display_body_layout.addWidget(reset_btn)

        drag_hint = QLabel("Drag any node to reposition")
        drag_hint.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        drag_hint.setWordWrap(True)
        self._display_body_layout.addWidget(drag_hint)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.HLine)
        sep0.setFrameShadow(QFrame.Sunken)
        self._display_body_layout.addWidget(sep0)

        # ---- Color-by mode -------------------------------------------
        # Two radios (Identity / Camera) sharing a parent → Qt auto-exclusive.
        # Identity = current default: nodes & edges colored by group identity.
        # Camera   = nodes use the camera palette; edges go neutral gray.
        mode_label = QLabel("Color by:")
        mode_label.setStyleSheet("font-weight: bold;")
        self._display_body_layout.addWidget(mode_label)

        mode_row = QWidget()
        mr = QHBoxLayout(mode_row)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.setSpacing(6)
        self._mode_identity_rb = QRadioButton("Identity")
        self._mode_camera_rb = QRadioButton("Camera")
        self._mode_identity_rb.setChecked(not self._color_by_camera)
        self._mode_camera_rb.setChecked(self._color_by_camera)
        # We hook BOTH radios — Qt fires toggled for each (one True, one
        # False). Handlers early-out on the False side to avoid double work.
        self._mode_identity_rb.toggled.connect(self._on_mode_identity_toggled)
        self._mode_camera_rb.toggled.connect(self._on_mode_camera_toggled)
        mr.addWidget(self._mode_identity_rb)
        mr.addWidget(self._mode_camera_rb)
        mr.addStretch(1)
        self._display_body_layout.addWidget(mode_row)

        sep_mode = QFrame()
        sep_mode.setFrameShape(QFrame.HLine)
        sep_mode.setFrameShadow(QFrame.Sunken)
        self._display_body_layout.addWidget(sep_mode)

        # ---- Weights toggle -------------------------------------------
        weights_chk = QCheckBox("Weights")
        weights_chk.setChecked(self._show_weights)
        weights_chk.toggled.connect(self._on_weights_toggled)
        weights_chk.setToolTip("Show numeric edge-weight labels on each edge")
        self._display_body_layout.addWidget(weights_chk)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        self._display_body_layout.addWidget(sep)

        # ---- Id section ----------------------------------------------
        id_header = QLabel("Id")
        id_header.setStyleSheet("font-weight: bold;")
        self._display_body_layout.addWidget(id_header)

        for g_idx in range(len(groups)):
            group = groups[g_idx]
            scores = _read_reproj_scores(group)
            avg = _avg_reproj_score(scores)

            # Container wraps top-row + (collapsed-by-default) detail panel.
            container = QWidget()
            cv = QVBoxLayout(container)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(1)

            # ---- top row: swatch | checkbox | avg score | ▸/▾ ----
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)

            color = _identity_color(self.session, g_idx)
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color}; "
                f"border: 1px solid #444; border-radius: 2px;"
            )
            rl.addWidget(swatch)

            ident = self.session.get_identity(g_idx)
            name = ident.name if ident is not None else f"id_{g_idx}"
            chk = QCheckBox(name)
            chk.setChecked(g_idx in self._visible_groups)
            chk.toggled.connect(
                lambda checked, gi=g_idx: self._on_group_toggled(gi, checked)
            )
            chk.setToolTip(
                f"Toggle visibility of group {g_idx} (nodes + edges + weights)"
            )
            rl.addWidget(chk)

            # Visual gap between the identity name and its numeric average.
            # Layout spacing is 4 px globally; a dedicated 12 px spacer here
            # gives the avg score room to breathe without changing the
            # swatch→checkbox gap or the stretch→button gap.
            rl.addSpacing(12)

            # Average reprojection error — directly to the right of the
            # identity name, before the stretch. Italic + a touch smaller
            # than the checkbox label so it reads as metadata, not a label.
            # No explicit color override → inherits the theme's text color,
            # which is white(ish) under a dark palette and dark under a
            # light palette.
            avg_lbl = QLabel(_format_score(avg))
            avg_lbl.setStyleSheet("font-size: 10pt; font-style: italic;")
            avg_lbl.setToolTip(
                "Mean reprojection error across all views for this group"
                if avg is not None
                else "No reprojection scores available for this group"
            )
            rl.addWidget(avg_lbl)

            rl.addStretch(1)

            # Expand button — small flat triangle, doubles as collapse.
            expand_btn = QToolButton()
            expand_btn.setAutoRaise(True)
            expand_btn.setCheckable(True)
            expand_btn.setFixedWidth(18)
            expand_btn.setStyleSheet("QToolButton { font-size: 9pt; }")
            expand_btn.setToolTip("Show per-view reprojection scores")
            # Disable the button when there's nothing to expand.
            expand_btn.setEnabled(bool(scores))
            rl.addWidget(expand_btn)

            cv.addWidget(row)

            # ---- detail panel: per-view scores, slightly indented ----
            detail = QWidget()
            dl = QVBoxLayout(detail)
            # 20 px left indent so per-view rows align under the checkbox
            # label, not under the swatch.
            dl.setContentsMargins(20, 0, 0, 2)
            dl.setSpacing(0)
            if scores:
                for cam in sorted(scores.keys()):
                    val = scores[cam]
                    try:
                        f = float(val)
                        score_str = _format_score(f)
                    except (TypeError, ValueError):
                        score_str = "—"
                    line = QLabel(f"{cam}: {score_str}")
                    # Bigger than before (10 pt) and color inherited from the
                    # theme palette — under a dark theme that comes out
                    # near-white instead of the previously hard-coded #555
                    # which read as nearly transparent.
                    line.setStyleSheet("font-size: 10pt;")
                    dl.addWidget(line)
            else:
                line = QLabel("(no scores)")
                line.setStyleSheet("font-size: 10pt; font-style: italic;")
                dl.addWidget(line)
            cv.addWidget(detail)

            # Initial expand state + button glyph.
            is_expanded = g_idx in self._expanded_groups
            detail.setVisible(is_expanded)
            expand_btn.setChecked(is_expanded)
            expand_btn.setText("▾" if is_expanded else "▸")
            expand_btn.toggled.connect(
                lambda checked, gi=g_idx, d=detail, b=expand_btn:
                    self._on_expand_toggled(gi, checked, d, b)
            )

            self._display_body_layout.addWidget(container)

        self._display_body_layout.addStretch(1)

    def _on_weights_toggled(self, checked: bool) -> None:
        self._show_weights = bool(checked)
        self._draw_current()

    def _on_mode_identity_toggled(self, checked: bool) -> None:
        # React only on the "selected" side of the toggle. The Camera radio
        # firing False as a side-effect is silenced by this guard.
        if not checked:
            return
        if not self._color_by_camera:
            return
        self._color_by_camera = False
        self._draw_current()

    def _on_mode_camera_toggled(self, checked: bool) -> None:
        if not checked:
            return
        if self._color_by_camera:
            return
        self._color_by_camera = True
        self._draw_current()

    def _on_group_toggled(self, group_idx: int, checked: bool) -> None:
        if checked:
            self._visible_groups.add(group_idx)
        else:
            self._visible_groups.discard(group_idx)
        self._draw_current()

    def _on_reset_layout(self) -> None:
        """Drop any user-dragged positions for the current frame and redraw
        with the default polygon layout."""
        self._node_positions.pop(self._frame_idx, None)
        self._draw_current()

    def _on_expand_toggled(
        self, group_idx: int, checked: bool, detail_panel: QWidget, btn: QToolButton
    ) -> None:
        """Show / hide a group's per-view score breakdown without rebuilding
        the whole display panel (so other expansions, the Weights toggle,
        and visibility checkboxes keep their state)."""
        if checked:
            self._expanded_groups.add(group_idx)
        else:
            self._expanded_groups.discard(group_idx)
        detail_panel.setVisible(checked)
        btn.setText("▾" if checked else "▸")

    # ------------------------------------------------------------------
    # Node drag — matplotlib mouse events
    # ------------------------------------------------------------------

    # Hit-test threshold in data coordinates. The axes span ~3.7 units
    # across and scatter markers are ~0.22 units in diameter at the
    # default figure size; 0.15 gives a comfortable click-target without
    # accidentally grabbing a neighbor when nodes are stacked closely
    # along a camera's tangent.
    _HIT_RADIUS = 0.15

    def _hit_test_node(self, event) -> Optional[tuple[str, int]]:
        """Return the (cam, track) of the closest node under `event`, or
        None if no node is within `_HIT_RADIUS`. Operates in data coords
        on the single subplot. `event.inaxes is None` for clicks outside
        the plot area."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        if not self._current_node_pos:
            return None
        x, y = float(event.xdata), float(event.ydata)
        best, best_d = None, float("inf")
        for node, pos in self._current_node_pos.items():
            dx, dy = x - float(pos[0]), y - float(pos[1])
            d = math.hypot(dx, dy)
            if d < best_d:
                best, best_d = node, d
        if best is not None and best_d <= self._HIT_RADIUS:
            return best
        return None

    def _on_mouse_press(self, event) -> None:
        """Begin a drag if the left button presses on top of a node."""
        if event.button != 1:
            return
        hit = self._hit_test_node(event)
        if hit is None:
            return
        self._drag_node = hit
        # Lock the grip offset so the node tracks the cursor at the click
        # point rather than snapping its center to the cursor — feels more
        # natural during drag.
        grip = np.array([float(event.xdata), float(event.ydata)])
        self._drag_offset = self._current_node_pos[hit] - grip
        self._canvas.setCursor(Qt.ClosedHandCursor)

    def _on_mouse_motion(self, event) -> None:
        """Either drag the active node or update the hover cursor."""
        # Active drag: update position + persist + redraw.
        if self._drag_node is not None:
            if event.inaxes is None or event.xdata is None or event.ydata is None:
                return
            cursor = np.array([float(event.xdata), float(event.ydata)])
            new_pos = cursor + (
                self._drag_offset
                if self._drag_offset is not None
                else np.zeros(2)
            )
            # Persist so the position survives toggles, color-mode flips,
            # group re-pushes, and frame round-trips.
            frame_overrides = self._node_positions.setdefault(self._frame_idx, {})
            frame_overrides[self._drag_node] = new_pos.copy()
            # Local stash for the next motion event's hit-test.
            self._current_node_pos[self._drag_node] = new_pos
            self._draw_current()
            return

        # Hover-cursor toggle when not dragging.
        hit = self._hit_test_node(event)
        if hit == self._hovered_node:
            return
        self._hovered_node = hit
        if hit is None:
            self._canvas.unsetCursor()
        else:
            self._canvas.setCursor(Qt.OpenHandCursor)

    def _on_mouse_release(self, event) -> None:
        if self._drag_node is None or event.button != 1:
            return
        self._drag_node = None
        self._drag_offset = None
        # Restore hover cursor if mouse is still over a node, else default.
        hit = self._hit_test_node(event)
        self._hovered_node = hit
        if hit is None:
            self._canvas.unsetCursor()
        else:
            self._canvas.setCursor(Qt.OpenHandCursor)

    # ------------------------------------------------------------------
    # Render orchestration
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Pull the bundle for the current frame, reset state if needed,
        rebuild the display panel, then draw.
        """
        stored = self.session.frame_tracker_groups.get(self._frame_idx)
        if not stored:
            self._show_no_data()
            return

        groups = stored.get("groups") or []
        if not groups:
            self._show_no_data()
            return

        # Reset visibility + expand state on frame change OR if the group
        # cardinality changed (eg. the user re-pushed the same frame with
        # different tracker output).
        if (
            self._frame_idx != self._last_rendered_frame
            or self._last_group_count != len(groups)
        ):
            self._visible_groups = set(range(len(groups)))
            self._expanded_groups = set()
        self._last_rendered_frame = self._frame_idx
        self._last_group_count = len(groups)

        self._no_data_label.hide()
        self._canvas.show()
        self._display_panel.show()
        self._rebuild_display_panel(groups)
        self._draw(groups, stored.get("adjacency_matrix"), stored.get("instance_list"))

    def _draw_current(self) -> None:
        """Re-draw using the currently stored bundle without touching the
        display-panel scaffold (used by toggle handlers)."""
        stored = self.session.frame_tracker_groups.get(self._frame_idx)
        if not stored:
            self._show_no_data()
            return
        groups = stored.get("groups") or []
        if not groups:
            self._show_no_data()
            return
        self._draw(groups, stored.get("adjacency_matrix"), stored.get("instance_list"))

    def _show_no_data(self) -> None:
        self._canvas.hide()
        self._display_panel.hide()
        self._no_data_label.show()
        self._canvas.figure.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # The actual rendering
    # ------------------------------------------------------------------

    def _draw(self, groups: list, adj, instance_list) -> None:
        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        # ---- map (cam, track) -> group_idx ----------------------------
        # >>> TRACKER attribute read: Group.cam_track is list[(cam, track)]. <<<
        node_to_group: dict[tuple[str, int], int] = {}
        for g_idx, g in enumerate(groups):
            for cam, track in g.cam_track:
                node_to_group[(cam, track)] = g_idx

        # ---- map matrix-index -> (cam, track) -------------------------
        # instance_list[k] = (track_idx, cam_name, points_ndarray)
        idx_to_node: dict[int, tuple[str, int]] = {}
        if instance_list is not None:
            for k, item in enumerate(instance_list):
                track_idx, cam_name = item[0], item[1]
                idx_to_node[k] = (cam_name, track_idx)

        # ---- visibility filter ----------------------------------------
        visible = self._visible_groups
        node_is_visible: dict[tuple[str, int], bool] = {
            node: (g_idx in visible)
            for node, g_idx in node_to_group.items()
        }

        # ---- collect cameras that have at least one visible node ------
        cams: list[str] = []
        cam_seen: set[str] = set()
        for g_idx, g in enumerate(groups):
            if g_idx not in visible:
                continue
            for cam, _t in g.cam_track:
                if cam not in cam_seen:
                    cam_seen.add(cam)
                    cams.append(cam)

        if not cams:
            # Everything toggled off — keep the canvas, draw a hint.
            self._current_node_pos = {}
            ax.text(0.5, 0.5, "All groups hidden",
                    ha="center", va="center", fontsize=14, color="#888888",
                    transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        k = max(len(cams), 1)

        # ---- polygon vertex positions per camera ----------------------
        polygon_pos: dict[str, np.ndarray] = {}
        for i, cam in enumerate(cams):
            theta = 2.0 * math.pi * i / k - math.pi / 2.0
            polygon_pos[cam] = np.array([math.cos(theta), math.sin(theta)])

        # ---- collect unique (cam, track) visible nodes per camera -----
        nodes_per_cam: dict[str, list[tuple[str, int]]] = {c: [] for c in cams}
        for node, vis in node_is_visible.items():
            if not vis:
                continue
            cam = node[0]
            if cam in nodes_per_cam and node not in nodes_per_cam[cam]:
                nodes_per_cam[cam].append(node)

        # Stable sort by track within each camera so layout is deterministic.
        for cam in cams:
            nodes_per_cam[cam].sort(key=lambda n: n[1])

        # ---- 2-D node positions: spread along the tangent at vertex ---
        node_pos: dict[tuple[str, int], np.ndarray] = {}
        for cam in cams:
            anchor = polygon_pos[cam]
            r_norm = float(np.linalg.norm(anchor)) or 1.0
            radial = anchor / r_norm
            tangent = np.array([-radial[1], radial[0]])
            ns = nodes_per_cam[cam]
            n = len(ns)
            step = 0.20
            for j, node in enumerate(ns):
                offset = (j - (n - 1) / 2.0) * step
                inward = 1.0 - 0.06 * (1 if n > 1 else 0)
                node_pos[node] = anchor * inward + offset * tangent

        # Override default layout with any user-dragged positions for this
        # frame. We re-assign rather than translate so successive drags
        # compose naturally. Camera-anchor labels stay at polygon vertices.
        user_overrides = self._node_positions.get(self._frame_idx, {})
        for node, override_pos in user_overrides.items():
            if node in node_pos:
                node_pos[node] = override_pos.copy()

        # Stash for the mouse-handler hit-test.
        self._current_node_pos = node_pos

        # ---- edges from adjacency_matrix -----------------------------
        # Iterate the upper triangle. Skip np.inf / NaN (no edge).
        edges_drawn = 0
        if adj is not None and instance_list is not None:
            adj_arr = np.asarray(adj)
            n = adj_arr.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    w = adj_arr[i, j]
                    if not np.isfinite(w):
                        continue
                    node_i = idx_to_node.get(i)
                    node_j = idx_to_node.get(j)
                    if node_i is None or node_j is None:
                        continue
                    if not node_is_visible.get(node_i, False):
                        continue
                    if not node_is_visible.get(node_j, False):
                        continue
                    p1 = node_pos.get(node_i)
                    p2 = node_pos.get(node_j)
                    if p1 is None or p2 is None:
                        continue
                    g_i = node_to_group.get(node_i)
                    g_j = node_to_group.get(node_j)
                    # Edge color depends on the current "Color by" mode:
                    #   - Identity mode: tinted with the group's identity
                    #     color (same group → tinted; cross-group → neutral,
                    #     which BFS shouldn't produce but we render honestly).
                    #   - Camera mode: cameras don't have a natural inter-
                    #     camera "edge identity" so edges go neutral gray.
                    if self._color_by_camera:
                        color = "#888888"
                    elif g_i == g_j and g_i is not None:
                        color = _identity_color(self.session, g_i)
                    else:
                        color = "#888888"
                    ax.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]],
                        color=color, linewidth=1.5, alpha=0.7, zorder=1,
                    )
                    if self._show_weights:
                        mx, my = (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5
                        ax.text(
                            mx, my, f"{float(w):.2f}",
                            ha="center", va="center", fontsize=7,
                            color="#222222",
                            bbox=dict(
                                boxstyle="round,pad=0.18",
                                facecolor="#ffffff", edgecolor="none", alpha=0.85,
                            ),
                            zorder=3,
                        )
                    edges_drawn += 1

        # ---- nodes -----------------------------------------------------
        for node, pos in node_pos.items():
            g_idx = node_to_group.get(node)
            if self._color_by_camera:
                # Color by the node's camera (= the polygon vertex it
                # belongs to). Group membership only affects visibility
                # and the invalid-group red border.
                color = _camera_color(self.session, node[0])
            else:
                color = (
                    _identity_color(self.session, g_idx)
                    if g_idx is not None else "#bbbbbb"
                )
            edgecolor, lw = "#222222", 1.0
            # Highlight nodes that participate in an invalid group with red.
            if g_idx is not None and not getattr(groups[g_idx], "valid", True):
                edgecolor, lw = "#cc0000", 2.0
            ax.scatter(
                [pos[0]], [pos[1]],
                s=480, c=[color], edgecolors=edgecolor, linewidths=lw, zorder=2,
            )
            ax.annotate(
                f"t{node[1]}", pos, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#ffffff", zorder=4,
            )

        # ---- camera-name labels at polygon vertices --------------------
        # In Camera color-mode the box outline matches the camera color so
        # the label itself doubles as a legend — no separate swatch list
        # required in the panel.
        for cam, pos in polygon_pos.items():
            if self._color_by_camera:
                edgecolor, linewidth = _camera_color(self.session, cam), 2.0
            else:
                edgecolor, linewidth = "#888", 0.8
            ax.annotate(
                cam, pos * 1.28,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#f4f4f4",
                          edgecolor=edgecolor, linewidth=linewidth),
                zorder=5,
            )

        ax.set_xlim(-1.85, 1.85)
        ax.set_ylim(-1.85, 1.85)
        ax.set_aspect("equal")
        self._canvas.draw_idle()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _identity_color(session: Session, group_idx: int | None) -> str:
    """Color for the node belonging to group `group_idx`. Identity-driven if
    registered, else deterministic track-palette fallback."""
    if group_idx is None:
        return "#bbbbbb"
    ident = session.get_identity(group_idx)
    if ident is not None:
        return ident.color
    return get_track_color(group_idx)


def _camera_color(session: Session, cam_name: str) -> str:
    """Stable color for a camera, indexed by its position in
    `session.camera_names()` (which preserves loader order). Unknown
    cameras get the neutral fallback so a hand-edited bundle that points at
    a stale camera name doesn't crash the renderer."""
    try:
        idx = session.camera_names().index(cam_name)
    except ValueError:
        return "#888888"
    return get_camera_color(idx)


def _read_reproj_scores(group) -> dict:
    """Return the per-view reprojection-error dict from a tracker.Group.

    Handles both the current `Group.reproj_score` (singular — see
    josh_source/tracker.py line 99) and a hypothetical future
    `Group.reproj_scores` (plural) without code changes here.

    Returns `{}` if the attribute is absent / not a dict.
    """
    # >>> TRACKER attribute access: Group.reproj_score (dict[cam, float]). <<<
    scores = (
        getattr(group, "reproj_scores", None)
        if getattr(group, "reproj_scores", None) is not None
        else getattr(group, "reproj_score", None)
    )
    if not isinstance(scores, dict):
        return {}
    return scores


def _avg_reproj_score(scores: dict) -> Optional[float]:
    """Mean of the finite numeric values in `scores`, or None if there are
    none (eg. empty dict, all-NaN, non-numeric)."""
    if not scores:
        return None
    vals: list[float] = []
    for v in scores.values():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _format_score(value: Optional[float]) -> str:
    """Two-decimal score, or em-dash when there's no usable value."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:.2f}"
