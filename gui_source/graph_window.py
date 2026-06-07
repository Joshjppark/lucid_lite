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
    * "Id" section — one toggle per identity in `sft.trackIds` at the
      rendered frame (ordered by stable identity_id). Turning an
      identity off removes its nodes AND any edges that involve those
      nodes (and therefore their weight labels). Hidden identities
      (group=None this frame) show up as disabled rows tagged "(hidden)".
    * Toggle state is keyed by identity_id and persisted per-frame, so
      an identity that vanishes for a frame and comes back keeps its
      visibility.

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

        # Display-panel state — global, not per-frame. These describe
        # *how* to render, not *what* the user has customized on a given
        # frame, so they survive frame navigation as-is.
        self._show_weights: bool = True
        # False = color nodes by identity (group_idx → identity color),
        # edges colored to match. True = color nodes by camera, edges
        # rendered neutral gray (cameras don't have inter-camera identity
        # by definition, so a coloured edge would be meaningless).
        self._color_by_camera: bool = False

        # ---- per-frame visual cache ----------------------------------
        # Four parallel dicts keyed by frame_idx. They hold whatever the
        # user has customized at that frame and persist across frame
        # navigation, group re-pushes that preserve topology, and
        # window close/reopen (in-memory; lives with the dialog).
        #
        # Cache values are keyed by *stable identity_id* (from
        # sft.trackIds), so an identity that disappears for a frame and
        # then comes back retains its toggle/expand state automatically.
        # The disabled-edges cache is keyed by frozenset({(cam, t),
        # (cam, t)}), also stable across re-pushes.
        #
        # Memory: only frames the user actively customized end up in
        # these dicts — unedited frames hit the default branch at draw
        # time and never allocate an entry.
        self._cached_visible_identities: dict[int, set[int]] = {}
        self._cached_expanded_identities: dict[int, set[int]] = {}
        self._cached_disabled_edges: dict[int, set[frozenset]] = {}
        self._cached_node_positions: dict[int, dict[tuple[str, int], np.ndarray]] = {}

        # Working copies for the currently rendered frame. Populated in
        # `_refresh` from the per-frame cache (or defaults if absent);
        # the toggle handlers mutate these and persist back to the cache.
        self._visible_identities: set[int] = set()
        self._expanded_identities: set[int] = set()

        # ---- transient render-time stashes (not cached) --------------
        # Set during _draw so the mouse handlers can hit-test against
        # the actual rendered positions. Keyed by (cam, track) /
        # frozenset({(cam,t), (cam,t)}). Re-populated every _draw call.
        self._current_node_pos: dict[tuple[str, int], np.ndarray] = {}
        self._current_edges: dict[frozenset, tuple[np.ndarray, np.ndarray]] = {}

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

        # Follow the main window's playhead so arrow keys + space-bar
        # playback (registered as application-wide QShortcuts on the
        # main window) keep the graph view in sync without having to
        # re-implement the keybindings here. Same approach as
        # Triangulation3DWindow. `parent` is the LucidLiteWindow.
        if parent is not None and hasattr(parent, "currentFrameChanged"):
            parent.currentFrameChanged.connect(self.set_frame)

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

    def _ident_pairs(self, groups: list, trackIds) -> list[tuple[int, object]]:
        """Return ordered (identity_id, Group | None) pairs to render.

        Source of truth is `sft.trackIds` (dict[identity_id ->
        TrackedIdentity]). Each TrackedIdentity holds the Group object
        it owns this frame, or None if the identity is occluded.

        Falls back to using positional g_idx as the identity when the
        bundle predates trackIds (or trackIds is None on an invalid
        frame). In that case all entries have a non-None group.

        Ordered by identity_id so the display panel stays stable across
        frames (an identity that vanishes for one frame and comes back
        slots back into the same row).
        """
        if trackIds:
            return [
                (int(ident_id), getattr(ti, "group", None))
                for ident_id, ti in sorted(trackIds.items())
            ]
        return [(g_idx, g) for g_idx, g in enumerate(groups)]

    def _rebuild_display_panel(self, groups: list, trackIds=None) -> None:
        """Repopulate the toggle rows based on the groups list. Restores
        toggle state from `self._show_weights` / `self._visible_identities`.
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

        # Iterate identities directly from sft.trackIds. Hidden
        # identities (group is None) get a dimmed row so the user sees
        # the full identity set, but their checkbox/expand are inert
        # since they contribute no nodes this frame.
        for ident_id, group in self._ident_pairs(groups, trackIds):
            scores = _read_reproj_scores(group) if group is not None else {}
            avg = _avg_reproj_score(scores)
            is_hidden = group is None

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

            color = _identity_color(self.session, ident_id)
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color}; "
                f"border: 1px solid #444; border-radius: 2px;"
                + (" opacity: 0.4;" if is_hidden else "")
            )
            rl.addWidget(swatch)

            ident = self.session.get_identity(ident_id)
            base_name = ident.name if ident is not None else f"id_{ident_id}"
            label = f"{base_name} (hidden)" if is_hidden else base_name
            chk = QCheckBox(label)
            chk.setChecked(ident_id in self._visible_identities)
            chk.setEnabled(not is_hidden)
            chk.toggled.connect(
                lambda checked, ii=ident_id: self._on_identity_toggled(ii, checked)
            )
            chk.setToolTip(
                f"Toggle visibility of identity {ident_id} (nodes + edges + weights)"
                if not is_hidden
                else f"Identity {ident_id} has no detection this frame"
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
                    line.setStyleSheet("font-size: 10pt;")
                    dl.addWidget(line)
            else:
                line = QLabel("(no scores)")
                line.setStyleSheet("font-size: 10pt; font-style: italic;")
                dl.addWidget(line)
            cv.addWidget(detail)

            # Initial expand state + button glyph.
            is_expanded = ident_id in self._expanded_identities
            detail.setVisible(is_expanded)
            expand_btn.setChecked(is_expanded)
            expand_btn.setText("▾" if is_expanded else "▸")
            expand_btn.toggled.connect(
                lambda checked, ii=ident_id, d=detail, b=expand_btn:
                    self._on_expand_toggled(ii, checked, d, b)
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

    def _on_identity_toggled(self, ident_id: int, checked: bool) -> None:
        if checked:
            self._visible_identities.add(ident_id)
        else:
            self._visible_identities.discard(ident_id)
        # Persist a copy so navigating away and back restores the toggle
        # state. We snapshot rather than alias so a later `_refresh` that
        # reads the cache doesn't pick up in-progress edits.
        self._cached_visible_identities[self._frame_idx] = set(self._visible_identities)
        self._draw_current()

    def _on_reset_layout(self) -> None:
        """Drop any user-dragged positions for the current frame and redraw
        with the default polygon layout."""
        self._cached_node_positions.pop(self._frame_idx, None)
        self._draw_current()

    def _on_expand_toggled(
        self, ident_id: int, checked: bool, detail_panel: QWidget, btn: QToolButton
    ) -> None:
        """Show / hide an identity's per-view score breakdown without rebuilding
        the whole display panel (so other expansions, the Weights toggle,
        and visibility checkboxes keep their state)."""
        if checked:
            self._expanded_identities.add(ident_id)
        else:
            self._expanded_identities.discard(ident_id)
        # Persist so the expand state survives frame navigation.
        self._cached_expanded_identities[self._frame_idx] = set(self._expanded_identities)
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
    # Tighter than the node radius — edges are 1.5 px lines, so the
    # cursor needs to be close. Still generous enough to hit comfortably
    # when zoomed to the default extent.
    _EDGE_HIT_RADIUS = 0.05

    def _hit_test_edge(self, event) -> Optional[frozenset]:
        """Return the edge_key of the segment under `event`, or None.

        Closest point-to-segment distance against every line drawn in
        the last _draw pass. Edge keys are frozensets of (cam, track)
        endpoint tuples, so they're stable regardless of which order
        the matrix iteration visits the endpoints.
        """
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return None
        if not self._current_edges:
            return None
        p = np.array([float(event.xdata), float(event.ydata)])
        best, best_d = None, float("inf")
        for key, (a, b) in self._current_edges.items():
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                d = float(np.linalg.norm(p - a))
            else:
                t = float(np.dot(p - a, ab) / denom)
                t = max(0.0, min(1.0, t))
                closest = a + t * ab
                d = float(np.linalg.norm(p - closest))
            if d < best_d:
                best, best_d = key, d
        if best is not None and best_d <= self._EDGE_HIT_RADIUS:
            return best
        return None

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
        """Begin a drag if the left button presses on top of a node;
        otherwise toggle the muted state of any edge under the cursor."""
        if event.button != 1:
            return
        hit = self._hit_test_node(event)
        if hit is not None:
            self._drag_node = hit
            # Lock the grip offset so the node tracks the cursor at the
            # click point rather than snapping its center to the cursor
            # — feels more natural during drag.
            grip = np.array([float(event.xdata), float(event.ydata)])
            self._drag_offset = self._current_node_pos[hit] - grip
            self._canvas.setCursor(Qt.ClosedHandCursor)
            return

        edge_hit = self._hit_test_edge(event)
        if edge_hit is not None:
            disabled = self._cached_disabled_edges.setdefault(self._frame_idx, set())
            if edge_hit in disabled:
                disabled.discard(edge_hit)
            else:
                disabled.add(edge_hit)
            self._draw_current()

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
            frame_overrides = self._cached_node_positions.setdefault(self._frame_idx, {})
            frame_overrides[self._drag_node] = new_pos.copy()
            # Local stash for the next motion event's hit-test.
            self._current_node_pos[self._drag_node] = new_pos
            self._draw_current()
            return

        # Hover-cursor toggle when not dragging. Nodes win over edges so
        # the cursor stays consistent with what a click would actually do.
        hit = self._hit_test_node(event)
        if hit == self._hovered_node:
            return
        self._hovered_node = hit
        if hit is not None:
            self._canvas.setCursor(Qt.OpenHandCursor)
        elif self._hit_test_edge(event) is not None:
            self._canvas.setCursor(Qt.PointingHandCursor)
        else:
            self._canvas.unsetCursor()

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

    def _ensure_caches(self) -> None:
        """Backfill cache / working-state attributes that may be missing
        on an autoreloaded instance whose `__init__` ran against an
        older field layout. No-op on a fresh instance."""
        for name in (
            "_cached_visible_identities",
            "_cached_expanded_identities",
            "_cached_disabled_edges",
            "_cached_node_positions",
        ):
            if not hasattr(self, name):
                setattr(self, name, {})
        for name in ("_visible_identities", "_expanded_identities"):
            if not hasattr(self, name):
                setattr(self, name, set())

    def _refresh(self) -> None:
        """Pull the bundle for the current frame, reset state if needed,
        rebuild the display panel, then draw.
        """
        self._ensure_caches()
        stored = self.session.frame_tracker_groups.get(self._frame_idx)
        if not stored:
            self._show_no_data()
            return

        groups = stored.get("groups") or []
        if not groups:
            self._show_no_data()
            return

        trackIds = stored.get("trackIds")

        # Load working state from the per-frame cache. Defaults: all
        # currently-present identities visible, nothing expanded.
        #
        # The cache is keyed by stable identity_id from sft.trackIds, so
        # an identity that disappears for a frame retains its toggle
        # state for when it reappears — no cardinality-based wipe needed.
        # We still intersect the cached set with the present identities
        # so the working `_visible_identities` only contains things that
        # actually exist this frame (avoids drawing references to gone
        # identities).
        present_idents = {ident_id for ident_id, _g in self._ident_pairs(groups, trackIds)}

        cached_vis = self._cached_visible_identities.get(self._frame_idx)
        if cached_vis is None:
            self._visible_identities = set(present_idents)
        else:
            self._visible_identities = set(cached_vis) & present_idents

        cached_exp = self._cached_expanded_identities.get(self._frame_idx)
        self._expanded_identities = set(cached_exp) if cached_exp else set()

        self._no_data_label.hide()
        self._canvas.show()
        self._display_panel.show()
        self._rebuild_display_panel(groups, trackIds=trackIds)
        self._draw(
            groups, stored.get("adjacency_matrix"), stored.get("instance_list"),
            trackIds=trackIds,
        )

    def _draw_current(self) -> None:
        """Re-draw using the currently stored bundle without touching the
        display-panel scaffold (used by toggle handlers)."""
        self._ensure_caches()
        stored = self.session.frame_tracker_groups.get(self._frame_idx)
        if not stored:
            self._show_no_data()
            return
        groups = stored.get("groups") or []
        if not groups:
            self._show_no_data()
            return
        self._draw(
            groups, stored.get("adjacency_matrix"), stored.get("instance_list"),
            trackIds=stored.get("trackIds"),
        )

    def _show_no_data(self) -> None:
        self._canvas.hide()
        self._display_panel.hide()
        self._no_data_label.show()
        self._canvas.figure.clear()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # The actual rendering
    # ------------------------------------------------------------------

    def _draw(self, groups: list, adj, instance_list, trackIds=None) -> None:
        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        # ---- iterate identities (the unit of truth) -------------------
        # Each entry is (identity_id, Group | None). Hidden identities
        # contribute no nodes/edges but still appear in the display panel.
        # >>> TRACKER attribute read: Group.cam_track is list[(cam, track)]. <<<
        ident_pairs = self._ident_pairs(groups, trackIds)
        node_to_ident: dict[tuple[str, int], int] = {}
        ident_to_group: dict[int, object] = {}
        for ident_id, group in ident_pairs:
            if group is None:
                continue
            ident_to_group[ident_id] = group
            for cam, track in group.cam_track:
                node_to_ident[(cam, track)] = ident_id

        # ---- map matrix-index -> (cam, track) -------------------------
        # instance_list[k] = (track_idx, cam_name, points_ndarray)
        idx_to_node: dict[int, tuple[str, int]] = {}
        if instance_list is not None:
            for k, item in enumerate(instance_list):
                track_idx, cam_name = item[0], item[1]
                idx_to_node[k] = (cam_name, track_idx)

        # ---- visibility filter ----------------------------------------
        visible = self._visible_identities
        node_is_visible: dict[tuple[str, int], bool] = {
            node: (ident_id in visible)
            for node, ident_id in node_to_ident.items()
        }

        # ---- collect cameras that have at least one visible node ------
        cams: list[str] = []
        cam_seen: set[str] = set()
        for ident_id, group in ident_pairs:
            if group is None or ident_id not in visible:
                continue
            for cam, _t in group.cam_track:
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
        user_overrides = self._cached_node_positions.get(self._frame_idx, {})
        for node, override_pos in user_overrides.items():
            if node in node_pos:
                node_pos[node] = override_pos.copy()

        # Stash for the mouse-handler hit-test.
        self._current_node_pos = node_pos

        # ---- edges from adjacency_matrix -----------------------------
        # Iterate the upper triangle. Skip np.inf / NaN (no edge).
        edges_drawn = 0
        # Reset the hit-test registry for the new draw pass.
        self._current_edges = {}
        disabled_edges = self._cached_disabled_edges.get(self._frame_idx, set())
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
                    ident_i = node_to_ident.get(node_i)
                    ident_j = node_to_ident.get(node_j)
                    edge_key = frozenset({node_i, node_j})
                    is_muted = edge_key in disabled_edges
                    # Edge color depends on the current "Color by" mode:
                    #   - Identity mode: tinted with the identity color
                    #     (same identity → tinted; cross-identity →
                    #     neutral, which BFS shouldn't produce but we
                    #     render honestly).
                    #   - Camera mode: cameras don't have a natural inter-
                    #     camera "edge identity" so edges go neutral gray.
                    # Muted (user-toggled-off) overrides both: light gray
                    # at low alpha, weight label dimmed to match.
                    if is_muted:
                        color = "#bbbbbb"
                        edge_alpha = 0.25
                    elif self._color_by_camera:
                        color = "#888888"
                        edge_alpha = 0.7
                    elif ident_i == ident_j and ident_i is not None:
                        color = _identity_color(self.session, ident_i)
                        edge_alpha = 0.7
                    else:
                        color = "#888888"
                        edge_alpha = 0.7
                    ax.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]],
                        color=color, linewidth=1.5, alpha=edge_alpha, zorder=1,
                    )
                    self._current_edges[edge_key] = (
                        np.asarray(p1, dtype=float),
                        np.asarray(p2, dtype=float),
                    )
                    if self._show_weights:
                        mx, my = (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5
                        label_color = "#888888" if is_muted else "#222222"
                        label_bg_alpha = 0.3 if is_muted else 0.85
                        ax.text(
                            mx, my, f"{float(w):.2f}",
                            ha="center", va="center", fontsize=7,
                            color=label_color,
                            bbox=dict(
                                boxstyle="round,pad=0.18",
                                facecolor="#ffffff", edgecolor="none",
                                alpha=label_bg_alpha,
                            ),
                            zorder=3,
                        )
                    edges_drawn += 1

        # ---- nodes -----------------------------------------------------
        for node, pos in node_pos.items():
            ident_id = node_to_ident.get(node)
            if self._color_by_camera:
                # Color by the node's camera (= the polygon vertex it
                # belongs to). Identity membership only affects visibility
                # and the invalid-group red border.
                color = _camera_color(self.session, node[0])
            else:
                color = (
                    _identity_color(self.session, ident_id)
                    if ident_id is not None else "#bbbbbb"
                )
            edgecolor, lw = "#222222", 1.0
            # Highlight nodes that participate in an invalid group with red.
            group_obj = ident_to_group.get(ident_id) if ident_id is not None else None
            if group_obj is not None and not getattr(group_obj, "valid", True):
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
