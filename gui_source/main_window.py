"""LUCID-Lite main window — assembles the camera dock area, view strip,
timeline, and assignment sidebar.

The video panels live inside a NESTED QMainWindow whose dock system gives
us close / drag-rearrange / drag-to-tabify / drag-to-float for free —
matching luc3d's Dockview pane manager. The outer main window keeps the
left view strip and the right assignment sidebar around that nested area.
"""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QAction, QColor, QDrag, QGuiApplication, QKeySequence, QPainter, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QSplitter, QTabBar, QVBoxLayout, QWidget,
)

from assignment_panel import IdentityAssignmentPanel
from playback_controls import PlaybackControls
from pose_data import Session
from timeline_widget import TimelineWidget
from video_panel import VideoPanelWidget
from view_strip import ViewStripWidget

SIDEBAR_DEFAULT_WIDTH = 320
SIDEBAR_MIN_WIDTH = 260
TIMELINE_DEFAULT_HEIGHT = 200
TIMELINE_MIN_HEIGHT = 60


# Custom MIME type used to identify lucid-lite video-tab drags. Putting our
# own MIME type on the QDrag prevents the grid from accepting unrelated
# drops (eg. files dragged from the file manager).
_TAB_MIME_TYPE = "application/x-lucid-video-tab"


class DraggableTabHeader(QWidget):
    """The tab-style header bar that hosts the camera name + close button
    AND starts a QDrag when the user click-drags on it.

    Drag flow:
      1. Left-press on the header records `_drag_start` (in local coords).
      2. mouseMoveEvent waits until movement exceeds Qt's startDragDistance
         (jitter threshold) before kicking off the drag.
      3. The QDrag carries the camera name on `_TAB_MIME_TYPE` and uses a
         pixmap of the header itself as the drag image. Hotspot is set to
         the original press point so the floating ghost sits naturally
         under the cursor.

    The close button is a child QPushButton, so Qt routes its clicks to it
    directly — the header's mousePressEvent never fires for those clicks,
    so close and drag don't conflict.
    """

    def __init__(self, cam_name: str, on_close_cb, parent=None):
        super().__init__(parent)
        self.cam_name = cam_name
        self._drag_start: QPoint | None = None

        self.setFixedHeight(22)
        self.setObjectName(f"tab_header::{cam_name}")
        self.setStyleSheet(
            "QWidget#" + self.objectName() + " { "
            "background-color: #2c2c2c; "
            "border-top: 1px solid #4a4a4a; "
            "border-left: 1px solid #4a4a4a; "
            "border-right: 1px solid #4a4a4a; }"
        )
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(f"Drag to move {cam_name} elsewhere in the grid")

        hl = QHBoxLayout(self)
        hl.setContentsMargins(8, 0, 2, 0)
        hl.setSpacing(4)

        title = QLabel(cam_name, self)
        title.setStyleSheet(
            "QLabel { color: #dddddd; font-weight: bold; font-size: 9pt; "
            "background: transparent; border: none; }"
        )
        hl.addWidget(title)
        hl.addStretch(1)

        close_btn = QPushButton("×", self)
        close_btn.setFixedSize(18, 18)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setToolTip(f"Close {cam_name} (re-open from the Views strip)")
        close_btn.setStyleSheet(
            "QPushButton { color: #bbb; background: transparent; border: none; "
            "font-size: 14pt; font-weight: bold; padding: 0; margin: 0; }"
            "QPushButton:hover { color: #ffffff; background: #c04848; "
            "border-radius: 3px; }"
        )
        close_btn.clicked.connect(on_close_cb)
        hl.addWidget(close_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or self._drag_start is None:
            return super().mouseMoveEvent(event)
        # Wait for jitter threshold before starting the drag.
        if (event.position().toPoint() - self._drag_start).manhattanLength() \
                < QApplication.startDragDistance():
            return
        self._begin_drag()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def _begin_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_TAB_MIME_TYPE, self.cam_name.encode("utf-8"))
        drag.setMimeData(mime)
        # Compact drag pixmap (~130 px wide) instead of grabbing the whole
        # header — when tabs span the full panel width the floater can
        # otherwise obscure neighbouring drop targets. Hotspot is near the
        # left edge so the cursor visually sits inside the floater.
        pix = self._make_drag_pixmap()
        drag.setPixmap(pix)
        drag.setHotSpot(QPoint(12, pix.height() // 2))
        drag.exec(Qt.MoveAction)
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)

    def _make_drag_pixmap(self) -> QPixmap:
        """Build the narrow tab-pill shown beside the cursor during drag."""
        w, h = 130, 22
        pix = QPixmap(w, h)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(QColor("#2c2c2c"))
            painter.setPen(QColor("#5b8def"))  # accent border = "in flight"
            painter.drawRoundedRect(0, 0, w - 1, h - 1, 3, 3)
            painter.setPen(QColor("#dddddd"))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(
                pix.rect().adjusted(10, 0, -8, 0),
                Qt.AlignVCenter | Qt.AlignLeft,
                self.cam_name,
            )
        finally:
            painter.end()
        return pix


class _DropOverlay(QWidget):
    """Translucent quadrant indicator drawn over the panel area of a
    VideoTabContainer while a tab is being dragged over it.

    Sits as a child of the tab container, positioned just below the
    header. `WA_TransparentForMouseEvents` keeps it from intercepting
    drag events so the container still receives them.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._zone: str | None = None

    def set_zone(self, zone: str | None) -> None:
        """Switch which quadrant is highlighted (or `None` to hide)."""
        if zone == self._zone:
            return
        self._zone = zone
        if zone is None:
            self.hide()
        else:
            self.show()
            self.raise_()
        self.update()

    def paintEvent(self, event):
        if self._zone is None:
            return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        if self._zone == "west":
            rect = QRect(0, 0, w // 2, h)
        elif self._zone == "east":
            rect = QRect(w - w // 2, 0, w // 2, h)
        elif self._zone == "north":
            rect = QRect(0, 0, w, h // 2)
        else:  # south
            rect = QRect(0, h - h // 2, w, h // 2)

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(rect, QColor(91, 141, 239, 90))   # translucent blue
            painter.setPen(QColor(91, 141, 239, 220))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        finally:
            painter.end()


class VideoTabContainer(QWidget):
    """Tab-styled wrapper around a single VideoPanelWidget.

    Header bar shows the camera name + close button (×) and is also the
    drag handle. Clicking close hides this container and emits
    `closed(cam_name)`.

    Also acts as a *drop target* — accepting a video-tab drag from
    another `DraggableTabHeader` and reporting it as one of four edge
    zones (`west`, `east`, `north`, `south`) to a callback registered
    via `set_zone_drop_handler`. The cursor's nearest edge picks the
    zone (so cursor near top → north → split horizontally and stack
    above; near left → west → insert to the left within the same row;
    etc.). A `_DropOverlay` child paints translucent feedback over the
    chosen quadrant while the cursor is moving.
    """
    closed = Signal(str)  # camera name

    def __init__(self, cam_name: str, panel: VideoPanelWidget, parent=None):
        super().__init__(parent)
        self.cam_name = cam_name
        self.panel = panel
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAcceptDrops(True)
        self._on_zone_drop = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = DraggableTabHeader(cam_name, self._on_close_clicked, self)
        root.addWidget(self.header, stretch=0)
        root.addWidget(panel, stretch=1)

        # Overlay child — geometry tracks the panel area (header excluded)
        # so the highlighted zone reads as "where in the panel the new tab
        # will land" rather than overdrawing the title bar.
        self._overlay = _DropOverlay(self)
        self._overlay.hide()

    def _on_close_clicked(self) -> None:
        self.hide()
        self.closed.emit(self.cam_name)

    # ---- drop-zone API -------------------------------------------------

    def set_zone_drop_handler(self, cb) -> None:
        """Install a callback `cb(src_cam, target_cam, zone)` invoked
        when a tab is dropped onto this container."""
        self._on_zone_drop = cb

    def resizeEvent(self, event):
        super().resizeEvent(event)
        hh = self.header.height()
        self._overlay.setGeometry(0, hh, self.width(), max(0, self.height() - hh))

    def _compute_zone(self, pos: QPoint) -> str:
        """Nearest-edge picker. Returns one of west/east/north/south."""
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return "east"
        rx = pos.x() / w
        ry = pos.y() / h
        distances = {
            "west": rx, "east": 1.0 - rx,
            "north": ry, "south": 1.0 - ry,
        }
        return min(distances, key=distances.get)

    def _is_compatible_drag(self, event) -> bool:
        """Reject non-tab drags and self-drops."""
        if not event.mimeData().hasFormat(_TAB_MIME_TYPE):
            return False
        src_cam = bytes(event.mimeData().data(_TAB_MIME_TYPE)).decode("utf-8")
        return src_cam != self.cam_name

    def dragEnterEvent(self, event):
        if not self._is_compatible_drag(event):
            event.ignore()
            return
        self._overlay.set_zone(self._compute_zone(event.position().toPoint()))
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if not self._is_compatible_drag(event):
            event.ignore()
            return
        self._overlay.set_zone(self._compute_zone(event.position().toPoint()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._overlay.set_zone(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        if not self._is_compatible_drag(event):
            return
        src_cam = bytes(event.mimeData().data(_TAB_MIME_TYPE)).decode("utf-8")
        zone = self._compute_zone(event.position().toPoint())
        self._overlay.set_zone(None)
        if self._on_zone_drop is not None:
            self._on_zone_drop(src_cam, self.cam_name, zone)
        event.acceptProposedAction()


class VideoGridContainer(QWidget):
    """Drop-aware QWidget that hosts the nested-splitter video grid.

    Accepts any drop carrying `_TAB_MIME_TYPE` (= a video-tab drag started
    by `DraggableTabHeader`). The actual move logic lives on the main
    window — `set_drop_handler` installs a callback that receives the
    dropped camera name + the local position so the main window can pick
    the right row/column.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._on_drop = None  # set by set_drop_handler

    def set_drop_handler(self, cb) -> None:
        self._on_drop = cb

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_TAB_MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_TAB_MIME_TYPE):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(_TAB_MIME_TYPE):
            return
        cam = bytes(event.mimeData().data(_TAB_MIME_TYPE)).decode("utf-8")
        if self._on_drop is not None:
            self._on_drop(cam, event.position().toPoint())
        event.acceptProposedAction()


class LucidLiteWindow(QMainWindow):
    # Emitted after clamp/dedupe in set_current_frame — notebooks can subscribe
    # to live frame navigation via window.currentFrameChanged.connect(...).
    currentFrameChanged = Signal(int)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"LUCID-Lite — {session.folder.name if session.folder else 'no session'}"
        )

        self.session = session
        self._current_frame = session.min_frame
        # Bookkeeping: each VideoPanelWidget is wrapped in a tab-styled
        # container so it can be hidden/restored via the view strip.
        # `_video_panels[cam]` is the actual panel; `_video_tabs[cam]` is
        # its wrapper (carries the title bar + close button).
        self._video_panels: dict[str, VideoPanelWidget] = {}
        self._video_tabs: dict[str, VideoTabContainer] = {}
        # Records (row, col) for each camera so restoring a closed cell
        # puts it back where it was in the grid.
        self._video_positions: dict[str, tuple[int, int]] = {}

        self._build_central()
        self._add_menus()
        self._size_to_screen()

        self.statusBar().showMessage(
            f"Loaded {len(session.frame_groups)} frames across "
            f"{len(session.camera_names())} cameras"
        )

    # ---- layout -------------------------------------------------------

    def _build_central(self) -> None:
        central = QWidget(self)
        central.setObjectName("central_root")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ----- Horizontal split: [view strip | video grid | assignment]
        splitter = QSplitter(Qt.Horizontal, central)
        splitter.setObjectName("main_hsplit")
        splitter.setChildrenCollapsible(False)

        # Left view strip (small icon + cam name).
        self.view_strip = ViewStripWidget(self.session, splitter)
        self.view_strip.setObjectName("view_strip")
        self.view_strip.viewActivated.connect(self._activate_view)

        # Center: a regular grid of tab-styled video containers. Each
        # container has a name + close button at the top; clicking close
        # hides the container and dims the view strip; clicking the strip
        # row re-shows it.
        # Video area: nested QSplitters so closing a panel makes its
        # neighbours expand (luc3d's dynamic tiling behavior). Outer
        # vertical splitter holds one horizontal splitter per row.
        # VideoGridContainer (not plain QWidget) accepts tab drops.
        self._video_grid_container = VideoGridContainer(splitter)
        self._video_grid_container.setObjectName("video_grid_container")
        self._video_grid_container.setStyleSheet("background-color: #181818;")
        self._video_grid_container.set_drop_handler(self._handle_tab_drop)
        gc_layout = QVBoxLayout(self._video_grid_container)
        gc_layout.setContentsMargins(0, 0, 0, 0)
        gc_layout.setSpacing(0)
        self._video_v_splitter = QSplitter(Qt.Vertical, self._video_grid_container)
        self._video_v_splitter.setChildrenCollapsible(False)
        self._video_v_splitter.setHandleWidth(3)
        gc_layout.addWidget(self._video_v_splitter)
        # Per-row horizontal splitter (populated in _build_video_grid).
        self._video_row_splitters: list[QSplitter] = []
        self._build_video_grid()

        # Right side: identity assignment panel (unchanged).
        self.assignment = IdentityAssignmentPanel(self.session, splitter)
        self.assignment.setObjectName("assignment_panel")
        self.assignment.setMinimumWidth(SIDEBAR_MIN_WIDTH)

        splitter.addWidget(self.view_strip)
        splitter.addWidget(self._video_grid_container)
        splitter.addWidget(self.assignment)
        # Stretch: view strip fixed, grid grows, assignment fixed-ish.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 1)

        timeline_container = QWidget()
        timeline_container.setObjectName("timeline_container")
        timeline_container.setMinimumHeight(TIMELINE_MIN_HEIGHT)
        tl_row = QHBoxLayout(timeline_container)
        tl_row.setContentsMargins(0, 0, 0, 0)
        tl_row.setSpacing(0)

        self.timeline_tabs = QTabBar(timeline_container)
        self.timeline_tabs.setObjectName("timeline_color_tabs")
        self.timeline_tabs.setShape(QTabBar.RoundedWest)
        self.timeline_tabs.addTab("Track")
        self.timeline_tabs.addTab("Identities")
        self.timeline_tabs.setCurrentIndex(0 if self.session.color_mode == "track" else 1)
        self.timeline_tabs.currentChanged.connect(self._on_timeline_tab_changed)
        self.session.color_mode_changed.connect(self._sync_timeline_tab)

        # Playback controls (Prev / Play-Pause / Next) sit between the color
        # tab bar and the timeline. They drive frame navigation through the
        # main window's set_current_frame so all views stay in sync.
        self.playback = PlaybackControls(
            fps=self._effective_fps(),
            parent=timeline_container,
        )
        self.playback.set_frame_range(self.session.min_frame, self.session.max_frame)
        self.playback.set_current_frame(self._current_frame)
        self.playback.frameSeekRequested.connect(self.set_current_frame)

        self.timeline = TimelineWidget(self.session, timeline_container)
        self.timeline.frameSeekRequested.connect(self.set_current_frame)

        tl_row.addWidget(self.timeline_tabs, stretch=0)
        tl_row.addWidget(self.playback, stretch=0)
        tl_row.addWidget(self.timeline, stretch=1)

        # Vertical splitter so the user can shrink/grow the timeline strip.
        v_splitter = QSplitter(Qt.Vertical, central)
        v_splitter.setObjectName("main_vsplit")
        v_splitter.setChildrenCollapsible(False)
        v_splitter.addWidget(splitter)
        v_splitter.addWidget(timeline_container)
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        # Slim handle so the drag affordance is visible but unobtrusive.
        v_splitter.setHandleWidth(6)

        outer.addWidget(v_splitter, stretch=1)

        self.setCentralWidget(central)
        self._splitter = splitter
        self._v_splitter = v_splitter

    def _effective_fps(self) -> float:
        """Pick the FPS to drive playback from. Falls back to 30 if no
        decoder has reported an FPS yet."""
        for panel in self._video_panels.values():
            fps = getattr(panel, "fps", None)
            if fps:
                return float(fps)
        return 30.0

    def _build_video_grid(self) -> None:
        """Populate the nested-splitter grid with one VideoTabContainer per
        camera, in row-major order over `_grid_dims(n)`.

        Layout shape:

            video_v_splitter (Qt.Vertical)
            ├── row_split[0] (Qt.Horizontal)
            │   ├── VideoTabContainer (cam 0)
            │   ├── VideoTabContainer (cam 1)
            │   └── ...
            ├── row_split[1]
            │   └── ...

        Splitter-based layout (vs the rigid QGridLayout) gives us the
        luc3d "neighbour panels grow when one is closed" behavior: when
        a tab is detached via `_on_tab_closed`, the row's splitter
        rebalances over its remaining children; if the entire row
        becomes empty, the row-splitter itself hides and the vertical
        splitter redistributes the freed height.
        """
        cam_names = self.session.camera_names()
        n = len(cam_names)
        if n == 0:
            return
        rows, cols = self._grid_dims(n)

        # Pre-create one horizontal splitter per row.
        for r in range(rows):
            row_split = QSplitter(Qt.Horizontal, self._video_v_splitter)
            row_split.setChildrenCollapsible(False)
            row_split.setHandleWidth(3)
            self._video_v_splitter.addWidget(row_split)
            self._video_row_splitters.append(row_split)

        # Make + place each tab.
        for i, cam_name in enumerate(cam_names):
            panel = VideoPanelWidget(self.session, cam_name, parent=self._video_grid_container)
            panel.frameSeekRequested.connect(self.set_current_frame)
            self._video_panels[cam_name] = panel

            tab = VideoTabContainer(cam_name, panel, parent=self._video_grid_container)
            tab.closed.connect(self._on_tab_closed)
            tab.set_zone_drop_handler(self._handle_zone_drop)
            self._video_tabs[cam_name] = tab

            r, c = divmod(i, cols)
            self._video_positions[cam_name] = (r, c)
            self._video_row_splitters[r].addWidget(tab)

    @staticmethod
    def _grid_dims(n: int) -> tuple[int, int]:
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        return rows, cols

    # ---- view-strip ↔ video-tab coordination --------------------------

    def _on_tab_closed(self, cam_name: str) -> None:
        """Tab's × button → detach from its row splitter, hide it, dim the
        view-strip row. Detaching (rather than just hiding in place) is
        what makes the row splitter rebalance over remaining children —
        that's the dynamic re-sizing the user asked for.
        """
        tab = self._video_tabs.get(cam_name)
        if tab is None:
            return
        tab.hide()
        tab.setParent(None)  # unparent → row splitter loses a child → rebalance

        # If the row is now empty, hide the row splitter so the vertical
        # splitter redistributes that height to the surviving rows.
        r, _c = self._video_positions.get(cam_name, (0, 0))
        if 0 <= r < len(self._video_row_splitters):
            row_split = self._video_row_splitters[r]
            if row_split.count() == 0:
                row_split.hide()

        if hasattr(self, "view_strip"):
            self.view_strip.set_view_visible(cam_name, False)

    # ---- drag/drop rearrangement -------------------------------------

    def _handle_zone_drop(
        self, src_cam: str, target_cam: str, zone: str
    ) -> None:
        """Drop callback installed on every `VideoTabContainer`.

        Routes a dropped tab according to which edge of the target tab
        the cursor was closest to:

          * **west** / **east**  → insert into the target's row, left
            or right of the target column. Same-row reorder reuses
            `_move_tab_to`'s index-shift correction.
          * **north** / **south** → insert a *new horizontal row* into
            the vertical splitter above or below the target's row,
            then move the source tab into it (as its sole child).

        Self-drops and "north/south of own only-child row" no-op so
        users don't accidentally collapse-and-recreate the same layout.
        """
        if src_cam == target_cam:
            return
        tgt_row, tgt_col = self._locate_tab(target_cam)
        if tgt_row < 0:
            return

        if zone in ("west", "east"):
            new_idx = tgt_col if zone == "west" else tgt_col + 1
            self._move_tab_to(src_cam, tgt_row, new_idx)
            return

        # north / south → create a fresh row in the vertical splitter
        src_row, _ = self._locate_tab(src_cam)
        if src_row == tgt_row:
            src_split = self._video_row_splitters[src_row]
            if src_split.count() == 1:
                # Source is the only tab in this row; "stack above/below
                # myself" would just shift the same single panel around.
                return
        new_row_idx = tgt_row if zone == "north" else tgt_row + 1
        self._insert_new_row(new_row_idx)
        # _move_tab_to walks the splitter list to find the (possibly
        # shifted) source row, so we only need the new target index.
        self._move_tab_to(src_cam, new_row_idx, 0)

    def _locate_tab(self, cam: str) -> tuple[int, int]:
        """Walk the row splitters to find `cam`'s current (row, col).

        Returns (-1, -1) if the tab is detached (closed). Authoritative
        over `_video_positions`, which is only synced after explicit
        moves and can lag for stale entries.
        """
        for r, rs in enumerate(self._video_row_splitters):
            for c in range(rs.count()):
                w = rs.widget(c)
                if isinstance(w, VideoTabContainer) and w.cam_name == cam:
                    return r, c
        return -1, -1

    def _insert_new_row(self, at_index: int) -> QSplitter:
        """Insert a fresh horizontal row-splitter at `at_index` in the
        vertical splitter. All rows at or below that index slide down
        by one; `_video_positions` rows are shifted to match so that
        closed-tab restores still land in the correct row.
        """
        # Shift stored positions first — do this before inserting so the
        # new (empty) row doesn't get claimed by any restore.
        for cam, (r, c) in list(self._video_positions.items()):
            if r >= at_index:
                self._video_positions[cam] = (r + 1, c)

        new_row = QSplitter(Qt.Horizontal, self._video_v_splitter)
        new_row.setChildrenCollapsible(False)
        new_row.setHandleWidth(3)
        self._video_v_splitter.insertWidget(at_index, new_row)
        self._video_row_splitters.insert(at_index, new_row)
        return new_row

    def _handle_tab_drop(self, cam_name: str, pos_in_container: QPoint) -> None:
        """Drop callback installed on `VideoGridContainer`.

        Resolves `pos_in_container` (in grid-container coordinates) to a
        target (row, insertion_index) within the nested splitter grid and
        invokes `_move_tab_to`. Drops that miss every visible row are
        no-ops; rebuilding hidden rows by drop is a future enhancement.
        """
        global_pos = self._video_grid_container.mapToGlobal(pos_in_container)
        for row_idx, row_split in enumerate(self._video_row_splitters):
            if row_split.isHidden() or row_split.count() == 0:
                continue
            local = row_split.mapFromGlobal(global_pos)
            if not row_split.rect().contains(local):
                continue
            insert_idx = self._compute_drop_idx_x(row_split, local.x())
            self._move_tab_to(cam_name, row_idx, insert_idx)
            return

    def _compute_drop_idx_x(self, row_split: QSplitter, x: int) -> int:
        """Return the index `insertWidget` should use to land an inbound
        tab at horizontal position `x` (row-local). Drops before the
        first child whose horizontal midpoint is greater than `x`; falls
        off the end if `x` is past every child."""
        for i in range(row_split.count()):
            w = row_split.widget(i)
            if not w.isVisible():
                continue
            mid = w.x() + w.width() / 2
            if x < mid:
                return i
        return row_split.count()

    def _move_tab_to(self, cam_name: str, target_row: int, target_idx: int) -> None:
        """Move `cam_name`'s tab to (`target_row`, `target_idx`) in the
        grid. Handles same-row reordering (with index-shift correction
        for the removed source) and cross-row moves; updates the
        `_video_positions` map for affected rows so future close/restore
        cycles preserve the *current* visible order (not the initial one).
        """
        tab = self._video_tabs.get(cam_name)
        if tab is None:
            return
        src_split = tab.parent()
        src_row = -1
        if isinstance(src_split, QSplitter):
            for i, rs in enumerate(self._video_row_splitters):
                if rs is src_split:
                    src_row = i
                    break

        target_split = self._video_row_splitters[target_row]

        if isinstance(src_split, QSplitter):
            src_idx = src_split.indexOf(tab)
            if src_split is target_split:
                # Same-row reorder: removing the source shifts subsequent
                # indices down by one, so we adjust the target index.
                if src_idx < target_idx:
                    target_idx -= 1
                if src_idx == target_idx:
                    return  # no-op move
            tab.setParent(None)
            if src_split is not target_split and src_split.count() == 0:
                # Source row went empty — collapse it so the vertical
                # splitter redistributes the freed height.
                src_split.hide()

        if target_split.isHidden():
            target_split.show()
        # Clamp index so we never index past current count.
        target_idx = max(0, min(target_idx, target_split.count()))
        target_split.insertWidget(target_idx, tab)
        tab.show()

        # Resync stored positions for affected rows so close/restore
        # preserves the user's new layout.
        self._sync_row_positions(target_row)
        if src_row >= 0 and src_row != target_row:
            self._sync_row_positions(src_row)

        if hasattr(self, "view_strip"):
            self.view_strip.set_view_visible(cam_name, True)

    def _sync_row_positions(self, row_idx: int) -> None:
        """Refresh `_video_positions` for every visible tab in `row_idx`
        to reflect the splitter's current left-to-right order."""
        split = self._video_row_splitters[row_idx]
        for i in range(split.count()):
            w = split.widget(i)
            if isinstance(w, VideoTabContainer):
                self._video_positions[w.cam_name] = (row_idx, i)

    def _activate_view(self, cam_name: str) -> None:
        """Show / raise the camera's tab container — called from a click on
        a view-strip row. If the tab was detached, re-insert it at its
        original column position within its row splitter (and re-show the
        row splitter if it had been hidden).
        """
        tab = self._video_tabs.get(cam_name)
        if tab is None:
            return

        # Re-attach if currently detached.
        if tab.parent() is None:
            r, c = self._video_positions.get(cam_name, (0, 0))
            if 0 <= r < len(self._video_row_splitters):
                row_split = self._video_row_splitters[r]
                if row_split.isHidden():
                    row_split.show()
                # Insert at the index whose left neighbours all have a
                # smaller original column than ours, so the tab returns to
                # its original layout slot regardless of what's currently
                # visible.
                insert_idx = row_split.count()
                for i in range(row_split.count()):
                    sibling = row_split.widget(i)
                    if isinstance(sibling, VideoTabContainer):
                        sib_c = self._video_positions.get(sibling.cam_name, (r, 0))[1]
                        if sib_c > c:
                            insert_idx = i
                            break
                row_split.insertWidget(insert_idx, tab)

        if tab.isHidden():
            tab.show()
        tab.raise_()

        if hasattr(self, "view_strip"):
            self.view_strip.set_view_visible(cam_name, True)

        panel = self._video_panels.get(cam_name)
        if panel is not None:
            panel.setFocus(Qt.MouseFocusReason)

    def _size_to_screen(self) -> None:
        # 3-pane splitter sizes: [view strip | dock host | assignment].
        # View strip is fixed-ish (~120 px), assignment ~320, dock host gets
        # the rest.
        VS = 120
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 1000)
            self._splitter.setSizes(
                [VS, 1600 - VS - SIDEBAR_DEFAULT_WIDTH, SIDEBAR_DEFAULT_WIDTH]
            )
            self._v_splitter.setSizes([1000 - TIMELINE_DEFAULT_HEIGHT, TIMELINE_DEFAULT_HEIGHT])
            return
        geom = screen.availableGeometry()
        self.resize(geom.size())
        self.move(geom.topLeft())
        center_w = max(200, geom.width() - VS - SIDEBAR_DEFAULT_WIDTH)
        self._splitter.setSizes([VS, center_w, SIDEBAR_DEFAULT_WIDTH])
        top_h = max(TIMELINE_MIN_HEIGHT + 100,
                    geom.height() - TIMELINE_DEFAULT_HEIGHT)
        self._v_splitter.setSizes([top_h, TIMELINE_DEFAULT_HEIGHT])

    def _on_timeline_tab_changed(self, idx: int) -> None:
        self.session.set_color_mode("track" if idx == 0 else "identity")

    def _sync_timeline_tab(self, mode: str) -> None:
        target = 0 if mode == "track" else 1
        if self.timeline_tabs.currentIndex() != target:
            self.timeline_tabs.blockSignals(True)
            try:
                self.timeline_tabs.setCurrentIndex(target)
            finally:
                self.timeline_tabs.blockSignals(False)

    def _add_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_act = QAction("&Open Session Folder…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._open_session_dialog)
        file_menu.addAction(open_act)

        load_ll_act = QAction("Load &Identity Assignments…", self)
        load_ll_act.triggered.connect(self._load_lucid_labels_dialog)
        file_menu.addAction(load_ll_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    # ---- frame navigation --------------------------------------------

    def set_current_frame(self, frame_idx: int) -> None:
        lo, hi = self.session.min_frame, self.session.max_frame
        frame_idx = max(lo, min(hi, int(frame_idx)))
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        for panel in self._video_panels.values():
            panel.set_current_frame(frame_idx)
        self.timeline.set_current_frame(frame_idx)
        self.assignment.set_current_frame(frame_idx)
        if hasattr(self, "playback"):
            self.playback.set_current_frame(frame_idx)
        self.statusBar().showMessage(f"Frame {frame_idx}")
        self.currentFrameChanged.emit(frame_idx)

    def keyPressEvent(self, event) -> None:
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        if event.key() == Qt.Key_Right:
            self.set_current_frame(self._current_frame + step)
        elif event.key() == Qt.Key_Left:
            self.set_current_frame(self._current_frame - step)
        elif event.key() == Qt.Key_Home:
            self.set_current_frame(self.session.min_frame)
        elif event.key() == Qt.Key_End:
            self.set_current_frame(self.session.max_frame)
        elif event.key() == Qt.Key_Space:
            self.playback.toggle_play()
        else:
            super().keyPressEvent(event)

    # ---- menu handlers -----------------------------------------------

    def _open_session_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Session Folder")
        if not path:
            return
        try:
            new_session = Session.load_from_folder(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"{type(exc).__name__}: {exc}")
            return
        new_win = LucidLiteWindow(new_session)
        new_win.show()
        self.close()

    def _load_lucid_labels_dialog(self) -> None:
        from lucid_labels import load_lucid_labels
        try:
            load_lucid_labels(self.session, None)  # type: ignore[arg-type]
        except NotImplementedError as exc:
            self.statusBar().showMessage(f"LucidLabels: {exc}", 8000)
