"""LUCID-Lite main window — assembles the camera grid, timeline, and assignment sidebar."""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog, QGridLayout, QHBoxLayout, QMainWindow, QMessageBox,
    QSplitter, QTabBar, QVBoxLayout, QWidget,
)

from assignment_panel import IdentityAssignmentPanel
from playback_controls import PlaybackControls
from pose_data import Session
from timeline_widget import TimelineWidget
from video_panel import VideoPanelWidget

SIDEBAR_DEFAULT_WIDTH = 320
SIDEBAR_MIN_WIDTH = 260
TIMELINE_DEFAULT_HEIGHT = 200
TIMELINE_MIN_HEIGHT = 60


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
        self._video_panels: dict[str, VideoPanelWidget] = {}

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

        splitter = QSplitter(Qt.Horizontal, central)
        splitter.setObjectName("main_hsplit")
        splitter.setChildrenCollapsible(False)

        grid_container = QWidget(splitter)
        grid_container.setObjectName("video_grid_container")
        grid_layout = QGridLayout(grid_container)
        grid_layout.setObjectName("video_grid")
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(2)
        self._build_video_grid(grid_layout)

        self.assignment = IdentityAssignmentPanel(self.session, splitter)
        self.assignment.setObjectName("assignment_panel")
        self.assignment.setMinimumWidth(SIDEBAR_MIN_WIDTH)

        splitter.addWidget(grid_container)
        splitter.addWidget(self.assignment)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

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

    def _build_video_grid(self, grid: QGridLayout) -> None:
        cam_names = self.session.camera_names()
        n = len(cam_names)
        if n == 0:
            return
        rows, cols = self._grid_dims(n)
        for i, cam_name in enumerate(cam_names):
            panel = VideoPanelWidget(self.session, cam_name, parent=self)
            panel.frameSeekRequested.connect(self.set_current_frame)
            self._video_panels[cam_name] = panel
            r, c = divmod(i, cols)
            grid.addWidget(panel, r, c)
        for r in range(rows):
            grid.setRowStretch(r, 1)
        for c in range(cols):
            grid.setColumnStretch(c, 1)

    @staticmethod
    def _grid_dims(n: int) -> tuple[int, int]:
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        return rows, cols

    def _size_to_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 1000)
            self._splitter.setSizes([1600 - SIDEBAR_DEFAULT_WIDTH, SIDEBAR_DEFAULT_WIDTH])
            self._v_splitter.setSizes([1000 - TIMELINE_DEFAULT_HEIGHT, TIMELINE_DEFAULT_HEIGHT])
            return
        geom = screen.availableGeometry()
        self.resize(geom.size())
        self.move(geom.topLeft())
        target_left = max(0, geom.width() - SIDEBAR_DEFAULT_WIDTH)
        self._splitter.setSizes([target_left, SIDEBAR_DEFAULT_WIDTH])
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
