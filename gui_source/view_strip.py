"""Left-side view strip — one row per camera with a color swatch + name.

Mirrors the role of luc3d's `populateViewStrip` (ui/sessions-panes.js).
Click a row to focus / un-hide that camera's dock widget; the main window
hooks `viewActivated` to bring the panel forward.

Rows show their corresponding camera's status:
  * normal foreground — dock is visible
  * dimmed foreground — dock is closed (re-clickable to bring back)
  * accented background — dock is the currently raised (active) one

The strip is intentionally lightweight: a `QListWidget` with custom-painted
swatch icons. It rebuilds on `session.frame_groups_changed` so loading a
new session updates the row list without main-window orchestration.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame, QLabel, QListWidget, QListWidgetItem, QSizePolicy, QVBoxLayout,
    QWidget,
)

from colors import get_camera_color
from pose_data import Session


# Strip width range. Tight enough to feel like a side rail, wide enough to
# show the longest camera names (~6 chars in this codebase).
_STRIP_MIN_W = 112
_STRIP_MAX_W = 168
_SWATCH_PX = 22  # icon square size; QListWidget setIconSize uses (W, H)


class ViewStripWidget(QWidget):
    """Sidebar listing every camera in `session.camera_names()`.

    Signals:
        viewActivated(str) — emitted with a camera name when the user
            clicks a row. Connect to the main window's "show this dock"
            handler.
    """
    viewActivated = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setMinimumWidth(_STRIP_MIN_W)
        self.setMaximumWidth(_STRIP_MAX_W)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # Header (matches the "Display" label styling in the graph window
        # for visual continuity across side-panels).
        title = QLabel("Views")
        title.setStyleSheet("font-weight: bold; font-size: 10pt;")
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # Row list — QListWidget gives us hover/selected styling and
        # keyboard navigation for free.
        self._list = QListWidget(self)
        self._list.setIconSize(QSize(_SWATCH_PX, _SWATCH_PX))
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setUniformItemSizes(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.currentItemChanged.connect(self._on_current_item_changed)
        root.addWidget(self._list, stretch=1)

        # Hint line at the bottom — explains the dim-row convention.
        hint = QLabel("Click to focus")
        hint.setStyleSheet("color: #888; font-style: italic; font-size: 8pt;")
        root.addWidget(hint)

        # Cache: cam_name -> QListWidgetItem so status updates are O(1).
        self._items: dict[str, QListWidgetItem] = {}
        # State: which cameras' docks are currently visible.
        self._visible_cams: set[str] = set()

        self._populate()
        # Repopulate if the session swaps cameras (eg. loading another folder).
        session.frame_groups_changed.connect(self._populate)

    # ---- external API --------------------------------------------

    def set_view_visible(self, cam_name: str, visible: bool) -> None:
        """Update the row's dim/normal foreground to reflect whether the
        camera's dock is currently shown. Closing the dock dims the row;
        re-showing un-dims it."""
        if visible:
            self._visible_cams.add(cam_name)
        else:
            self._visible_cams.discard(cam_name)
        item = self._items.get(cam_name)
        if item is None:
            return
        item.setForeground(
            QColor("#dddddd") if visible else QColor("#777777")
        )

    def set_active_view(self, cam_name: str | None) -> None:
        """Highlight the currently-raised dock's row in the strip.

        Pass None to clear the highlight (eg. when every dock is hidden).
        We block the signal during the update so we don't re-emit
        viewActivated and loop the focus call back into the main window.
        """
        self._list.blockSignals(True)
        try:
            if cam_name is None:
                self._list.setCurrentItem(None)
                return
            item = self._items.get(cam_name)
            if item is not None:
                self._list.setCurrentItem(item)
        finally:
            self._list.blockSignals(False)

    # ---- internals -----------------------------------------------

    def _populate(self) -> None:
        self._list.clear()
        self._items.clear()
        for cam_idx, cam_name in enumerate(self.session.camera_names()):
            item = QListWidgetItem(self._color_icon(cam_idx), cam_name)
            item.setData(Qt.UserRole, cam_name)
            item.setToolTip(cam_name)
            # Assume all visible at startup; main window can call
            # set_view_visible(cam, False) for any pre-closed cameras.
            item.setForeground(QColor("#dddddd"))
            self._list.addItem(item)
            self._items[cam_name] = item
            self._visible_cams.add(cam_name)

    @staticmethod
    def _color_icon(cam_idx: int) -> QIcon:
        """Build a small filled-square icon in the camera's palette color."""
        pix = QPixmap(_SWATCH_PX, _SWATCH_PX)
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QColor("#222222"))
            painter.setBrush(QColor(get_camera_color(cam_idx)))
            # Small inset so the border is visible against the swatch fill.
            painter.drawRoundedRect(1, 1, _SWATCH_PX - 2, _SWATCH_PX - 2, 3, 3)
        finally:
            painter.end()
        return QIcon(pix)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        cam = item.data(Qt.UserRole)
        if cam:
            self.viewActivated.emit(str(cam))

    def _on_current_item_changed(self, cur, _prev) -> None:
        # Also fire on keyboard arrow navigation — same effect as a click.
        if cur is None:
            return
        cam = cur.data(Qt.UserRole)
        if cam:
            self.viewActivated.emit(str(cam))
