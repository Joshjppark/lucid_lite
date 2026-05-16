"""SLEAP-like timeline: per (camera, track) rows with colored segments.

Custom QWidget + paintEvent. One row per (camera, track) ordered camera-first,
matching timeline.js:719–754. Click to seek; horizontal scroll via wheel.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QFontMetricsF, QPainter, QPen, QPolygonF,
)
from PySide6.QtWidgets import QWidget

from colors import get_identity_color, get_track_color
from pose_data import Session

ROW_HEIGHT = 12
ROW_GAP = 1
VIEW_GROUP_GAP = 8
LEFT_MARGIN = 110
TOP_MARGIN = 8
BOTTOM_MARGIN = 24


class TimelineWidget(QWidget):
    frameSeekRequested = Signal(int)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_frame: int = session.min_frame
        self._segments: dict[tuple[str, int], list[tuple[int, int]]] = {}
        self._row_layout: list[tuple[str, int, float]] = []  # (camera, track_idx, y)
        self._dragging: bool = False

        self.setMinimumHeight(140)
        self.setStyleSheet("background-color: #1e1e1e; color: #ccc;")

        session.frame_groups_changed.connect(self._rebuild)
        session.identity_map_changed.connect(self.update)
        session.identities_changed.connect(self.update)
        session.tracks_changed.connect(self._rebuild)
        session.color_mode_changed.connect(self.update)

        self._rebuild()

    # ---- public API ---------------------------------------------------

    def set_current_frame(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self.update()

    # ---- segment precompute ------------------------------------------

    def _rebuild(self) -> None:
        session = self.session
        # Build track occupancy from frame_groups.
        per_key: dict[tuple[str, int], list[int]] = {}
        for frame_idx in session.frame_indices:
            fg = session.frame_groups[frame_idx]
            for cam_name, insts in fg.instances.items():
                for inst in insts:
                    if inst.track_idx is None:
                        continue
                    per_key.setdefault((cam_name, inst.track_idx), []).append(frame_idx)

        # Condense into contiguous runs.
        self._segments = {
            key: _runs(sorted(set(frames))) for key, frames in per_key.items()
        }

        # Row layout: sorted by camera name then track_idx.
        self._row_layout = []
        y = float(TOP_MARGIN)
        prev_cam: str | None = None
        for (cam, track_idx) in sorted(self._segments.keys()):
            if prev_cam is not None and cam != prev_cam:
                y += VIEW_GROUP_GAP
            self._row_layout.append((cam, track_idx, y))
            y += ROW_HEIGHT + ROW_GAP
            prev_cam = cam

        self.setMinimumHeight(int(y + BOTTOM_MARGIN + 16))
        self.update()

    # ---- geometry helpers --------------------------------------------

    def _frame_range(self) -> tuple[int, int]:
        if not self.session.frame_groups:
            return 0, 1
        lo = self.session.min_frame
        hi = max(self.session.max_frame, lo + 1)
        return lo, hi

    def _content_x_range(self) -> tuple[float, float]:
        return float(LEFT_MARGIN), float(self.width() - 8)

    def _frame_to_x(self, frame: float) -> float:
        lo, hi = self._frame_range()
        x0, x1 = self._content_x_range()
        if hi == lo:
            return x0
        return x0 + (frame - lo) * (x1 - x0) / (hi - lo)

    def _x_to_frame(self, x: float) -> int:
        lo, hi = self._frame_range()
        x0, x1 = self._content_x_range()
        if x1 == x0:
            return lo
        f = lo + (x - x0) * (hi - lo) / (x1 - x0)
        return max(lo, min(hi, int(round(f))))

    # ---- paint --------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        # Row labels + bars
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        fm = QFontMetricsF(font)

        x0, x1 = self._content_x_range()

        for cam_name, track_idx, y in self._row_layout:
            # Row background
            painter.fillRect(QRectF(x0, y, x1 - x0, ROW_HEIGHT), QColor("#252525"))

            # Label
            label = f"{cam_name}/t{track_idx}"
            painter.setPen(QColor("#aaa"))
            painter.drawText(
                QRectF(4, y, LEFT_MARGIN - 8, ROW_HEIGHT),
                Qt.AlignVCenter | Qt.AlignRight,
                label,
            )

            # Determine color from session.color_mode. Identity mode uses the
            # global track_identity_map only (per-frame overrides don't change
            # the row hue). Gray fallback for missing assignments.
            if self.session.color_mode == "identity":
                ident_id = self.session.track_identity_map.get(f"{cam_name}:{track_idx}")
                ident = self.session.get_identity(ident_id) if ident_id is not None else None
                color_hex = ident.color if ident else "#888888"
            else:
                color_hex = get_track_color(track_idx)
            painter.setBrush(QColor(color_hex))
            painter.setPen(Qt.NoPen)

            for (seg_start, seg_end) in self._segments.get((cam_name, track_idx), []):
                sx = self._frame_to_x(seg_start)
                ex = self._frame_to_x(seg_end + 1)
                painter.drawRect(QRectF(sx, y + 1, max(1.0, ex - sx), ROW_HEIGHT - 2))

        # Playhead
        self._draw_playhead(painter, x0, x1)

        # Frame axis labels
        self._draw_frame_axis(painter, fm)

    def _draw_playhead(self, painter: QPainter, x0: float, x1: float) -> None:
        lo, hi = self._frame_range()
        if not (lo <= self._current_frame <= hi):
            return
        x = self._frame_to_x(self._current_frame + 0.5)
        pen = QPen(QColor("#ffffff"))
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, TOP_MARGIN - 4),
                         QPointF(x, self.height() - BOTTOM_MARGIN))
        # Triangle pointer
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(Qt.NoPen)
        tri = QPolygonF([
            QPointF(x - 4, TOP_MARGIN - 4),
            QPointF(x + 4, TOP_MARGIN - 4),
            QPointF(x, TOP_MARGIN + 2),
        ])
        painter.drawPolygon(tri)

    def _draw_frame_axis(self, painter: QPainter, fm) -> None:
        lo, hi = self._frame_range()
        painter.setPen(QColor("#888"))
        y = self.height() - BOTTOM_MARGIN + 2
        x0, x1 = self._content_x_range()
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        # Tick every ~200 pixels
        n_ticks = max(2, int((x1 - x0) / 160))
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            f = int(round(lo + frac * (hi - lo)))
            x = self._frame_to_x(f)
            painter.drawLine(QPointF(x, y), QPointF(x, y + 4))
            painter.drawText(QPointF(x + 2, y + 14), str(f))

    # ---- mouse --------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._seek_to_mouse(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._seek_to_mouse(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def _seek_to_mouse(self, mouse_x: float) -> None:
        if mouse_x < LEFT_MARGIN:
            return
        frame = self._x_to_frame(mouse_x)
        self._current_frame = frame
        self.frameSeekRequested.emit(frame)
        self.update()


def _runs(sorted_frames: list[int]) -> list[tuple[int, int]]:
    if not sorted_frames:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append((start, prev))
            start = prev = f
    runs.append((start, prev))
    return runs
