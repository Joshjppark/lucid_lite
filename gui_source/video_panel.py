"""Per-camera video panel: letterboxed frame + pose overlay.

Composited in paintEvent with one QImage (video) + QPainter overlay calls.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

import overlay_renderer
from pose_data import Session
from video_decoder import DecodeWorker, OnDemandVideoDecoder


class VideoPanelWidget(QWidget):
    frameSeekRequested = Signal(int)   # user keyboard-nav (forwarded to main window)

    def __init__(self, session: Session, camera_name: str, parent=None):
        super().__init__(parent)
        self.session = session
        self.camera_name = camera_name
        self._current_frame: int = session.min_frame
        self._pixmap: QPixmap | None = None
        self._video_w: int = 0
        self._video_h: int = 0

        # Open video decoder if we have a path.
        video_path = session.video_paths.get(camera_name)
        self._decoder: OnDemandVideoDecoder | None = None
        self._worker: DecodeWorker | None = None
        if video_path is not None:
            try:
                self._decoder = OnDemandVideoDecoder(video_path)
                self._video_w = self._decoder.width
                self._video_h = self._decoder.height
                self._worker = DecodeWorker(self._decoder, self)
                self._worker.frame_ready.connect(self._on_frame_ready)
            except Exception as exc:
                print(f"[{camera_name}] failed to open {video_path}: {exc}")

        # Fallback size if video didn't open — read calibration.
        if self._video_w == 0:
            for cam in session.cameras:
                if cam.name == camera_name and cam.size[0] > 0:
                    self._video_w, self._video_h = cam.size
                    break
            if self._video_w == 0:
                self._video_w, self._video_h = 1280, 720

        self.setMinimumSize(320, 180)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: #111;")

        # Listen to model changes.
        session.identity_map_changed.connect(self.update)
        session.identities_changed.connect(self.update)
        session.tracks_changed.connect(self.update)
        session.color_mode_changed.connect(self.update)
        session.appearance_changed.connect(self.update)

        # Initial frame request.
        self._request_frame(self._current_frame)

    # ---- public API ---------------------------------------------------

    @property
    def fps(self) -> float | None:
        """Video FPS from the decoder, or None if no video opened."""
        if self._decoder is None:
            return None
        return float(self._decoder.fps) if self._decoder.fps else None

    def set_current_frame(self, frame_idx: int) -> None:
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        self._request_frame(frame_idx)
        self.update()  # repaint overlay immediately even before image arrives

    # ---- decoder plumbing --------------------------------------------

    def _request_frame(self, frame_idx: int) -> None:
        if self._worker is None:
            return
        self._worker.request(frame_idx)

    def _on_frame_ready(self, frame_idx: int, image: QImage) -> None:
        if frame_idx != self._current_frame:
            return
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    # ---- rendering ----------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cell_w, cell_h = self.width(), self.height()
        rect = _letterbox(self._video_w, self._video_h, cell_w, cell_h)

        # Clear background
        painter.fillRect(self.rect(), QColor("#111"))

        # Video pixmap
        if self._pixmap is not None:
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        else:
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             f"{self.camera_name}\n(no video)")

        # Overlay — map (x,y) in video pixels to panel coords
        def v2p(x: float, y: float) -> QPointF:
            px = rect.x() + (x / self._video_w) * rect.width()
            py = rect.y() + (y / self._video_h) * rect.height()
            return QPointF(px, py)

        overlay_renderer.draw_overlay_for_camera(
            painter, self.session,
            self.session.frame_group(self._current_frame),
            self.camera_name, self._current_frame, v2p,
        )

        # Camera name badge
        painter.setPen(QColor("#ffffff"))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, 20, self.camera_name)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Right:
            self.frameSeekRequested.emit(self._current_frame + 1)
        elif key == Qt.Key_Left:
            self.frameSeekRequested.emit(self._current_frame - 1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._decoder is not None:
            self._decoder.close()
        super().closeEvent(event)


def _letterbox(video_w: int, video_h: int, cell_w: int, cell_h: int) -> QRectF:
    if video_w <= 0 or video_h <= 0 or cell_w <= 0 or cell_h <= 0:
        return QRectF(0, 0, cell_w, cell_h)
    video_ar = video_w / video_h
    cell_ar = cell_w / cell_h
    if video_ar > cell_ar:
        css_w = float(cell_w)
        css_h = cell_w / video_ar
    else:
        css_h = float(cell_h)
        css_w = cell_h * video_ar
    x = (cell_w - css_w) / 2
    y = (cell_h - css_h) / 2
    return QRectF(x, y, css_w, css_h)
