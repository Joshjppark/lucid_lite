"""Per-camera video panel: letterboxed frame + pose overlay.

Composited in paintEvent with one QImage (video) + QPainter overlay calls.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QPushButton, QWidget

import overlay_renderer
from pose_data import Session
from video_decoder import DecodeWorker, OnDemandVideoDecoder

# Zoom limits (multiplicative). 0.5×–20× covers "fit to panel zoomed out" to
# "single-pixel inspection". Wheel step is 1.15× per notch — feels natural
# without overshooting.
_ZOOM_MIN = 0.5
_ZOOM_MAX = 20.0
_ZOOM_STEP = 1.05


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

        # ---- zoom / pan state ----------------------------------------
        # `_zoom` is multiplicative around the letterbox center; `_pan_*`
        # is an additional translation in panel coordinates. Applied
        # together via QPainter transforms in paintEvent, so the overlay
        # picks up the same transform as the video pixmap automatically.
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # Drag-to-pan state.
        self._panning: bool = False
        self._pan_grab_x: float = 0.0
        self._pan_grab_y: float = 0.0
        self._pan_start_x: float = 0.0
        self._pan_start_y: float = 0.0

        # Floating "Reset Zoom" button, child of self so it tracks resizes.
        # Repositioned in resizeEvent. Hidden unless zoom/pan is non-default.
        self._unzoom_btn = QPushButton("Reset Zoom", self)
        self._unzoom_btn.setStyleSheet(
            "QPushButton { background-color: rgba(0,0,0,180); color: #fff; "
            "padding: 3px 8px; border: 1px solid #aaa; border-radius: 3px; "
            "font-size: 9pt; }"
            "QPushButton:hover { background-color: rgba(40,40,40,220); }"
        )
        self._unzoom_btn.clicked.connect(self.reset_zoom)
        self._unzoom_btn.hide()

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
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        cell_w, cell_h = self.width(), self.height()
        rect = _letterbox(self._video_w, self._video_h, cell_w, cell_h)

        # Clear background — drawn in panel coords, NOT inside the zoom
        # transform, so the dark border around a zoomed-in frame stays put.
        painter.fillRect(self.rect(), QColor("#111"))

        # Apply zoom + pan as a painter transform. The transform is
        # additive: `pan` first (in panel coords), then scale around the
        # letterbox center. Both the video pixmap and the overlay draw
        # commands go through this transform, so the pose overlay zooms
        # together with the frame underneath.
        had_transform = (self._zoom != 1.0) or self._pan_x or self._pan_y
        if had_transform:
            cx, cy = rect.center().x(), rect.center().y()
            painter.translate(self._pan_x, self._pan_y)
            painter.translate(cx, cy)
            painter.scale(self._zoom, self._zoom)
            painter.translate(-cx, -cy)

        # Video pixmap.
        if self._pixmap is not None:
            painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        else:
            painter.setPen(QColor("#555"))
            painter.drawText(rect, Qt.AlignCenter,
                             f"{self.camera_name}\n(no video)")

        # Overlay — v2p maps to letterbox coords. The painter transform
        # above takes care of zoom/pan.
        def v2p(x: float, y: float) -> QPointF:
            px = rect.x() + (x / self._video_w) * rect.width()
            py = rect.y() + (y / self._video_h) * rect.height()
            return QPointF(px, py)

        overlay_renderer.draw_overlay_for_camera(
            painter, self.session,
            self.session.frame_group(self._current_frame),
            self.camera_name, self._current_frame, v2p,
        )

        # Reset to panel coords for chrome (camera name badge).
        if had_transform:
            painter.resetTransform()

        # Camera-name badge (kept small — the dock title bar shows the name
        # too, but the in-canvas badge survives floating/un-titled docks
        # and is useful for screenshots).
        painter.setPen(QColor("#ffffff"))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, 18, self.camera_name)

    # ---- zoom + pan input -------------------------------------------

    def wheelEvent(self, event) -> None:
        """Cursor-anchored zoom on mouse wheel. Holds the video coord under
        the cursor stable across the zoom."""
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)

        factor = _ZOOM_STEP if delta > 0 else (1.0 / _ZOOM_STEP)
        new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return event.accept()
        # Actual factor after clamping.
        s = new_zoom / self._zoom

        # Compute new pan so that the panel point under the cursor maps to
        # the same video point as before. Derivation in the changelog: keep
        # `(px - cx - pan)/scale` invariant.
        rect = _letterbox(self._video_w, self._video_h, self.width(), self.height())
        cx, cy = rect.center().x(), rect.center().y()
        pos = event.position()
        px, py = pos.x(), pos.y()
        self._pan_x = s * self._pan_x + (1.0 - s) * (px - cx)
        self._pan_y = s * self._pan_y + (1.0 - s) * (py - cy)
        self._zoom = new_zoom

        self._update_unzoom_btn()
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        # Right-click drag OR Shift+left-click drag = pan. Plain left-click
        # is reserved so future selection / drag-instance work can hook in.
        is_pan = (
            event.button() == Qt.RightButton
            or (event.button() == Qt.LeftButton
                and event.modifiers() & Qt.ShiftModifier)
        )
        if is_pan:
            self._panning = True
            pos = event.position()
            self._pan_grab_x = pos.x()
            self._pan_grab_y = pos.y()
            self._pan_start_x = self._pan_x
            self._pan_start_y = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            pos = event.position()
            self._pan_x = self._pan_start_x + (pos.x() - self._pan_grab_x)
            self._pan_y = self._pan_start_y + (pos.y() - self._pan_grab_y)
            self._update_unzoom_btn()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() in (Qt.RightButton, Qt.LeftButton):
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click anywhere on the panel resets zoom + pan — matches
        the convention used in luc3d's viewport."""
        if event.button() == Qt.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def reset_zoom(self) -> None:
        """Restore the default (no-zoom, no-pan) view."""
        if self._zoom == 1.0 and self._pan_x == 0.0 and self._pan_y == 0.0:
            return
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_unzoom_btn()
        self.update()

    def _update_unzoom_btn(self) -> None:
        """Show the Reset Zoom button only when there's something to reset.

        Uses `isHidden()` (inverted) rather than `isVisible()` because the
        latter returns False whenever any ancestor is hidden — which makes
        the toggle flaky during construction and in headless tests.
        `isHidden()` reflects the most recent setVisible() call directly.
        """
        active = bool((self._zoom != 1.0) or self._pan_x or self._pan_y)
        currently_shown = not self._unzoom_btn.isHidden()
        if active != currently_shown:
            self._unzoom_btn.setVisible(active)

    def resizeEvent(self, event) -> None:
        # Pin the Reset Zoom button to the top-right corner.
        btn = self._unzoom_btn
        margin = 6
        btn.adjustSize()
        btn.move(self.width() - btn.width() - margin, margin)
        super().resizeEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Right:
            self.frameSeekRequested.emit(self._current_frame + 1)
        elif key == Qt.Key_Left:
            self.frameSeekRequested.emit(self._current_frame - 1)
        elif key == Qt.Key_0 and (event.modifiers() & Qt.ControlModifier):
            # Ctrl+0 = reset zoom (mirrors browser convention).
            self.reset_zoom()
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
