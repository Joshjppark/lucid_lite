"""Playback controls strip: Prev / Play-Pause / Next.

Drives `LucidLiteWindow.set_current_frame` from a QTimer that ticks at the FPS
of the loaded videos. The widget itself stays small and stateless beyond a
play/pause flag — the main window owns the frame counter.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget


_BTN_QSS = """
QPushButton {
    background-color: #2b2b2b;
    color: #ddd;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 12px;
    min-width: 28px;
}
QPushButton:hover { background-color: #3a3a3a; }
QPushButton:pressed { background-color: #4a4a4a; }
QPushButton:disabled { color: #666; border-color: #2a2a2a; }
"""


class PlaybackControls(QWidget):
    """Three-button transport bar that emits seek requests.

    Signals
    -------
    frameSeekRequested(int)
        Emitted on each tick of the play timer (with the next frame number)
        and on prev/next button clicks.
    """

    frameSeekRequested = Signal(int)

    def __init__(self, fps: float = 30.0, parent=None):
        super().__init__(parent)

        self._fps: float = max(1.0, float(fps) if fps else 30.0)
        self._current_frame: int = 0
        self._min_frame: int = 0
        self._max_frame: int = 0
        self._is_playing: bool = False

        self.setStyleSheet(_BTN_QSS)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(2)

        self.prev_btn = QPushButton("⏮")
        self.prev_btn.setToolTip("Previous frame (Left)")
        self.prev_btn.clicked.connect(self._on_prev)

        self.play_btn = QPushButton("▶")
        self.play_btn.setToolTip("Play / Pause (Space)")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._on_play_toggled)

        self.next_btn = QPushButton("⏭")
        self.next_btn.setToolTip("Next frame (Right)")
        self.next_btn.clicked.connect(self._on_next)

        h.addWidget(self.prev_btn)
        h.addWidget(self.play_btn)
        h.addWidget(self.next_btn)

        # The timer ticks once per video-frame. interval is recomputed via
        # set_fps() whenever the upstream FPS becomes known.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._on_tick)
        self._timer.setInterval(int(round(1000.0 / self._fps)))

    # ---- external sync ------------------------------------------------

    def set_fps(self, fps: float) -> None:
        """Update tick rate (called when a video decoder finishes opening)."""
        if not fps or fps <= 0:
            return
        self._fps = float(fps)
        new_interval = max(1, int(round(1000.0 / self._fps)))
        self._timer.setInterval(new_interval)

    def set_frame_range(self, lo: int, hi: int) -> None:
        self._min_frame = int(lo)
        self._max_frame = int(hi)

    def set_current_frame(self, frame_idx: int) -> None:
        """Called by the main window after a seek lands — keeps internal
        counter in step so play resumes from the right place."""
        self._current_frame = int(frame_idx)
        # If we've reached the end while playing, stop.
        if self._is_playing and self._current_frame >= self._max_frame:
            self.pause()

    # ---- public actions -----------------------------------------------

    def play(self) -> None:
        if self._is_playing:
            return
        if self._current_frame >= self._max_frame:
            self._current_frame = self._min_frame
            self.frameSeekRequested.emit(self._current_frame)
        self._is_playing = True
        self.play_btn.setChecked(True)
        self.play_btn.setText("⏸")
        self._timer.start()

    def pause(self) -> None:
        if not self._is_playing:
            self.play_btn.setChecked(False)
            self.play_btn.setText("▶")
            return
        self._is_playing = False
        self._timer.stop()
        self.play_btn.setChecked(False)
        self.play_btn.setText("▶")

    def toggle_play(self) -> None:
        if self._is_playing:
            self.pause()
        else:
            self.play()

    # ---- handlers -----------------------------------------------------

    def _on_play_toggled(self, checked: bool) -> None:
        # Always derive from current state — checked may not reflect intent
        # if the previous click was a programmatic setChecked.
        if checked:
            self.play()
        else:
            self.pause()

    def _on_prev(self) -> None:
        target = max(self._min_frame, self._current_frame - 1)
        if target == self._current_frame:
            return
        self.frameSeekRequested.emit(target)

    def _on_next(self) -> None:
        target = min(self._max_frame, self._current_frame + 1)
        if target == self._current_frame:
            return
        self.frameSeekRequested.emit(target)

    def _on_tick(self) -> None:
        next_frame = self._current_frame + 1
        if next_frame > self._max_frame:
            self.pause()
            return
        self.frameSeekRequested.emit(next_frame)
