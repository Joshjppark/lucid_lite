"""PyAV-backed on-demand video decoder with frame cache.

Mirrors the role of video.js' OnDemandVideoDecoder but simpler: one decoder
per camera, seek-and-decode-forward for frame-accurate access.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage

try:
    # shiboken6 ships with PySide6 — used to detect deleted QObjects so
    # background decode tasks don't try to emit through a dead C++ object.
    from shiboken6 import isValid as _qt_is_valid  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    def _qt_is_valid(_obj) -> bool:  # type: ignore[misc]
        return True


class OnDemandVideoDecoder:
    """Thread-safe frame-accurate seek-and-decode. Per camera."""

    def __init__(self, path: Path, cache_size: int = 30):
        self.path = Path(path)
        self._container: av.container.InputContainer | None = None
        self._stream = None
        self._lock = threading.Lock()
        self._cache: "OrderedDict[int, QImage]" = OrderedDict()
        self._cache_size = cache_size
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        self._open()

    def _open(self) -> None:
        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        rate = self._stream.average_rate or self._stream.base_rate
        self._fps = float(rate) if rate else 30.0
        self._n_frames = int(self._stream.frames) or 0
        self._width = int(self._stream.codec_context.width)
        self._height = int(self._stream.codec_context.height)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def get_frame(self, frame_idx: int) -> QImage | None:
        """Return a QImage for frame_idx (RGB888). Thread-safe."""
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]

            try:
                img = self._decode(frame_idx)
            except Exception:
                return None

            if img is not None:
                self._cache[frame_idx] = img
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return img

    def _decode(self, frame_idx: int) -> QImage | None:
        assert self._container is not None and self._stream is not None
        tb = self._stream.time_base
        target_ts = int(frame_idx / self._fps / tb)
        self._container.seek(target_ts, stream=self._stream, any_frame=False, backward=True)

        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            cur_idx = int(round(float(frame.pts * tb) * self._fps))
            if cur_idx >= frame_idx:
                arr = frame.to_ndarray(format="rgb24")
                return _numpy_to_qimage(arr)
        return None

    def close(self) -> None:
        with self._lock:
            if self._container is not None:
                self._container.close()
                self._container = None


def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    buf = arr.tobytes()
    qimg = QImage(buf, w, h, 3 * w, QImage.Format_RGB888)
    # Detach from the underlying numpy buffer so it survives.
    return qimg.copy()


class _DecodeTask(QRunnable):
    def __init__(self, worker: "DecodeWorker", frame_idx: int, generation: int):
        super().__init__()
        self.worker = worker
        self.frame_idx = frame_idx
        self.generation = generation
        self.setAutoDelete(True)

    def run(self) -> None:
        # The worker is a QObject parented to a VideoPanelWidget. If that
        # widget was destroyed while we were decoding (common when a Jupyter
        # notebook lets the window get garbage-collected, or on app shutdown
        # while a task is still queued), emitting through it raises
        # "RuntimeError: Signal source has been deleted". Bail out cleanly.
        if not _qt_is_valid(self.worker):
            return
        img = self.worker.decoder.get_frame(self.frame_idx)
        # Drop if a newer request superseded us.
        if self.generation != self.worker.generation:
            return
        if img is None:
            return
        if not _qt_is_valid(self.worker):
            return
        try:
            self.worker.frame_ready.emit(self.frame_idx, img)
        except RuntimeError:
            # Worker was deleted between the isValid() check and the emit.
            pass


class DecodeWorker(QObject):
    """Qt-signal wrapper around OnDemandVideoDecoder that runs decodes in a thread pool."""

    frame_ready = Signal(int, QImage)

    def __init__(self, decoder: OnDemandVideoDecoder, parent: QObject | None = None):
        super().__init__(parent)
        self.decoder = decoder
        self.generation = 0
        self._pool = QThreadPool.globalInstance()

    def request(self, frame_idx: int) -> None:
        self.generation += 1
        task = _DecodeTask(self, frame_idx, self.generation)
        self._pool.start(task)
