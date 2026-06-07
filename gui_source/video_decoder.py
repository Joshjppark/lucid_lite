"""PyAV-backed on-demand video decoder with per-camera loader thread.

Architecture (mirrors sleap.gui.widgets.video_worker.FrameLoaderThread):

  panel ──request──▶ FrameLoaderThread (one per camera)
                    │   queue.Queue[int]
                    │   ─ coalesces to most-recent on each wake-up
                    │   OnDemandVideoDecoder (owned by this thread)
                    │   ─ PyAV container; sequential-iter optimization
                    └──frame_ready──▶ panel._on_frame_ready (GUI thread)

Why dedicated QThread instead of QThreadPool:

  * PyAV containers aren't thread-safe to share across threads. SLEAP
    sidesteps this by deepcopying the Video object; we sidestep it by
    constructing the OnDemandVideoDecoder inside the worker's run().
  * Per-camera isolation: with N cameras and a pool sized at M < N
    threads, slow decodes on one camera back up the queue for others.
    One thread per camera = no cross-camera head-of-line blocking.
  * Coalescing playback bursts: when playback at FPS pushes requests
    faster than decode can keep up, only the freshest request runs.
    With a thread pool that's hard — every queued task runs at least
    partially. With our run loop, we drain the queue first and only
    process the latest pending frame_idx.
"""
from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def probe_video_metadata(path: Path) -> tuple[int, int, float, int]:
    """Open `path` just to read (width, height, fps, n_frames), then close.

    Used by panels at construction time so they can size themselves
    before the per-camera loader thread is started. Avoids needing
    cross-thread access to the loader's decoder.
    """
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 30.0
        n_frames = int(stream.frames) or 0
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
    finally:
        container.close()
    return width, height, fps, n_frames


class OnDemandVideoDecoder:
    """Single-thread-owned frame-accurate seek-and-decode.

    Lives entirely inside one FrameLoaderThread — no longer takes a
    lock since only the owning thread calls it. Kept as its own class
    (rather than folded into the QThread) so the seek/decode logic can
    be tested headlessly without spinning up Qt.

    Sequential-read optimization: keeps the PyAV decode iterator alive
    across `get_frame` calls. If the next request is forward and within
    `_seek_threshold` frames of the last decoded position, we pull from
    the existing iterator instead of issuing another `container.seek`
    (the seek + keyframe rewind + decode-forward is the dominant cost
    for HD video and dominates playback latency if done per frame).
    Random seeks (backwards or large forward jumps) still take the
    seek path.
    """

    def __init__(self, path: Path, cache_size: int = 64):
        self.path = Path(path)
        self._container: av.container.InputContainer | None = None
        self._stream = None
        self._cache: "OrderedDict[int, QImage]" = OrderedDict()
        self._cache_size = cache_size
        self._fps: float = 30.0
        self._n_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        # Sequential-decode state. Reset on seek / decode error / close.
        self._iter = None
        self._last_idx: int = -1
        # Tunable: if `frame_idx - last_idx` exceeds this, prefer seek
        # over walking frames forward. PyAV keyframe distances are
        # typically 30–60 for typical encodes, so 32 is a safe threshold
        # — beyond that, seeking to the nearest keyframe and decoding
        # forward from there is faster than continuing the current iter.
        self._seek_threshold: int = 32
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
        """Return a QImage for frame_idx (RGB888). Single-thread access."""
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]

        try:
            img = self._decode(frame_idx)
        except Exception:
            # Decode iterator can be left in a bad state by a partial
            # read or codec error — drop it so the next call re-seeks.
            self._iter = None
            self._last_idx = -1
            return None

        if img is not None:
            self._cache[frame_idx] = img
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return img

    def _decode(self, frame_idx: int) -> QImage | None:
        assert self._container is not None and self._stream is not None
        tb = self._stream.time_base

        # Decide whether to reuse the existing decode iterator (cheap —
        # just pulls the next frame from the codec) or pay for a seek
        # (expensive — flushes decoder state, rewinds to a keyframe,
        # decodes forward). Reuse only when stepping forward by a small
        # amount; backwards or large jumps require seek.
        need_seek = (
            self._iter is None
            or frame_idx <= self._last_idx
            or frame_idx - self._last_idx > self._seek_threshold
        )
        if need_seek:
            target_ts = int(frame_idx / self._fps / tb)
            self._container.seek(
                target_ts, stream=self._stream, any_frame=False, backward=True,
            )
            self._iter = self._container.decode(self._stream)

        for frame in self._iter:
            if frame.pts is None:
                continue
            cur_idx = int(round(float(frame.pts * tb) * self._fps))
            self._last_idx = cur_idx
            if cur_idx >= frame_idx:
                arr = frame.to_ndarray(format="rgb24")
                return _numpy_to_qimage(arr)
        # Stream exhausted — invalidate iterator so the next call seeks.
        self._iter = None
        return None

    def close(self) -> None:
        self._iter = None
        self._last_idx = -1
        if self._container is not None:
            self._container.close()
            self._container = None


def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    buf = arr.tobytes()
    qimg = QImage(buf, w, h, 3 * w, QImage.Format_RGB888)
    # Detach from the underlying numpy buffer so it survives.
    return qimg.copy()


class FrameLoaderThread(QThread):
    """One dedicated decode thread per camera.

    Run loop blocks on a `queue.Queue[int]` of frame indices. When a
    request arrives, drains the queue to keep only the most recent
    pending index — older requests are silently dropped without ever
    paying for the decode. This is the playback-latency trick: during
    a burst (e.g. user scrubs the timeline) only the freshest frame
    is decoded.

    The OnDemandVideoDecoder is constructed inside `run()` so the PyAV
    container is opened on the worker thread and never touched from
    the GUI thread.
    """

    frame_ready = Signal(int, QImage)

    def __init__(self, video_path: Path, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self._queue: queue.Queue[int] = queue.Queue()
        self._stop = threading.Event()
        # Decoder lives on the worker thread; constructed in run().
        self._decoder: OnDemandVideoDecoder | None = None

    # ---- GUI-thread API --------------------------------------------------

    def request(self, frame_idx: int) -> None:
        """Queue a frame request. Safe to call from any thread."""
        if self._stop.is_set():
            return
        self._queue.put(int(frame_idx))

    def stop(self) -> None:
        """Tell the thread to exit and wait for it. Safe to call twice."""
        self._stop.set()
        # Nudge the run loop out of its queue.get() timeout.
        try:
            self._queue.put_nowait(-1)
        except Exception:
            pass
        self.quit()
        if not self.wait(2000):
            self.terminate()
            self.wait()

    # ---- worker-thread body ---------------------------------------------

    def run(self) -> None:
        try:
            self._decoder = OnDemandVideoDecoder(self.video_path)
        except Exception as exc:
            print(f"[FrameLoaderThread] failed to open {self.video_path}: {exc}")
            return

        try:
            while not self._stop.is_set():
                try:
                    frame_idx = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                # Coalesce: keep the most recent enqueued index. Older
                # requests are silently dropped — playback always shows
                # the freshest available frame, never a stale backlog.
                while True:
                    try:
                        frame_idx = self._queue.get_nowait()
                    except queue.Empty:
                        break

                if frame_idx < 0 or self._stop.is_set():
                    continue

                img = self._decoder.get_frame(frame_idx)
                if img is None or self._stop.is_set():
                    continue
                self.frame_ready.emit(frame_idx, img)
        finally:
            if self._decoder is not None:
                self._decoder.close()
                self._decoder = None
