# GUI ↔ Notebook Communication — Design (embedded-kernel)

**Status:** approved. Pattern B (background-thread kernel), `ipykernel` dep accepted, default connection UX.

Supersedes the earlier ZMQ PAIR proposal. The GUI runs an embedded Jupyter kernel on a background thread; notebooks attach to that kernel and share the GUI's Python interpreter. No wire format, no RPC vocabulary — the notebook simply calls the same `Session` / `LucidLiteWindow` methods the GUI does.

## 1. Architecture

- **`ipykernel`** embedded in the GUI process, launched on a background `threading.Thread` at startup when `--comms` is passed (opt-in).
- Qt event loop continues to own the **main thread** via `app.exec()`.
- Notebook cells execute on the **kernel thread**. Qt's automatic cross-thread signal delivery (`QueuedConnection` under the hood) means mutations like `session.set_frame_identity(...)` from a cell repaint the UI correctly without any extra plumbing.
- The kernel writes a standard connection file (`kernel-<pid>.json`) into `jupyter --runtime-dir`. GUI prints the absolute path to stderr on launch. Users attach via `jupyter console --existing`, `jupyter qtconsole --existing`, or a notebook client pointed at that file.
- One process, one interpreter. Kernel is a daemon thread — dies with the GUI.

**New dep:** `ipykernel>=6.0` (added to `pyproject.toml`).

## 2. Threading — Pattern B

- `ipykernel.kernelapp.IPKernelApp` is a singleton. We start it from a worker thread.
- It calls `signal.signal(SIGINT, ...)` during init, which fails off the main thread. Mitigation: monkey-patch `app.init_signal = lambda: None` before `initialize()`.
- The kernel uses tornado/asyncio. We create a fresh `asyncio` event loop on the worker thread via `asyncio.new_event_loop()` + `asyncio.set_event_loop()` before `initialize()`.
- `user_ns` seeded with `session`, `window`, `app`, and `comms` (the module itself), so a freshly attached notebook has immediate access.

## 3. Data structures (convenience adapters, not a wire protocol)

Because the notebook shares the interpreter, `Session` and `FrameGroup` are directly accessible — no serialization is required. The dataclasses below exist as *optional snapshot helpers* for users who want a clean, stable, identity-resolved view.

File: `gui_source/comms.py`.

```python
@dataclass
class InstanceRecord:
    points: list[tuple[float, float] | None]  # per-node (x,y) or None
    track_idx: int | None                     # from Instance.track_idx
    identity_id: int | None                   # resolved via Session.get_identity_id_for_track(...)
    score: float                              # from Instance.score
    type: str                                 # "predicted" | "user"
    metadata: dict                            # empty by default; caller's scratch space

@dataclass
class FrameRecord:
    video_id: str                             # camera_name
    frame_idx: int
    instances: list[InstanceRecord]           # linked (has track_idx)
    unlinked_instances: list[InstanceRecord]  # unlinked — same frame+camera, no track

@dataclass
class FrameBundle:
    """All FrameRecords at one frame_idx, keyed by camera."""
    frame_idx: int
    records: dict[str, FrameRecord]           # camera_name -> record
```

Helpers:
- `instance_to_record(session, inst, frame_idx, camera_name) -> InstanceRecord`
- `session_to_frame_record(session, frame_idx, camera_name) -> FrameRecord`
- `session_to_bundle(session, frame_idx) -> FrameBundle` — per user request.
- `session_to_bundles(session, start, end) -> list[FrameBundle]` — inclusive range.

`InstanceRecord.metadata` is purely caller-owned — the GUI neither reads nor writes it. Keeps `Instance` untouched in `pose_data.py`.

## 4. Notebook → GUI identity updates

No RPC. The notebook calls existing methods directly:

```python
session.set_frame_identity(frame_idx, camera_name, track_idx, identity_id)
session.set_global_track_identity(camera_name, track_idx, identity_id)
session.add_identity(name, color)
session.add_track(name)
```

Thread-affinity: these mutate dicts (GIL-safe for atomic ops) and emit Qt signals from the kernel thread. Qt's auto-connection ships those to the UI's main-thread slots as `QueuedConnection`, triggering repaints.

**Avoid** calling direct widget methods (`window.timeline.update()`) from a notebook cell — those are not thread-safe. Touch `Session`, not widgets.

## 5. Live frame-change push

One small addition in `main_window.py`:

```python
currentFrameChanged = Signal(int)

def set_current_frame(self, frame_idx):
    # ... existing clamp + dedupe ...
    self.currentFrameChanged.emit(frame_idx)
```

Notebook side:

```python
window.currentFrameChanged.connect(lambda f: print("frame:", f))
```

Qt routes this cross-thread automatically.

## 6. GUI-side plumbing

- **`main.py`** migrates from bare `sys.argv[1]` to `argparse`:
  - Positional `folder` (optional — falls back to `QFileDialog`).
  - `--comms` (flag, no value) — start the embedded kernel.
  - `--help`.
- **`gui_source/comms.py`** exposes `start_embedded_kernel(user_ns: dict)` which spawns the worker thread.
- `main.py` calls it once the `LucidLiteWindow` is constructed, passing `{"app": app, "window": window, "session": session, "comms": comms_module}` as the seed namespace.
- Connection-file path logged to stderr.

## 7. `comms.ipynb`

Minimal demo. Assumes the user has launched `uv run python gui_source/main.py --comms /path/to/session` and attached the notebook to the printed connection file.

Cells:
1. **Introspect the namespace** — `session`, `window`, `comms` are already defined.
2. **Snapshot a frame** — `bundle = comms.session_to_bundle(session, session.min_frame); print(bundle)`.
3. **Pass-through "algorithm"** — function returning `identity_id` verbatim per instance.
4. **Write identity updates back** — `session.set_frame_identity(...)` in a loop; observe the GUI repaint live.
5. **Subscribe to frame changes** — `window.currentFrameChanged.connect(print)`; scrub the timeline in the GUI, watch the notebook print frame indices.

## 8. Shutdown / lifecycle

- Kernel thread is a daemon; dies with the process.
- Closing the GUI exits `app.exec()` and the process. Attached notebook clients see "kernel died" and can reconnect after re-launching the GUI.
- No explicit kernel.shutdown required; we trade clean-shutdown niceties for simplicity.

## 9. Risks

- **Direct widget access off main thread.** Convention-enforced; convention may be broken by a user. Document in the notebook.
- **Blocking cells** — an infinite-loop cell blocks the kernel thread, not the Qt thread; GUI stays responsive. Interrupt via `jupyter console`'s Ctrl-C or the notebook UI.
- **Native crashes** — a segfault in PyAV/h5py/native code inside a cell takes down the GUI. Acceptable for a dev tool.
- **`ipykernel` signal-handler init** — we patch `init_signal` to no-op; if ipykernel changes its init order this could regress. Covered by a smoke-test that starts the kernel in a thread.
- **Identity/track map concurrent reads** — notebook thread reads `frame_identity_map` while main thread mutates. Dict read during write is GIL-safe for simple operations; iteration under mutation is not. Convention: prefer `session_to_bundle(...)` snapshots for read-heavy analysis rather than walking live dicts.

## 10. Files touched

- **new** `gui_source/comms.py` — dataclasses, helpers, `start_embedded_kernel`.
- `gui_source/main.py` — argparse migration, wire up `start_embedded_kernel`.
- `gui_source/main_window.py` — add `currentFrameChanged = Signal(int)`, emit in `set_current_frame`.
- `pyproject.toml` — `ipykernel>=6.0`.
- **new** `comms.ipynb` — demo notebook.

No changes to `pose_data.py`, `video_panel.py`, `timeline_widget.py`, `assignment_panel.py`, `overlay_renderer.py`.
