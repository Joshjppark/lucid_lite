# LUCID-Lite — Project Handoff

Read-only Python companion to the LUCID web app. Loads a single per-camera SLP session; renders video panels with pose overlays, a SLEAP-style timeline, and an identity/track assignment panel. No 3D, no reprojection, no geometry editing, no export.

**Status:** MVP implemented. All code lives in `python_tracking_dev/gui_source/` (paths below are relative to the LUCID repo root). Developed on a **headless remote server** — imports, adapter, and `QMainWindow` construction are verified under `QT_QPA_PLATFORM=offscreen`, but no visual rendering has been validated. Next step is running it locally against a real session folder and iterating on anything that looks wrong.

---

## 1. Quick start (local dev)

### 1a. Prerequisites

- **Python ≥ 3.11** (for `tomllib`)
- A desktop / display (this is a Qt app)
- FFmpeg libraries (PyAV wheels bundle them on most platforms, but on Linux you may need `libavcodec`/`libavformat` system packages)

### 1b. Install

```bash
# From the repo root, or anywhere:
pip install PySide6 av sleap-io numpy
```

### 1c. Run

```bash
# From the repo root:
python python_tracking_dev/gui_source/main.py                 # folder picker opens
python python_tracking_dev/gui_source/main.py /path/to/session
```

### 1d. Session folder layout (required)

```
session_folder/
├── calibration.toml            # optional; calibration.json also accepted
├── <camera_name>/
│   ├── *.mp4                   # one video per cam; first match wins
│   └── *.slp                   # one SLP per cam; first match wins
├── <camera_name>/
│   └── ...
```

Missing calibration is OK — cameras are inferred from subfolder names and `session.has_calibration` stays `False`. Camera subfolder name is matched to the calibration entry's `name` case-insensitively, with `cam_` / `camera_` prefixes stripped (see `session_loader._cam_key`).

---

## 2. What's implemented

All 14 files already exist in `python_tracking_dev/gui_source/`. Flat layout — no `__init__.py`, plain module imports (`from pose_data import Session`). If you add packaging later, you'll need to switch to relative imports or add a `src/` layout.

| File | Purpose | Maps to (LUCID JS) |
|------|---------|--------------------|
| **`main.py`** | CLI entry — run this | — |
| `main_window.py` | `LucidLiteWindow(QMainWindow)`: camera docks (top), timeline (bottom), assignment (right), menus, keyboard nav | `index.html` outer layout |
| `pose_data.py` | `Session` (QObject + signals), `FrameGroup`, `InstanceGroup`, `Instance`, `UnlinkedInstance`, `Skeleton`, `Camera`, `Identity` | `pose-data.js` |
| `session_loader.py` | `load_session_from_folder` + `rebuild_instance_groups` (bucket instances by `track_idx`) | `index.html:12483–12999`, `slp-merge.js:151–180` |
| `calibration.py` | `calibration.toml` / `.json` → `Camera[]` | `file-io.js:125–250` |
| `slp_reader.py` | `sleap-io.load_slp(...)` → lite `Instance` / `FrameGroup` | `slp-import-worker.js:84–459` |
| `video_decoder.py` | `OnDemandVideoDecoder` (PyAV, LRU cache, seek-and-decode) + `DecodeWorker` (QThreadPool) | `video.js` |
| `video_panel.py` | `VideoPanelWidget`: `paintEvent` composes letterboxed pixmap + pose overlay | `index.html:14613–14725`, `index.html:11585–11620` (letterbox) |
| `overlay_renderer.py` | `draw_overlay_for_camera` — edges, nodes, labels | `overlays.js:369–562`, `overlays.js:1570` |
| `timeline_widget.py` | Custom `QWidget` + `paintEvent`; one row per `(camera, track)`; click to seek | `timeline.js` |
| `assignment_panel.py` | Right-dock `QTableWidget` with per-row identity combobox + "New Identity" / "New Track" buttons | `interaction.js` identity paths |
| `new_identity_dialog.py` | `NewIdentityDialog` (name + color picker), `NewTrackDialog` (name) | — (new) |
| `colors.py` | 20-color Green-Armytage palette shared by overlay + timeline | `overlays.js:11–32` |
| `lucid_labels.py` | `Protocol` + `load_lucid_labels()` — raises `NotImplementedError` until schema lands | — (new) |
| `README.md` | User-facing usage doc | — |

### Signals (all defined on `Session`)

- `frame_groups_changed` — fires after bulk load or structural change
- `identity_map_changed` — fires from `set_frame_identity` / `set_global_track_identity`
- `identities_changed` — fires from `add_identity`
- `tracks_changed` — fires from `add_track`

Widgets subscribe to the relevant signals in their `__init__` and call `self.update()` on trigger.

### Keyboard

| Key               | Action                |
|-------------------|-----------------------|
| `← / →`           | Step 1 frame          |
| `Shift + ← / →`   | Step 10 frames        |
| `Home / End`      | Jump to first / last  |

Handled in `LucidLiteWindow.keyPressEvent`.

---

## 3. What was tested on the remote (headless)

Everything below ran under `QT_QPA_PLATFORM=offscreen`:

- **Byte-compile:** `python -m py_compile *.py` — no syntax errors.
- **Import graph:** all 14 modules import cleanly; no circular imports (Session's `load_from_folder` lazy-imports `session_loader`).
- **SLP adapter:** `slp_reader.merge_slp_into_session` on `python_tracking_dev/claude-sleap-files.slp` produced 1000 `FrameGroup`s, 2 tracks, and `rebuild_instance_groups` produced 2 `InstanceGroup`s at frame 0.
- **Window construction:** `LucidLiteWindow(session)` builds, `set_current_frame(5)` runs, timeline precomputes segments for `(cam_test, 0)` and `(cam_test, 1)`. Harmless warning: `"This plugin does not support propagateSizeHints()"` — Qt offscreen platform only, goes away on a real display.

---

## 4. Untested / known caveats — the first thing to check locally

Because headless dev can't validate rendering, these are the likely hot spots:

1. **Video decode actually showing frames.** `OnDemandVideoDecoder._decode` (in `video_decoder.py`) seeks by `container.seek(target_ts, stream=..., any_frame=False, backward=True)` then decodes forward until `cur_idx >= frame_idx`. PyAV seek quirks are codec-dependent; H.264 should work, H.265/HEVC sometimes needs `any_frame=True`. If the video panel stays black with "(no video)" text, the PyAV open-or-decode path is failing — the exception is printed once on `VideoPanelWidget.__init__`.
2. **Letterbox math.** `video_panel._letterbox` mirrors `index.html:11601–11607`. Should center the video in the cell. If it looks stretched or misaligned, compare against the JS source.
3. **Overlay coordinate mapping.** `video_panel.paintEvent` builds a `v2p` lambda that maps video pixel coords → panel coords. Overlay should sit directly on top of the body. If it's offset, the transform is wrong.
4. **Identity color propagation.** Changing an identity in the assignment dock should repaint the overlay (via `identity_map_changed` signal) and timeline row hue. Wired through but not visually confirmed.
5. **Multi-camera grid layout.** Cameras live in a `QGridLayout` packed `ceil(sqrt(N))` columns wide. Cells share space evenly via row/column stretch; with very small windows (< 1366×768) and 6+ cameras the cells can hit the per-panel `setMinimumSize(320, 180)` floor — drag the splitter handle to give the grid more room.
6. **Timeline scrolling / zoom.** Currently no zoom or horizontal scroll — the full frame range is always stretched across the widget. For long sessions (10k+ frames) segments will be 1px wide. See §6 for the fix.
7. **Frame indexing across multiple videos in one SLP.** `slp_reader` assumes one SLP per camera. If an SLP contains labels from several videos, they all bucket into the same `FrameGroup` — that's a misuse of the loader, but worth a defensive warning (see §6).
8. **`Session.load_from_folder` is synchronous.** A large session blocks the UI thread during load. Move to a worker thread if it becomes annoying (`QThread` + `Session.frame_groups_changed.emit()` from the main thread).

---

## 5. Reference map — LUCID web app source

The Python port intentionally mirrors the LUCID JS app. When in doubt about behavior, read the JS file it was ported from. Key files (all in repo root):

| JS file | What to look at |
|---------|-----------------|
| `pose-data.js` | Lines 4–94 Skeleton; 97–277 Camera; 280–352 Instance; 375–444 FrameGroup; 464–541 InstanceGroup; 544–1110 Session (identity / track maps, `getIdentityIdForTrack` at 668–676) |
| `file-io.js` | 125–159 parseCalibrationTOML; 228–250 parseCalibrationJSON; 1430–1503 SLP export (not ported — read only if you implement export later) |
| `slp-import-worker.js` | 84–459 `parseSlp` (SLP → frames / instances); 331–385 FrameGroup building |
| `slp-merge.js` | 151–180 `rebuildInstanceGroupsForFrames` — direct analog of `session_loader.rebuild_instance_groups` |
| `index.html` | 12524–12576 folder scanning; 12608–12654 calibration matching; 11585–11620 letterbox / canvas fitting; 14613–14725 `VideoPaneRenderer` (per-panel structure); 3283–3323 identity-change callback chain |
| `overlays.js` | 11–32 TRACK_COLORS (exact palette Lucid-Lite uses); 117–125 color-by-track/identity; 369–562 `drawSkeleton`; 1075–1222 instance labels; 1570 `drawFrameOverlays` |
| `timeline.js` | 568–754 `_buildTrackSegments` (the core of our timeline precompute); 719–754 row ordering (camera-first); 1374–1403 playhead draw; 1542–1701 mouse seek |
| `video.js` | 32–184 `OnDemandVideoDecoder`; 394–430 seek lock (we use a `threading.Lock` for the same reason) |
| `interaction.js` | 98–587 identity/track assignment state machine; 929, 2006 unlinked-instance flows |
| `viewport3d.js`, `triangulation.js` | Intentionally **not** ported — 3D is out of scope |

---

## 6. Suggested next work (priorities)

Roughly in order. None are blocking — the MVP runs today.

### High priority (do once you see it working locally)

1. **Verify the full video-plus-overlay render on a real session folder.** Fix any of §4's hot spots that actually show up.
2. **Defensive warning when an SLP references multiple videos.** In `slp_reader.merge_slp_into_session`, inspect `labels.videos`; if length > 1, log a warning (per-camera SLPs should have exactly one video entry).
3. **Move session loading off the UI thread.** Wrap `Session.load_from_folder` in a `QThread`; show a modal progress dialog. Pattern: emit `frame_groups_changed` only once at the end so the timeline doesn't rebuild mid-load.

### Medium priority

4. **Timeline zoom + horizontal scroll.** `timeline_widget` currently fits `min_frame..max_frame` across the full width. Add `wheelEvent` zoom (scale around cursor x) and middle-drag pan, plus horizontal overflow. Mirror `timeline.js:1542–1701`.
5. **Per-frame override UX.** The assignment panel shows `scope = frame | global` but there's no toggle. Add a right-click row menu: "Apply globally" (→ `session.set_global_track_identity`) / "Clear frame override".
6. **Per-panel zoom / pan on video panels.** Deferred from the original plan §3.6. Store `(scale, offset_x, offset_y)` per panel; transform both pixmap draw and overlay `v2p`. Wheel = zoom, middle-drag = pan.
7. **Playback.** Add a play/pause toggle that advances `current_frame` on a `QTimer` at `decoder.fps`. Sync all cameras by keying off one "master" frame.

### Lower priority

8. **`LucidLabels` schema + reader.** Fill in `lucid_labels.py` once the upstream format is decided. The entry point is already wired (`main_window._load_lucid_labels_dialog`).
9. **Tests.** Add `pytest` for `session_loader`, `slp_reader`, `pose_data.Session.get_identity_id_for_track` (frame override vs global fallback is easy to get wrong).
10. **`pyproject.toml` + editable install.** Lets you `python -m lucid_lite` instead of the long path. Also formalizes the dep list.
11. **Save / restore splitter sizes** via QSettings so the user's chosen sidebar width persists between sessions (the grid is auto-tiled, so only the splitter state needs saving).

---

## 7. Decisions already locked in (approved by user)

From the original plan §9 — all 10 were approved before MVP implementation:

1. Framework: **PySide6**
2. Project location: **`python_tracking_dev/gui_source/`** (flat, no package)
3. Python **≥ 3.11** (uses `tomllib`)
4. Video backend: **PyAV** (frame-accurate seek required)
5. Panel layout: **adaptive `QGridLayout` of camera views on the left, assignment sidebar on the right via `QSplitter`** (replaced the original `QDockWidget`-per-camera decision in the layout refactor)
6. Video-panel zoom/pan: **deferred to post-MVP** (item 6 above)
7. Timeline rows: **one row per `(camera, track)`** — ordered camera-first
8. Cross-view grouping button: **skipped**; shared identity implicitly groups tracks across cameras via `Session.track_identity_map`
9. `LucidLabels`: **stub raising `NotImplementedError`** until upstream schema lands
10. Testing: **manual UI**; no automated tests in MVP

If you revisit any of these, note the change here so the reasoning doesn't get lost.

---

## 8. Out of scope (don't re-open without discussion)

- Editing node positions, creating new `Instance`s
- 3D viewer / `viewport3d.js` equivalent
- Triangulation / reprojection (`triangulation.js`)
- SLP export (the round-trip side of `file-io.js`)
- Multi-session loading
- Undo/redo for identity/track edits

---

## 9. Gotchas worth remembering

- **Dual-parent `Instance` refs** — `FrameGroup.instances[cam]` and `InstanceGroup.instances[cam]` hold the **same** Python object. If you mutate an Instance through one, the other sees it. This mirrors `pose-data.js` and is load-bearing for `rebuild_instance_groups`.
- **`InstanceGroup.instances` is singular per camera** — `dict[str, Instance]`, overwrites on `add_instance`. This is intentional (matches JS) but means if two instances in the same camera share a `track_idx`, only the last one survives in the 3D-side group. 2D side (`FrameGroup.instances[cam]` = `list[Instance]`) preserves all.
- **Track index vs identity id** — `Instance.track_idx` indexes `Session.tracks`. `Identity.id` is an auto-incrementing integer with no relation to any list position. Don't use one where the other is expected.
- **Per-frame identity overrides take precedence over global** — see `Session.get_identity_id_for_track`. Assignment panel writes to the per-frame map; there's no UI (yet) to write to the global map.
- **Splitter sizes reset on every launch.** `_size_to_screen` recomputes `[grid, sidebar]` based on the primary screen each time the window opens, so any user resize of the splitter handle is lost on relaunch. Persist via QSettings if/when desired (see §6 item 11).
- **PyAV seek flags** — `any_frame=False, backward=True` asks for the nearest keyframe at-or-before the target, and we decode forward until we hit the exact frame. Some codecs misreport keyframe positions; if you see wrong frames, try `any_frame=True`.
