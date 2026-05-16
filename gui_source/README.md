# LUCID-Lite

Read-only Python companion to the LUCID web app. Loads a single per-camera SLP
session and shows multi-view video + pose overlays, a SLEAP-style timeline,
and an identity/track assignment panel.

## Requirements

- Python ≥ 3.11
- `PySide6`, `PyAV` (`av`), `sleap-io`, `numpy`
- For the embedded-kernel notebook workflow (`--comms`): `ipykernel`, `jupyter-console`.
  JupyterLab / classic Notebook are not bundled — `uv add jupyterlab` (or
  `uv add notebook`) if you want those clients.

Installed via `uv sync` in the project root; use `uv run python …` to invoke.

## Run

```bash
# From the project root (where pyproject.toml lives):
uv run python gui_source/main.py /path/to/session_folder

# No-arg: a folder picker opens on launch.
uv run python gui_source/main.py

# With an embedded Jupyter kernel so a notebook can attach (see
# "Jupyter notebook integration" below):
uv run python gui_source/main.py /path/to/session_folder --comms
```

### CLI flags

| Flag       | Meaning                                                              |
|------------|----------------------------------------------------------------------|
| `folder`   | Positional; session folder to open. Omit for a folder picker.        |
| `--comms`  | Start an embedded Jupyter kernel on a background thread.             |
| `--help`   | Show argparse help.                                                  |

## Expected session folder layout

```
session_folder/
├── calibration.toml            (optional; .json also accepted)
├── <camera_name>/
│   ├── *.mp4
│   └── *.slp
└── ...
```

Missing calibration is allowed — cameras are inferred from subfolder names and
3D features stay disabled.

## Keyboard

| Key              | Action                 |
|------------------|------------------------|
| ← / →            | Step frame by 1        |
| Shift + ← / →    | Step frame by 10       |
| Home / End       | Jump to first / last   |

## Layout

- On launch the window resizes to the primary screen's available geometry.
- **Left:** an adaptive grid of camera views (`ceil(sqrt(N))` columns).
- **Right:** the identity/track assignment sidebar (default 320 px), resizable
  via the splitter handle.
- **Bottom:** the timeline, spanning the full window width.

## Files

| File                    | Purpose                                  |
|-------------------------|------------------------------------------|
| `main.py`               | **Entry point** — run this              |
| `main_window.py`        | `QMainWindow` assembly                   |
| `pose_data.py`          | `Session`, `FrameGroup`, `InstanceGroup`, `Instance`, … |
| `session_loader.py`     | Folder scan → `Session`                  |
| `calibration.py`        | TOML / JSON calibration parse            |
| `slp_reader.py`         | `sleap-io` → `Instance` adapter          |
| `video_decoder.py`      | PyAV on-demand decoder + cache           |
| `video_panel.py`        | Per-camera `QWidget` (letterbox + overlay) |
| `overlay_renderer.py`   | Pose skeleton draw                       |
| `timeline_widget.py`    | SLEAP-like timeline                      |
| `assignment_panel.py`   | Identity/track assignment sidebar        |
| `new_identity_dialog.py`| Create-identity + create-track dialogs   |
| `colors.py`             | Shared 20-color palette                  |
| `lucid_labels.py`       | External-label binding stub              |
| `comms.py`              | Embedded-kernel launcher + snapshot dataclasses/helpers |

## Jupyter notebook integration

Launching with `--comms` starts an embedded `ipykernel` on a background thread;
notebooks attach to that kernel and **share the GUI's Python interpreter**. A
notebook cell that calls `session.set_frame_identity(...)` mutates the same
`Session` object the GUI is rendering, and the overlay/timeline repaint live.

See `comms.ipynb` (at the project root) for a 5-cell demo and step-by-step
attach instructions. Full design rationale is in
`prompts/plans/gui-notebook-comms.md`.

Quick start:

```bash
# Terminal 1 — start the GUI with the kernel:
uv run python gui_source/main.py /path/to/session --comms
# stderr will print:
#   [comms] connection file: /Users/.../jupyter/runtime/kernel-<pid>.json
#   [comms] attach with: jupyter console --existing <path>

# Terminal 2 — attach a REPL to the running kernel:
uv run jupyter console --existing
# You get a Python prompt with session, window, app, comms already defined.
```

In `comms.py` the exported snapshot API is:
`session_to_bundle(session, frame_idx) -> FrameBundle`,
`session_to_bundles(session, start, end) -> list[FrameBundle]`,
plus the `InstanceRecord` / `FrameRecord` / `FrameBundle` dataclasses. These
are convenience adapters — notebooks can also touch `session.frame_groups`,
`session.identities`, etc. directly.

## Out of scope (by design)

Editing node positions, creating new instances, 3D viewer, triangulation,
reprojection, SLP export, multi-session loading.
