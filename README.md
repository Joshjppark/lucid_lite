# lucid_lite

Multi-view pose tracking GUI plus a triangulation-based identity tracker.
See [`gui_source/README.md`](gui_source/README.md) for the GUI internals.

This project is the staging ground for changes that will eventually be
upstreamed into `sleap-3d`. **Do not depend on or install any `sleap-3d`
packages here** — keep this repo standalone until the migration happens.

## Install

```bash
uv sync
```

## Run the GUI

```bash
uv run python gui_source/main.py /path/to/session_folder
```

Omit the folder to get a picker. Add `--comms` to start an embedded Jupyter
kernel so a notebook can attach to the running process.

From a notebook (after `--comms` or in a fresh kernel that imports the GUI):

```python
from gui_source import main

app, window = main.main([
    "main.py",
    "/path/to/session_folder",
])
session = window.session
```

## Run the tracker

`MultiFrameTrack` runs the per-frame triangulation + identity matching across a
range of frames in the session loaded by the GUI window.

```python
from josh_source import tracker

node_weights = {
    "Nose":   0.7, "Ear_R":  0.7, "Ear_L":  0.7,
    "TTI":    1.0, "TailTip": 0.0,
    "Head":   1.0, "Trunk":  0.8,
    "Tail_0": 0.0, "Tail_1": 0.0, "Tail_2": 0.0,
    "Shoulder_left":  0.7, "Shoulder_right": 0.7,
    "Haunch_left":    0.7, "Haunch_right":   0.7,
    "Neck":   0.7,
}

mft = tracker.MultiFrameTrack(
    window=window,
    node_weights=node_weights,
    # start=0, end=None, max_ids=None,   # all optional
)
mft.track(verbose=True)
mft.push_assignments_to_gui()   # write the identity assignments back into the GUI
```

Results live on the instance:
- `mft.frames` — list of `SingleFrameTrack` per processed frame
- `mft.trackIds` — list of `dict[identity_id -> TrackedIdentity]` per frame
- `mft.visible_ids`, `mft.invalid_instances`, `mft.nonmatch_instances`,
  `mft.nonmatch_groups`
