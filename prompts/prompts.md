

##
# Prompts 1 **UI Fix: Window and Layout**

**Window Initialization:**
Set the initial window size to match the display/monitor resolution of the viewing device.

**Layout Change:**
Move the info panel to the right side of the window. Video grid views should occupy the left side, with the info panel as a vertical sidebar on the right. The video grid should support multiple rows and columns (e.g., a 2×2 or 3×2 grid) rather than a single horizontal row, adapting to the number of loaded camera views. Visually, this should look like luc3d without the 3D viewer panel.


```
┌─────────────────────────────────────────────────────────────────────────┐
│  File  Triangulate  [+ Instance]  [Sessions]    [Group] [Edit Group]    │
├─────────────────────────────────────────────┬───────────────────────────┤
│                                             │  Info Panel              │
│  ┌──────────────┐  ┌──────────────┐         │                         │
│  │              │  │              │         │  ┌─ Instances ─────────┐│
│  │    camA      │  │    camB      │         │  │ Grouped Instances   ││
│  │              │  │              │         │  │  track_0            ││
│  │              │  │              │         │  │    UserInstance     ││
│  └──────────────┘  └──────────────┘         │  │    ReprojInstance   ││
│                                             │  │  track_1            ││
│  ┌──────────────┐  ┌──────────────┐         │  │    UserInstance     ││
│  │              │  │              │         │  │                     ││
│  │    camC      │  │    camD      │         │  │ Ungrouped Instances ││
│  │              │  │              │         │  │  camA: instance_0   ││
│  │              │  │              │         │  └─────────────────────┘│
│  └──────────────┘  └──────────────┘         │                         │
│                                             │                         │
├─────────────────────────────────────────────┴──────────────────────────┤
│  Timeline                          [Tracks] [IDs] [Both] [Reprojs] [TV]│
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ track_0  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │ track_1  ░░░░░░░░░░░████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ◄━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►   │
│  Frame: 42/1800          30 FPS  [Speed]     Camera: camA  Labeled: 5  │
└─────────────────────────────────────────────────────────────────────────┘
```

##
# Prompt 2  **Bug Fix: New Identity and Track Buttons**


The `New Identity` and `New Track` buttons are not functional — they throw `AttributeError` because `NewIdentityDialog` and `NewTrackDialog` use `dlg.Accepted` instead of `dlg.accepted`. Fix the attribute references.

Error output:
```
AttributeError: 'NewIdentityDialog' object has no attribute 'Accepted'. Did you mean: 'accepted'?
AttributeError: 'NewTrackDialog' object has no attribute 'Accepted'. Did you mean: 'accepted'?
```
File: `/Users/joshuapark/Documents/talmolab/repos/python_tracking_dev/gui_source/assignment_panel.py`, lines 145 and 151.

**Feature: New Track Tab in Info Panel**
Add a `Track/ID` tab to the info panel containing:

- `New Identity` and `New Track` buttons (move from current location).
- **Tracks Legend Table:** Lists all tracks with their associated colors.
- **Identities Legend Table:** Lists all identities with their associated colors.

Both tables should update in real time as tracks/identities are created or modified.

**Feature: Color-By Toggle**
Add a toggle button in the `Track` tab that switches instance coloring between:
- **Color by Track:** Instances colored by their track assignment.
- **Color by Identity:** Instances colored by their identity assignment.

When coloring instances by track or identity, any instance that lacks the corresponding assignment (no track when coloring by track, no identity when coloring by identity) should default to gray.


Timeline
1. Make multiple tabs in the Timeline, one with Track and one with Identities. The tabs can be added on the left side of timeline (there is a bit of open space there)
2. 3 columns (the way it is currently)


##
# Prompt 3  **Feature: GUI ↔ Jupyter Notebook Communication System**

The purpose of the Python GUI is to export instance positions and track data in real time to a Jupyter notebook for:
1. Visualizing the performance of current identity assignment algorithms.
2. Developing and testing new identity assignment algorithms.

**Step 1: Propose Communication Architecture**
Propose a communication protocol between the GUI and notebook (e.g., shared memory, ZMQ sockets, callback hooks, or a shared Python object). The system must support real-time bidirectional communication while both the GUI and notebook are running.

**Step 2: GUI → Notebook Data Structure**
Design a data structure for sending frame data from the GUI to the notebook with the following requirements:
- Supports single-frame or multi-frame payloads (list-like structure).
- Per frame: video identifier, frame index, and a list of associated instances.
- Per instance: node positions, track assignment, identity assignment, and a score field.
- Per instance: an extensible metadata dictionary for future use.

Review the existing data structures in the Python GUI and reuse as much as possible rather than creating new ones. Propose minimal additions or wrappers where needed.

**Step 3: Notebook → GUI Identity Updates**
The notebook receives frame data (which may contain null or placeholder identity values), processes it through an algorithm, and sends updated identity assignments back to the GUI. The GUI should apply these updates and reflect them in the viewer in real time.

**Step 4: Create `comms.ipynb`**
Create a minimal notebook at `comms.ipynb` that demonstrates:
1. Launching or connecting to the running GUI.
2. Receiving frame data from the GUI.
3. Processing identity assignments (stub — can be pass-through initially).
4. Sending updated identity assignments back to the GUI.

**Write the full communication design and data structure proposal to `prompts/plans/gui-notebook-comms.md` before implementing. Do not implement until approved.**


## General Implementation Guideliens:
* use a subagent to read over the prompt and plan
* use another subagent to find faults in the plan and resolve arising issues
* ask questions to me as needed.

Answers to your questions:


I actually don't like ZeroMQ PAIR socket. I personally dislike sending json requests for a connection because the purpose of maing a python gui is so that it can share a kernel with a python notebook. So instead, provide your thoughts on the following idea:
```
Embed a kernel in the GUI process. This is usually the nicest option when it fits. In your GUI's Python process, call IPython.embed_kernel() (or use ipykernel.kernelapp.IPKernelApp). It prints connection info; you then run jupyter console --existing <connection-file> or open a notebook pointing at that kernel. The notebook now shares the GUI's interpreter — all its globals, open file handles, live Qt widgets, model state, everything. You can poke at self.app.current_frame from a notebook cell while the GUI keeps rendering. The catch is threading: the GUI event loop and the kernel's event loop have to coexist, which is straightforward for Qt 
```
