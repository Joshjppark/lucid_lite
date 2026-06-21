"""Push tracker-computed identities into the lucid_lite session.

Originally inlined in tracking_test.ipynb; extracted here so the notebook
stays clean and the helpers are importable from other notebooks/scripts.
"""
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gui_source.colors import next_palette_color
from gui_source.graph_window import GroupGraphWindow
from gui_source.pose_data import Identity

from josh_source import tracker


DEFAULT_PALETTE: dict[int, tuple[str, str]] = {
    0: ("id_0", "#e41a1c"),
    1: ("id_1", "#377eb8"),
    2: ("id_2", "#4daf4a"),
    3: ("id_3", "#984ea3"),
}

DEFAULT_NODE_WEIGHTS: dict[str, float] = {
    "Nose":   1, "Ear_R":  1, "Ear_L":  1,
    "TTI":    1, "TailTip": 1,
    "Head":   1, "Trunk":  1,
    "Tail_0": 1, "Tail_1": 1, "Tail_2": 1,
    "Shoulder_left":  1, "Shoulder_right": 1,
    "Haunch_left":    1, "Haunch_right":   1,
    "Neck": 1,
}


def push_frame_assignments(
    window,
    frame_idx: int,
    palette: dict[int, tuple[str, str]] | None = None,
    node_weights: dict[str, float] | None = None,
    prev_trackIds: dict | None = None,
    next_avail_id: int = 0,
    max_id: int | None = None,
    seek_to_frame: bool = True,
):
    """Compute identities at `frame_idx` and push them into the session.

    Constructs josh_source.tracker.SingleFrameTrack — which internally runs
    _calc_edge_weights → _run_bfs → identity assignment (_init_identities
    for frame 0 or when prev_trackIds is None, _match_prev_groups otherwise)
    and populates sft.trackIds: dict[int, TrackedIdentity] keyed by stable
    identity id. The viewer is seeked to frame_idx and switched to identity
    color mode.

    Returns (groups, assignments, adjacency_matrix, instance_list, sft).
    `groups` is sft.groups (visible-only); `assignments` is built from
    sft.trackIds, so an identity with `group is None` (occluded) writes no
    per-frame override and its prior color stays carried-forward.
    """
    session = window.session

    # 0a. Frame must exist.
    fg = session.frame_group(frame_idx)
    if fg is None:
        raise ValueError(
            f"frame {frame_idx} not in session "
            f"(range {session.min_frame}..{session.max_frame})"
        )

    # 0b. Default node_weights = ones over the project skeleton.
    if node_weights is None:
        node_weights = DEFAULT_NODE_WEIGHTS

    # 0c. Persistent ProjCache across calls within the same kernel session
    #     so we don't recompute fundamental matrices every push.
    if not hasattr(push_frame_assignments, "_proj_cache"):
        push_frame_assignments._proj_cache = tracker.ProjCache()
    proj_cache = push_frame_assignments._proj_cache

    # 1. Run the tracker. __init__ handles _calc_edge_weights → _run_bfs →
    #    identity assignment. prev_trackIds=None forces _init_identities
    #    (fresh ids 0..N-1); a non-None dict forces _match_prev_groups so
    #    identities stay consistent across consecutive frames.
    sft = tracker.SingleFrameTrack(
        fg, session.cameras,
        node_weights=node_weights, proj_cache=proj_cache,
        prev_trackIds=prev_trackIds, next_avail_id=next_avail_id,
        max_ids=max_id,
    )

    # 2. Derive assignments from sft.trackIds: identity_id -> (cam, track).
    #    Identities with no current detection (group is None) are skipped —
    #    their existing per-frame overrides should NOT be overwritten with
    #    a stale group, so they fall through to step 6's -1 sentinel logic
    #    or remain unmapped if they had no detection here at all.
    #
    #    Invalid frame: SingleFrameTrack sets trackIds=None when its
    #    group-formation step couldn't produce a consistent assignment.
    #    Treat as "no assignments" so every visible pair falls through to
    #    step 6's -1 sentinel; the bundle stored in step 8 still lets the
    #    graph view show the broken state. The caller (track_pusher) is
    #    responsible for NOT advancing prev_trackIds past this frame, so
    #    the next valid frame matches against the last valid trackIds.
    assignments: dict[tuple[str, int], int] = {}
    if sft.trackIds is not None:
        for ident_id, ti in sft.trackIds.items():
            if ti.group is None:
                continue
            for cam, track in ti.group.cam_track:
                assignments[(cam, track)] = ident_id

    # 3. Snapshot every visible (cam, track) at this frame so we know which
    #    instances need the -1 sentinel (didn't land in any group).
    visible_pairs: list[tuple[str, int]] = []
    for cam in session.camera_names():
        seen: set[int] = set()
        for inst in fg.get_instances(cam):
            if inst.track_idx is None or inst.track_idx in seen:
                continue
            seen.add(inst.track_idx)
            visible_pairs.append((cam, inst.track_idx))

    with session.batch_updates():
        # 4. Create any identity_id the tracker produced that doesn't exist.
        needed_ids = set(assignments.values())
        existing_ids = {i.id for i in session.identities}
        added = needed_ids - existing_ids
        for ident_id in sorted(added):
            if palette and ident_id in palette:
                name, color = palette[ident_id]
            else:
                name = f"id_{ident_id}"
                color = next_palette_color(len(session.identities))
            session.identities.append(Identity(id=ident_id, name=name, color=color))
            session._identity_counter = max(session._identity_counter, ident_id + 1)
        if added:
            session._emit("identities_changed")

        # 5. Reset prior per-frame overrides at this frame.
        prefix = f"{frame_idx}:"
        for key in [k for k in session.frame_identity_map if k.startswith(prefix)]:
            del session.frame_identity_map[key]

        # 6. -1 sentinel for visible pairs not covered by any group.
        assigned = set(assignments.keys())
        for (cam, track) in visible_pairs:
            if (cam, track) not in assigned:
                session.frame_identity_map[f"{frame_idx}:{cam}:{track}"] = -1

        # 7. Write the tracker-derived assignments verbatim.
        for (cam, track), ident_id in assignments.items():
            session.frame_identity_map[f"{frame_idx}:{cam}:{track}"] = ident_id

        session._emit("identity_map_changed")

        # 8. Store the full bundle so graph_window can draw edges from the
        #    matrix and label them with weights. sft.groups is filtered to
        #    visible-only (identities with group is None are dropped).
        session.set_groups_for_frame(
            frame_idx,
            sft.groups,
            adjacency_matrix=sft.adjacency_matrix,
            instance_list=sft.instance_list,
            trackIds=sft.trackIds,
        )

        # 9. Identity-mode coloring.
        session.set_color_mode("identity")

    # 10. Seek outside batch_updates so decode + repaint happen cleanly.
    #     Bulk callers (track_pusher) pass seek_to_frame=False on every
    #     intermediate frame and True only on the last one, so the panels
    #     don't pay one decode request per swept frame.
    if seek_to_frame:
        window.set_current_frame(frame_idx)

    return sft.groups, assignments, sft.adjacency_matrix, sft.instance_list, sft


def reset_identities(window) -> None:
    """Wipe every identity off the session.

    Clears session.identities, the per-frame overrides
    (frame_identity_map), the global track→identity map
    (track_identity_map), and rewinds the identity-id counter to 0. Leaves
    tracks, frame data, and tracker-pushed group bundles
    (frame_tracker_groups) intact — those describe the underlying
    detections, not identities. Emits identities_changed and
    identity_map_changed inside batch_updates so the assignment panel and
    every video overlay redraw exactly once.
    """
    session = window.session
    with session.batch_updates():
        session.identities.clear()
        session.frame_identity_map.clear()
        session.track_identity_map.clear()
        session._identity_counter = 0
        session._emit("identities_changed")
        session._emit("identity_map_changed")


def track_pusher(
    window,
    frame_indices: Iterable[int],
    palette: dict[int, tuple[str, str]] | None = None,
    node_weights: dict[str, float] | None = None,
    show_graph: bool = False,
    verbose: bool | None = None,
    prev_trackIds: dict | None = None,
    next_avail_id: int = 0,
    max_id: int | None = None,
) -> list[dict]:
    """Push assignments for one or many frames.

    Threads sft.trackIds forward across iterations so identity ids stay
    stable across consecutive frames — frame K's tracker is constructed
    with prev_trackIds = sft_{K-1}.trackIds (so SingleFrameTrack invokes
    _match_prev_groups instead of _init_identities). The first frame
    uses whatever `prev_trackIds` argument was passed in (default
    None → fresh identity assignment).

    Params:
    - window: LucidLiteWindow returned by gui_source.main.main(...).
    - frame_indices: iterable of frames to iterate. Accepts `range(...)`,
      `list[int]`, or any other Iterable[int]; materialized internally so
      len() and -1 indexing are available regardless of input type.
    - palette: identity-id -> (name, hex_color) override; default DEFAULT_PALETTE.
    - node_weights: skeleton-node weights for SingleFrameTrack; default DEFAULT_NODE_WEIGHTS.
    - show_graph: if True, opens a GroupGraphWindow at the *last* processed frame.
    - verbose: per-frame diagnostic prints. If None, defaults to True for
               single-frame inputs and False for multi-frame sweeps.
    - prev_trackIds: optional initial trackIds dict to seed the first frame.
      Pass a prior sweep's `results[-1]["trackIds"]` to continue identity
      propagation across separate calls.
    - max_id: optional upper bound on identity ids forwarded to the tracker.

    Returns: list[dict] aligned to the materialized frame_indices with
    keys 'frame_idx', 'groups', 'assignments', 'adjacency_matrix',
    'instance_list', 'sft', 'trackIds'.
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    if node_weights is None:
        node_weights = DEFAULT_NODE_WEIGHTS

    # Materialize once so len() and -1 indexing work for range / list /
    # any other Iterable[int]. range(0) and empty lists both become [].
    frame_list = list(frame_indices)
    n = len(frame_list)

    if verbose is None:
        verbose = n == 1

    session = window.session
    results: list[dict] = []
    iterator = tqdm(frame_list, desc="push") if n > 1 else frame_list

    # Single outer batch context around the whole sweep. Session
    # ._batch_depth counts entries, so the inner batch_updates inside
    # push_frame_assignments just bumps depth and never emits — every
    # queued signal fires exactly once when this outer context exits.
    # During a long sweep that's the difference between N rebuilds of
    # the assignment panel / video overlays and 1.
    with session.batch_updates():
        for i, frame_idx in enumerate(iterator):
            is_last = i == n - 1
            groups, assignments, adj, inst_list, sft = push_frame_assignments(
                window, frame_idx,
                palette=palette, node_weights=node_weights,
                prev_trackIds=prev_trackIds, next_avail_id=next_avail_id,
                max_id=max_id,
                # Skip the per-frame seek + decode on every intermediate
                # frame — only the last frame in the sweep actually moves
                # the playhead, which is what the user sees.
                seek_to_frame=is_last,
            )
            next_avail_id = sft.next_avail_id
            # Forward-propagate only when this frame produced a valid
            # solution. When SingleFrameTrack couldn't resolve identities
            # it leaves sft.trackIds=None; in that case we keep
            # prev_trackIds pointing at the LAST valid frame so the next
            # valid frame matches against real identities instead of
            # starting fresh from _init_identities.
            if sft.trackIds is not None:
                prev_trackIds = sft.trackIds
            results.append({
                "frame_idx": frame_idx,
                "groups": groups,
                "assignments": assignments,
                "adjacency_matrix": adj,
                "instance_list": inst_list,
                "sft": sft,
                "trackIds": sft.trackIds,
            })

            if verbose:
                print(f"trackIds @ {frame_idx}:")
                if sft.trackIds is None:
                    print("  INVALID — sft.trackIds is None; prev_trackIds preserved")
                else:
                    for ident_id, ti in sft.trackIds.items():
                        if ti.group is None:
                            print(f"  id={ident_id}  HIDDEN  frames_hidden={ti.frames_hidden}")
                        else:
                            print(f"  id={ident_id}  valid={ti.group.valid}  cam_track={ti.group.cam_track}")
                print()
                print(f"adjacency_matrix : shape={adj.shape}, "
                      f"finite edges={int(np.isfinite(adj).sum() // 2)}")
                print(f"instance_list    : {len(inst_list)} entries")
                print()
                print("identities       :", [(i.id, i.name, i.color) for i in session.identities])
                print("color_mode       :", session.color_mode)
                print("current_frame    :", window._current_frame)
                print("stored bundle?   :", frame_idx in session.frame_tracker_groups)
                print("map@frame_idx    :", {k: v for k, v in session.frame_identity_map.items()
                                             if k.startswith(f"{frame_idx}:")})

                fg = session.frame_group(frame_idx)
                print()
                print("overlay resolution:")
                for cam in session.camera_names():
                    for inst in fg.get_instances(cam):
                        rid = session.get_identity_id_for_track(frame_idx, cam, inst.track_idx)
                        ident = session.get_identity(rid)
                        print(f"  {cam:>6} t{inst.track_idx} -> id={rid} ident={ident}")

    if show_graph and frame_list:
        last_frame = frame_list[-1]
        gw = GroupGraphWindow(session, last_frame, parent=window)
        gw.show()

    return results


def swap_detector(source) -> tuple[int, dict[int, list[dict]]]:
    """Detect identity ↔ SLEAP-track swaps across consecutive frames.

    A "swap" here is: for some persistent lucid_lite identity `id_X` and
    some camera `cam`, the underlying SLEAP track index that lucid_lite
    bound to `id_X.cam` changes from frame to frame. The lucid_lite
    identity stays the same (so motmetrics IDF1 won't fire), but the
    tracker has matched a different raw SLEAP detection to that identity
    — which is what `_match_prev_groups`' Hungarian solves for, and what
    you'd flag in the GUI as "the wrong mouse is wearing id_0 in cam back
    now".

    Hidden identities (group is None at frame F) are NOT compared at F —
    their last-known per-cam track is carried forward in the state so the
    comparison resumes against the last frame they were visible. This
    means a swap is reported on the *reappearance frame*, not on every
    hidden frame in between.

    Params:
    - source: one of
        - `MultiFrameTrack` (uses `.frames`, each a `SingleFrameTrack`
          with `.frame_idx` and `.trackIds`)
        - list of dicts from `track_pusher` (uses `frame_idx` and
          `trackIds` keys)

    Returns:
        (n_swaps, swaps) where
        - n_swaps: total number of (frame, identity, cam) swap events
        - swaps: dict[frame_idx -> list[ {id, cam, prev_track, new_track} ]]
          Frames with zero swaps are omitted from the dict.
    """
    # Normalize input to an ordered iterable of (frame_idx, trackIds).
    if hasattr(source, "frames"):
        pairs = [(sft.frame_idx, sft.trackIds) for sft in source.frames]
    else:
        pairs = [(r["frame_idx"], r["trackIds"]) for r in source]

    swaps: dict[int, list[dict]] = {}
    # last seen (cam -> track_idx) per identity. Persists across hidden
    # frames so we compare a reappearance against the last visible frame.
    last_cam_track: dict[int, dict[str, int]] = {}

    for frame_idx, trackIds in pairs:
        if trackIds is None:
            continue

        for ident_id, ti in trackIds.items():
            if ti.group is None:
                # hidden this frame — leave its last_cam_track entry intact.
                continue

            curr_map = {cam: tidx for cam, tidx in ti.group.cam_track}
            prev_map = last_cam_track.get(ident_id)

            if prev_map is not None:
                for cam, new_track in curr_map.items():
                    prev_track = prev_map.get(cam)
                    if prev_track is None or prev_track == new_track:
                        continue
                    swaps.setdefault(frame_idx, []).append({
                        "id": ident_id,
                        "cam": cam,
                        "prev_track": prev_track,
                        "new_track": new_track,
                    })

            # Update state: merge so cams not present this frame keep
            # their prior value (carry-forward across partial visibility).
            merged = dict(prev_map) if prev_map else {}
            merged.update(curr_map)
            last_cam_track[ident_id] = merged

    n_swaps = sum(len(v) for v in swaps.values())
    return n_swaps, swaps


def gt_swap_detector(
    source,
    gt_dir,
    cameras: list[str] | None = None,
    max_distance: float = 50.0,
    return_metrics: bool = False,
):
    """GT-grounded swap detection via motmetrics SWITCH events.

    For each camera, pairs lucid_lite's predicted identities (per
    frame, per cam) against GT identities loaded from
    `<gt_dir>/<cam>/*.predictions.proofread.slp.analysis.h5` and runs a
    `MOTAccumulator`. A SWITCH = a frame where some GT id binds to a
    different predicted id than it did on its previous matched frame
    (the canonical "true swap" measure — same definition as
    `run_lucid_gt_first10k_session72.py`).

    Predicted centroids per cam come from `sft.instance_list`: nanmean
    of the 2D keypoints for the instance whose (cam, sleap_track_idx)
    pair matches `cam_track` of the lucid identity.

    Params:
    - source: MultiFrameTrack or track_pusher results list.
    - gt_dir: Path (or str) to the session GT mirror containing per-cam
      subdirs; each must have one `*.predictions.proofread.slp.analysis.h5`.
    - cameras: subset of cam names to score. Default = every cam with
      both predictions and a GT h5.
    - max_distance: pixel gate (motmetrics treats anything beyond as a
      non-match). Passed to `norm2squared_matrix` as `max_d2 = d**2`.
    - return_metrics: if True, also return a per-cam dict of
      idf1/idp/idr/mota/recall/precision/switches.

    Returns:
        (n_switches, switches_by_frame) — or
        (n_switches, switches_by_frame, per_cam_metrics) if
        return_metrics=True.

        switches_by_frame: {frame_idx -> [ {cam, gt_id, pred_id} ]}.
        Frames with zero switches omitted.
    """
    import h5py
    import motmetrics as mm

    # Normalize input → list of SingleFrameTrack
    if hasattr(source, "frames"):
        frames = source.frames
    else:
        frames = [r["sft"] for r in source]
    if not frames:
        empty: tuple = (0, {}, {}) if return_metrics else (0, {})
        return empty

    # Discover cams present in pred (any instance_list entry) and gt.
    pred_cams: set[str] = set()
    for sft in frames:
        for _, cam, _ in sft.instance_list:
            pred_cams.add(cam)

    gt_dir = Path(gt_dir)
    cam_to_gt: dict[str, Path] = {}
    for cam in pred_cams:
        cdir = gt_dir / cam
        if not cdir.is_dir():
            continue
        h5s = sorted(cdir.glob("*.predictions.proofread.slp.analysis.h5"))
        if h5s:
            cam_to_gt[cam] = h5s[0]

    if cameras is None:
        cameras = sorted(cam_to_gt)
    else:
        cameras = [c for c in cameras if c in cam_to_gt]
    if not cameras:
        empty = (0, {}, {}) if return_metrics else (0, {})
        return empty

    # GT centroids per cam: (n_frames, n_animals, 2).
    gt_centroids_by_cam: dict[str, np.ndarray] = {}
    for cam in cameras:
        with h5py.File(cam_to_gt[cam], "r") as f:
            # tracks: (n_animals, 2, n_nodes, n_frames) -> (n_frames, n_animals, n_nodes, 2)
            tr = f["tracks"][:].transpose(3, 0, 2, 1)
        gt_centroids_by_cam[cam] = np.nanmean(tr, axis=2)

    # Predicted centroids per cam: {cam: {frame_idx: {ident_id: (2,) array}}}.
    pred_by_cam: dict[str, dict[int, dict[int, np.ndarray]]] = {c: {} for c in cameras}
    for sft in frames:
        if sft.trackIds is None:
            continue
        ct_to_id: dict[tuple[str, int], int] = {}
        for ident_id, ti in sft.trackIds.items():
            if ti.group is None:
                continue
            for cam, t in ti.group.cam_track:
                ct_to_id[(cam, t)] = ident_id
        for t, cam, pts in sft.instance_list:
            if cam not in pred_by_cam:
                continue
            ident_id = ct_to_id.get((cam, t))
            if ident_id is None:
                continue
            cent = np.nanmean(pts, axis=0)
            if np.isnan(cent).any():
                continue
            pred_by_cam[cam].setdefault(sft.frame_idx, {})[ident_id] = cent

    # Per-cam motmetrics → SWITCH events.
    mh = mm.metrics.create()
    switches_by_frame: dict[int, list[dict]] = {}
    per_cam_metrics: dict[str, dict] = {}

    for cam in cameras:
        acc = mm.MOTAccumulator(auto_id=False)
        gt_cents = gt_centroids_by_cam[cam]
        n_gt_frames = gt_cents.shape[0]

        for sft in frames:
            fi = sft.frame_idx
            if fi < n_gt_frames:
                fg = gt_cents[fi]
                gt_ids = [i for i in range(fg.shape[0]) if not np.isnan(fg[i]).any()]
                gt_pts = np.array([fg[i] for i in gt_ids]) if gt_ids else np.empty((0, 2))
            else:
                gt_ids, gt_pts = [], np.empty((0, 2))

            pred_dict = pred_by_cam[cam].get(fi, {})
            pred_ids = list(pred_dict.keys())
            pred_pts = (np.array([pred_dict[i] for i in pred_ids])
                        if pred_ids else np.empty((0, 2)))

            if gt_pts.size and pred_pts.size:
                dists = mm.distances.norm2squared_matrix(
                    gt_pts, pred_pts, max_d2=max_distance ** 2,
                )
            else:
                dists = np.empty((len(gt_ids), len(pred_ids)))

            acc.update(gt_ids, pred_ids, dists, frameid=fi)

        ev = acc.events
        sw = ev[ev["Type"] == "SWITCH"]
        for (frame_idx, _event_idx), row in sw.iterrows():
            switches_by_frame.setdefault(int(frame_idx), []).append({
                "cam": cam,
                "gt_id": int(row["OId"]),
                "pred_id": int(row["HId"]),
            })

        if return_metrics:
            s = mh.compute(acc, metrics=[
                "idf1", "idp", "idr", "mota", "num_switches",
                "recall", "precision",
            ], name=cam)
            per_cam_metrics[cam] = {
                "idf1": float(s.idf1.iloc[0]),
                "idp":  float(s.idp.iloc[0]),
                "idr":  float(s.idr.iloc[0]),
                "mota": float(s.mota.iloc[0]),
                "switches":  int(s.num_switches.iloc[0]),
                "recall":    float(s.recall.iloc[0]),
                "precision": float(s.precision.iloc[0]),
            }

    n_switches = sum(len(v) for v in switches_by_frame.values())
    if return_metrics:
        return n_switches, switches_by_frame, per_cam_metrics
    return n_switches, switches_by_frame
