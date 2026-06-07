"""Push tracker-computed identities into the lucid_lite session.

Originally inlined in tracking_test.ipynb; extracted here so the notebook
stays clean and the helpers are importable from other notebooks/scripts.
"""
from collections.abc import Iterable

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
        prev_trackIds=prev_trackIds, max_ids=max_id,
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
                prev_trackIds=prev_trackIds, max_id=max_id,
                # Skip the per-frame seek + decode on every intermediate
                # frame — only the last frame in the sweep actually moves
                # the playhead, which is what the user sees.
                seek_to_frame=is_last,
            )
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
