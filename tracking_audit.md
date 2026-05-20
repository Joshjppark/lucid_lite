# Tracking Audit — Duplicate Identity Bug

**Audience**: LUCID-3D maintainers + reviewers porting the fix back to `tracker.js`.
**Scope**: `luc3d/pose/tracker.js` algorithm, verified end-to-end against the
Python port in `lucid_lite/luc3d_tracker_helper.py`.
**Dataset**: `/Users/joshuapark/Documents/talmolab/lucid_folders/10072022145420_small`
(8 calibrated cameras, 6 with labeled instances, 4 animals, 1800 frames).

---

## TL;DR

The `trackAll` algorithm runs to completion and produces a clean
`session.identities` list (4 identities matching the number of animals).
However, the **viewer renders two skeletons in the same camera using the
same identity color** at scattered frames (33 frames in this dataset; the
first is frame 147). The duplication is **purely a rendering / lookup
fallback artifact**: at affected frames, one of the colliding tracks has
no per-frame identity override and the renderer falls back to the global
`trackIdentityMap`, which still holds a value from an earlier frame.
Random coincidence with another track's identity at that camera produces
the visible duplicate.

The fix is a **per-frame uniqueness guard at the end of
`matchFrameInstances`**: every visible `(camera, track)` pair at a frame
that did not receive a per-frame override gets an explicit "no identity"
sentinel (`-1`). The viewer renders those instances in track-color
instead of colliding with another instance's identity-color.

```
Before fix:  4 identities, 33 frames with visible duplicates
After  fix:  4 identities, 0 frames with visible duplicates
```

The bug-reproducing path is preserved behind
`enforce_uniqueness=False` for regression testing.

---

## 1. Tracking algorithm overview

The tracker takes a multi-camera session (4–8 calibrated views per
frame, each carrying SLEAP-derived 2-D pose instances) and decides
**which instances across different cameras correspond to the same animal**
at every frame, then assigns each cross-camera group an `Identity` that
remains stable across time.

```
┌──────────────────────────────────────────────────────────────────────┐
│  track_all(session, num_animals)                                     │
│   │                                                                  │
│   │  for frame_idx in session.frame_indices:                         │
│   │      │                                                           │
│   │      ▼                                                           │
│   │  match_frame_instances(frame_group, cameras, session, …)         │
│   │      │                                                           │
│   │      ├── collect_instances(frame_group, cameras)                 │
│   │      ├── match_pairwise(...)  ──▶ list of group dicts            │
│   │      │       │                                                   │
│   │      │       │     ▶  bootstrap on 2 densest cameras             │
│   │      │       │     ▶  refine top-N via cross_view_score          │
│   │      │       │     ▶  >0.05 filter (unconstrained mode)          │
│   │      │       │     ▶  graft remaining cameras via reprojection   │
│   │      │       │                                                   │
│   │      ├── reorder_groups_by_prev(...)  ──▶ groups aligned to prev │
│   │      │       (4-signal Hungarian)                                │
│   │      │                                                           │
│   │      ├── triangulate_group(...)  ──▶ targets3d                   │
│   │      ├── identity assignment (vote → fallback → spawn)           │
│   │      └── ★ per-frame uniqueness guard ★                          │
│   │                                                                  │
│   ▼                                                                  │
│   identities + track_identity_map + frame_identity_map               │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 Identity bookkeeping

Three pieces of state live on `Session`:

| Field | Type | Purpose |
|---|---|---|
| `identities` | `list[Identity]` | The set of known animals. Wiped at the start of `track_all`. |
| `track_identity_map` | `dict["cam:track", id]` | **Global** mapping. Last-write-wins across frames; reflects whichever frame most-recently assigned this `(camera, track)`. |
| `frame_identity_map` | `dict["frame:cam:track", id]` | **Per-frame override**. Authoritative for that frame. |

The viewer's lookup (matches `pose-data.js:getIdentityForTrack`) is:

```python
def get_identity_for_track(self, track_idx, camera_name, frame_idx):
    # 1. Per-frame override wins
    frame_key = f"{frame_idx}:{camera_name}:{track_idx}"
    if frame_key in self.frame_identity_map:
        return self.get_identity(self.frame_identity_map[frame_key])
    # 2. Else direct global at (cam, track)
    return self.get_identity(self.track_identity_map.get(f"{camera_name}:{track_idx}"))
    # 3. (JS only — broader suffix-scan fallback omitted for clarity)
```

Per-camera per-frame **uniqueness is implicit** — no code path inside
`matchFrameInstances` enforces it. The algorithm assumes that writing one
identity per kept group is sufficient.

### 1.2 `match_pairwise`

Given the active cameras with their instances at a frame:

1. **Sort cameras by candidate count.** Bootstrap on the two cameras with
   the most visible instances. They become `activeCams[0]`, `activeCams[1]`.
2. **Score every cross-view pair** between those two cameras with a fast
   `epipolar_score` (`exp(-mean_perp_dist / 10)`).
3. **Hungarian** on the resulting `(n_a, n_b)` cost matrix
   (`cost = -score`).
4. **Refine the top candidates** by recomputing with the full
   `cross_view_score = 0.4·epipolar + 0.6·reprojection_OKS`.
5. **Filter**:
   - Constrained: take the top `num_animals`.
   - Unconstrained: keep matches with `score > 0.05`.
6. **Solo padding** (unconstrained only): each instance of cam1 / cam2
   not matched in step 4 becomes its own singleton group.
7. **Graft remaining cameras** in order. For each subsequent camera, build
   a `(n_groups, n_insts_c3)` cost matrix where each group's row is the
   reprojection distance from its 3-D triangulated pose to the new
   camera's instances. Hungarian, threshold `< 100 px` to graft. Add
   unmatched instances as new singletons (unconstrained).

```python
def match_pairwise(frame_group, session, caches, num_animals=0, prev_assignments=None):
    cam_instances, cam_map, active_cams = collect_instances(frame_group, session.cameras)

    cams_by_count = sorted(active_cams, key=lambda c: -len(cam_instances[c]))
    best1, best2 = cams_by_count[0], cams_by_count[1]
    ...

    # Fast epipolar score matrix
    score_matrix = np.zeros((n_a, n_b))
    for a in range(n_a):
        for b in range(n_b):
            s = epipolar_score(insts1[a], cam1, insts2[b], cam2, caches)
            if prev_assignments is not None:
                # +0.3 bonus when both tracks were the same identity last frame
                pid_a = prev_assignments.get(f"{best1}:{insts1[a].track_idx}")
                pid_b = prev_assignments.get(f"{best2}:{insts2[b].track_idx}")
                if pid_a is not None and pid_b is not None and pid_a == pid_b:
                    s += 0.3
            score_matrix[a, b] = s

    assignment = hungarian(-score_matrix)

    # Refine top-N with full reprojection score, then filter
    ...
    if num_animals:
        matches = matches[:num_animals]
    else:
        matches = [m for m in matches if m["score"] > 0.05]

    # Build groups, then solo padding, then cam-graft
    ...
```

Critically: **every visible `(cam, track)` at the frame ends up in at
least one group output by `match_pairwise`**. Either as a member of a
multi-camera matched group, or as a 1-camera singleton.

### 1.3 `reorder_groups_by_prev`

To keep identity labels stable over time, this step Hungarian-aligns
the current frame's groups to the **previous frame's `targets3d`** using
four signals (each normalized to a `[0, 1]`-ish range, then averaged):

| Signal | Meaning |
|---|---|
| (1) Reprojection of prev 3-D into current view | `exp(-mean_px_dist/50)` |
| (2) 3-D Euclidean distance between prev and current keypoints | `exp(-mean_3d_dist/30)` |
| (3) Cross-view OKS between current 3-D reprojected into prev view and prev's 2-D | OKS-Gaussian σ = 50 px |
| (4) Track-identity continuity (2× weight) | fraction of `(cam, track)` in this group whose prev assignment matches this target's identity |

Cost matrix is `(n, n)` where `n = max(n_t, n_g)`, **padded to a square
with cost = 1000 for invalid `(ti ≥ n_t)` or `(gi ≥ n_g)` cells**:

```python
def reorder_groups_by_prev(groups, prev_targets3d, cam_map, prev_assignments, caches):
    n_t = len(prev_targets3d)
    n_g = len(groups)
    n = max(n_t, n_g)
    ...
    cost = np.full((n, n), 1000.0)
    for ti in range(n_t):
        for gi in range(n_g):
            ...
            cost[ti, gi] = -(score / score_count) if score_count > 0 else 0.0

    assignment = hungarian(cost)

    # Faithful port of JS reorderGroupsByPrevTargets (tracker.js:357–364):
    #
    #   var usedGroups = new Set(assignment.filter(g => g >= 0 && g < nGroups));
    #
    # When n_g > n_t the padded targets (ti >= n_t) consume real groups
    # via their cost=1000 rows. JS treats those padded-claim groups as
    # "used" so the leftover-append loop SKIPS them → those groups get
    # silently dropped from `reordered`.
    n = len(assignment)
    used = {int(assignment[i]) for i in range(n)
            if 0 <= int(assignment[i]) < n_g}

    reordered = []
    for ti in range(n_t):
        gi = assignment[ti]
        reordered.append(groups[gi] if 0 <= gi < n_g else {})
    for gi in range(n_g):
        if gi not in used:
            reordered.append(groups[gi])
    return reordered
```

The drop is **silent**: groups that were perfectly valid outputs of
`match_pairwise` are quietly removed if they were unlucky enough to be
the column the Hungarian assigned to a padded row.

### 1.4 Identity assignment loop

For each surviving group, in order:

```python
for g_i, group in enumerate(groups):
    if not group:
        continue

    identity = None

    # Vote phase — pick the prev identity most-supported by this group
    if prev_assignments is not None:
        votes = {}
        for cn, inst in group.items():
            pid = prev_assignments.get(f"{cn}:{inst.track_idx}")
            if pid is not None:
                votes[pid] = votes.get(pid, 0) + 1
        # Best vote that isn't already used at this frame
        best_id, best_vote = None, -1
        for vid, cnt in votes.items():
            if cnt > best_vote and vid not in used_ids:
                best_vote = cnt
                best_id = vid
        if best_id is not None:
            identity = session.get_identity(best_id)

    # Fallback: first identity in the first-N pool not yet used
    if identity is None:
        max_id = num_animals if num_animals else len(session.identities)
        for ei in range(min(max_id, len(session.identities))):
            if session.identities[ei].id not in used_ids:
                identity = session.identities[ei]
                break

    # Unconstrained: spawn a fresh identity
    if identity is None and not num_animals:
        identity = session.add_identity(f"id_{len(session.identities)}")

    if identity is None:
        continue

    used_ids.add(identity.id)
    targets3d[g_i]["identityId"] = identity.id

    # ★ THE WRITES — only fire for groups that survived reorder ★
    for cn, inst in group.items():
        if inst.track_idx is None:
            continue
        session.track_identity_map[f"{cn}:{inst.track_idx}"] = identity.id
        if per_frame:
            session.set_frame_identity(frame_group.frame_idx, cn,
                                       inst.track_idx, identity.id)
        assignments[f"{cn}:{inst.track_idx}"] = identity.id
```

`used_ids` enforces uniqueness **across groups within this frame**, but
**only for groups that survived reorder**. Tracks whose group was dropped
get *no* writes at all — no per-frame override, no fresh global, nothing.
Their global mapping retains whatever some earlier frame happened to set
it to.

---

## 2. The duplicate-identity bug

### 2.1 Symptom

In the LUCID web viewer, when coloring by identity, two skeletons in the
**same camera view at the same frame** render in the **same identity
color**. Example from the user's session (no `project.slp`, frame 148,
camera `midL`):

```
midL:track_0  →  id_3   (rendered red)
midL:track_3  →  id_3   (rendered red)   ← duplicate
```

### 2.2 Reproduction in Python

The bug was reproduced end-to-end in
`luc3d_tracker_helper.py` (`enforce_uniqueness=False`). 33 frames in
`10072022145420_small` exhibit at least one per-camera identity collision
in the viewer's lookup path.

```python
import luc3d_tracker_helper as lt
lt.track_all(session, num_animals=None, enforce_uniqueness=False)

# Audit duplicates via the same path the viewer uses
dups = 0
for fi in sorted(session.frame_groups):
    fg = session.frame_group(fi)
    for cam, insts in fg.instances.items():
        by_id = {}
        for inst in insts:
            if inst.track_idx is None:
                continue
            ident = session.get_identity_for_track(inst.track_idx, cam, fi)
            if ident is None:
                continue
            by_id.setdefault(ident.id, []).append(inst.track_idx)
        for gid, ts in by_id.items():
            if len(ts) > 1:
                dups += 1
print(dups)
# → 33
```

### 2.3 Root cause walk-through (frame 148)

| Stage | Output |
|---|---|
| `match_pairwise` at frame 148 | 4 groups produced. One of them, call it `G_X`, contains `{midL: track_0, ...}`. |
| `reorder_groups_by_prev` cost matrix | `(n_t=3, n_g=4)` padded to `(4, 4)`. Padded row `ti=3` has cost 1000 everywhere. |
| Hungarian | Padded `ti=3` claims `G_X`. `used = {…, G_X}`. |
| Leftover-append loop | Skips `G_X` (it's in `used`). **`G_X` is dropped.** |
| Identity-assignment loop | Iterates kept groups only. `midL:0` is in `G_X` → **no per-frame override written**. |

State at end of frame 148 processing:

```
frame_identity_map["148:midL:0"]   →  (absent)
frame_identity_map["148:midL:1"]   →  0
frame_identity_map["148:midL:3"]   →  3

track_identity_map["midL:0"]       →  3   (set by some EARLIER frame; never updated this frame)
track_identity_map["midL:1"]       →  0
track_identity_map["midL:3"]       →  1
```

Viewer lookups at frame 148:

```
get_identity_for_track(track_idx=0, cam="midL", frame=148)
  step 1: "148:midL:0" not in frame_identity_map  → fall through
  step 2: "midL:0" in track_identity_map → returns id_3   ← STALE GLOBAL

get_identity_for_track(track_idx=3, cam="midL", frame=148)
  step 1: "148:midL:3" → returns id_3                    ← CORRECT
```

Both tracks resolve to `id_3` → visual duplicate.

### 2.4 Why `usedIds` does not prevent this

`usedIds` only sees groups that survive reorder. The dropped group's
identity-assignment phase **never runs**, so its track's old global
value is the one the viewer encounters. There is no point in the
algorithm where the question "is this `(cam, track)` at this frame
unique within its camera?" is asked.

### 2.5 Why `getIdentityIdForTrack` audits looked clean earlier

`getIdentityIdForTrack` / `get_identity_id_for_track` was being run
against the **per-frame map only**, and didn't reveal the stale-global
fallback. The viewer's actual color path is `getIdentityForTrack`, which
falls through to the global map. Auditing through that function
surfaces the duplicate.

### 2.6 Confirmed in JS console

Probe of frame 148 midL in the live JS viewer:

```js
const out = [];
for (const t of [0, 1, 3]) {
  out.push({
    track: t,
    perFrame: s.frameIdentityMap.has(`148:midL:${t}`)
              ? s.frameIdentityMap.get(`148:midL:${t}`) : '(none)',
    global:   s.trackIdentityMap.has(`midL:${t}`)
              ? s.trackIdentityMap.get(`midL:${t}`) : '(none)'
  });
}
out
```

Returned:

```
{ track: 0, perFrame: '(none)', global: 3 }
{ track: 1, perFrame: 0,        global: 0 }
{ track: 3, perFrame: 3,        global: 1 }
```

Byte-identical to Python's reproduction.

---

## 3. Proposed solution

### 3.1 The fix

At the end of `matchFrameInstances` / `match_frame_instances`, after the
per-group writes, sweep every visible `(cam, track)` at this frame. For
any without a per-frame override, write an explicit **uniqueness
sentinel** (`-1`). The viewer's identity lookup short-circuits on this
sentinel — returning `null` instead of falling through to the global —
so the affected track renders in track-color and never collides with
another track's identity-color.

The algorithm is **otherwise unchanged**: `reorder_groups_by_prev` still
drops the same groups, identity counts remain identical, performance
impact is negligible (one extra `O(n_visible)` sweep per frame).

### 3.2 Python implementation

`luc3d_tracker_helper.py:match_frame_instances` — guard inserted after
the per-group writes, before `return`:

```python
# ------------------------------------------------------------------
# Per-frame uniqueness guard (THE FIX)
# ------------------------------------------------------------------
# When reorder_groups_by_prev drops a group claimed by a padded
# Hungarian row, the (cam, track) entries in that dropped group never
# got per-frame overrides written above. The viewer's identity-color
# path then falls back to track_identity_map (global), which still
# holds stale values from earlier frames. If that stale value happens
# to coincide with an identity already used by another track in the
# same camera at this frame, two skeletons render in the same color.
#
# Fix: for every visible (cam, track) at this frame that did not get
# a per-frame override above, write an explicit sentinel (-1). The
# viewer's get_identity_for_track / get_identity_id_for_track paths
# short-circuit on this value (returning None), so the track renders
# in track-color instead of colliding.
#
# Set enforce_uniqueness=False to disable this guard and faithfully
# reproduce the JS bug for debugging / before-after comparisons.
if per_frame and enforce_uniqueness:
    fi = frame_group.frame_idx
    written = set(assignments.keys())  # "cam:track" strings
    for cam, insts in frame_group.instances.items():
        for inst in insts:
            if inst.track_idx is None:
                continue
            key = f"{cam}:{inst.track_idx}"
            if key in written:
                continue
            # Only write the sentinel when there's a global value that
            # would *otherwise* leak in via fallback. If no global value,
            # the lookup naturally returns None — no sentinel needed.
            if key in session.track_identity_map:
                session.set_frame_identity(fi, cam, inst.track_idx, -1)
    # Cover unlinked instances too (rare in SLP imports but possible).
    for cam, ul_list in frame_group.unlinked_instances.items():
        for ul in ul_list:
            if ul.instance.track_idx is None:
                continue
            key = f"{cam}:{ul.instance.track_idx}"
            if key in written:
                continue
            if key in session.track_identity_map:
                session.set_frame_identity(fi, cam, ul.instance.track_idx, -1)
```

`gui_source/pose_data.py` — lookup updated to honor the sentinel:

```python
def get_identity_id_for_track(
    self, frame_idx: int, camera_name: str, track_idx: int | None
) -> int | None:
    """Strict lookup: per-frame override → global (cam, track) → None.

    A per-frame override of `-1` is the uniqueness sentinel written
    by match_frame_instances to suppress fallback to a stale global.
    It means "no identity at this frame for this (cam, track)".
    """
    if track_idx is None:
        return None
    per_frame_key = f"{frame_idx}:{camera_name}:{track_idx}"
    if per_frame_key in self.frame_identity_map:
        v = self.frame_identity_map[per_frame_key]
        return None if v is None or v < 0 else v
    return self.track_identity_map.get(f"{camera_name}:{track_idx}")


def get_identity_for_track(
    self, track_idx, camera_name=None, frame_idx=None,
):
    """Permissive lookup mirroring JS getIdentityForTrack (used by the
    overlay renderer)."""
    if track_idx is None:
        return None
    if frame_idx is not None and camera_name is not None:
        frame_key = f"{frame_idx}:{camera_name}:{track_idx}"
        if frame_key in self.frame_identity_map:
            v = self.frame_identity_map[frame_key]
            if v is None or v < 0:
                return None             # sentinel — explicit "no identity"
            return self.get_identity(v)
    # … direct global, then cross-camera suffix-scan (JS-equivalent fallback)
```

### 3.3 Verification on `10072022145420_small`

```
[BUG MODE   — enforce_uniqueness=False]  identities=4   viewer-dup frames=33
[FIX MODE   — enforce_uniqueness=True ]  identities=4   viewer-dup frames=0
```

Frame 148 midL specifically:

```
midL:0  perFrame=-1  global=3   →  viewer: (no identity, track color)
midL:1  perFrame=0   global=0   →  viewer: id_0
midL:3  perFrame=3   global=1   →  viewer: id_3   ← unique
```

### 3.4 JS port

Two-file change in `luc3d/`:

#### `luc3d/pose/tracker.js` — append the guard inside `matchFrameInstances`

Insert after the existing per-group identity-write loop (before `return`):

```js
// ---- Per-frame uniqueness guard ----
// See lucid_lite/tracking_audit.md §2 for the root-cause analysis.
// Reorder may drop a real group claimed by a padded Hungarian row;
// the (cam, track) entries in that dropped group never get per-frame
// overrides, so the viewer falls back to a stale global value. If two
// tracks at the same camera at this frame resolve to the same identity
// via different paths (one per-frame, one stale-global), the viewer
// draws them in the same color.
//
// Fix: write an explicit sentinel (-1) per-frame override for every
// visible (cam, track) at this frame that didn't get one above.
// getIdentityForTrack / getIdentityIdForTrack short-circuit on this
// value so the track renders in track-color rather than colliding.
if (opts.perFrame && session.setFrameIdentity) {
    var SENTINEL_NO_IDENTITY = -1;
    var written = new Set(assignments.keys());
    for (var ci = 0; ci < cameras.length; ci++) {
        var camName = cameras[ci].name;
        var linked = frameGroup.getInstances(camName);
        if (linked) {
            for (var li = 0; li < linked.length; li++) {
                var inst = linked[li];
                if (inst.trackIdx == null) continue;
                var key = camName + ':' + inst.trackIdx;
                if (written.has(key)) continue;
                if (session.trackIdentityMap.has(key)) {
                    session.setFrameIdentity(fi, camName, inst.trackIdx,
                                             SENTINEL_NO_IDENTITY);
                }
            }
        }
        var unlinked = frameGroup.getUnlinkedInstances(camName);
        if (unlinked) {
            for (var ui = 0; ui < unlinked.length; ui++) {
                var ul = unlinked[ui];
                if (ul.instance.trackIdx == null) continue;
                var ukey = camName + ':' + ul.instance.trackIdx;
                if (written.has(ukey)) continue;
                if (session.trackIdentityMap.has(ukey)) {
                    session.setFrameIdentity(fi, camName, ul.instance.trackIdx,
                                             SENTINEL_NO_IDENTITY);
                }
            }
        }
    }
}
```

#### `luc3d/pose/pose-data.js` — short-circuit the sentinel in both lookups

```js
getIdentityForTrack(trackIdx, cameraName, frameIdx) {
    // Check per-frame override first
    if (frameIdx != null && cameraName) {
        var frameKey = frameIdx + ':' + cameraName + ':' + trackIdx;
        if (this.frameIdentityMap.has(frameKey)) {
            var frameIdVal = this.frameIdentityMap.get(frameKey);
            // -1 is the uniqueness sentinel — explicit "no identity at
            // this frame, do not fall back to global or suffix-scan."
            if (frameIdVal == null || frameIdVal < 0) return null;
            return this.getIdentity(frameIdVal);
        }
    }
    // … existing logic for per-frame-without-camera, direct global,
    // cross-camera suffix-scan …
}

getIdentityIdForTrack(cameraName, trackIdx, frameIdx) {
    if (frameIdx != null) {
        var frameKey = frameIdx + ':' + cameraName + ':' + trackIdx;
        if (this.frameIdentityMap.has(frameKey)) {
            var frameIdVal = this.frameIdentityMap.get(frameKey);
            if (frameIdVal == null || frameIdVal < 0) return null;
            return frameIdVal;
        }
    }
    var globalVal = this.trackIdentityMap.get(cameraName + ':' + trackIdx);
    return globalVal != null ? globalVal : null;
}
```

### 3.5 Why not "fix the root cause" instead?

The "root cause" — `reorder_groups_by_prev` dropping a real high-scoring
group in favor of a singleton — is itself a downstream effect of the
4-signal Hungarian cost surface. Investigating that is a separate effort
because:

1. The drop is **algorithmically correct** under the current cost
   formulation: padded rows have constant cost 1000, and when ties exist
   among optima, the JV solver picks one. The "wrong" group dropped is
   simply Hungarian's tie-break choice.
2. Keeping all groups (i.e., reverting the drop) **changes the number of
   identities** the unconstrained tracker creates: from 4 (matching JS)
   to 8 (creating one extra transient identity per dropped-singleton
   frame). That violates the algorithm's documented contract that
   `num_animals=None` should infer the correct count from the densest
   frame.
3. The drop is **rare** in practice (33 / 1800 ≈ 1.8% of frames) and
   has **no effect on the global identity assignment** — only on the
   per-frame coverage. The sentinel guard fully restores correct
   rendering without disturbing identity counts or 3-D triangulation.

If a future investigation does revisit the cost formulation, the
sentinel guard remains a useful invariant: it guarantees that
**per-frame uniqueness within a camera holds, regardless of what reorder
chooses to drop**.

### 3.6 Regression test surface

The Python port retains both modes for regression testing:

```python
# Reproduce the JS bug exactly (33 duplicate-frame audit)
lt.track_all(session, num_animals=None, enforce_uniqueness=False)

# Apply the fix (default; 0 duplicates)
lt.track_all(session, num_animals=None, enforce_uniqueness=True)
```

The LUCID-Lite GUI calls `track_all` with the default, so the **Track
Frames** button now produces visually-correct identities. The dialog
does not expose the toggle; toggling is reserved for notebook-based
debugging.

### 3.7 Other notes that came out of this audit

The following are unrelated to the duplicate bug but were uncovered
during the investigation and are recorded here for completeness:

- **JS Hungarian tie-breaking ≠ scipy's.** The Python port replaces
  `scipy.optimize.linear_sum_assignment` with a direct translation of
  `triangulation.js:hungarianAlgorithm` (Jonker-Volgenant with
  pad-to-square zero fill, `SENTINEL = 1e15` for non-finite entries).
  Without this, the bootstrap pairwise matches and graft assignments
  occasionally diverge, producing identity counts off by 1–2.

- **DLT triangulation is vectorized.** `triangulate_group` now builds
  one `(n_kp, 2C, 4)` tensor per group and dispatches a single batched
  `np.linalg.svd`, replacing per-keypoint SVDs. Numerically identical to
  the per-keypoint loop (padding zero rows leaves `AᵀA` unchanged) and
  ~30 % faster on the full sweep.

- **Session signal batching.** `Session.batch_updates()` suppresses
  per-mutation Qt signals during `track_all` and emits each unique
  signal once on exit. Mirrors the JS pattern of writing to
  `Map`/`trackIdentityMap` without observers; without batching, the
  GUI's assignment panel rebuilt ~22 `QComboBox`es per identity write
  and dominated the sweep time (89 % of total).

- **`project.slp` loader.** The JS app reads the root-level
  `project.slp` directly. The lucid-lite loader was updated to do the
  same when the file is present; falls back to per-camera
  `.analysis.h5` exports otherwise. Did not affect the duplicate bug —
  `side/sideL` cameras have zero labeled frames in this dataset's
  `project.slp` too, so the input is identical either way.

---

## 4. File map

```
lucid_lite/
├── luc3d_tracker_helper.py
│     ├── match_pairwise             (§1.2 / lines 437–579)
│     ├── reorder_groups_by_prev     (§1.3 / lines 692–810)
│     ├── match_frame_instances      (§1.4 + FIX / lines 813–965)
│     ├── track_all                  (lines 966–1043)
│     ├── hungarian                  (§3.7 / lines 393–566)
│     ├── triangulate_group          (§3.7 / lines 353–390)
│     └── triangulate_dlt_batch      (§3.7 / lines 121–172)
│
└── gui_source/
      └── pose_data.py
            ├── get_identity_id_for_track   (§3.2 — fix)
            ├── get_identity_for_track      (§3.2 — fix)
            └── Session.batch_updates       (§3.7)

luc3d/pose/
├── tracker.js                       (port target for §3.4)
└── pose-data.js                     (port target for §3.4)
```
