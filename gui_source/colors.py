"""Shared color palette — matches overlays.js TRACK_COLORS (Green-Armytage 20-color)."""

TRACK_COLORS = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#5b8def", "#f38181",
    "#aa96da", "#fcbad3", "#a8d8ea", "#ffaaa5", "#98dfea",
    "#c3aed6", "#ffb6b9", "#bbded6", "#fae3d9", "#8ac6d1",
    "#ff9a76", "#b8e0d2", "#d6eadf", "#eac4d5", "#95b8d1",
]
# Index 3 changed from "#95e1d3" (mint) -> "#5b8def" (medium blue) so it no
# longer clashes with index 1 (#4ecdc4 teal). All other entries are the
# Green-Armytage 20-color palette mirrored from overlays.js:11–32.


def get_track_color(track_idx: int | None) -> str:
    if track_idx is None or track_idx < 0:
        return "#888888"
    return TRACK_COLORS[track_idx % len(TRACK_COLORS)]


def get_identity_color(identity_id: int | None, fallback_track_idx: int | None = None) -> str:
    if identity_id is None or identity_id < 0:
        return get_track_color(fallback_track_idx)
    return TRACK_COLORS[identity_id % len(TRACK_COLORS)]


def next_palette_color(n_existing: int) -> str:
    return TRACK_COLORS[n_existing % len(TRACK_COLORS)]
