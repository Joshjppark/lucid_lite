"""Pose overlay drawing — skeleton edges + nodes + optional label text.

Mirrors overlays.js drawSkeleton; read-only (no drag previews, no selection halos).
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from colors import get_identity_color, get_track_color
from pose_data import FrameGroup, Instance, Session


# Fallback defaults if a Session is unavailable. Live values come from
# Session.node_size / Session.edge_width and are controlled by the sliders
# in the Track/ID tab.
NODE_RADIUS = 4
EDGE_WIDTH = 2.0


def draw_overlay_for_camera(
    painter: QPainter,
    session: Session,
    frame_group: FrameGroup | None,
    camera_name: str,
    current_frame_idx: int,
    video_to_panel,  # callable (x, y) -> QPointF
) -> None:
    if frame_group is None or session.skeleton is None:
        return

    node_radius = float(getattr(session, "node_size", NODE_RADIUS))
    edge_width = float(getattr(session, "edge_width", EDGE_WIDTH))
    show_labels = bool(getattr(session, "show_identity_labels", True))

    # Instances attached to this camera (linked via track_idx).
    #
    # Identity-mode color is resolved through `get_identity_id_for_track`
    # (strict per-frame → global lookup). Note that the JS viewer uses
    # `getGroupColor` which checks `group.identityId` first — that
    # surfaces a real JS bug when the SLP file has stale `identity_idx`
    # values saved into instance_groups: track_all updates the per-frame
    # and global maps but does NOT clear `group.identityId`, so the
    # viewer can still render duplicate identities at frames where the
    # SLP had per-frame collisions baked in (eg. frame 149 midL on
    # 10072022145420_small). Python doesn't reproduce that because our
    # session_loader builds InstanceGroups fresh with identity_id=None.
    # Fix applied in tracker.js is the wipe of group.identityId at the
    # start of trackAll (see luc3d issue documented in commit msg).
    for inst in frame_group.get_instances(camera_name):
        ident_id = session.get_identity_id_for_track(
            current_frame_idx, camera_name, inst.track_idx
        )
        ident = session.get_identity(ident_id)
        if session.color_mode == "identity":
            color_hex = ident.color if ident else "#888888"
        else:
            color_hex = get_track_color(inst.track_idx)
        _draw_instance(painter, inst, session.skeleton, color_hex, video_to_panel,
                       label=_instance_label(inst, ident) if show_labels else "",
                       node_radius=node_radius, edge_width=edge_width)

    # Unlinked instances (no track yet) — rare in SLP imports but we draw them
    # so they remain visible if later added.
    for ul in frame_group.unlinked_instances.get(camera_name, []):
        if session.color_mode == "identity":
            color_hex = "#888888"
        else:
            color_hex = get_track_color(ul.instance.track_idx)
        _draw_instance(painter, ul.instance, session.skeleton, color_hex,
                       video_to_panel, label="unlinked",
                       node_radius=node_radius, edge_width=edge_width)


def _instance_label(inst: Instance, ident) -> str:
    # Only show identity names. Track-index fallbacks (eg. "t0", "t1") and
    # the "?" placeholder used to render as overlay text; both are now
    # suppressed by returning an empty string, which `_draw_instance` skips
    # via the `if anchor is not None and label` guard below.
    if ident is not None:
        return ident.name
    return ""


def _draw_instance(painter, inst: Instance, skeleton, color_hex: str,
                   video_to_panel, label: str,
                   node_radius: float = NODE_RADIUS,
                   edge_width: float = EDGE_WIDTH) -> None:
    color = QColor(color_hex)
    pen = QPen(color, edge_width)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)

    # Edges
    pts_panel: list[QPointF | None] = []
    for p in inst.points:
        pts_panel.append(video_to_panel(p[0], p[1]) if p is not None else None)

    for src, dst in skeleton.edges:
        if src < len(pts_panel) and dst < len(pts_panel):
            a, b = pts_panel[src], pts_panel[dst]
            if a is not None and b is not None:
                painter.drawLine(a, b)

    # Nodes
    painter.setBrush(color)
    painter.setPen(QPen(QColor("#000000"), 1))
    r = max(0.5, float(node_radius))
    for pp in pts_panel:
        if pp is not None:
            painter.drawEllipse(pp, r, r)

    # Label at first visible node
    anchor = next((p for p in pts_panel if p is not None), None)
    if anchor is not None and label:
        painter.setPen(QPen(QColor("#ffffff"), 1))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(anchor + QPointF(6, -6), label)
