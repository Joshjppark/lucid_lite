"""Pose overlay drawing — skeleton edges + nodes + optional label text.

Mirrors overlays.js drawSkeleton; read-only (no drag previews, no selection halos).
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from colors import get_identity_color, get_track_color
from pose_data import FrameGroup, Instance, Session


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

    # Instances attached to this camera (linked via track_idx).
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
                       label=_instance_label(inst, ident))

    # Unlinked instances (no track yet) — rare in SLP imports but we draw them
    # so they remain visible if later added.
    for ul in frame_group.unlinked_instances.get(camera_name, []):
        if session.color_mode == "identity":
            color_hex = "#888888"
        else:
            color_hex = get_track_color(ul.instance.track_idx)
        _draw_instance(painter, ul.instance, session.skeleton, color_hex,
                       video_to_panel, label="unlinked")


def _instance_label(inst: Instance, ident) -> str:
    if ident is not None:
        return ident.name
    if inst.track_idx is not None:
        return f"t{inst.track_idx}"
    return "?"


def _draw_instance(painter, inst: Instance, skeleton, color_hex: str,
                   video_to_panel, label: str) -> None:
    color = QColor(color_hex)
    pen = QPen(color, EDGE_WIDTH)
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
    for pp in pts_panel:
        if pp is not None:
            painter.drawEllipse(pp, NODE_RADIUS, NODE_RADIUS)

    # Label at first visible node
    anchor = next((p for p in pts_panel if p is not None), None)
    if anchor is not None and label:
        painter.setPen(QPen(QColor("#ffffff"), 1))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(anchor + QPointF(6, -6), label)
