"""Stub for external LucidLabels identity-assignment payloads.

Upstream schema is TBD — this module reserves the binding point. The GUI wires
a "File → Load Identity Assignments…" menu item that calls load_lucid_labels;
until the schema lands, it surfaces NotImplementedError in the status bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pose_data import Session


@dataclass
class FrameIdentityAssignment:
    frame_idx: int
    camera_name: str
    track_idx: int
    identity_id: int


class LucidLabels(Protocol):
    """Future per-frame identity-label payload. Fields TBD."""

    def get_frame_assignments(self) -> list[FrameIdentityAssignment]: ...


def load_lucid_labels(session: "Session", labels: LucidLabels) -> None:
    raise NotImplementedError(
        "LucidLabels binding pending upstream definition. "
        "Expected: LucidLabels.get_frame_assignments() -> list[FrameIdentityAssignment]."
    )
