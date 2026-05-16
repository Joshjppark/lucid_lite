"""Embedded-kernel comms between the lucid-lite GUI and a Jupyter notebook.

See prompts/plans/gui-notebook-comms.md for the design.

Two responsibilities:

1. Snapshot helpers (InstanceRecord / FrameRecord / FrameBundle + session_to_*
   functions). Optional — notebooks can also touch Session directly.
2. start_embedded_kernel(user_ns) — spawns an ipykernel on a daemon thread so
   a notebook (or `jupyter console --existing`) can attach and share this
   process's interpreter. The Qt event loop keeps running on the main thread.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

from pose_data import Instance, Session


# ---------------------------------------------------------------------------
# Snapshot dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InstanceRecord:
    points: list[Optional[tuple[float, float]]]
    track_idx: Optional[int]
    identity_id: Optional[int]
    score: float
    type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FrameRecord:
    video_id: str
    frame_idx: int
    instances: list[InstanceRecord] = field(default_factory=list)
    unlinked_instances: list[InstanceRecord] = field(default_factory=list)


@dataclass
class FrameBundle:
    """All FrameRecords at one frame_idx, keyed by camera name."""
    frame_idx: int
    records: dict[str, FrameRecord] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def instance_to_record(
    session: Session,
    inst: Instance,
    frame_idx: int,
    camera_name: str,
) -> InstanceRecord:
    identity_id = session.get_identity_id_for_track(
        frame_idx, camera_name, inst.track_idx
    )
    return InstanceRecord(
        points=list(inst.points),
        track_idx=inst.track_idx,
        identity_id=identity_id,
        score=inst.score,
        type=inst.type,
        metadata={},
    )


def session_to_frame_record(
    session: Session,
    frame_idx: int,
    camera_name: str,
) -> FrameRecord:
    fg = session.frame_group(frame_idx)
    record = FrameRecord(video_id=camera_name, frame_idx=frame_idx)
    if fg is None:
        return record
    for inst in fg.get_instances(camera_name):
        record.instances.append(instance_to_record(session, inst, frame_idx, camera_name))
    for ul in fg.unlinked_instances.get(camera_name, []):
        record.unlinked_instances.append(
            instance_to_record(session, ul.instance, frame_idx, camera_name)
        )
    return record


def session_to_bundle(session: Session, frame_idx: int) -> FrameBundle:
    """Snapshot every camera's FrameRecord at one frame_idx."""
    bundle = FrameBundle(frame_idx=frame_idx)
    for cam in session.camera_names():
        bundle.records[cam] = session_to_frame_record(session, frame_idx, cam)
    return bundle


def session_to_bundles(
    session: Session,
    start: int | None = None,
    end: int | None = None,
) -> list[FrameBundle]:
    """Inclusive [start, end] range; defaults to the whole session."""
    lo = session.min_frame if start is None else start
    hi = session.max_frame if end is None else end
    return [session_to_bundle(session, i) for i in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# Embedded-kernel launcher
# ---------------------------------------------------------------------------

def start_embedded_kernel(user_ns: dict) -> threading.Thread:
    """Start an ipykernel on a daemon background thread.

    The kernel shares this process's interpreter, so notebook cells see the
    same `session`, `window`, etc. passed in `user_ns`. Qt continues to own
    the main thread.

    Returns the kernel thread. The connection-file path is printed to stderr
    once the kernel is up. Attach via `jupyter console --existing`, a
    notebook's "Existing kernel" picker, or `%connect_info`.
    """
    ready = threading.Event()
    conn_holder: dict[str, str] = {}

    def run() -> None:
        # ipykernel uses asyncio; new loop per worker thread.
        asyncio.set_event_loop(asyncio.new_event_loop())

        from ipykernel.kernelapp import IPKernelApp

        app = IPKernelApp.instance()
        # IPKernelApp.init_signal calls signal.signal, which only works on the
        # main thread. We're on a worker thread; disable it.
        app.init_signal = lambda: None
        app.initialize([])
        # Inject user namespace via the shell — in ipykernel 7 `app.kernel` is
        # not reliably populated until `app.start()` has entered its async
        # main loop, but `app.shell` is available right after `initialize()`.
        app.shell.push(user_ns)
        conn_holder["path"] = app.connection_file
        ready.set()
        try:
            app.start()
        except Exception as exc:  # kernel loop crashed
            print(f"[comms] embedded kernel exited: {exc!r}", file=sys.stderr)

    thread = threading.Thread(target=run, name="lucid-embedded-kernel", daemon=True)
    thread.start()
    # Wait briefly for connection_file to exist so we can print it.
    if ready.wait(timeout=5.0):
        path = conn_holder.get("path", "<unknown>")
        print(
            f"[comms] embedded Jupyter kernel running\n"
            f"[comms]   connection file: {path}\n"
            f"[comms]   attach with: jupyter console --existing {path}",
            file=sys.stderr,
        )
    else:
        print("[comms] embedded kernel did not report ready within 5s",
              file=sys.stderr)
    return thread
