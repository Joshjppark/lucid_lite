"""Tracker UI helpers — the 'Track Frames' dialog and a progress-driven runner.

`TrackFramesDialog` collects an optional `num_animals` from the user.
`run_track_all_with_progress` calls `luc3d_tracker_helper.track_all` and pumps
its progress callback into a `QProgressDialog` so the GUI stays responsive
and the user can cancel mid-sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressDialog, QVBoxLayout, QWidget,
)


class TrackFramesDialog(QDialog):
    """Tiny modal: 'Number of animals (optional)' + OK/Cancel.

    Result is exposed via `num_animals()` — `None` if left empty, else int.
    """

    def __init__(self, parent: QWidget | None = None,
                 default_value: Optional[int] = None):
        super().__init__(parent)
        self.setWindowTitle("Track Frames")
        self.setModal(True)

        root = QVBoxLayout(self)

        root.addWidget(QLabel(
            "Run identity assignment on every frame in the session."
        ))

        hint = QLabel(
            "Leave 'Number of animals' empty to let the tracker decide based "
            "on the densest frame."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        root.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Number of animals:"))
        # QLineEdit (not QSpinBox) so we can distinguish "empty" from 0.
        self.num_edit = QLineEdit()
        self.num_edit.setPlaceholderText("(auto)")
        self.num_edit.setMaximumWidth(80)
        if default_value is not None:
            self.num_edit.setText(str(int(default_value)))
        row.addWidget(self.num_edit)
        row.addStretch()
        root.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _on_accept(self) -> None:
        text = self.num_edit.text().strip()
        if text:
            try:
                v = int(text)
            except ValueError:
                QMessageBox.warning(
                    self, "Track Frames",
                    f"'{text}' isn't a valid integer.",
                )
                return
            if v < 1:
                QMessageBox.warning(
                    self, "Track Frames",
                    "Number of animals must be ≥ 1 (leave empty for auto).",
                )
                return
        self.accept()

    def num_animals(self) -> Optional[int]:
        text = self.num_edit.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None


def _ensure_helper_on_path() -> None:
    """`luc3d_tracker_helper.py` lives in the repo root, not in `gui_source/`.
    Add the repo root to `sys.path` so the import below resolves regardless
    of where the script was launched from."""
    here = Path(__file__).resolve().parent          # gui_source/
    repo_root = here.parent                          # lucid_lite/
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def run_track_all_with_progress(
    session,
    parent: QWidget | None,
    num_animals: Optional[int] = None,
    per_frame: bool = True,
    clear_existing: bool = True,
) -> Optional[list]:
    """Run `luc3d_tracker_helper.track_all` synchronously with a progress bar.

    Returns the history list on completion, `None` if the user cancelled or
    an error occurred (after surfacing it via QMessageBox).
    """
    _ensure_helper_on_path()
    try:
        import luc3d_tracker_helper as lt
    except Exception as exc:  # pragma: no cover
        QMessageBox.critical(
            parent, "Track Frames",
            f"Could not import luc3d_tracker_helper:\n{type(exc).__name__}: {exc}",
        )
        return None

    n_frames = len(session.frame_groups)
    if n_frames == 0:
        QMessageBox.information(
            parent, "Track Frames", "Session has no frames to track.",
        )
        return None

    progress = QProgressDialog(
        "Tracking frames…", "Cancel", 0, n_frames, parent,
    )
    progress.setWindowTitle("Track Frames")
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(True)
    progress.setAutoReset(True)
    progress.setValue(0)

    cancelled = {"flag": False}

    def on_progress(k: int, fi: int, result: dict) -> None:
        # Called once per processed frame. `setValue` pumps events, so the
        # Cancel button gets a chance to be clicked.
        progress.setValue(k + 1)
        n_ids = len(session.identities)
        n_assn = len(result.get("assignments") or [])
        progress.setLabelText(
            f"Tracking frame {fi}   ({k + 1}/{n_frames})\n"
            f"identities: {n_ids}    assignments: {n_assn}"
        )
        if progress.wasCanceled():
            cancelled["flag"] = True
            # Raise to break out of the long-running loop.
            raise _UserCancel()

    try:
        history = lt.track_all(
            session,
            num_animals=num_animals,
            per_frame=per_frame,
            on_progress=on_progress,
            clear_existing=clear_existing,
        )
    except _UserCancel:
        progress.close()
        QMessageBox.information(
            parent, "Track Frames",
            "Cancelled. Identities created up to this point are kept.",
        )
        return None
    except Exception as exc:
        progress.close()
        import traceback
        traceback.print_exc()
        QMessageBox.critical(
            parent, "Track Frames",
            f"{type(exc).__name__}: {exc}",
        )
        return None

    progress.setValue(n_frames)
    return history


class _UserCancel(Exception):
    """Sentinel raised from the progress callback to break out of track_all."""
    pass
