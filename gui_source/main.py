"""
LUCID-Lite entry point.

Run:
    python main.py                            # launches with open-folder dialog
    python main.py /path/to/session_folder    # loads that session immediately
    python main.py /path/to/session_folder --comms
        # also starts an embedded Jupyter kernel so a notebook can attach
        # and share this process's Python interpreter. The kernel's
        # connection-file path is printed to stderr.

Two-step API for notebook use:
    labels = main.make_labels(folder, prefer='proofread')   # headless
    app, window = main.make_window(labels)                  # opens GUI

`make_labels` returns a `MultiviewLabels`, which the tracker can consume
directly (no GUI required). Pass it back to `make_window` only if you want
to visualize tracking results in the lucid GUI.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget


def _error(parent: QWidget, title: str, message: str) -> None:
    print(f"[{title}] {message}", file=sys.stderr)
    box = QMessageBox(QMessageBox.Critical, title, message, QMessageBox.Ok, parent)
    box.exec()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lucid-lite")
    p.add_argument(
        "folder", nargs="?", default=None,
        help="Session folder to open (optional; a picker opens otherwise).",
    )
    p.add_argument(
        "--prefer", choices=("proofread", "predictions"), default="proofread",
        help="Per-cam label source preference (default: proofread).",
    )
    p.add_argument(
        "--comms", action="store_true",
        help="Start an embedded Jupyter kernel so a notebook can attach "
             "to this process (see prompts/plans/gui-notebook-comms.md).",
    )
    return p.parse_args(argv[1:])


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("LUCID-Lite")
    return app


def make_labels(
    folder: "str | Path | None" = None,
    *,
    prefer: str = "proofread",
    start: int | None = None,
    end: int | None = None,
    videos_dir: "str | Path | None" = None,
    verbose: bool = True,
):
    """Build a `MultiviewLabels` from a session folder. Headless — no Qt.

    Params:
      folder: session folder. If None, opens a folder picker (requires Qt).
      prefer: 'proofread' (default) or 'predictions' — which per-cam file to
              prefer. Falls back to the other if the preferred is missing.
      start, end: optional frame subset, half-open `[start, end)`. The GUI's
              playback / timeline / panel navigation derive their bounds
              from session.min_frame / session.max_frame, so these limits
              propagate everywhere automatically. The underlying mp4 files
              are NOT trimmed — the decoder can still seek any frame — only
              the GUI's reachable range is restricted.
      videos_dir: optional sibling folder to discover .mp4/.avi files from.
              Use this when `folder` is a labels-only mirror (e.g.
              slap_GT/<date>/<sid>/) without videos — point at the folder
              that holds them. Each cam subdir under `videos_dir` is
              searched for `*.mp4` then `*.avi`. If None, videos are
              auto-discovered inside `folder` (the default behavior).
      verbose: print the discovered file per cam + the resolved source_kind.

    Returns a `MultiviewLabels` (see josh_source.multiview_labels).
    """
    from josh_source.multiview_labels import MultiviewLabels

    if folder is None:
        # Need a Qt app + picker.
        app = _ensure_qapp()
        anchor = QWidget()
        picked = QFileDialog.getExistingDirectory(anchor, "Open LUCID Session Folder")
        if not picked:
            return None
        folder = picked

    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    return MultiviewLabels.from_folder(
        folder, prefer=prefer, start=start, end=end,
        videos_dir=videos_dir, verbose=verbose,
    )


def make_window(labels, *, comms: bool = False):
    """Open the lucid-lite GUI on top of a `MultiviewLabels`.

    Returns (app, window). The notebook caller MUST keep `window` alive (it
    owns the per-camera VideoPanelWidgets/FrameLoaderThreads — if it gets
    garbage-collected, in-flight decodes will crash).
    """
    app = _ensure_qapp()

    import comms as comms_mod
    from main_window import LucidLiteWindow

    window = LucidLiteWindow(labels.session)
    window.show()

    if comms:
        comms_mod.start_embedded_kernel({
            "app": app,
            "window": window,
            "session": labels.session,
            "labels": labels,
            "comms": comms_mod,
        })

    return app, window


def main(source: "list[str] | str | Path | object | None" = None, *,
         labels: "object | None" = None, comms: bool = False,
         prefer: str = "proofread"):
    """Launch the GUI.

    `source` is one of:
      * a `MultiviewLabels` (or anything with `.session`) — use as-is;
      * a CLI argv `list[str]` (e.g. ``["main.py", folder, "--comms"]``);
      * a folder `str`/`Path` — build labels from that folder via make_labels;
      * `None` — open a folder picker.

    `labels` is preserved for backwards compatibility — if supplied, the
    folder is still resolved for calibration + videos but detections come
    from the provided multi-video `sleap_io.Labels`. (Legacy path; prefer
    the two-step `make_labels` / `make_window` API.)
    """
    app = _ensure_qapp()
    anchor = QWidget()

    want_comms = comms
    folder: Path | None = None
    prefer_arg = prefer

    # MultiviewLabels (or anything with .session) passed in directly.
    if source is not None and hasattr(source, "session") \
            and not isinstance(source, (list, tuple, str, Path)):
        mv_labels = source
    else:
        # argv list / folder path / None
        if isinstance(source, (str, Path)):
            folder = Path(source).expanduser().resolve()
        else:
            args = _parse_args(source if source is not None else ["main.py"])
            want_comms = want_comms or args.comms
            prefer_arg = args.prefer
            if args.folder is not None:
                folder = Path(args.folder).expanduser().resolve()

        if folder is not None and not folder.is_dir():
            _error(anchor, "LUCID-Lite", f"Not a directory: {folder}")
            return 2

        if folder is None:
            picked = QFileDialog.getExistingDirectory(anchor, "Open LUCID Session Folder")
            if not picked:
                return 0
            folder = Path(picked)

        if labels is not None:
            # Legacy path: an sio.Labels was provided and the GUI builds a
            # session via the old labels module so we don't lose that hook.
            import labels as labels_mod
            try:
                session = labels_mod.build_session(folder, labels)
            except Exception as exc:
                traceback.print_exc()
                _error(anchor, "Load failed", f"{type(exc).__name__}: {exc}")
                return 1

            from main_window import LucidLiteWindow
            window = LucidLiteWindow(session)
            window.show()

            if want_comms:
                import comms as comms_mod
                comms_mod.start_embedded_kernel({
                    "app": app, "window": window,
                    "session": session, "comms": comms_mod,
                })
            return app, window

        try:
            mv_labels = make_labels(folder, prefer=prefer_arg, verbose=True)
        except Exception as exc:
            traceback.print_exc()
            _error(anchor, "Load failed", f"{type(exc).__name__}: {exc}")
            return 1

    return make_window(mv_labels, comms=want_comms)


if __name__ == "__main__":
    _result = main(sys.argv)
    if isinstance(_result, tuple):
        _app, _window = _result
        raise SystemExit(_app.exec())
    raise SystemExit(int(_result) if _result is not None else 0)
