"""
LUCID-Lite entry point.

Run:
    python main.py                            # launches with open-folder dialog
    python main.py /path/to/session_folder    # loads that session immediately
    python main.py /path/to/session_folder --comms
        # also starts an embedded Jupyter kernel so a notebook can attach
        # and share this process's Python interpreter. The kernel's
        # connection-file path is printed to stderr.
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
        "--comms", action="store_true",
        help="Start an embedded Jupyter kernel so a notebook can attach "
             "to this process (see prompts/plans/gui-notebook-comms.md).",
    )
    return p.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # app = QApplication(argv)
    app.setApplicationName("LUCID-Lite")

    # Anchor widget so QMessageBox/QFileDialog always have a live parent —
    # parent=None before any top-level is shown segfaults PySide6 on macOS.
    anchor = QWidget()

    import comms
    from main_window import LucidLiteWindow
    from pose_data import Session

    folder: Path | None = None
    if args.folder is not None:
        folder = Path(args.folder).expanduser().resolve()
        if not folder.is_dir():
            _error(anchor, "LUCID-Lite", f"Not a directory: {folder}")
            return 2

    if folder is None:
        picked = QFileDialog.getExistingDirectory(anchor, "Open LUCID Session Folder")
        if not picked:
            return 0
        folder = Path(picked)

    try:
        session = Session.load_from_folder(folder)
    except Exception as exc:
        traceback.print_exc()
        _error(anchor, "Load failed", f"{type(exc).__name__}: {exc}")
        return 1

    window = LucidLiteWindow(session)
    window.show()

    if args.comms:
        comms.start_embedded_kernel({
            "app": app,
            "window": window,
            "session": session,
            "comms": comms,
        })

    # Always hand back (app, window). When invoked from a Jupyter notebook
    # with `%gui qt`, control returns to the caller immediately — if `window`
    # isn't kept alive by the caller it gets garbage-collected, child
    # VideoPanelWidgets (and their per-camera FrameLoaderThreads) die and
    # in-flight decodes crash on emit. Keep the window reference alive in
    # the notebook:
    # Notebook caller:  app, window = main.main([])    # keep `window`!
    # CLI caller below runs app.exec() itself.
    return app, window


if __name__ == "__main__":
    _result = main(sys.argv)
    if isinstance(_result, tuple):
        _app, _window = _result
        raise SystemExit(_app.exec())
    raise SystemExit(int(_result) if _result is not None else 0)
