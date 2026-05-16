"""Simple dialogs for creating a new Identity or Track."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from colors import next_palette_color
from pose_data import Session


class NewIdentityDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Identity")
        self.session = session

        default_name = f"id_{len(session.identities)}"
        default_color = next_palette_color(len(session.identities))

        self.name_edit = QLineEdit(default_name)
        self._color = default_color
        self.color_btn = QPushButton()
        self.color_btn.setFixedWidth(64)
        self._apply_color_btn_style()
        self.color_btn.clicked.connect(self._pick_color)

        color_row = QWidget()
        crow = QHBoxLayout(color_row)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.addWidget(self.color_btn)
        crow.addWidget(QLabel(self._color))
        self._color_label = crow.itemAt(1).widget()

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Color:", color_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self, "Pick Identity Color")
        if c.isValid():
            self._color = c.name()
            self._apply_color_btn_style()
            self._color_label.setText(self._color)

    def _apply_color_btn_style(self) -> None:
        self.color_btn.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #333; min-height: 20px;"
        )

    def result_identity_spec(self) -> tuple[str, str]:
        return self.name_edit.text().strip() or f"id_{len(self.session.identities)}", self._color


class NewTrackDialog(QDialog):
    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Track")
        self.session = session
        self.name_edit = QLineEdit(f"track_{len(session.tracks)}")

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def track_name(self) -> str:
        return self.name_edit.text().strip() or f"track_{len(self.session.tracks)}"
