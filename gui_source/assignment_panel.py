"""Right-sidebar panel: shows (camera, track) pairs in the current frame and
lets the user assign identities, create identities, and create tracks.

Mirrors the feature scope described in prompts/plans/lucid-lite.md §5.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from colors import get_track_color
from new_identity_dialog import NewIdentityDialog, NewTrackDialog
from pose_data import Session

NONE_ID = -1   # sentinel for "no identity"


class IdentityAssignmentPanel(QWidget):
    identityAssignmentChanged = Signal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_frame: int = session.min_frame
        self._populating: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._build_assignments_tab()
        self._build_track_id_tab()

        # Assignments-tab refresh: all four model signals feed _refresh.
        session.identity_map_changed.connect(self._refresh)
        session.identities_changed.connect(self._refresh)
        session.tracks_changed.connect(self._refresh)
        session.frame_groups_changed.connect(self._refresh)

        # Track/ID-tab legends refresh on their specific signals.
        session.tracks_changed.connect(self._refresh_tracks_legend)
        session.identities_changed.connect(self._refresh_identities_legend)

        # Color-by radios stay in sync with external changes.
        session.color_mode_changed.connect(self._sync_color_mode_ui)

        self._refresh()
        self._refresh_tracks_legend()
        self._refresh_identities_legend()
        self._sync_color_mode_ui(session.color_mode)

    # ---- tab builders -------------------------------------------------

    def _build_assignments_tab(self) -> None:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self.frame_label = QLabel(f"Frame {self._current_frame}")
        self.frame_label.setStyleSheet("font-weight: bold;")
        v.addWidget(self.frame_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Camera", "Track", "Identity", "Scope"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)

        self._tabs.addTab(tab, "Assignments")

    def _build_track_id_tab(self) -> None:
        tab = QWidget()
        v = QVBoxLayout(tab)

        btn_row = QHBoxLayout()
        self.new_ident_btn = QPushButton("+ New Identity")
        self.new_track_btn = QPushButton("+ New Track")
        self.new_ident_btn.clicked.connect(self._create_identity)
        self.new_track_btn.clicked.connect(self._create_track)
        btn_row.addWidget(self.new_ident_btn)
        btn_row.addWidget(self.new_track_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Color by:"))
        self.color_mode_track_btn = QRadioButton("Track")
        self.color_mode_id_btn = QRadioButton("Identity")
        self._color_mode_group = QButtonGroup(self)
        self._color_mode_group.addButton(self.color_mode_track_btn)
        self._color_mode_group.addButton(self.color_mode_id_btn)
        self._color_mode_group.setExclusive(True)
        self.color_mode_track_btn.toggled.connect(self._on_mode_track_toggled)
        self.color_mode_id_btn.toggled.connect(self._on_mode_id_toggled)
        mode_row.addWidget(self.color_mode_track_btn)
        mode_row.addWidget(self.color_mode_id_btn)
        mode_row.addStretch()
        v.addLayout(mode_row)

        v.addWidget(QLabel("Tracks"))
        self.tracks_table = QTableWidget(0, 2)
        self.tracks_table.setHorizontalHeaderLabels(["Color", "Track"])
        self.tracks_table.verticalHeader().setVisible(False)
        self.tracks_table.setColumnWidth(0, 36)
        self.tracks_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tracks_table)

        v.addWidget(QLabel("Identities"))
        self.identities_table = QTableWidget(0, 3)
        self.identities_table.setHorizontalHeaderLabels(["Color", "Name", "ID"])
        self.identities_table.verticalHeader().setVisible(False)
        self.identities_table.setColumnWidth(0, 36)
        self.identities_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.identities_table)

        self._tabs.addTab(tab, "Track/ID")

    # ---- external API -------------------------------------------------

    def set_current_frame(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self.frame_label.setText(f"Frame {frame_idx}")
        self._refresh()

    # ---- refresh / populate ------------------------------------------

    def _refresh(self) -> None:
        self._populating = True
        try:
            fg = self.session.frame_group(self._current_frame)
            rows: list[tuple[str, int]] = []
            if fg is not None:
                for cam_name in sorted(fg.instances.keys()):
                    seen_tracks: set[int] = set()
                    for inst in fg.instances[cam_name]:
                        if inst.track_idx is None or inst.track_idx in seen_tracks:
                            continue
                        seen_tracks.add(inst.track_idx)
                        rows.append((cam_name, inst.track_idx))

            self.table.setRowCount(len(rows))
            for row_i, (cam_name, track_idx) in enumerate(rows):
                self.table.setItem(row_i, 0, _ro_item(cam_name))
                track_name = (
                    self.session.tracks[track_idx]
                    if 0 <= track_idx < len(self.session.tracks)
                    else f"t{track_idx}"
                )
                self.table.setItem(row_i, 1, _ro_item(f"{track_idx} ({track_name})"))

                combo = QComboBox()
                self._fill_identity_combo(combo)

                effective_id = self.session.get_identity_id_for_track(
                    self._current_frame, cam_name, track_idx
                )
                combo.setCurrentIndex(self._find_combo_index(combo, effective_id))

                combo.currentIndexChanged.connect(
                    lambda _idx, r=row_i, cam=cam_name, tr=track_idx, cb=combo:
                        self._on_identity_changed(r, cam, tr, cb)
                )
                self.table.setCellWidget(row_i, 2, combo)

                # Scope: per-frame override vs global
                per_key = f"{self._current_frame}:{cam_name}:{track_idx}"
                scope = "frame" if per_key in self.session.frame_identity_map else "global"
                self.table.setItem(row_i, 3, _ro_item(scope))

                # Color swatch in the Camera cell background
                ident = self.session.get_identity(effective_id)
                if ident is not None:
                    item = self.table.item(row_i, 0)
                    item.setBackground(QColor(ident.color))
        finally:
            self._populating = False

    def _refresh_tracks_legend(self) -> None:
        tracks = self.session.tracks
        self.tracks_table.setRowCount(len(tracks))
        for i, name in enumerate(tracks):
            swatch = _ro_item("")
            swatch.setBackground(QColor(get_track_color(i)))
            self.tracks_table.setItem(i, 0, swatch)
            self.tracks_table.setItem(i, 1, _ro_item(name))

    def _refresh_identities_legend(self) -> None:
        idents = self.session.identities
        self.identities_table.setRowCount(len(idents))
        for i, ident in enumerate(idents):
            swatch = _ro_item("")
            swatch.setBackground(QColor(ident.color))
            self.identities_table.setItem(i, 0, swatch)
            self.identities_table.setItem(i, 1, _ro_item(ident.name))
            self.identities_table.setItem(i, 2, _ro_item(str(ident.id)))

    def _fill_identity_combo(self, combo: QComboBox) -> None:
        combo.clear()
        combo.addItem("— (none) —", NONE_ID)
        for ident in self.session.identities:
            combo.addItem(f"{ident.name}", ident.id)

    def _find_combo_index(self, combo: QComboBox, identity_id: int | None) -> int:
        target = NONE_ID if identity_id is None else identity_id
        for i in range(combo.count()):
            if combo.itemData(i) == target:
                return i
        return 0

    # ---- handlers -----------------------------------------------------

    def _on_identity_changed(self, row: int, cam_name: str, track_idx: int, combo: QComboBox) -> None:
        if self._populating:
            return
        new_id = combo.currentData()
        new_id = None if new_id == NONE_ID else int(new_id)
        self.session.set_frame_identity(self._current_frame, cam_name, track_idx, new_id)
        self.identityAssignmentChanged.emit()

    def _create_identity(self) -> None:
        dlg = NewIdentityDialog(self.session, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name, color = dlg.result_identity_spec()
            self.session.add_identity(name, color)

    def _create_track(self) -> None:
        dlg = NewTrackDialog(self.session, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.session.add_track(dlg.track_name())

    def _on_mode_track_toggled(self, checked: bool) -> None:
        if checked:
            self.session.set_color_mode("track")

    def _on_mode_id_toggled(self, checked: bool) -> None:
        if checked:
            self.session.set_color_mode("identity")

    def _sync_color_mode_ui(self, mode: str) -> None:
        self.color_mode_track_btn.blockSignals(True)
        self.color_mode_id_btn.blockSignals(True)
        try:
            self.color_mode_track_btn.setChecked(mode == "track")
            self.color_mode_id_btn.setChecked(mode == "identity")
        finally:
            self.color_mode_track_btn.blockSignals(False)
            self.color_mode_id_btn.blockSignals(False)


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item
