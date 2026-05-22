"""Right-sidebar panel: shows (camera, track) pairs in the current frame and
lets the user assign identities, create identities, and create tracks.

Mirrors the feature scope described in prompts/plans/lucid-lite.md §5.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QRadioButton, QSlider, QTableWidget,
    QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
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

        # Tree layout: camera names are top-level nodes (bold), tracks at
        # that frame are indented children. Column 0 holds the camera name
        # at the top level and "track_idx (track_name)" at the leaf level.
        # Identity combo lives in column 1, scope text in column 2.
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Camera / Track", "Identity", "Scope"])
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(16)
        self.tree.setUniformRowHeights(True)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        v.addWidget(self.tree)

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

        track_row = QHBoxLayout()
        self.track_frames_btn = QPushButton("Track Frames")
        self.track_frames_btn.setToolTip(
            "Run luc3d identity assignment on every frame of this session."
        )
        self.track_frames_btn.clicked.connect(self._track_frames)
        track_row.addWidget(self.track_frames_btn)
        track_row.addStretch()
        v.addLayout(track_row)

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

        # Identity-name label visibility. Off → overlays still draw skeleton
        # edges + nodes + node markers but no text. Drives
        # Session.show_identity_labels via set_show_identity_labels, which
        # emits appearance_changed → every video panel repaints.
        labels_row = QHBoxLayout()
        labels_row.addWidget(QLabel("Labels:"))
        self.show_id_labels_chk = QCheckBox("Show identity names")
        self.show_id_labels_chk.setChecked(bool(self.session.show_identity_labels))
        self.show_id_labels_chk.setToolTip(
            "Show or hide the identity-name text drawn next to each instance "
            "in the video overlay."
        )
        self.show_id_labels_chk.toggled.connect(self._on_show_id_labels_toggled)
        labels_row.addWidget(self.show_id_labels_chk)
        labels_row.addStretch()
        v.addLayout(labels_row)

        # ---- skeleton appearance sliders (node radius + edge width) -----
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        v.addWidget(sep)

        v.addWidget(QLabel("Skeleton appearance"))

        node_row = QHBoxLayout()
        node_row.addWidget(QLabel("Node size"))
        self.node_size_slider = QSlider(Qt.Horizontal)
        # Slider int range 1..40 maps to node radius 0.5..20.0 px (step 0.5).
        self.node_size_slider.setRange(1, 40)
        self.node_size_slider.setValue(int(round(self.session.node_size * 2)))
        self.node_size_slider.setToolTip("Skeleton node circle radius (video px)")
        self.node_size_value = QLabel(f"{self.session.node_size:.1f}")
        self.node_size_value.setMinimumWidth(32)
        self.node_size_slider.valueChanged.connect(self._on_node_size_changed)
        node_row.addWidget(self.node_size_slider, stretch=1)
        node_row.addWidget(self.node_size_value)
        v.addLayout(node_row)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge width"))
        self.edge_width_slider = QSlider(Qt.Horizontal)
        # Slider int range 1..40 maps to edge width 0.5..20.0 px (step 0.5).
        self.edge_width_slider.setRange(1, 40)
        self.edge_width_slider.setValue(int(round(self.session.edge_width * 2)))
        self.edge_width_slider.setToolTip("Skeleton edge stroke width (video px)")
        self.edge_width_value = QLabel(f"{self.session.edge_width:.1f}")
        self.edge_width_value.setMinimumWidth(32)
        self.edge_width_slider.valueChanged.connect(self._on_edge_width_changed)
        edge_row.addWidget(self.edge_width_slider, stretch=1)
        edge_row.addWidget(self.edge_width_value)
        v.addLayout(edge_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        v.addWidget(sep2)

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
            # Show every known camera once, even if no instances at this
            # frame. Order follows session.camera_names() (loader order) so
            # the panel matches the video grid layout.
            cam_names = self.session.camera_names()
            self.tree.clear()

            for cam_name in cam_names:
                cam_item = QTreeWidgetItem([cam_name, "", ""])
                cam_font = cam_item.font(0)
                cam_font.setBold(True)
                cam_item.setFont(0, cam_font)
                # Camera rows are headers — not selectable, not editable.
                cam_item.setFlags(cam_item.flags() & ~Qt.ItemIsSelectable)
                self.tree.addTopLevelItem(cam_item)
                cam_item.setExpanded(True)

                # Unique tracks visible in this camera at the current frame.
                tracks_here: list[int] = []
                if fg is not None:
                    seen: set[int] = set()
                    for inst in fg.get_instances(cam_name):
                        if inst.track_idx is None or inst.track_idx in seen:
                            continue
                        seen.add(inst.track_idx)
                        tracks_here.append(inst.track_idx)

                if not tracks_here:
                    # Reserve visual space under empty cameras with a muted
                    # placeholder row — non-selectable, non-interactive.
                    ph = QTreeWidgetItem(["(no instances)", "", ""])
                    ph_font = ph.font(0)
                    ph_font.setItalic(True)
                    ph.setFont(0, ph_font)
                    ph.setForeground(0, QColor("#888888"))
                    ph.setFlags(ph.flags() & ~Qt.ItemIsSelectable & ~Qt.ItemIsEnabled)
                    cam_item.addChild(ph)
                    continue

                for track_idx in tracks_here:
                    track_name = (
                        self.session.tracks[track_idx]
                        if 0 <= track_idx < len(self.session.tracks)
                        else f"t{track_idx}"
                    )
                    track_item = QTreeWidgetItem(
                        [f"{track_idx} ({track_name})", "", ""]
                    )
                    track_item.setFlags(track_item.flags() & ~Qt.ItemIsSelectable)
                    cam_item.addChild(track_item)

                    combo = QComboBox()
                    self._fill_identity_combo(combo)
                    effective_id = self.session.get_identity_id_for_track(
                        self._current_frame, cam_name, track_idx
                    )
                    combo.setCurrentIndex(self._find_combo_index(combo, effective_id))
                    combo.currentIndexChanged.connect(
                        lambda _idx, cam=cam_name, tr=track_idx, cb=combo:
                            self._on_identity_changed(cam, tr, cb)
                    )
                    self.tree.setItemWidget(track_item, 1, combo)

                    # Scope: per-frame override vs global.
                    per_key = f"{self._current_frame}:{cam_name}:{track_idx}"
                    scope = (
                        "frame"
                        if per_key in self.session.frame_identity_map
                        else "global"
                    )
                    track_item.setText(2, scope)

                    # Identity color swatch behind the track label.
                    ident = self.session.get_identity(effective_id)
                    if ident is not None:
                        track_item.setBackground(0, QColor(ident.color))
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

    def _on_identity_changed(self, cam_name: str, track_idx: int, combo: QComboBox) -> None:
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

    def _track_frames(self) -> None:
        """Open the Track Frames dialog, then run track_all over the session."""
        from track_dialog import TrackFramesDialog, run_track_all_with_progress

        dlg = TrackFramesDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        num_animals = dlg.num_animals()

        # Disable the button while running so a second click can't re-enter.
        self.track_frames_btn.setEnabled(False)
        try:
            history = run_track_all_with_progress(
                self.session,
                parent=self,
                num_animals=num_animals,
            )
        finally:
            self.track_frames_btn.setEnabled(True)

        if history is None:
            return
        # track_all already fires session signals as it runs; this final
        # refresh just makes sure the assignments table matches the new
        # identity map for the currently-selected frame.
        self._refresh()
        self._refresh_identities_legend()
        self.identityAssignmentChanged.emit()

    def _on_mode_track_toggled(self, checked: bool) -> None:
        if checked:
            self.session.set_color_mode("track")

    def _on_mode_id_toggled(self, checked: bool) -> None:
        if checked:
            self.session.set_color_mode("identity")

    def _on_node_size_changed(self, slider_val: int) -> None:
        # Map slider [1..40] -> radius [0.5..20.0] in 0.5 increments.
        radius = slider_val / 2.0
        self.node_size_value.setText(f"{radius:.1f}")
        self.session.set_node_size(radius)

    def _on_edge_width_changed(self, slider_val: int) -> None:
        # Map slider [1..40] -> width [0.5..20.0] in 0.5 increments.
        width = slider_val / 2.0
        self.edge_width_value.setText(f"{width:.1f}")
        self.session.set_edge_width(width)

    def _on_show_id_labels_toggled(self, checked: bool) -> None:
        self.session.set_show_identity_labels(checked)

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
