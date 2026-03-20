"""
Sidebar panel with three sections:
  1. Display  – scalar-field selector, layer toggles
  2. Obstacle Editor – tree of placed obstacles, add / remove
  3. Simulation  – CPU/GPU selector, Run button, info bar
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QCheckBox,
    QLabel, QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem,
    QHBoxLayout, QProgressBar, QScrollArea, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QStackedWidget,
)

from config import SimConfig, ObstacleDef, PlacedObstacle, PRECONFIGURED_OBSTACLES


class SidebarPanel(QWidget):
    """Right-hand sidebar with display options, obstacle editor, sim controls."""

    scalar_field_changed = pyqtSignal(str)
    layer_toggled = pyqtSignal(str, bool)          # (layer_name, visible)
    run_simulation = pyqtSignal()
    obstacles_changed = pyqtSignal()

    def __init__(self, config: SimConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.placed_obstacles: list = []            # list[PlacedObstacle]

        self.setFixedWidth(310)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._layout = QVBoxLayout(inner)
        self._layout.setAlignment(Qt.AlignTop)

        self._build_display_section()
        self._build_obstacle_section()
        self._build_simulation_section()
        self._build_info_section()

        self._layout.addStretch()
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ================================================================== #
    #  1. Display section                                                 #
    # ================================================================== #
    def _build_display_section(self):
        grp = QGroupBox("Display")
        vbox = QVBoxLayout(grp)

        # scalar field selector
        vbox.addWidget(QLabel("Scalar Field:"))
        self.combo_scalar = QComboBox()
        self.combo_scalar.addItems(["h", "speed", "vorticity"])
        self.combo_scalar.currentTextChanged.connect(self.scalar_field_changed.emit)
        vbox.addWidget(self.combo_scalar)

        # layer toggles
        vbox.addWidget(QLabel("Layers:"))
        self._layer_checks = {}
        for name, label in [("surface", "Water Surface"),
                            ("glyphs", "Velocity Arrows"),
                            ("contours", "Vorticity Contours"),
                            ("streamlines", "Streamlines")]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, n=name: self.layer_toggled.emit(n, checked))
            vbox.addWidget(cb)
            self._layer_checks[name] = cb

        self._layout.addWidget(grp)

    # ================================================================== #
    #  2. Obstacle editor                                                 #
    # ================================================================== #
    def _build_obstacle_section(self):
        grp = QGroupBox("Obstacle Editor")
        vbox = QVBoxLayout(grp)

        # tree showing placed obstacles
        vbox.addWidget(QLabel("Placed Obstacles:"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Type", "X (m)", "Y (m)"])
        self.tree.setColumnWidth(0, 110)
        self.tree.setMaximumHeight(140)
        self.tree.itemDoubleClicked.connect(self._on_obstacle_dblclick)
        vbox.addWidget(self.tree)

        # obstacle type selector
        vbox.addWidget(QLabel("Add Obstacle:"))
        self.combo_obstacle = QComboBox()
        for od in PRECONFIGURED_OBSTACLES:
            self.combo_obstacle.addItem(f"{od.name}  ({od.kind})")
        vbox.addWidget(self.combo_obstacle)

        # position inputs
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("X:"))
        self.edit_x = QLineEdit("5.0")
        self.edit_x.setFixedWidth(60)
        pos_row.addWidget(self.edit_x)
        pos_row.addWidget(QLabel("Y:"))
        self.edit_y = QLineEdit("5.0")
        self.edit_y.setFixedWidth(60)
        pos_row.addWidget(self.edit_y)
        vbox.addLayout(pos_row)

        self.btn_add = QPushButton("Add Obstacle")
        self.btn_add.clicked.connect(self._add_obstacle)
        vbox.addWidget(self.btn_add)

        self._layout.addWidget(grp)

    def _add_obstacle(self):
        idx = self.combo_obstacle.currentIndex()
        defn = PRECONFIGURED_OBSTACLES[idx]
        try:
            x = float(self.edit_x.text())
            y = float(self.edit_y.text())
        except ValueError:
            self._show_info("Invalid X or Y value.", error=True)
            return

        # bounds check
        if not (0 < x < self.config.domain_width):
            self._show_info(f"X must be between 0 and {self.config.domain_width:.1f} m", error=True)
            return
        if not (0 < y < self.config.domain_height):
            self._show_info(f"Y must be between 0 and {self.config.domain_height:.1f} m", error=True)
            return

        obs = PlacedObstacle(defn, x, y)
        self.placed_obstacles.append(obs)
        self._refresh_tree()
        self.obstacles_changed.emit()

    def _on_obstacle_dblclick(self, item, column):
        """Double-click a row → confirm delete dialog."""
        idx = self.tree.indexOfTopLevelItem(item)
        if idx < 0:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Remove Obstacle")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"Remove '{self.placed_obstacles[idx].definition.name}'?"))
        btns = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec_() == QDialog.Accepted:
            self.placed_obstacles.pop(idx)
            self._refresh_tree()
            self.obstacles_changed.emit()

    def _refresh_tree(self):
        self.tree.clear()
        for obs in self.placed_obstacles:
            item = QTreeWidgetItem([
                obs.definition.name,
                f"{obs.x:.2f}",
                f"{obs.y:.2f}",
            ])
            self.tree.addTopLevelItem(item)

    # ================================================================== #
    #  3. Simulation controls                                             #
    # ================================================================== #
    def _build_simulation_section(self):
        grp = QGroupBox("Simulation")
        vbox = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU", "GPU"])
        self.combo_backend.currentTextChanged.connect(self._on_backend_changed)
        row.addWidget(self.combo_backend)
        vbox.addLayout(row)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #2a7fff; color: white; "
            "font-weight: bold; padding: 6px; }")
        self.btn_run.clicked.connect(self.run_simulation.emit)
        vbox.addWidget(self.btn_run)

        self._layout.addWidget(grp)

    def _on_backend_changed(self, text):
        self.config.use_gpu = (text == "GPU")

    # ================================================================== #
    #  4. Info bar (progress / status)                                    #
    # ================================================================== #
    def _build_info_section(self):
        grp = QGroupBox("Information")
        vbox = QVBoxLayout(grp)

        self.info_stack = QStackedWidget()

        # page 0: status label
        self.lbl_status = QLabel("Ready. Add obstacles and click Run Simulation.")
        self.lbl_status.setWordWrap(True)
        self.info_stack.addWidget(self.lbl_status)

        # page 1: progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.info_stack.addWidget(self.progress_bar)

        vbox.addWidget(self.info_stack)
        self._layout.addWidget(grp)

    # ---- public helpers ---------------------------------------------- #
    def show_progress(self, current: int, total: int):
        self.info_stack.setCurrentIndex(1)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def show_status(self, text: str):
        self.info_stack.setCurrentIndex(0)
        self.lbl_status.setText(text)

    def _show_info(self, text: str, error: bool = False):
        if error:
            self.lbl_status.setStyleSheet("color: red;")
        else:
            self.lbl_status.setStyleSheet("")
        self.show_status(text)

    def set_controls_enabled(self, enabled: bool):
        """Disable / re-enable controls during simulation."""
        for w in (self.btn_add, self.btn_run, self.combo_backend,
                  self.combo_obstacle, self.edit_x, self.edit_y, self.tree):
            w.setEnabled(enabled)
