"""
Main application window.

Layout:
  +-------------------+-----------------------------------+
  |   Sidebar (310px) |     VTK Render Window              |
  +-------------------+-----------------------------------+
  |            Bottom Slider Panel                         |
  +-------------------------------------------------------+
"""
import vtk
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QMessageBox)

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from config import SimConfig
from visualization.vtk_pipeline import VTKPipeline
from gui.sidebar_panel import SidebarPanel
from gui.slider_panel import SliderPanel
from gui.simulation_worker import SimulationWorker
from gui.live_worker import LiveSimWorker


class MainWindow(QMainWindow):
    def __init__(self, config: SimConfig):
        super().__init__()
        self.config = config
        self._worker = None
        self._live_worker = None
        self._pending_live_frame = None

        self._live_render_timer = QTimer(self)
        self._live_render_timer.setSingleShot(True)
        self._live_render_timer.setInterval(int(1000 / max(1, config.live_preview_max_fps)))
        self._live_render_timer.timeout.connect(self._drain_live_frame)

        self.setWindowTitle("CS 530 – River Hydraulics Visualization")
        self.resize(1400, 800)

        # ---- central widget ----------------------------------------- #
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_row = QHBoxLayout()
        top_row.setSpacing(0)

        # sidebar
        self.sidebar = SidebarPanel(config)
        top_row.addWidget(self.sidebar)

        # VTK render widget
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        top_row.addWidget(self.vtk_widget, stretch=1)

        root_layout.addLayout(top_row, stretch=1)

        # bottom slider
        self.slider_panel = SliderPanel()
        self.slider_panel.set_enabled_all(False)
        root_layout.addWidget(self.slider_panel)

        # ---- VTK renderer ------------------------------------------- #
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.15, 0.15, 0.20)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self.pipeline = VTKPipeline(config, self.renderer)

        # ---- wire signals -------------------------------------------- #
        self.sidebar.run_simulation.connect(self._on_run_simulation)
        self.sidebar.scalar_field_changed.connect(self.pipeline.set_scalar_field)
        self.sidebar.layer_toggled.connect(self.pipeline.set_layer_visibility)
        self.sidebar.live_preview_toggled.connect(self._on_live_toggled)
        self.sidebar.obstacles_changed.connect(self._on_obstacles_changed)

        self.slider_panel.frame_changed.connect(self._on_frame_changed)
        self.slider_panel.playback_toggled.connect(self.pipeline.set_animating)

        # ---- kick-start VTK interactor ------------------------------ #
        iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        iren.Initialize()
        self.pipeline.setup_coordinate_display(iren)
        self.vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------ #
    #  Live preview                                                       #
    # ------------------------------------------------------------------ #
    def _on_live_toggled(self, enabled: bool):
        if enabled:
            self._start_live_preview()
        else:
            self._stop_live_preview()

    def _start_live_preview(self):
        self._stop_live_preview()
        self.sidebar.set_live_preview_mode(True)

        worker = LiveSimWorker(
            self.config,
            self.sidebar.placed_obstacles,
            parent=self,
        )

        # Set up the pipeline in live mode BEFORE starting the worker
        self.pipeline.start_live_mode(
            worker.live_nx, worker.live_ny,
            worker.live_dx, worker.live_dy,
            self.sidebar.placed_obstacles,
        )
        self.pipeline.setup_coordinate_display(
            self.vtk_widget.GetRenderWindow().GetInteractor())
        self.vtk_widget.GetRenderWindow().Render()

        worker.frame_ready.connect(self._on_live_frame)
        worker.status.connect(self.sidebar.show_status)
        worker.start()
        self._live_worker = worker

    def _stop_live_preview(self):
        self._live_render_timer.stop()
        self._pending_live_frame = None
        if self._live_worker and self._live_worker.isRunning():
            self._live_worker.stop()
            self._live_worker.wait()
        self._live_worker = None
        self.sidebar.set_live_preview_mode(False)
        self.pipeline.stop_live_mode()

    def _on_live_frame(self, frame_data):
        self._pending_live_frame = frame_data
        if not self._live_render_timer.isActive():
            self._live_render_timer.start()

    def _drain_live_frame(self):
        if self._pending_live_frame is None:
            return
        frame_data = self._pending_live_frame
        self._pending_live_frame = None
        self.pipeline.update_live_frame(frame_data, render=False)
        self.vtk_widget.GetRenderWindow().Render()

        if self._pending_live_frame is not None:
            self._live_render_timer.start()

    def _on_obstacles_changed(self):
        """Restart live preview when obstacles are added/removed."""
        if self.sidebar.chk_live.isChecked():
            self._start_live_preview()

    # ------------------------------------------------------------------ #
    #  Simulation lifecycle                                               #
    # ------------------------------------------------------------------ #
    def _on_run_simulation(self):
        if self._worker and self._worker.isRunning():
            return

        # Stop live preview — Taichi can only have one active context
        was_live = self.sidebar.chk_live.isChecked()
        if was_live:
            self.sidebar.chk_live.setChecked(False)  # triggers _stop_live_preview
        self._stop_live_preview()

        self.sidebar.set_controls_enabled(False)
        self.slider_panel.set_enabled_all(False)
        self.slider_panel.reset()

        self._worker = SimulationWorker(
            self.config,
            self.sidebar.placed_obstacles,
            parent=self,
        )
        self._worker.progress.connect(self._on_sim_progress)
        self._worker.status.connect(self.sidebar.show_status)
        self._worker.finished_ok.connect(self._on_sim_finished)
        self._worker.finished_err.connect(self._on_sim_error)
        self._worker.start()

    def _on_sim_progress(self, current, total):
        self.sidebar.show_progress(current, total)

    def _on_sim_finished(self, data_dir, num_frames):
        self.sidebar.set_controls_enabled(True)
        self.sidebar.show_status("Simulation complete. Loading visualisation…")

        self.pipeline.load_simulation(
            data_dir, num_frames, self.sidebar.placed_obstacles)
        self.pipeline.setup_coordinate_display(
            self.vtk_widget.GetRenderWindow().GetInteractor())

        self.slider_panel.set_num_frames(num_frames, self.config.export_interval)
        self.slider_panel.set_enabled_all(True)

        self.vtk_widget.GetRenderWindow().Render()
        self.sidebar.show_status("Ready.")

    def _on_sim_error(self, msg):
        self.sidebar.set_controls_enabled(True)
        self.sidebar.show_status(f"Error: {msg}")
        QMessageBox.critical(self, "Simulation Error", msg)

    # ------------------------------------------------------------------ #
    #  Frame navigation                                                   #
    # ------------------------------------------------------------------ #
    def _on_frame_changed(self, idx):
        self.pipeline.set_frame(idx)

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        self._stop_live_preview()
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self.vtk_widget.GetRenderWindow().Finalize()
        del self.vtk_widget
        super().closeEvent(event)
