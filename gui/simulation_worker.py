"""
QThread worker that runs the Taichi SWE simulation off the main thread.
Emits progress signals so the GUI stays responsive.
"""
import os
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from config import SimConfig, DATA_DIR


class SimulationWorker(QThread):
    """Runs one full simulation in a background thread."""

    progress = pyqtSignal(int, int)         # (current_frame, total_frames)
    status = pyqtSignal(str)                # status text
    finished_ok = pyqtSignal(str, int)      # (data_dir, num_frames)
    finished_err = pyqtSignal(str)          # error message

    def __init__(self, config: SimConfig, obstacles: list, parent=None):
        super().__init__(parent)
        self.config = config
        self.obstacles = list(obstacles)

    def run(self):
        try:
            cfg = self.config
            self.status.emit("Initialising Taichi…")

            from simulation.swe_solver import SWESolver, init_taichi
            from simulation.obstacles import build_bed_elevation
            from simulation.data_export import export_frame

            backend = init_taichi(cfg.use_gpu)
            self.status.emit(f"Backend: {backend}  — building bed…")

            b = build_bed_elevation(cfg, self.obstacles)

            solver = SWESolver(
                cfg.nx, cfg.ny, cfg.dx, cfg.dy,
                cfg.dt, cfg.g, cfg.nu, cfg.h0, cfg.u0,
            )
            solver.set_bed(b)
            solver.initialize()

            # clear old frames
            out = DATA_DIR
            os.makedirs(out, exist_ok=True)
            for f in os.listdir(out):
                if f.endswith(".vti"):
                    os.remove(os.path.join(out, f))

            num_frames = cfg.num_frames
            steps = cfg.steps_per_frame

            self.status.emit("Running simulation…")
            for frame in range(num_frames):
                for _ in range(steps):
                    solver.step()
                data = solver.get_frame_data()
                export_frame(data, cfg, frame, out)
                self.progress.emit(frame + 1, num_frames)

            self.status.emit("Simulation complete.")
            self.finished_ok.emit(out, num_frames)

        except Exception as e:
            self.finished_err.emit(f"{e}\n{traceback.format_exc()}")
