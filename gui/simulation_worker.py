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

            from simulation.swe_solver import (
                SWESolver,
                init_taichi,
                is_cuda_stream_error,
            )
            from simulation.obstacles import build_bed_elevation
            from simulation.data_export import FrameExporter

            backend_attempts = [cfg.use_gpu]
            if cfg.use_gpu:
                backend_attempts.append(False)

            last_error = None
            for use_gpu in backend_attempts:
                try:
                    backend = init_taichi(use_gpu)
                    self.status.emit(f"Backend: {backend}  — building bed…")

                    run_cfg = cfg if use_gpu == cfg.use_gpu else SimConfig(**{
                        **cfg.__dict__,
                        "use_gpu": use_gpu,
                    })
                    b = build_bed_elevation(run_cfg, self.obstacles)

                    solver = SWESolver(
                        run_cfg.nx, run_cfg.ny, run_cfg.dx, run_cfg.dy,
                        run_cfg.dt, run_cfg.g, run_cfg.nu, run_cfg.h0, run_cfg.u0,
                    )
                    solver.set_bed(b)
                    solver.initialize()

                    # clear old frames
                    out = DATA_DIR
                    os.makedirs(out, exist_ok=True)
                    for f in os.listdir(out):
                        if f.endswith(".vti"):
                            os.remove(os.path.join(out, f))
                    exporter = FrameExporter(run_cfg, out)

                    num_frames = run_cfg.num_frames
                    steps = run_cfg.steps_per_frame

                    self.status.emit("Running simulation…")
                    for frame in range(num_frames):
                        for _ in range(steps):
                            solver.step()
                        data = solver.get_frame_data()
                        exporter.export_frame(data, frame)
                        self.progress.emit(frame + 1, num_frames)

                    self.status.emit(f"Simulation complete ({backend}).")
                    self.finished_ok.emit(out, num_frames)
                    return
                except Exception as e:
                    last_error = e
                    if use_gpu and is_cuda_stream_error(e):
                        self.status.emit(
                            "CUDA stream error detected; retrying simulation on CPU.")
                        continue
                    raise

            if last_error:
                raise last_error

        except Exception as e:
            self.finished_err.emit(f"{e}\n{traceback.format_exc()}")
