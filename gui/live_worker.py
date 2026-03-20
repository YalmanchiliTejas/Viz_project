"""
Background thread that runs the SWE solver continuously at reduced resolution
for interactive preview.
"""
from PyQt5.QtCore import QThread, pyqtSignal

from config import SimConfig


class LiveSimWorker(QThread):
    """Continuously runs the solver and pushes frames for live display."""

    frame_ready = pyqtSignal(object)       # dict of numpy arrays
    status = pyqtSignal(str)

    def __init__(self, config: SimConfig, obstacles: list, parent=None):
        super().__init__(parent)
        self.config = config
        self.obstacles = list(obstacles)
        self._running = True

        # Reduced resolution for real-time performance
        self.live_nx = min(config.nx, config.live_preview_nx)
        self.live_ny = min(config.ny, config.live_preview_ny)
        self.live_dx = config.domain_width / self.live_nx
        self.live_dy = config.domain_height / self.live_ny

    def run(self):
        try:
            from simulation.swe_solver import SWESolver, init_taichi
            from simulation.obstacles import build_bed_elevation

            cfg = self.config
            backend = init_taichi(cfg.use_gpu)
            self.status.emit(
                "Live preview "
                f"({backend}) — {self.live_nx}×{self.live_ny}. "
                "Showing low-resolution versions of the selected layers "
                "before the full simulation runs."
            )

            # Build bed at preview resolution
            live_cfg = SimConfig(
                nx=self.live_nx, ny=self.live_ny,
                dx=self.live_dx, dy=self.live_dy,
                h0=cfg.h0, u0=cfg.u0, nu=cfg.nu, g=cfg.g,
                dt=cfg.dt, use_gpu=cfg.use_gpu,
            )
            b = build_bed_elevation(live_cfg, self.obstacles)

            solver = SWESolver(
                self.live_nx, self.live_ny,
                self.live_dx, self.live_dy,
                cfg.dt, cfg.g, cfg.nu, cfg.h0, cfg.u0,
            )
            solver.set_bed(b)
            solver.initialize()
            self.frame_ready.emit(solver.get_frame_data())

            frame_interval = 1.0 / max(1, cfg.live_preview_max_fps)
            steps_per_display = max(1, int(frame_interval / cfg.dt))

            while self._running:
                for _ in range(steps_per_display):
                    if not self._running:
                        return
                    solver.step()
                data = solver.get_frame_data()
                self.frame_ready.emit(data)

        except Exception as e:
            self.status.emit(f"Live preview error: {e}")

    def stop(self):
        self._running = False
