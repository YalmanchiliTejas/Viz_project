import os
from dataclasses import dataclass


@dataclass
class SimConfig:
    """Simulation configuration matching the CS 530 proposal Table 1."""
    nx: int = 512
    ny: int = 256
    dx: float = 0.039
    dy: float = 0.039
    h0: float = 0.5
    u0: float = 1.0
    nu: float = 0.0005
    g: float = 9.81
    dt: float = 0.002
    export_interval: float = 0.1
    sim_time: float = 20.0
    use_gpu: bool = False
    warp_scale: float = 3.0
    live_preview_nx: int = 128
    live_preview_ny: int = 64
    live_preview_max_fps: int = 8
    live_preview_range_update_interval: int = 4

    @property
    def num_frames(self) -> int:
        return int(self.sim_time / self.export_interval)

    @property
    def steps_per_frame(self) -> int:
        return int(self.export_interval / self.dt)

    @property
    def domain_width(self) -> float:
        return self.nx * self.dx

    @property
    def domain_height(self) -> float:
        return self.ny * self.dy


@dataclass
class ObstacleDef:
    """Definition of a preconfigured obstacle type."""
    name: str
    kind: str       # "rock" or "log"
    radius: float   # sphere radius (rock) or cross-section radius (log)
    height: float   # peak bed elevation
    length: float = 0.0    # log length (ignored for rocks)
    angle: float = 90.0    # rotation angle in degrees (logs only)


@dataclass
class PlacedObstacle:
    """An obstacle placed at a specific location in the domain."""
    definition: ObstacleDef
    x: float
    y: float


# --------------- Preconfigured obstacle catalogue ---------------
PRECONFIGURED_OBSTACLES = [
    ObstacleDef("Small Rock",  "rock", radius=0.4,  height=0.20),
    ObstacleDef("Medium Rock", "rock", radius=0.7,  height=0.30),
    ObstacleDef("Large Rock",  "rock", radius=1.0,  height=0.40),
    ObstacleDef("Boulder",     "rock", radius=1.5,  height=0.40),
    ObstacleDef("Small Log",   "log",  radius=0.15, height=0.40, length=1.5),
    ObstacleDef("Large Log",   "log",  radius=0.20, height=0.50, length=3.0),
]

# Default output directory for VTI frames
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames")
