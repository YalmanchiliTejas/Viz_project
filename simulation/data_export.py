"""
Export simulation frames to VTK XML ImageData (.vti) files.
"""
import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from config import SimConfig


class FrameExporter:
    """Reuse VTK export state across frames to avoid per-frame allocations."""

    SCALAR_FIELDS = ("h", "vx", "vy", "speed", "vorticity", "pressure")

    def __init__(self, config: SimConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        nx, ny = config.nx, config.ny
        n = nx * ny

        self.image = vtk.vtkImageData()
        self.image.SetDimensions(nx, ny, 1)
        self.image.SetSpacing(config.dx, config.dy, 1.0)
        self.image.SetOrigin(0, 0, 0)

        point_data = self.image.GetPointData()
        self.scalar_buffers = {}
        self.scalar_arrays = {}
        for name in self.SCALAR_FIELDS:
            arr = numpy_to_vtk(np.zeros(n, dtype=np.float32), deep=True)
            arr.SetName(name)
            point_data.AddArray(arr)
            self.scalar_arrays[name] = arr
            self.scalar_buffers[name] = vtk_to_numpy(arr)

        velocity = numpy_to_vtk(np.zeros((n, 3), dtype=np.float32), deep=True)
        velocity.SetName("velocity")
        point_data.AddArray(velocity)
        self.velocity_array = velocity
        self.velocity_buffer = vtk_to_numpy(velocity)

        point_data.SetActiveScalars("h")
        point_data.SetActiveVectors("velocity")

        self.writer = vtk.vtkXMLImageDataWriter()
        self.writer.SetInputData(self.image)

    def export_frame(self, frame_data: dict, frame_idx: int) -> str:
        """Write one frame as a .vti file and return the path."""
        for name in self.SCALAR_FIELDS:
            np.copyto(
                self.scalar_buffers[name],
                frame_data[name].flatten(order="F").astype(np.float32, copy=False),
            )
            self.scalar_arrays[name].Modified()

        vx = frame_data["vx"].flatten(order="F").astype(np.float32, copy=False)
        vy = frame_data["vy"].flatten(order="F").astype(np.float32, copy=False)
        self.velocity_buffer[:, 0] = vx
        self.velocity_buffer[:, 1] = vy
        self.velocity_buffer[:, 2] = 0.0
        self.velocity_array.Modified()
        self.image.Modified()

        path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.vti")
        self.writer.SetFileName(path)
        self.writer.Write()
        return path


def export_frame(frame_data: dict, config: SimConfig,
                 frame_idx: int, output_dir: str) -> str:
    """Compatibility wrapper for one-off exports."""
    exporter = FrameExporter(config, output_dir)
    return exporter.export_frame(frame_data, frame_idx)
