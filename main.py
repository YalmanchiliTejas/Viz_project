#!/usr/bin/env python3
"""
CS 530 – River Hydraulics Visualization Tool
=============================================
Entry point.  Run with:

    python main.py              # default (CPU)
    python main.py --gpu        # attempt GPU backend

The application opens a PyQt5 window with:
  - VTK 3D render window (water surface, glyphs, contours, streamlines)
  - Sidebar (scalar-field selector, obstacle editor, simulation controls)
  - Bottom animation slider (play/pause, step, frame scrubber)
"""
import sys
import argparse
import os

# Ensure the project root is on the path so that `config`, `simulation`, etc.
# can be imported as top-level packages.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from config import SimConfig
from gui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(
        description="CS 530 River Hydraulics Visualisation Tool")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU backend for Taichi (falls back to CPU)")
    parser.add_argument("--nx", type=int, default=512,
                        help="Grid cells in x (default 512)")
    parser.add_argument("--ny", type=int, default=256,
                        help="Grid cells in y (default 256)")
    parser.add_argument("--sim-time", type=float, default=20.0,
                        help="Simulation duration in seconds (default 20)")
    args = parser.parse_args()

    config = SimConfig(
        nx=args.nx,
        ny=args.ny,
        sim_time=args.sim_time,
        use_gpu=args.gpu,
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
