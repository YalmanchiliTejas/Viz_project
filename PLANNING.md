# CS 530 - River Hydraulics Visualization Tool: Implementation Plan

## Overview
Interactive visualization tool for simulating and visualizing river hydraulic features
(eddies, recirculating holes, turbulent waves) caused by obstacles in a stream.

- **Simulation**: Taichi-based Shallow Water Equations (SWE) solver (CPU/GPU)
- **Visualization**: VTK pipeline (3D surface, velocity glyphs, vorticity contours, streamlines)
- **GUI**: PyQt5 with sidebar controls, animation slider, and VTK render window

## Architecture

```
main.py                         Entry point, arg parsing
config.py                       SimConfig, ObstacleDef, PlacedObstacle, presets
simulation/
  swe_solver.py                 Taichi SWE finite-volume solver (Rusanov flux)
  obstacles.py                  Bed elevation builder + VTK obstacle meshes
  data_export.py                Frame → .vti export
visualization/
  vtk_pipeline.py               Full VTK pipeline (surface, glyphs, contours, streamlines)
gui/
  main_window.py                QMainWindow layout
  sidebar_panel.py              Display, Obstacle Editor, Simulation controls
  slider_panel.py               Play/Pause, Step, Frame slider
  simulation_worker.py          QThread for non-blocking simulation
data/frames/                    Runtime .vti output
```

## Data Flow

```
User places obstacles → sidebar_panel → config.obstacles
User clicks "Run Simulation"
  → simulation_worker (QThread)
    → swe_solver (Taichi, CPU or GPU)
    → data_export → data/frames/frame_XXXX.vti (200 frames)
  → finished signal
Main thread loads VTI series
  → vtk_pipeline reads frames
  → renderer displays surface, glyphs, contours, streamlines, obstacles
User controls animation via slider_panel
User toggles layers / scalar field via sidebar_panel
```

## Simulation Parameters (Table 1)

| Parameter          | Value                  |
|--------------------|------------------------|
| Domain resolution  | 512 × 256              |
| dx, dy             | 0.039 m                |
| h0                 | 0.5 m                  |
| u0                 | 0.5 m/s (x-direction)  |
| ν (viscosity)      | 0.001 m²/s             |
| g                  | 9.81 m/s²              |
| dt                 | 0.0088 s               |
| Export interval     | 0.1 s                  |
| Simulation time    | 20 s → 200 frames      |

## VTK Pipeline Design

1. **Surface**: reader → ImageDataGeometryFilter → WarpScalar(h) → Mapper(color by field) → Actor
2. **Glyphs**: reader → ExtractVOI(16x subsample) → GeomFilter → WarpScalar(h) → Glyph3D(arrows) → Actor
3. **Contours**: reader → ContourFilter(vorticity) → Transform(z-offset) → TubeFilter → Actor
4. **Streamlines**: reader → StreamTracer(velocity seeds) → Transform(z-offset) → TubeFilter → Actor
5. **Obstacles**: SphereSource/CylinderSource → Transform(position) → Actor (per obstacle)

## Preconfigured Obstacles

| Name         | Type | Radius | Height | Length |
|--------------|------|--------|--------|--------|
| Small Rock   | rock | 0.4 m  | 0.20 m | —      |
| Medium Rock  | rock | 0.7 m  | 0.30 m | —      |
| Large Rock   | rock | 1.0 m  | 0.40 m | —      |
| Small Log    | log  | 0.15 m | 0.40 m | 1.5 m  |
| Large Log    | log  | 0.20 m | 0.50 m | 3.0 m  |

## GUI Layout

```
+-------------------+-----------------------------------+
|   Sidebar (300px) |     VTK Render Window              |
|                   |     (QVTKRenderWindowInteractor)   |
|  [Display]        |     + colorbar + coord annotation  |
|  [Obstacle Editor]|                                    |
|  [Sim Controls]   |                                    |
|  [Info Bar]       |                                    |
+-------------------+-----------------------------------+
|  [<<] [▶/▮▮] [>>]  |====slider====|  Frame: 0 / 200   |
+-------------------------------------------------------+
```

## Key Design Decisions

- **Rusanov (local Lax-Friedrichs) flux** for SWE solver: robust, handles shocks
- **Double-buffered fields** in Taichi: h, hu, hv ↔ h_new, hu_new, hv_new
- **Fixed z-offset** for contours/streamlines above warped surface (simple, robust)
- **Semi-transparent surface** (opacity 0.7) so submerged obstacles are visible
- **QThread simulation worker** keeps GUI responsive during long simulations
- **ti.reset() + ti.init()** pattern allows CPU↔GPU switching between runs
