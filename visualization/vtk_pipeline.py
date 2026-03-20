"""
VTK visualization pipeline for the SWE simulation data.

Builds four visualization layers on a shared vtkRenderer:
  1. Water surface  – WarpScalar coloured by user-selected scalar field
  2. Velocity glyphs – subsampled arrows oriented by flow
  3. Vorticity contours – iso-contour lines of vorticity
  4. Streamlines – seeded at the domain inlet
Plus obstacle actors (rocks / logs).
"""
import os
import vtk

from config import SimConfig, PlacedObstacle
from simulation.obstacles import create_obstacle_actor


class VTKPipeline:
    """Manages the full VTK rendering pipeline."""

    SCALAR_FIELDS = ("h", "speed", "vorticity")
    SCALAR_LABELS = {"h": "Height (m)", "speed": "Speed (m/s)",
                     "vorticity": "Vorticity (1/s)"}

    def __init__(self, config: SimConfig, renderer: vtk.vtkRenderer):
        self.config = config
        self.renderer = renderer
        self.data_dir = ""
        self.num_frames = 0
        self.active_field = "h"

        # visibility flags
        self.show_surface = True
        self.show_glyphs = True
        self.show_contours = True
        self.show_streamlines = True

        # VTK objects (populated by setup_pipeline)
        self.reader = vtk.vtkXMLImageDataReader()
        self.surface_actor = None
        self.glyph_actor = None
        self.contour_actor = None
        self.streamline_actor = None
        self.obstacle_actors: list = []
        self.scalar_bar = None
        self.corner_annotation = None

        # color maps
        self._build_color_maps()
        # estimated scalar ranges – updated after first frame load
        self.scalar_ranges = {"h": (0.0, 1.0), "speed": (0.0, 1.5),
                              "vorticity": (-10.0, 10.0)}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def load_simulation(self, data_dir: str, num_frames: int,
                        obstacles: list):
        """Load a completed simulation and build the pipeline."""
        self.data_dir = data_dir
        self.num_frames = num_frames

        self._load_frame(0)
        self._estimate_ranges()
        self._build_pipeline()
        self._add_obstacles(obstacles)
        self._setup_camera()

    def set_frame(self, idx: int):
        """Switch the reader to a different time frame and re-render."""
        if not self._load_frame(idx):
            return
        self._refresh_active_arrays()
        self.renderer.GetRenderWindow().Render()

    def set_scalar_field(self, field_name: str):
        """Change the colour-mapped scalar field on the surface."""
        if field_name not in self.SCALAR_FIELDS:
            return
        self.active_field = field_name
        lo, hi = self.scalar_ranges[field_name]
        self._surface_mapper.SelectColorArray(field_name)
        self._surface_mapper.SetScalarRange(lo, hi)
        self._surface_mapper.SetLookupTable(self.ctfs[field_name])
        if self.scalar_bar:
            self.scalar_bar.SetLookupTable(self.ctfs[field_name])
            self.scalar_bar.SetTitle(self.SCALAR_LABELS[field_name])
        self.renderer.GetRenderWindow().Render()

    def set_layer_visibility(self, layer: str, visible: bool):
        actors = {
            "surface": self.surface_actor,
            "glyphs": self.glyph_actor,
            "contours": self.contour_actor,
            "streamlines": self.streamline_actor,
        }
        actor = actors.get(layer)
        if actor:
            actor.SetVisibility(visible)
            self.renderer.GetRenderWindow().Render()

    def setup_coordinate_display(self, interactor):
        """Add a corner annotation that tracks mouse world coordinates."""
        self.corner_annotation = vtk.vtkCornerAnnotation()
        self.corner_annotation.SetText(2, "X: --  Y: --")
        self.corner_annotation.GetTextProperty().SetFontSize(14)
        self.corner_annotation.GetTextProperty().SetColor(1, 1, 1)
        self.renderer.AddViewProp(self.corner_annotation)

        picker = vtk.vtkWorldPointPicker()

        def _on_mouse_move(obj, event):
            x, y = interactor.GetEventPosition()
            picker.Pick(x, y, 0, self.renderer)
            pos = picker.GetPickPosition()
            self.corner_annotation.SetText(
                2, f"X: {pos[0]:.2f} m   Y: {pos[1]:.2f} m")
            interactor.GetRenderWindow().Render()

        interactor.AddObserver("MouseMoveEvent", _on_mouse_move)

    def clear(self):
        """Remove all actors from the renderer."""
        self.renderer.RemoveAllViewProps()
        self.surface_actor = None
        self.glyph_actor = None
        self.contour_actor = None
        self.streamline_actor = None
        self.obstacle_actors.clear()
        self.scalar_bar = None
        self.corner_annotation = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _load_frame(self, idx: int) -> bool:
        path = os.path.join(self.data_dir, f"frame_{idx:04d}.vti")
        if not os.path.exists(path):
            return False
        self.reader.SetFileName(path)
        self.reader.Update()
        self._refresh_active_arrays()
        return True

    def _refresh_active_arrays(self):
        data = self.reader.GetOutput()
        if data:
            data.GetPointData().SetActiveScalars("h")
            data.GetPointData().SetActiveVectors("velocity")

    def _estimate_ranges(self):
        """Scan the first frame to set reasonable colour-map ranges."""
        data = self.reader.GetOutput()
        if not data:
            return
        pd = data.GetPointData()
        for name in self.SCALAR_FIELDS:
            arr = pd.GetArray(name)
            if arr:
                lo, hi = arr.GetRange()
                if name == "vorticity":
                    mx = max(abs(lo), abs(hi), 0.1)
                    lo, hi = -mx, mx
                elif lo == hi:
                    hi = lo + 1.0
                self.scalar_ranges[name] = (lo, hi)

    # ---- colour maps ------------------------------------------------- #
    def _build_color_maps(self):
        self.ctfs = {}

        # height – blue gradient
        ctf_h = vtk.vtkColorTransferFunction()
        ctf_h.AddRGBPoint(0.0, 0.05, 0.05, 0.40)
        ctf_h.AddRGBPoint(0.25, 0.10, 0.20, 0.70)
        ctf_h.AddRGBPoint(0.50, 0.15, 0.45, 0.90)
        ctf_h.AddRGBPoint(0.75, 0.30, 0.70, 1.00)
        ctf_h.AddRGBPoint(1.0, 0.60, 0.90, 1.00)
        self.ctfs["h"] = ctf_h

        # speed – blue → yellow → red
        ctf_s = vtk.vtkColorTransferFunction()
        ctf_s.AddRGBPoint(0.0, 0.10, 0.10, 0.80)
        ctf_s.AddRGBPoint(0.4, 0.10, 0.70, 0.90)
        ctf_s.AddRGBPoint(0.7, 1.00, 0.90, 0.20)
        ctf_s.AddRGBPoint(1.0, 0.90, 0.10, 0.10)
        self.ctfs["speed"] = ctf_s

        # vorticity – diverging blue-white-red
        ctf_v = vtk.vtkColorTransferFunction()
        ctf_v.SetColorSpaceToDiverging()
        ctf_v.AddRGBPoint(-10.0, 0.231, 0.298, 0.753)
        ctf_v.AddRGBPoint(0.0, 0.865, 0.865, 0.865)
        ctf_v.AddRGBPoint(10.0, 0.706, 0.016, 0.150)
        self.ctfs["vorticity"] = ctf_v

    # ---- pipeline construction --------------------------------------- #
    def _build_pipeline(self):
        self.clear()
        self._build_surface()
        self._build_glyphs()
        self._build_contours()
        self._build_streamlines()
        self._build_scalar_bar()

    def _build_surface(self):
        geom = vtk.vtkImageDataGeometryFilter()
        geom.SetInputConnection(self.reader.GetOutputPort())

        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(geom.GetOutputPort())
        warp.SetScaleFactor(self.config.warp_scale)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(warp.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(self.active_field)
        lo, hi = self.scalar_ranges[self.active_field]
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs[self.active_field])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.75)
        actor.SetVisibility(self.show_surface)

        self.renderer.AddActor(actor)
        self.surface_actor = actor
        self._surface_mapper = mapper
        self._surface_warp = warp

    def _build_glyphs(self):
        sub = vtk.vtkExtractVOI()
        sub.SetInputConnection(self.reader.GetOutputPort())
        sub.SetSampleRate(16, 16, 1)

        geom = vtk.vtkImageDataGeometryFilter()
        geom.SetInputConnection(sub.GetOutputPort())

        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(geom.GetOutputPort())
        warp.SetScaleFactor(self.config.warp_scale)

        # assign velocity as active vectors after warp (arrays propagate)
        assign_v = vtk.vtkAssignAttribute()
        assign_v.SetInputConnection(warp.GetOutputPort())
        assign_v.Assign("velocity", vtk.vtkDataSetAttributes.VECTORS,
                        vtk.vtkAssignAttribute.POINT_DATA)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)
        arrow.SetShaftRadius(0.03)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(assign_v.GetOutputPort())
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(0.4)
        glyph.OrientOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.15, 0.15, 0.15)
        actor.SetVisibility(self.show_glyphs)

        self.renderer.AddActor(actor)
        self.glyph_actor = actor

    def _build_contours(self):
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(self.reader.GetOutputPort())
        contour.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "vorticity")
        lo, hi = self.scalar_ranges["vorticity"]
        contour.GenerateValues(12, lo, hi)

        # elevate contours above surface
        z_offset = self.config.h0 * self.config.warp_scale + 0.15
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, z_offset)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(contour.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.02)
        tube.SetNumberOfSides(6)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.SetScalarRange(lo, hi)
        mapper.SetLookupTable(self.ctfs["vorticity"])

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetVisibility(self.show_contours)

        self.renderer.AddActor(actor)
        self.contour_actor = actor

    def _build_streamlines(self):
        # assign velocity vectors explicitly
        assign_v = vtk.vtkAssignAttribute()
        assign_v.SetInputConnection(self.reader.GetOutputPort())
        assign_v.Assign("velocity", vtk.vtkDataSetAttributes.VECTORS,
                        vtk.vtkAssignAttribute.POINT_DATA)

        # seed line along inlet (x ≈ 1 cell in)
        seeds = vtk.vtkLineSource()
        seeds.SetPoint1(self.config.dx * 2, self.config.dy * 5, 0)
        seeds.SetPoint2(self.config.dx * 2,
                        self.config.domain_height - self.config.dy * 5, 0)
        seeds.SetResolution(20)

        tracer = vtk.vtkStreamTracer()
        tracer.SetInputConnection(assign_v.GetOutputPort())
        tracer.SetSourceConnection(seeds.GetOutputPort())
        tracer.SetMaximumPropagation(self.config.domain_width * 2)
        tracer.SetIntegrationDirectionToForward()
        tracer.SetIntegratorTypeToRungeKutta4()
        tracer.SetMaximumNumberOfSteps(4000)

        # elevate above surface
        z_offset = self.config.h0 * self.config.warp_scale + 0.25
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, z_offset)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(tracer.GetOutputPort())
        tf.SetTransform(transform)

        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.025)
        tube.SetNumberOfSides(6)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.95, 0.95, 0.95)
        actor.GetProperty().SetOpacity(0.8)
        actor.SetVisibility(self.show_streamlines)

        self.renderer.AddActor(actor)
        self.streamline_actor = actor

    def _build_scalar_bar(self):
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(self.ctfs[self.active_field])
        bar.SetTitle(self.SCALAR_LABELS[self.active_field])
        bar.SetNumberOfLabels(5)
        bar.SetWidth(0.08)
        bar.SetHeight(0.4)
        bar.SetPosition(0.90, 0.05)
        bar.GetTitleTextProperty().SetFontSize(12)
        bar.GetTitleTextProperty().SetColor(1, 1, 1)
        bar.GetLabelTextProperty().SetColor(1, 1, 1)

        self.renderer.AddActor2D(bar)
        self.scalar_bar = bar

    # ---- obstacle actors --------------------------------------------- #
    def _add_obstacles(self, obstacles: list):
        for obs in obstacles:
            actor = create_obstacle_actor(obs, self.config.warp_scale)
            self.renderer.AddActor(actor)
            self.obstacle_actors.append(actor)

    # ---- camera ------------------------------------------------------ #
    def _setup_camera(self):
        w = self.config.domain_width
        h = self.config.domain_height
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(w / 2, -h * 0.6, h * 1.2)
        cam.SetFocalPoint(w / 2, h / 2, 0)
        cam.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.renderer.SetBackground(0.15, 0.15, 0.20)
