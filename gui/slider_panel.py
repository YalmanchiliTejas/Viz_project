"""
Bottom animation-control panel: Play/Pause, step buttons, frame slider.
"""
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton,
                             QSlider, QLabel, QSpinBox)


class SliderPanel(QWidget):
    """Horizontal bar with transport controls and a frame slider."""

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.num_frames = 0
        self._playing = False

        # ---- widgets ------------------------------------------------- #
        self.btn_back = QPushButton("⏪")
        self.btn_play = QPushButton("▶")
        self.btn_fwd  = QPushButton("⏩")
        for b in (self.btn_back, self.btn_play, self.btn_fwd):
            b.setFixedWidth(40)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)

        self.lbl_frame = QLabel("Frame: 0 / 0")
        self.lbl_time  = QLabel("t = 0.00 s")

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(10)
        self.spin_fps.setPrefix("FPS: ")
        self.spin_fps.setFixedWidth(90)

        # ---- layout -------------------------------------------------- #
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self.btn_back)
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_fwd)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.lbl_frame)
        layout.addWidget(self.lbl_time)
        layout.addWidget(self.spin_fps)

        # ---- timer --------------------------------------------------- #
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # ---- connections --------------------------------------------- #
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_fwd.clicked.connect(self._step_forward)
        self.btn_back.clicked.connect(self._step_back)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin_fps.valueChanged.connect(self._update_timer_interval)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def set_num_frames(self, n: int, dt_export: float = 0.1):
        self.num_frames = n
        self._dt_export = dt_export
        self.slider.setMaximum(max(0, n - 1))
        self.slider.setValue(0)
        self._update_labels(0)

    def reset(self):
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("▶")
        self.slider.setValue(0)

    def set_enabled_all(self, enabled: bool):
        for w in (self.btn_back, self.btn_play, self.btn_fwd,
                  self.slider, self.spin_fps):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #
    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._timer.stop()
            self.btn_play.setText("▶")
        else:
            self._playing = True
            self._update_timer_interval()
            self._timer.start()
            self.btn_play.setText("⏸")

    def _step_forward(self):
        v = self.slider.value()
        if v < self.slider.maximum():
            self.slider.setValue(v + 1)

    def _step_back(self):
        v = self.slider.value()
        if v > 0:
            self.slider.setValue(v - 1)

    def _tick(self):
        v = self.slider.value()
        if v < self.slider.maximum():
            self.slider.setValue(v + 1)
        else:
            self._toggle_play()  # stop at end

    def _on_slider(self, value):
        self._update_labels(value)
        self.frame_changed.emit(value)

    def _update_labels(self, frame):
        self.lbl_frame.setText(f"Frame: {frame} / {self.num_frames}")
        t = frame * getattr(self, "_dt_export", 0.1)
        self.lbl_time.setText(f"t = {t:.2f} s")

    def _update_timer_interval(self):
        fps = self.spin_fps.value()
        self._timer.setInterval(int(1000 / fps))
