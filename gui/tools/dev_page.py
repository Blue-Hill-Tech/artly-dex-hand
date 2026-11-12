"""The Dev Mode page: developer settings, opened from the Dev Mode button at the main page's top right.

- Camera pipeline: which tracker the camera mode runs (shown only when there is more than one; this repo has
  MediaPipe only).
- GeoRT models: the trained run per hand and tracking source (glove / camera), from that hand's folder
  (retargetor/GeoRT_checkpoint/left_hand or right_hand), or the newest one there; a choice applies at once (the tracking restarts) for this run, Save writes it to selection.json.
- Motor gains: each hand's teleop torque kp / kd (the motor started with Start Left / Start Right), from the next motor start;
  Make default puts a hand back to its saved values, Save writes them into its config (torque.kp / torque.kd).
- Torque demo safety stop: on (default) or off. On stops the torque demo (and the calibration's learning passes) when
  a fingertip is more than max_tip_error_mm from its target; the joint over-speed stop is always on.
- Hand adapters (a choice applies to this run; Save makes it the hand's default, written into its configs): the CAN FD /
  RS-485 adapter each hand's motor and demos open, picked from the adapters plugged in now
  (/dev/serial/by-id, rescanned every second while the page is shown), or None: that hand is not used (no tracking,
  no motor, no demo). The choice is passed to the processes as AAC_SERIAL_DEVICE (teleop/hand_control/
  hand_controller_aac.load_aac_config) and applies the next time a motor or demo starts.

The page only shows and emits the choices (`pipeline_changed`, `adapter_changed`); the main page applies and
remembers them.
"""
import re
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget)

SIDES = ("Left", "Right")
GEORT_SLOTS = (("glove", "Left"), ("glove", "Right"), ("camera", "Left"), ("camera", "Right"))
# (key, label, unit, maximum, step, decimals) of the motor gain bars
GAIN_BARS = (("kp", "kp", "mA/deg", 20.0, 0.1, 1), ("kd", "kd", "mA·s/deg", 1.5, 0.01, 2))
SPIN_STYLE = """
QDoubleSpinBox { border: 1px solid #D9D3BF; border-radius: 10px; background: #F3EFE2; color: #222; font-size: 16px;
                 padding: 4px 8px; min-width: 70px; }
"""
SLIDER_STYLE = """
QSlider::groove:horizontal { height: 6px; background: #E4DFCD; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #F2D65A; border-radius: 3px; }
QSlider::handle:horizontal { background: #FAF8EF; border: 1px solid #CFC8B4; width: 18px; height: 18px; margin: -7px 0;
                             border-radius: 9px; }
QSlider::handle:horizontal:hover { background: #F9E076; border-color: #E8CB55; }
"""
RESCAN_MS = 1000
CONTENT_WIDTH_PX = 680
TITLE_STYLE = "font-size: 28px; color: #222;"
SECTION_STYLE = "font-size: 19px; color: #222; font-weight: 600;"
TEXT_STYLE = "font-size: 15px; color: #5E584A;"
SMALL_STYLE = "font-size: 13px; color: #8A8472;"
WARNING_STYLE = "font-size: 15px; color: #B5452F;"
BACK_STYLE = """
QPushButton { border: 1px solid #D9D3BF; border-radius: 17px; background: #F3EFE2; color: #222; font-size: 16px;
              padding: 6px 22px; }
QPushButton:hover { background: #F9E076; border-color: #E8CB55; }
"""
SEGMENT_STYLE = """
QPushButton { border: 1px solid #D9D3BF; background: #F3EFE2; color: #222; font-size: 16px; padding: 7px 24px; }
QPushButton#first { border-top-left-radius: 19px; border-bottom-left-radius: 19px; border-right: none; }
QPushButton#middle { border-right: none; }
QPushButton#last { border-top-right-radius: 19px; border-bottom-right-radius: 19px; }
QPushButton:hover { background: #F7EBB5; }
QPushButton:checked { background: #F9E076; border-color: #E8CB55; }
"""
COMBO_STYLE = """
QComboBox { border: 1px solid #D9D3BF; border-radius: 14px; background: #F3EFE2; color: #222; font-size: 16px;
            padding: 8px 16px; min-width: 480px; }
QComboBox:hover { background: #F7EBB5; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView { background: #FAF8EF; border: 1px solid #D9D3BF; color: #222; font-size: 16px;
                              selection-background-color: #F9E076; selection-color: #222; padding: 4px; }
"""
ADAPTER_ROW_STYLE = "QLabel { background: #F3EFE2; border: 1px solid #E4DFCD; border-radius: 12px; padding: 10px 14px; }"


def adapter_serial(path):
    """The adapter's serial number (usb-<maker>_<model>_<serial>-if00), or its file name."""
    name = Path(path).name
    stem = name.rpartition("-if")[0] if "-if" in name else name
    return stem.rpartition("_")[2] or name


def adapter_model(path):
    """A readable maker / model from the by-id name, e.g. 'ElmueSoft netcult.ch elmue Slcan 2.5 - Openlight Labs'."""
    name = Path(path).name
    stem = name.rpartition("-if")[0] if "-if" in name else name
    stem = stem[4:] if stem.startswith("usb-") else stem
    model = stem.rpartition("_")[0] or stem
    return re.sub(r"\s+", " ", model.replace("_", " ")).strip()


def adapter_kind(path):
    """What the adapter is, for the list: "CAN FD" (an slcan / CANable adapter), "RS-485" (a USB serial bridge) or ""."""
    name = Path(path).name.lower()
    if "slcan" in name or "canable" in name or "candlelight" in name:
        return "CAN FD"
    if any(word in name for word in ("wch", "ch340", "ch341", "ftdi", "cp210", "single_serial", "rs485", "rs-485")):
        return "RS-485"
    return ""


class DevPage(QWidget):
    back = pyqtSignal()
    pipeline_changed = pyqtSignal(str)          # a PIPELINES key
    adapter_changed = pyqtSignal(str, object)   # side, adapter path or None (the hand is not used)
    tip_stop_changed = pyqtSignal(bool)         # the torque demo's fingertip-distance safety stop on / off
    friction_fine_changed = pyqtSignal(bool)    # Calibrate friction: the fine calibration (True) or the normal one
    save_requested = pyqtSignal()               # Save: make the chosen adapters the hands' defaults
    gains_changed = pyqtSignal(str, float, float)   # side, kp, kd (from the next motor start)
    gains_save_requested = pyqtSignal()         # Save: write the chosen gains into the hands' configs
    checkpoint_changed = pyqtSignal(str, str, object)   # source group ("glove"/"camera"), side, run name or None (newest)
    checkpoints_save_requested = pyqtSignal()   # Save: make the chosen GeoRT runs the defaults

    def __init__(self, corner, pipelines, pipeline, adapters, defaults, list_adapters, tip_stop=True,
                 pins=None, list_runs=None, gains=None, gain_overrides=None, friction_fine=False, parent=None):
        """pins: {(source group, side): GeoRT run name} chosen now (missing = newest); list_runs(side): the runs
        (teleop/manus/geort/checkpoints.list_runs)."""
        """corner: the window buttons; pipelines: {key: name}; adapters: {side: path or None} (current choice);
        defaults: {side: the config's adapter path}; list_adapters(): the adapter paths plugged in now."""
        super().__init__(parent)
        self.adapters, self.defaults, self.list_adapters = dict(adapters), dict(defaults), list_adapters
        self.saved = dict(adapters)                  # the defaults now in effect: Save is offered when the choice differs
        self.pins, self.saved_pins = dict(pins or {}), dict(pins or {})
        self.list_runs = list_runs or (lambda side: [])
        # {side: {"kp", "kd"}} chosen now / saved in the configs; gain_overrides: {side: {joint: {kp/kd}}} (shown only)
        self.gains = {side: dict(values) for side, values in (gains or {}).items()}
        self.saved_gains = {side: dict(values) for side, values in (gains or {}).items()}
        self.gain_overrides = dict(gain_overrides or {})
        self.attached = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QGridLayout()
        top.setContentsMargins(20, 14, 0, 0)
        top.setColumnStretch(0, 1)
        top.setColumnStretch(2, 1)
        back = QPushButton("‹  Back")
        back.setStyleSheet(BACK_STYLE)
        back.setCursor(Qt.PointingHandCursor)
        back.setToolTip("Back to the hands")
        back.clicked.connect(self.back)
        top.addWidget(back, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        title = QLabel("Dev Mode")
        title.setStyleSheet(TITLE_STYLE)
        top.addWidget(title, 0, 1, Qt.AlignHCenter | Qt.AlignTop)
        top.addWidget(corner, 0, 2, Qt.AlignRight | Qt.AlignTop)
        layout.addLayout(top)

        body = QVBoxLayout()
        body.setSpacing(10)
        # The sections scroll (the page is taller than a small window); the top bar stays.
        holder = QWidget()
        holder.setFixedWidth(CONTENT_WIDTH_PX)
        holder.setLayout(body)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 30, 0, 40)
        centred = QHBoxLayout()
        centred.addStretch(1)
        centred.addWidget(holder)
        centred.addStretch(1)
        content_layout.addLayout(centred)
        content_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
                             "QScrollBar:vertical { width: 10px; background: transparent; }"
                             "QScrollBar::handle:vertical { background: #D9D3BF; border-radius: 5px; min-height: 40px; }"
                             "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")
        layout.addWidget(scroll, 1)

        # Camera pipeline: a choice only when there is more than one tracker
        self.pipeline_buttons = {}
        if len(pipelines) > 1:
            body.addWidget(self._label("Camera pipeline", SECTION_STYLE))
            body.addWidget(self._label("The tracker the Camera mode runs. Changing it restarts the camera.", TEXT_STYLE))
            segments = QHBoxLayout()
            segments.setSpacing(0)
            group = QButtonGroup(self)
            group.setExclusive(True)
            keys = list(pipelines)
            for i, key in enumerate(keys):
                button = QPushButton(pipelines[key])
                button.setCheckable(True)
                button.setCursor(Qt.PointingHandCursor)
                button.setObjectName("first" if i == 0 else "last" if i == len(keys) - 1 else "middle")
                button.setStyleSheet(SEGMENT_STYLE)
                button.clicked.connect(lambda _=False, key=key: self.pipeline_changed.emit(key))
                group.addButton(button)
                segments.addWidget(button)
                self.pipeline_buttons[key] = button
            segments.addStretch(1)
            body.addLayout(segments)
            self.set_pipeline(pipeline)

        # GeoRT models
        if self.pipeline_buttons:
            body.addSpacing(26)
        self.geort_title = self._label("GeoRT models", SECTION_STYLE)
        body.addWidget(self.geort_title)
        body.addWidget(self._label("The trained model that turns your hand into the robot's joint angles, per hand and "
                                   "tracking source. Each hand lists the runs in its own folder: "
                                   "retargetor/GeoRT_checkpoint/left_hand and right_hand. Newest: the latest run there. "
                                   "A choice applies at once (the tracking restarts, about a second) for this run; Save "
                                   "makes it the default.", TEXT_STYLE))
        runs_grid = QGridLayout()
        runs_grid.setHorizontalSpacing(18)
        runs_grid.setVerticalSpacing(10)
        self.run_combos = {}
        for i, (group, side) in enumerate(GEORT_SLOTS):
            name = QLabel(f"{side} hand · {group.capitalize()}")
            name.setStyleSheet("font-size: 17px; color: #222;")
            runs_grid.addWidget(name, i, 0, Qt.AlignVCenter)
            combo = QComboBox()
            combo.setStyleSheet(COMBO_STYLE.replace("min-width: 480px", "min-width: 420px"))   # a wider label column
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(30)
            combo.setCursor(Qt.PointingHandCursor)
            combo.activated.connect(lambda _index, group=group, side=side: self._run_picked(group, side))
            runs_grid.addWidget(combo, i, 1, Qt.AlignLeft | Qt.AlignVCenter)
            self.run_combos[(group, side)] = combo
        runs_grid.setColumnStretch(2, 1)
        body.addLayout(runs_grid)
        runs_save = QHBoxLayout()
        runs_save.setSpacing(14)
        self.runs_save_button = QPushButton("Save as default")
        self.runs_save_button.setStyleSheet(BACK_STYLE + "QPushButton:disabled { background: #EEEADD; color: #A8A292; "
                                                         "border-color: #E4DFCD; }")
        self.runs_save_button.setCursor(Qt.PointingHandCursor)
        self.runs_save_button.setToolTip("Make the chosen GeoRT models the defaults (retargetor/GeoRT_checkpoint/"
                                         "selection.json), for the GUI and the teleop sender")
        self.runs_save_button.clicked.connect(self.checkpoints_save_requested)
        runs_save.addWidget(self.runs_save_button)
        self.runs_status = self._label("", SMALL_STYLE)
        runs_save.addWidget(self.runs_status, 1)
        body.addLayout(runs_save)
        self.refresh_runs()

        # Motor gains
        if self.gains:
            body.addSpacing(26)
            body.addWidget(self._label("Motor gains", SECTION_STYLE))
            body.addWidget(self._label("Stiffness (kp) and damping (kd) of each hand's motor in torque mode (Start Left / "
                                       "Start Right). Applies the next time that motor starts, for this run; Make default "
                                       "puts a hand back to its saved values; Save writes them into its config.",
                                       TEXT_STYLE))
            self.gain_widgets = {}
            for side in SIDES:
                if side not in self.gains:
                    continue
                head = QHBoxLayout()
                title = QLabel(f"{side} hand")
                title.setStyleSheet("font-size: 17px; color: #222;")
                head.addWidget(title)
                head.addStretch(1)
                reset = QPushButton("Make default")
                reset.setStyleSheet(BACK_STYLE + "QPushButton { font-size: 14px; padding: 4px 16px; border-radius: 13px; }"
                                    "QPushButton:disabled { background: #EEEADD; color: #A8A292; border-color: #E4DFCD; }")
                reset.setCursor(Qt.PointingHandCursor)
                reset.clicked.connect(lambda _=False, side=side: self.reset_gains(side))
                head.addWidget(reset)
                body.addSpacing(6)
                body.addLayout(head)
                widgets = dict(reset=reset)
                grid = QGridLayout()
                grid.setHorizontalSpacing(14)
                grid.setVerticalSpacing(6)
                for row, (key, label, unit, maximum, step, decimals) in enumerate(GAIN_BARS):
                    name = QLabel(label)
                    name.setStyleSheet("font-size: 16px; color: #5E584A;")
                    slider = QSlider(Qt.Horizontal)
                    slider.setStyleSheet(SLIDER_STYLE)
                    slider.setRange(0, int(round(maximum / step)))
                    slider.setCursor(Qt.PointingHandCursor)
                    spin = QDoubleSpinBox()
                    spin.setStyleSheet(SPIN_STYLE)
                    spin.setRange(0.0, maximum)
                    spin.setSingleStep(step)
                    spin.setDecimals(decimals)
                    unit_label = QLabel(unit)
                    unit_label.setStyleSheet(SMALL_STYLE)
                    slider.valueChanged.connect(lambda v, side=side, key=key, step=step: self._gain_from_slider(side, key, v * step))
                    spin.valueChanged.connect(lambda v, side=side, key=key: self._gain_from_spin(side, key, v))
                    grid.addWidget(name, row, 0)
                    grid.addWidget(slider, row, 1)
                    grid.addWidget(spin, row, 2)
                    grid.addWidget(unit_label, row, 3)
                    widgets[key] = (slider, spin, step)
                grid.setColumnStretch(1, 1)
                body.addLayout(grid)
                overrides = self.gain_overrides.get(side) or {}
                if overrides:
                    body.addWidget(self._label("Own values kept for: " + ", ".join(
                        f"{joint} " + " ".join(f"{k} {v:g}" for k, v in values.items())
                        for joint, values in overrides.items()) + " (torque.joints in the config).", SMALL_STYLE))
                self.gain_widgets[side] = widgets
                self._show_gains(side)
            gains_save = QHBoxLayout()
            gains_save.setSpacing(14)
            self.gains_save_button = QPushButton("Save as default")
            self.gains_save_button.setStyleSheet(BACK_STYLE + "QPushButton:disabled { background: #EEEADD; "
                                                              "color: #A8A292; border-color: #E4DFCD; }")
            self.gains_save_button.setCursor(Qt.PointingHandCursor)
            self.gains_save_button.setToolTip("Write these kp / kd into each hand's torque config (torque.kp, torque.kd)")
            self.gains_save_button.clicked.connect(self.gains_save_requested)
            gains_save.addWidget(self.gains_save_button)
            self.gains_status = self._label("", SMALL_STYLE)
            gains_save.addWidget(self.gains_status, 1)
            body.addSpacing(6)
            body.addLayout(gains_save)
            self._update_gains_save()

        # Torque demo safety stop
        body.addSpacing(26)
        body.addWidget(self._label("Torque demo safety stop", SECTION_STYLE))
        body.addWidget(self._label("On: the torque demo and its calibration stop (torque off, the hand held where it is) "
                                   "when a fingertip is more than 30 mm from its target. Off: they keep going. The stop "
                                   "for a joint moving too fast stays on either way. Applies from the next demo start.",
                                   TEXT_STYLE))
        toggle = QHBoxLayout()
        toggle.setSpacing(0)
        self.tip_stop_buttons = {}
        stop_group = QButtonGroup(self)
        stop_group.setExclusive(True)
        for value, text, name in ((True, "On", "first"), (False, "Off", "last")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setObjectName(name)
            button.setStyleSheet(SEGMENT_STYLE + "QPushButton { min-width: 60px; }")
            button.clicked.connect(lambda _=False, value=value: self._tip_stop_picked(value))
            stop_group.addButton(button)
            toggle.addWidget(button)
            self.tip_stop_buttons[value] = button
        toggle.addStretch(1)
        body.addLayout(toggle)
        self.tip_stop = None
        self.set_tip_stop(tip_stop)

        # Friction calibration: normal or fine
        body.addSpacing(26)
        body.addWidget(self._label("Friction calibration", SECTION_STYLE))
        body.addWidget(self._label("What a hand's Calibrate friction button runs. Normal: 3 angles per joint, 5 mA "
                                   "steps, validated at 20 deg/s (mean error up to 1.5 deg). Fine: 10 angles, 2 mA "
                                   "steps, every reading 3 times; validated at 10, 20 and 40 deg/s at two positions "
                                   "(mean error up to 1.0 deg), refined in 0.05 steps. Fine takes several times as "
                                   "long. Applies from the next calibration start.", TEXT_STYLE))
        toggle = QHBoxLayout()
        toggle.setSpacing(0)
        self.friction_fine_buttons = {}
        fine_group = QButtonGroup(self)
        fine_group.setExclusive(True)
        for value, text, name in ((False, "Normal", "first"), (True, "Fine", "last")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setObjectName(name)
            button.setStyleSheet(SEGMENT_STYLE + "QPushButton { min-width: 80px; }")
            button.clicked.connect(lambda _=False, value=value: self._friction_fine_picked(value))
            fine_group.addButton(button)
            toggle.addWidget(button)
            self.friction_fine_buttons[value] = button
        toggle.addStretch(1)
        body.addLayout(toggle)
        self.friction_fine = None
        self.set_friction_fine(friction_fine)

        # Hand adapters
        body.addSpacing(26)
        body.addWidget(self._label("Hand adapters", SECTION_STYLE))
        body.addWidget(self._label("The adapter each hand's motor and demos use. None: that hand is not used. A "
                                   "choice applies from the next motor or demo start, for this run only; Save makes "
                                   "it the hand's default (written into its config files).", TEXT_STYLE))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)
        self.combos = {}
        for i, side in enumerate(SIDES):
            name = QLabel(f"{side} hand")
            name.setStyleSheet("font-size: 17px; color: #222;")
            grid.addWidget(name, i, 0, Qt.AlignVCenter)
            combo = QComboBox()
            combo.setStyleSheet(COMBO_STYLE)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(34)
            combo.setCursor(Qt.PointingHandCursor)
            combo.activated.connect(lambda _index, side=side: self._picked(side))
            grid.addWidget(combo, i, 1, Qt.AlignLeft | Qt.AlignVCenter)
            self.combos[side] = combo
        grid.setColumnStretch(2, 1)
        body.addLayout(grid)
        self.warning = self._label("", WARNING_STYLE)
        body.addWidget(self.warning)
        save_row = QHBoxLayout()
        save_row.setSpacing(14)
        self.save_button = QPushButton("Save as default")
        self.save_button.setStyleSheet(BACK_STYLE + "QPushButton:disabled { background: #EEEADD; color: #A8A292; "
                                                    "border-color: #E4DFCD; }")
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.setToolTip("Make the chosen adapters the hands' defaults: written into each hand's config "
                                    "files, so they are used from now on, also from the terminal")
        self.save_button.clicked.connect(self.save_requested)
        save_row.addWidget(self.save_button)
        self.save_status = self._label("", SMALL_STYLE)
        save_row.addWidget(self.save_status, 1)
        body.addLayout(save_row)

        # Connected adapters
        body.addSpacing(26)
        self.list_title = self._label("Connected adapters", SECTION_STYLE)
        body.addWidget(self.list_title)
        self.list_box = QVBoxLayout()
        self.list_box.setSpacing(8)
        body.addLayout(self.list_box)

        self.timer = QTimer(self)
        self.timer.setInterval(RESCAN_MS)
        self.timer.timeout.connect(self.rescan)
        self.rescan(force=True)

    @staticmethod
    def _label(text, style):
        label = QLabel(text)
        label.setStyleSheet(style)
        label.setWordWrap(True)
        return label

    def showEvent(self, event):
        super().showEvent(event)
        self.rescan(force=True)
        self.refresh_runs()                          # runs copied in while the GUI was open show up too
        self.timer.start()

    # --- GeoRT models ------------------------------------------------------------------------------------------------
    def refresh_runs(self):
        """Rebuild the GeoRT model lists from the runs on disk and the current choice."""
        counts = []
        for side in SIDES:
            try:
                runs = list(self.list_runs(side))
            except OSError:
                runs = []
            counts.append(f"{len(runs)} {side.lower()}")
            for group in ("glove", "camera"):
                combo = self.run_combos[(group, side)]
                combo.blockSignals(True)
                combo.clear()
                usable = [run for run in runs if not run.get("wrong_hand")]
                newest = usable[-1]["label"] if usable else f"no run in {side.lower()}_hand/"
                combo.addItem(f"Newest   (now: {newest})", "")
                combo.setItemData(0, f"The latest run in retargetor/GeoRT_checkpoint/{side.lower()}_hand/",
                                  Qt.ToolTipRole)
                chosen = self.pins.get((group, side))
                names = [run["name"] for run in runs]
                for run in runs + ([dict(name=chosen, label=f"{chosen}   (missing)", training={})]
                                   if chosen and chosen not in names else []):
                    notes = ["default"] if run["name"] == self.saved_pins.get((group, side)) else []
                    if run.get("wrong_hand"):
                        other = "left" if side == "Right" else "right"
                        notes.append(f"trained for the {other} hand: move it to {other}_hand/")
                    combo.addItem(run["label"] + (f"   ({', '.join(notes)})" if notes else ""), run["name"])
                    training = run["training"]
                    tip = run["name"] + ("\n" + ", ".join(f"{k} {v}" for k, v in training.items()) if training else "")
                    combo.setItemData(combo.count() - 1, tip, Qt.ToolTipRole)
                combo.setCurrentIndex(max(combo.findData(chosen or ""), 0))
                combo.blockSignals(False)
        self.geort_title.setText(f"GeoRT models   ({', '.join(counts)} runs)")
        self._update_runs_save()

    def _run_picked(self, group, side):
        run = self.run_combos[(group, side)].currentData() or None
        if run != self.pins.get((group, side)):
            if run:
                self.pins[(group, side)] = run
            else:
                self.pins.pop((group, side), None)
            self.checkpoint_changed.emit(group, side, run)
            self.refresh_runs()

    def set_runs_saved(self, saved, message="", error=False):
        self.saved_pins = dict(saved)
        self.refresh_runs()
        self.runs_status.setStyleSheet(WARNING_STYLE if error else SMALL_STYLE)
        self.runs_status.setText(message)

    def _update_runs_save(self):
        unsaved = [slot for slot in GEORT_SLOTS if self.pins.get(slot) != self.saved_pins.get(slot)]
        self.runs_save_button.setEnabled(bool(unsaved))
        if unsaved:
            self.runs_status.setStyleSheet(SMALL_STYLE)
            self.runs_status.setText("Not saved: " + ", ".join(f"{side.lower()} {group}" for group, side in unsaved)
                                     + " (this run only).")
        elif self.runs_status.text().startswith("Not saved"):
            self.runs_status.setText("")

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def set_pipeline(self, pipeline):
        if pipeline in self.pipeline_buttons:
            self.pipeline_buttons[pipeline].setChecked(True)

    # --- motor gains -------------------------------------------------------------------------------------------------
    def _show_gains(self, side):
        """Put the side's chosen kp / kd on its bars and boxes (without echoing a change back)."""
        widgets = self.gain_widgets[side]
        for key, *_ in GAIN_BARS:
            slider, spin, step = widgets[key]
            value = self.gains[side][key]
            for widget in (slider, spin):
                widget.blockSignals(True)
            slider.setValue(int(round(value / step)))
            spin.setValue(value)
            for widget in (slider, spin):
                widget.blockSignals(False)
        widgets["reset"].setEnabled(self.gains[side] != self.saved_gains.get(side))

    def _gain_from_slider(self, side, key, value):
        self._set_gain(side, key, round(value, 4))

    def _gain_from_spin(self, side, key, value):
        self._set_gain(side, key, round(value, 4))

    def _set_gain(self, side, key, value):
        if abs(self.gains[side][key] - value) < 1e-9:
            return
        self.gains[side][key] = value
        self._show_gains(side)
        self.gains_changed.emit(side, self.gains[side]["kp"], self.gains[side]["kd"])
        self._update_gains_save()

    def reset_gains(self, side):
        """Make default: the side's kp / kd back to the values saved in its config."""
        self.gains[side] = dict(self.saved_gains[side])
        self._show_gains(side)
        self.gains_changed.emit(side, self.gains[side]["kp"], self.gains[side]["kd"])
        self._update_gains_save()

    def set_gains_saved(self, saved, message="", error=False):
        self.saved_gains = {side: dict(values) for side, values in saved.items()}
        for side in self.gain_widgets:
            self._show_gains(side)
        self._update_gains_save()
        self.gains_status.setStyleSheet(WARNING_STYLE if error else SMALL_STYLE)
        self.gains_status.setText(message)

    def _update_gains_save(self):
        unsaved = [side for side in SIDES if side in self.gains and self.gains[side] != self.saved_gains.get(side)]
        self.gains_save_button.setEnabled(bool(unsaved))
        if unsaved:
            self.gains_status.setStyleSheet(SMALL_STYLE)
            self.gains_status.setText("Not saved: " + " and ".join(
                f"{side.lower()} hand (saved kp {self.saved_gains[side]['kp']:g}, kd {self.saved_gains[side]['kd']:g})"
                for side in unsaved) + ".")
        elif self.gains_status.text().startswith("Not saved"):
            self.gains_status.setText("")

    def set_tip_stop(self, on):
        self.tip_stop = bool(on)
        self.tip_stop_buttons[self.tip_stop].setChecked(True)

    def _tip_stop_picked(self, on):
        if on != self.tip_stop:
            self.tip_stop = on
            self.tip_stop_changed.emit(on)

    def set_friction_fine(self, on):
        self.friction_fine = bool(on)
        self.friction_fine_buttons[self.friction_fine].setChecked(True)

    def _friction_fine_picked(self, on):
        if on != self.friction_fine:
            self.friction_fine = on
            self.friction_fine_changed.emit(on)

    def set_adapter(self, side, path):
        self.adapters[side] = path
        self.rescan(force=True)

    def set_saved(self, saved, defaults=None, message="", error=False):
        """After a save (or at start): the defaults now in effect, the configs' adapters, and what to say."""
        self.saved = dict(saved)
        if defaults is not None:
            self.defaults = dict(defaults)
        self.save_status.setStyleSheet(WARNING_STYLE if error else SMALL_STYLE)
        self.save_status.setText(message)
        self.rescan(force=True)

    def _update_save(self):
        unsaved = [side for side in SIDES if self.adapters.get(side) != self.saved.get(side)]
        self.save_button.setEnabled(bool(unsaved))
        if unsaved:
            self.save_status.setStyleSheet(SMALL_STYLE)
            self.save_status.setText("Not saved: " + " and ".join(f"{side.lower()} hand" for side in unsaved)
                                     + " (this run only).")

    def _picked(self, side):
        path = self.combos[side].currentData() or None
        if path != self.adapters.get(side):
            self.adapters[side] = path
            self.adapter_changed.emit(side, path)
            self.rescan(force=True)

    def rescan(self, force=False):
        """Re-list the adapters plugged in; the combos and the list are rebuilt when something changed."""
        try:
            attached = list(self.list_adapters())
        except OSError:
            attached = []
        if attached == self.attached and not force:
            return
        self.attached = attached
        for side, combo in self.combos.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(f"None (don't use the {side.lower()} hand)", "")
            chosen = self.adapters.get(side)
            paths = list(attached) + ([chosen] if chosen and chosen not in attached else [])
            for path in paths:
                kind = adapter_kind(path)
                text = f"{adapter_serial(path)}" + (f"   {kind} adapter" if kind else f"   {adapter_model(path)}")
                notes = []
                if path == self.defaults.get(side):
                    notes.append("config default")
                if path not in attached:
                    notes.append("not plugged in")
                combo.addItem(text + (f"   ({', '.join(notes)})" if notes else ""), path)
                combo.setItemData(combo.count() - 1, f"{adapter_model(path)}\n{path}", Qt.ToolTipRole)
            index = combo.findData(chosen or "")
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)
        warnings = []
        left, right = self.adapters.get("Left"), self.adapters.get("Right")
        if left and left == right:
            warnings.append("Both hands use the same adapter: only one of them can run at a time.")
        for side in SIDES:
            chosen = self.adapters.get(side)
            if chosen and chosen not in attached:
                warnings.append(f"The {side.lower()} hand's adapter ({adapter_serial(chosen)}) is not plugged in.")
            if chosen and adapter_kind(chosen) == "RS-485":
                warnings.append(f"The {side.lower()} hand's motor and demos talk CAN FD: an RS-485 adapter will not "
                                "work with them.")
        self.warning.setText("\n".join(warnings))
        self.warning.setVisible(bool(warnings))
        self._update_save()
        while self.list_box.count():
            widget = self.list_box.takeAt(0).widget()
            if widget is not None:
                widget.hide()                            # gone now, not only once Qt deletes it
                widget.setParent(None)
                widget.deleteLater()
        self.list_title.setText(f"Connected adapters ({len(attached)})")
        if not attached:
            self.list_box.addWidget(self._label("No adapter plugged in (nothing in /dev/serial/by-id).", TEXT_STYLE))
        for path in attached:
            users = [f"{side} hand" for side in SIDES if self.adapters.get(side) == path]
            kind = adapter_kind(path)
            used = (f"<span style='color:#7A5A00;'><b>{' and '.join(users)}</b></span>" if users
                    else "<span style='color:#8A8472;'>not used</span>")
            row = QLabel(f"<span style='font-size:16px; color:#222;'><b>{adapter_serial(path)}</b>"
                         f"{'&nbsp;&nbsp;' + kind if kind else ''}</span>"
                         f"<span style='font-size:15px;'>&nbsp;&nbsp;·&nbsp;&nbsp;{used}</span><br>"
                         f"<span style='font-size:12px; color:#8A8472;'>{adapter_model(path)}&nbsp;&nbsp;·&nbsp;&nbsp;"
                         f"{Path(path).name}</span>")
            row.setToolTip(path)
            row.setStyleSheet(ADAPTER_ROW_STYLE)
            row.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # a long name is clipped, not widening the page
            row.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.list_box.addWidget(row)
