"""Main window: frameless, the start-up splash, then the main page faded in on the cream background.

Without the system title bar the window is moved by dragging any empty background, maximised/restored by
double-clicking it, resized from the bottom-right grip, and minimised/maximised/closed with the small buttons in the
top-right corner of the main page. F11 toggles full screen, Ctrl+Q quits.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QPoint, QPropertyAnimation, QSettings, Qt, QTimer, QVariantAnimation
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtCore import QEasingCurve, pyqtSignal
from PyQt5.QtWidgets import (QAction, QActionGroup, QButtonGroup, QGraphicsOpacityEffect, QMenu, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                             QPushButton, QShortcut, QSizeGrip, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from splash import BACKGROUND, Splash

ROOT = Path(__file__).resolve().parents[2]       # agi-os (this file: gui/tools/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TITLE = "Artly-os"
LOGO_FILE = Path(__file__).resolve().parents[1] / "assets" / "artly_logo.png"
LOGO_HEIGHT_PX = 36
RECENTER_TOP_GAP_PX = 30          # space between the top bar and the Recenter button
FADE_MS = 160                     # one hand / controls fade (the switch is a fade out, then a fade in)
CAPTION_LIFT_PX = 40              # empty space under the "Start Left" / "Start Right" buttons
OUTER_STRETCH, HAND_STRETCH = 1, 4   # outer empty space vs each hand's view: more outer = hands closer to the middle

WINDOW_BUTTON_STYLE = """
QToolButton { border: none; background: transparent; color: #666; font-size: 15px; padding: 4px 10px; }
QToolButton:hover { background: #ECE8D9; color: #222; }
QToolButton#close:hover { background: #e5534b; color: white; }
"""

ROUND_BUTTON_STYLE = """
QPushButton { border: 1px solid #D9D3BF; border-radius: 24px; background: #F3EFE2; color: #222; font-size: 22px;
              padding: 0px; min-width: 48px; max-width: 48px; min-height: 48px; max-height: 48px; }
QPushButton:hover { background: #F9E076; border-color: #E8CB55; }
QPushButton:pressed, QPushButton:checked { background: #EFD25E; border-color: #D9BC45; }
"""

PILL_BUTTON_STYLE = """
QPushButton { border: 1px solid #D9D3BF; border-radius: 24px; background: #F3EFE2; color: #222;
              font-size: 20px; padding: 11px 44px; min-height: 24px; }
QPushButton:hover { background: #F9E076; border-color: #E8CB55; }
QPushButton:pressed, QPushButton:checked { background: #EFD25E; border-color: #D9BC45; }
QPushButton[active="true"] { background: #F9E076; border-color: #E8CB55; }
QPushButton[active="true"]:hover { background: #F4D964; }
QPushButton[active="true"]:pressed { background: #EFD25E; border-color: #D9BC45; }
QPushButton[unavailable="true"], QPushButton[unavailable="true"]:hover, QPushButton[unavailable="true"]:pressed {
    background: #DCD6C5; border-color: #CFC8B4; color: #9C9686; }
QToolTip { background: #FAF8EF; color: #3A372F; border: 1px solid #D9D3BF; padding: 8px 10px; font-size: 15px; }
"""

GLOVE = "Connect to glove"
POSITION_DEMO, TORQUE_DEMO, CALIBRATE = "Position demo", "Torque demo", "Calibrate torque demo"
FRICTION = "Calibrate friction"     # torque control's friction calibration (friction_calibrate.py), for Start the motor
# The "..." panel of a hand: one button each, top to bottom (its motor is the Start Left / Start Right button itself)
CONTROL_ACTIONS = (GLOVE, POSITION_DEMO, TORQUE_DEMO, CALIBRATE, FRICTION)
DEMO_KINDS = {POSITION_DEMO: "position", TORQUE_DEMO: "torque", CALIBRATE: "calibrate",
              FRICTION: "friction"}   # -> demo_process.DEMOS
SKELETON_LABELS = {True: "Hide skeleton", False: "Show skeleton"}   # by whether the skeleton is shown now
CONTROL_BUTTON_WIDTH_PX = 280
WARNING_STYLE = "font-size: 16px; color: #B5452F;"   # a problem the user must fix (warm red, readable on cream)
NOTICE_STYLE = "font-size: 16px; color: #5E584A;"    # a message that must outlive the live status line
ADAPTER_CHECK_MS = 1000                               # while warning about a missing adapter: look for it this often
CONTROL_SPACING_PX = 18
SIDES = ("Left", "Right")
SEARCH_GRACE_MS = 2500            # "Searching ..." this long after start-up, then "No glove found" for a missing one
GHOST_FADE_MS = 400               # the hand turning see-through / solid when its glove goes / comes
NOTE_STYLE = "font-size: 20px; color: #8A8472;"
NOTE_WIDTH_PX = 420
HINT_STYLE = "font-size: 14px; color: #8A8472;"
# Tracking mode shown in the mode bar: the MANUS glove or the webcam. Which tracker the camera mode runs is a Dev Mode
# choice (top right, remembered): MediaPipe, or WiLoR (a MANO hand fit on the GPU, teleop/manus/wilor/wilor_source.py).
MODES = {"glove": "Glove", "camera": "Camera"}
PIPELINES = {"mediapipe": "MediaPipe", "wilor": "WiLoR", "wilor_mini": "WiLoR-mini"}
PIPELINE_SOURCE = {"mediapipe": "camera", "wilor": "wilor", "wilor_mini": "wilor_mini"}   # what the hub runs
SETTINGS = ("artly.ai", "AAC Hand Demo")       # QSettings organisation / application (kept: the saved choices live here)
# Per source (what the hub and the glove workers run: "glove", "camera" = MediaPipe, "wilor"): the note while
# searching, the note when nothing is found, connect / disconnect labels.
SOURCE_TEXT = {
    "glove": dict(searching="Searching for the {side} glove ...",
                  missing="No {side} glove found.\nPlease connect the {side} MANUS glove.",
                  failed="Glove search failed: {error}", connect="Connect to glove", disconnect="Disconnect glove"),
    "wilor": dict(searching="Starting the camera (WiLoR: loading the model) ...",
                  missing="No {side} hand in view.\nShow your {side} hand to the camera.",
                  failed="Camera (WiLoR) failed: {error}", connect="Connect to camera", disconnect="Disconnect camera"),
    "wilor_mini": dict(searching="Starting the camera (WiLoR-mini: loading the model; the first start also downloads "
                                 "it, a few minutes) ...",
                       missing="No {side} hand in view.\nShow your {side} hand to the camera.",
                       failed="Camera (WiLoR-mini) failed: {error}", connect="Connect to camera",
                       disconnect="Disconnect camera"),
    "camera": dict(searching="Starting the camera ...",
                   missing="No {side} hand in view.\nShow your {side} hand to the camera.",
                   failed="Camera failed: {error}", connect="Connect to camera", disconnect="Disconnect camera"),
}
ALIGN_HINTS = {"none": "Hold your {side} hand flat and still, facing the camera, then click Align.",
               "collecting": "Aligning: hold still ...", "ok": ""}

MODE_BAR_STYLE = """
QPushButton { border: 1px solid #D9D3BF; background: #F3EFE2; color: #222; font-size: 17px; padding: 7px 26px; }
QPushButton#first { border-top-left-radius: 19px; border-bottom-left-radius: 19px; border-right: none; }
QPushButton#last { border-top-right-radius: 19px; border-bottom-right-radius: 19px; }
QPushButton:hover { background: #F7EBB5; }
QPushButton:checked { background: #F9E076; border-color: #E8CB55; }
QPushButton[unavailable="true"], QPushButton[unavailable="true"]:hover, QPushButton[unavailable="true"]:checked {
    background: #DCD6C5; border-color: #CFC8B4; color: #9C9686; }
QToolTip { background: #FAF8EF; color: #3A372F; border: 1px solid #D9D3BF; padding: 8px 10px; font-size: 15px; }
"""


class WindowButtons(QWidget):
    """Minimise, maximise/restore and close, for a window without a title bar."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setStyleSheet(WINDOW_BUTTON_STYLE)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for text, name, slot in (("–", "minimise", window.showMinimized), ("□", "maximise", window.toggle_maximised),
                                 ("×", "close", window.close)):
            button = QToolButton(self)
            button.setText(text)
            button.setObjectName(name)
            button.setToolTip(name.capitalize())
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(slot)
            layout.addWidget(button)


class Logo(QLabel):
    """The artly.ai logo, scaled to LOGO_HEIGHT_PX and sharp on high-DPI screens. Empty if the file is missing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        pixmap = QPixmap(str(LOGO_FILE))
        if pixmap.isNull():
            return
        ratio = self.devicePixelRatioF()
        scaled = pixmap.scaledToHeight(int(LOGO_HEIGHT_PX * ratio), Qt.SmoothTransformation)
        scaled.setDevicePixelRatio(ratio)
        self.setPixmap(scaled)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)     # dragging the logo moves the window like the background


DEV_BUTTON_STYLE = """
QPushButton { border: 1px solid #D9D3BF; border-radius: 15px; background: transparent; color: #666; font-size: 14px;
              padding: 5px 16px; }
QPushButton:hover { background: #ECE8D9; color: #222; }
QPushButton:pressed { background: #F9E076; border-color: #E8CB55; color: #222; }
"""
MENU_STYLE = """
QMenu { background: #FAF8EF; border: 1px solid #D9D3BF; border-radius: 10px; padding: 6px; font-size: 15px; }
QMenu::item { padding: 7px 28px 7px 30px; border-radius: 8px; color: #222; }
QMenu::item:selected { background: #F9E076; }
QMenu::item:disabled { color: #8A8472; background: transparent; }
QMenu::separator { height: 1px; background: #E4DFCD; margin: 5px 8px; }
"""


class DevModeButton(QPushButton):
    """Dev Mode: opens the developer page (dev_page.DevPage: camera pipeline, each hand's adapter)."""

    def __init__(self, parent=None):
        super().__init__("Dev Mode", parent)
        self.setStyleSheet(DEV_BUTTON_STYLE)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Developer settings: camera pipeline, hand adapters")


class ModeBar(QWidget):
    """Glove | Camera: the tracking source. `changed` is emitted with the mode key when the user picks the other one."""

    changed = pyqtSignal(str)

    def __init__(self, mode="glove", parent=None):
        super().__init__(parent)
        self.setStyleSheet(MODE_BAR_STYLE)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        keys = list(MODES)
        self.current = mode
        self.unavailable = {}                                    # mode -> why it cannot be used (hover notice)
        for i, key in enumerate(keys):
            button = QPushButton(MODES[key])
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setObjectName("first" if i == 0 else "last")
            button.setToolTip(f"Track the hands with the {MODES[key].lower()}")
            button.clicked.connect(lambda _=False, key=key: self._clicked(key))
            self.group.addButton(button)
            layout.addWidget(button)
            self.buttons[key] = button
        self.set_mode(mode)

    def set_mode(self, mode):
        self.current = mode
        self.buttons[mode].setChecked(True)

    def set_unavailable(self, mode, reason):
        """Grey the mode out with `reason` as its hover notice (clicks only show the notice); None makes it usable."""
        button = self.buttons[mode]
        if reason:
            self.unavailable[mode] = reason
            button.setToolTip(f"<b>{MODES[mode]} can't be used</b><br>{reason}")
            button.setCursor(Qt.ForbiddenCursor)
        else:
            self.unavailable.pop(mode, None)
            button.setToolTip(f"Track the hands with the {MODES[mode].lower()}")
            button.setCursor(Qt.PointingHandCursor)
        button.setProperty("unavailable", bool(reason))
        button.style().unpolish(button)                         # re-apply the stylesheet for the new property
        button.style().polish(button)

    def _clicked(self, key):
        if key in self.unavailable:
            self.set_mode(self.current)                         # the click must not move the selection
            from PyQt5.QtGui import QCursor
            from PyQt5.QtWidgets import QToolTip
            QToolTip.showText(QCursor.pos(), self.buttons[key].toolTip(), self.buttons[key])
            return
        self.changed.emit(key)


class HandPanel(QWidget):
    """One hand's 3D view with its buttons underneath: Start <side> (`start` with the side: starts / stops that hand's
    motor) and a round "..." (`selected` with the side: shows the hand's panel with the glove and demo buttons)."""

    selected = pyqtSignal(str)
    start = pyqtSignal(str)

    def __init__(self, side, parent=None):
        super().__init__(parent)
        from hand_view import GHOST_OPACITY, HandView   # VisPy + meshes: imported and built only when needed
        self.side = side
        self.view = HandView(side)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Show/hide the glove skeleton: above the hand, only while a glove is connected (faded in/out with it). It
        # keeps its place while hidden, so the hand does not move when it appears.
        self.skeleton_button = QPushButton(SKELETON_LABELS[True])
        self.skeleton_button.setStyleSheet(PILL_BUTTON_STYLE)
        self.skeleton_button.setCursor(Qt.PointingHandCursor)
        self.skeleton_button.setToolTip("Show or hide the glove skeleton over the model")
        policy = self.skeleton_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.skeleton_button.setSizePolicy(policy)
        self.skeleton_button.hide()
        # The reminder to connect the glove, in the same place (the hand is see-through while it shows).
        self.note = QLabel(f"Searching for the {side.lower()} glove ...")
        self.note.setAlignment(Qt.AlignCenter)
        self.note.setWordWrap(True)
        self.note.setFixedWidth(NOTE_WIDTH_PX)
        self.note.setStyleSheet(NOTE_STYLE)
        self.note.setAttribute(Qt.WA_TransparentForMouseEvents)
        # Camera mode: Align (a new camera alignment), next to the skeleton button, with what to do written under them.
        self.align_button = QPushButton("Align")
        self.align_button.setStyleSheet(PILL_BUTTON_STYLE)
        self.align_button.setCursor(Qt.PointingHandCursor)
        self.align_button.setToolTip("Hold the hand flat and still, facing the camera, then click: saves the camera "
                                     "alignment and, with MediaPipe, your flat hand (so a flat hand stays flat)")
        self.align_button.hide()
        self.hint = QLabel("")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet(HINT_STYLE)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        policy = self.hint.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.hint.setSizePolicy(policy)
        self.hint.hide()
        buttons = QHBoxLayout()
        buttons.setSpacing(CONTROL_SPACING_PX)
        buttons.addStretch(1)
        buttons.addWidget(self.align_button)
        buttons.addWidget(self.skeleton_button)
        buttons.addStretch(1)
        actions = QVBoxLayout()
        actions.setSpacing(6)
        actions.addLayout(buttons)
        actions.addWidget(self.hint)
        top = QGridLayout()
        top.addLayout(actions, 0, 0, Qt.AlignCenter)
        top.addWidget(self.note, 0, 0, Qt.AlignCenter)
        layout.addLayout(top)
        # The live 3D view, or a still image of it while the panel fades: Qt's opacity effect does not apply to an
        # OpenGL widget, so the fade runs on the snapshot and the live view is swapped back in afterwards.
        self.display = QStackedWidget()
        self.display.addWidget(self.view.widget)
        self.snapshot = QLabel()
        self.snapshot.setAlignment(Qt.AlignCenter)
        self.display.addWidget(self.snapshot)
        layout.addWidget(self.display, 1)
        # What the motor did (started, stopped, a missing adapter): shown here, as the "..." panel may be closed.
        self.message = QLabel("")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setFixedWidth(NOTE_WIDTH_PX + 60)
        self.message.setStyleSheet(NOTICE_STYLE)
        policy = self.message.sizePolicy()
        policy.setRetainSizeWhenHidden(True)                     # the buttons do not jump when it appears
        self.message.setSizePolicy(policy)
        self.message.hide()
        message_row = QHBoxLayout()
        message_row.addStretch(1)
        message_row.addWidget(self.message)
        message_row.addStretch(1)
        layout.addLayout(message_row)
        layout.addSpacing(8)
        self.start_button = QPushButton(f"Start {side}")
        self.start_button.setStyleSheet(PILL_BUTTON_STYLE)
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setToolTip(f"Start the {side.lower()} hand's motor (it follows your hand); click again to stop")
        self.start_button.clicked.connect(lambda: self.start.emit(self.side))
        self.more_button = QPushButton("⋯")
        self.more_button.setCheckable(True)
        self.more_button.setStyleSheet(ROUND_BUTTON_STYLE)
        self.more_button.setCursor(Qt.PointingHandCursor)
        self.more_button.setToolTip(f"More for the {side.lower()} hand: glove, position / torque demo, calibration "
                                    f"(click again for both hands)")
        self.more_button.clicked.connect(lambda: self.selected.emit(self.side))
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addStretch(1)
        row.addWidget(self.start_button)
        row.addWidget(self.more_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addSpacing(CAPTION_LIFT_PX)                       # raises the button off the bottom of the window
        self._ghost = None
        self._ghost_target = None
        self.ghost_opacity = GHOST_OPACITY
        self.view.set_opacity(GHOST_OPACITY)                     # no glove yet: see-through until one is found

    def set_motor_state(self, state):
        """The Start button for the motor's state: "idle" (Start <side>), "running" (Stop <side>, yellow) or
        "stopping"."""
        text = {"idle": f"Start {self.side}", "running": f"Stop {self.side}", "stopping": "Stopping ..."}[state]
        self.start_button.setText(text)
        self.start_button.setProperty("active", state == "running")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)

    def show_message(self, text, error=False):
        """A line about this hand's motor under the hand (red for a problem); empty text fades it out."""
        if text:
            self.message.setStyleSheet(WARNING_STYLE if error else NOTICE_STYLE)
            self.message.setText(text)
        fade_to(self.message, bool(text))

    def set_ghost(self, ghost):
        """See-through hand (no glove) or solid (glove found), animated."""
        target = self.ghost_opacity if ghost else 1.0
        if target == self._ghost_target:                         # already there or on its way: let it finish
            return
        self._ghost_target = target
        if self._ghost is not None:
            self._ghost.stop()
        if abs(self.view.opacity - target) < 1e-3:
            self.view.set_opacity(target)
            return
        self._ghost = QVariantAnimation(self)
        self._ghost.setDuration(GHOST_FADE_MS)
        self._ghost.setStartValue(float(self.view.opacity))
        self._ghost.setEndValue(float(target))
        self._ghost.setEasingCurve(QEasingCurve.InOutQuad)
        self._ghost.valueChanged.connect(lambda value: self.view.set_opacity(float(value)))
        self._ghost.start()

    def freeze(self):
        """Show a still image of the 3D view instead of the live view (for fading)."""
        frame = self.view.canvas.render()[..., :3].copy()
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        ratio = width / max(self.view.widget.width(), 1)
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(ratio)
        self.snapshot.setPixmap(pixmap)
        self.display.setCurrentWidget(self.snapshot)

    def thaw(self):
        """Back to the live 3D view."""
        self.display.setCurrentWidget(self.view.widget)


def fade(widget, start, end, done=None):
    """Fade `widget` from opacity `start` to `end` over FADE_MS, then remove the effect and call `done`."""
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(start)
    widget.setGraphicsEffect(effect)
    animation = QPropertyAnimation(effect, b"opacity", widget)
    animation.setDuration(FADE_MS)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(QEasingCurve.InOutQuad)

    def finish():
        if done is not None:
            done()
        widget.setGraphicsEffect(None)

    animation.finished.connect(finish)
    animation.start(QPropertyAnimation.DeleteWhenStopped)
    return animation


def fade_to(widget, visible):
    """Fade `widget` in (shown) or out (then hidden) unless it is already going that way; a change of mind midway
    turns the fade around from where it is."""
    if getattr(widget, "_wanted", widget.isVisible()) == visible:
        return
    widget._wanted = visible
    effect = widget.graphicsEffect()
    if isinstance(effect, QGraphicsOpacityEffect):
        start = effect.opacity()
    else:
        start = 1.0 if widget.isVisible() else 0.0
    widget.show()
    fade(widget, start, 1.0 if visible else 0.0, done=lambda: widget.setVisible(widget._wanted))


class ControlPanel(QWidget):
    """One hand's controls, on the page background, shown in the other hand's place when that hand is selected.

    A column of pill buttons in the middle of the area. `action` is emitted with (side, action name) on a click;
    add a button with add_action()."""

    action = pyqtSignal(str, str)

    def __init__(self, side, parent=None):
        super().__init__(parent)
        self.side = side
        self.buttons = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, CAPTION_LIFT_PX)
        outer.addStretch(1)
        self.column = QVBoxLayout()
        self.column.setSpacing(CONTROL_SPACING_PX)
        outer.addLayout(self.column)
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setFixedWidth(CONTROL_BUTTON_WIDTH_PX + 120)
        self.status.setStyleSheet("font-size: 14px; color: #8A8472;")
        status_row = QHBoxLayout()
        status_row.addStretch(1)
        status_row.addWidget(self.status)
        status_row.addStretch(1)
        # A warning of its own (the status line is rewritten by every tracking update): e.g. a missing adapter.
        self.warning = QLabel("")
        self.warning.setAlignment(Qt.AlignCenter)
        self.warning.setWordWrap(True)
        self.warning.setFixedWidth(CONTROL_BUTTON_WIDTH_PX + 120)
        self.warning.setStyleSheet(WARNING_STYLE)
        self.warning.hide()
        warning_row = QHBoxLayout()
        warning_row.addStretch(1)
        warning_row.addWidget(self.warning)
        warning_row.addStretch(1)
        outer.addSpacing(CONTROL_SPACING_PX)
        outer.addLayout(warning_row)
        outer.addSpacing(8)
        outer.addLayout(status_row)
        outer.addStretch(1)
        for name in CONTROL_ACTIONS:
            self.add_action(name)

    def set_label(self, name, text):
        self.buttons[name].setText(text)

    def set_active(self, name, active):
        """Yellow while what the button started is running (e.g. a demo)."""
        button = self.buttons[name]
        button.setProperty("active", bool(active))
        button.style().unpolish(button)                  # re-apply the stylesheet for the new property value
        button.style().polish(button)

    def warn(self, text):
        """Show a warning (fades in); an empty text fades it out."""
        self.notice(text, error=True)

    def notice(self, text, error=False):
        """A message on its own line, kept until replaced (the status line is rewritten by every tracking update):
        red for a problem, grey for information. Empty text fades it out."""
        if text:
            self.warning.setStyleSheet(WARNING_STYLE if error else NOTICE_STYLE)
            self.warning.setText(text)
        fade_to(self.warning, bool(text))

    def set_unavailable(self, name, reason):
        """Grey the button out with `reason` as its hover notice (a click only shows the notice); None makes it
        usable again."""
        button = self.buttons[name]
        if bool(reason) == bool(button.property("unavailable")) and (not reason or button.toolTip().endswith(reason)):
            return                                       # unchanged: no restyle on every tracking update
        button.setProperty("unavailable", bool(reason))
        button.setToolTip(f"<b>Not available now</b><br>{reason}" if reason else "")
        button.setCursor(Qt.ForbiddenCursor if reason else Qt.PointingHandCursor)
        button.style().unpolish(button)
        button.style().polish(button)

    def _clicked(self, name):
        button = self.buttons[name]
        if button.property("unavailable"):
            from PyQt5.QtGui import QCursor
            from PyQt5.QtWidgets import QToolTip
            QToolTip.showText(QCursor.pos(), button.toolTip(), button)
            return
        self.action.emit(self.side, name)

    def add_action(self, name):
        button = QPushButton(name)
        # fixed width: narrower side padding than the other pills, so "Stop the position demo" fits
        button.setStyleSheet(PILL_BUTTON_STYLE + "QPushButton { padding-left: 14px; padding-right: 14px; }")
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedWidth(CONTROL_BUTTON_WIDTH_PX)
        button.clicked.connect(lambda: self._clicked(name))
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        row.addStretch(1)
        self.column.addLayout(row)
        self.buttons[name] = button
        return button


class MainPage(QWidget):
    """The two hands side by side; the controls (controller, ellipse sender) come next."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background: {BACKGROUND.name()};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top bar: logo | (empty middle) | window buttons.
        top = QGridLayout()
        top.setContentsMargins(0, 14, 0, 0)                     # symmetric, so the middle column is truly centred
        top.setColumnStretch(0, 1)
        top.setColumnStretch(2, 1)
        logo = Logo()
        logo.setContentsMargins(20, 0, 0, 0)                     # the logo's inset lives in its own column
        top.addWidget(logo, 0, 0, Qt.AlignLeft | Qt.AlignTop)
        self.mode = "glove"                                      # a MODES key (the mode bar)
        # Glove mode needs the MANUS SDK beside the repo and the dongle's USB rule: without them it starts in Camera
        # mode and the Glove button is greyed out, saying why (teleop/manus/stream/manus_sdk_check.py).
        self.glove_problem = self._glove_problem()
        if self.glove_problem is not None:
            self.mode = "camera"
            print(f"[gui] glove mode unavailable: {self.glove_problem}", flush=True)
        self.settings = QSettings(*SETTINGS)
        pipeline = str(self.settings.value("camera_pipeline", "mediapipe"))
        self.pipeline = pipeline if pipeline in PIPELINES else "mediapipe"   # the camera mode's tracker (Dev Mode)
        self.mode_bar = ModeBar(self.mode)
        self.mode_bar.set_unavailable("glove", self.glove_problem)
        self.mode_bar.changed.connect(self.set_mode)
        top.addWidget(self.mode_bar, 0, 1, Qt.AlignHCenter | Qt.AlignTop)
        self.dev_button = DevModeButton()
        self.dev_button.clicked.connect(window.show_dev)
        self.dev_page = None                                     # set by MainWindow
        self.adapters = {side: self._saved_adapter(side) for side in SIDES}   # path, or None: the hand is not used
        self.saved_adapters = dict(self.adapters)                # the defaults; Dev Mode's Save changes them
        # GeoRT runs per (source group, side), from retargetor/GeoRT_checkpoint/selection.json; Dev Mode changes them
        # for this run (the tracking service is restarted with them) and its Save writes them back.
        from teleop.manus.geort.checkpoints import load_selection
        self.pins = load_selection()
        # Each hand's motor gains (teleop torque kp / kd): from its config; Dev Mode changes them for the next motor
        # start, its Save writes them back (motor_process.save_default_gains).
        from motor_process import config_gains
        self.gains, self.gain_overrides = {}, {}
        for side in SIDES:
            try:
                values = config_gains(side)
                self.gains[side] = dict(kp=values["kp"], kd=values["kd"])
                self.gain_overrides[side] = values["joints"]
            except Exception as error:
                print(f"[gui] {side} hand: cannot read its motor gains ({error})", flush=True)
        self.saved_gains = {side: dict(values) for side, values in self.gains.items()}
        self.saved_pins = dict(self.pins)
        # Dev Mode: the torque demo's fingertip-distance safety stop (on unless turned off there; remembered)
        self.tip_stop = str(self.settings.value("torque_demo_tip_stop", "true")).lower() != "false"
        self.friction_fine = str(self.settings.value("friction_calibration_fine", "false")).lower() == "true"
        corner = QHBoxLayout()
        corner.setContentsMargins(0, 0, 0, 0)
        corner.setSpacing(14)
        corner.addWidget(self.dev_button, 0, Qt.AlignVCenter)
        corner.addWidget(WindowButtons(window), 0, Qt.AlignTop)
        top.addLayout(corner, 0, 2, Qt.AlignRight | Qt.AlignTop)
        layout.addLayout(top)

        # Recenter: its own centred row, a little below the top bar (above the hands).
        self.recenter_button = QPushButton("Recenter")
        self.recenter_button.setStyleSheet(PILL_BUTTON_STYLE)
        self.recenter_button.setCursor(Qt.PointingHandCursor)
        self.recenter_button.setToolTip("Put both hands back to the start view")
        self.recenter_button.clicked.connect(self.recenter)
        row = QHBoxLayout()
        row.setContentsMargins(0, RECENTER_TOP_GAP_PX, 0, 0)
        row.addStretch(1)
        row.addWidget(self.recenter_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.hands = QHBoxLayout()
        self.hands.setContentsMargins(40, 10, 40, 10)
        self.hands.setSpacing(40)
        layout.addLayout(self.hands, 1)
        self.panels = {}
        self.controls = {}
        self.motors = {}                                         # side -> MotorProcess
        self.demos = {}                                          # side -> DemoProcess (one at a time per hand)
        self.adapter_watch = QTimer(self)                        # clears "connect the adapter" once it is plugged in
        self.adapter_watch.setInterval(ADAPTER_CHECK_MS)
        self.adapter_watch.timeout.connect(self._check_adapters)
        self.adapter_missing = set()                             # sides warned about a missing adapter
        self.link = None                                         # TrackingLink: the tracking service process
        self.snapshot = None                                     # its newest state (tracking_service.py)
        self.link_error = None                                   # the service died: its exit, for the notes
        self.pending_source = None                               # a source to start once the old service exits
        self.fake_sides = None                                   # tests: synthetic gloves in the service
        self.disconnecting = dict.fromkeys(SIDES, False)
        self.selected = None
        self.animating = False

        self._layout = layout
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        layout.addLayout(bottom)

    def recenter(self):
        """Both hands back to their start-up view."""
        for panel in self.panels.values():
            panel.view.recenter()

    def on_action(self, side, name):
        """A hand's control button was clicked."""
        if name == GLOVE:
            self.toggle_glove(side)
        elif name in DEMO_KINDS:
            self.toggle_demo(side, name)
        else:
            print(f"[gui] {side} hand: {name} (not wired yet)", flush=True)

    # --- the tracking service (its own process: source, retargeting, controller sessions) ---------------------
    def start_search(self):
        """Start the tracking service for the current source. The search, auto-connect, alignment and the controller
        sessions run there; this page only shows its state and sends it commands."""
        from tracking_link import TrackingLink
        link = self.link = TrackingLink(self)
        link.state.connect(lambda snapshot, link=link: self._state(link, snapshot))
        link.output.connect(lambda line: print(line, flush=True))
        link.stopped.connect(lambda code, link=link: self._link_stopped(link, code))
        self.snapshot, self.link_error = None, None
        self.disconnecting = dict.fromkeys(SIDES, False)
        link.start(self.source, self.fake_sides, checkpoints=self._pin_choice())
        for side in SIDES:
            if not self.used(side):
                link.send(cmd="disconnect", side=side)           # queued until the service is up
        for side in self.panels:
            self._refresh_glove(side)

    @property
    def source(self):
        """What the service runs: "glove", "camera" (MediaPipe), "wilor" or "wilor_mini"."""
        if self.mode == "glove":
            return "glove"
        return PIPELINE_SOURCE[self.pipeline]

    def set_mode(self, mode):
        """Glove <-> camera (the mode bar)."""
        self.mode_bar.set_mode(mode)
        self._switch(mode, self.pipeline)

    def set_pipeline(self, pipeline):
        """The camera mode's tracker (Dev Mode); remembered for the next start. Takes effect at once in camera mode."""
        if self.dev_page is not None:
            self.dev_page.set_pipeline(pipeline)
        self.settings.setValue("camera_pipeline", pipeline)
        self._switch(self.mode, pipeline)

    # --- each hand's adapter (Dev Mode) -------------------------------------------------------------------------------
    def _saved_adapter(self, side):
        """This hand's default adapter: None when saved as not used ("none"), otherwise the one in the hand's config
        (Dev Mode's Save writes it there). A path remembered by an earlier version of the GUI still counts."""
        value = self.settings.value(f"adapter/{side}", None)
        if value == "none":
            return None
        if value:
            return str(value)
        try:
            from motor_process import default_adapter
            return default_adapter(side)
        except Exception as error:
            print(f"[gui] {side} hand: cannot read its config's adapter ({error})", flush=True)
            return None

    def used(self, side):
        return self.adapters.get(side) is not None

    def set_adapter(self, side, path):
        """Dev Mode picked this hand's adapter (a path), or None: the hand is not used. For this run: the motor and
        demos open it from their next start (Save makes it the default). None also stops this hand's motor / demo and
        its tracking."""
        before = self.adapters.get(side)
        self.adapters[side] = path
        print(f"[gui] {side} hand adapter: {path or 'none (hand not used)'}", flush=True)
        controls = self.controls.get(side)
        if path is None:
            demo, motor = self.demos.get(side), self.motors.get(side)
            if demo is not None and demo.running():
                demo.stop()                                      # back to rest, out of torque mode
            if motor is not None and motor.running():
                self._send(cmd="detach_motor", side=side)
                QTimer.singleShot(300, motor.stop)
            self._send(cmd="disconnect", side=side)
        elif before is None:
            self._send(cmd="connect", side=side)                 # back in use: track it again
        if controls is not None:
            self._clear_adapter_warning(side)
            controls.notice("" if path else f"The {side.lower()} hand is not used (Dev Mode).")
        if side in self.panels:
            self._refresh_glove(side)

    def set_tip_stop(self, on):
        """Dev Mode: the torque demo's fingertip-distance safety stop on / off, from the next demo start."""
        self.tip_stop = bool(on)
        self.settings.setValue("torque_demo_tip_stop", "true" if on else "false")
        print(f"[gui] torque demo fingertip-distance safety stop: {'on' if on else 'OFF'}", flush=True)

    def set_friction_fine(self, on):
        """Dev Mode: Calibrate friction runs the fine calibration (friction_calibrate.py --fine), from its next start."""
        self.friction_fine = bool(on)
        self.settings.setValue("friction_calibration_fine", "true" if on else "false")
        print(f"[gui] friction calibration: {'FINE' if on else 'normal'}", flush=True)

    # --- GeoRT models (Dev Mode) -----------------------------------------------------------------------------------
    def _pin_choice(self):
        from teleop.manus.geort.checkpoints import SIDES as GEORT_SIDES, SOURCES
        return {(group, side): self.pins.get((group, side)) for group in SOURCES for side in GEORT_SIDES}

    def set_checkpoint(self, group, side, run):
        """Dev Mode chose a GeoRT run (None: the newest) for this hand and source group: the tracking service is
        restarted with it at once (a running motor stays on, idle, and is driven again when the hand is back)."""
        if run:
            self.pins[(group, side)] = run
        else:
            self.pins.pop((group, side), None)
        print(f"[gui] GeoRT model, {side.lower()} hand, {group}: {run or 'newest'}", flush=True)
        self._restart_tracking()

    def _restart_tracking(self):
        link = self.link
        if link is not None and link.running():
            self.pending_source = self.source
            self.snapshot = None
            for side in self.panels:
                self._refresh_glove(side)
            link.stop()
        elif self.panels:
            self.start_search()

    def save_checkpoints(self):
        """Dev Mode's Save: the chosen GeoRT runs become the defaults (selection.json), for the GUI and the sender."""
        from teleop.manus.geort.checkpoints import save_selection
        try:
            path = save_selection(self._pin_choice())
        except Exception as error:
            print(f"[gui] GeoRT selection NOT saved: {error}", flush=True)
            if self.dev_page is not None:
                self.dev_page.set_runs_saved(self.saved_pins, f"Not saved: {error}", error=True)
            return
        self.saved_pins = dict(self.pins)
        print(f"[gui] GeoRT selection saved to {path}", flush=True)
        if self.dev_page is not None:
            self.dev_page.set_runs_saved(self.saved_pins, f"Saved as default ({path.name}).")

    def set_gains(self, side, kp, kd):
        """Dev Mode: this hand's motor kp / kd, used from its next motor start."""
        self.gains[side] = dict(kp=kp, kd=kd)
        motor = self.motors.get(side)
        if motor is not None and motor.running() and side in self.panels:
            self.panels[side].show_message(f"New motor gains (kp {kp:g}, kd {kd:g}): click Stop {side}, then Start "
                                           f"{side} to use them.")

    def save_gains(self):
        """Dev Mode's Save: write each changed hand's kp / kd into its teleop torque config."""
        from motor_process import save_default_gains
        done, problems = [], []
        for side, values in self.gains.items():
            if values == self.saved_gains.get(side):
                continue
            try:
                changed = save_default_gains(side, values["kp"], values["kd"])
                self.saved_gains[side] = dict(values)
                done.append(f"{side.lower()} hand kp {values['kp']:g}, kd {values['kd']:g}"
                            + (f" ({Path(changed).name})" if changed else ""))
                print(f"[gui] {side} hand motor gains saved: kp {values['kp']:g}, kd {values['kd']:g} -> {changed}",
                      flush=True)
            except Exception as error:
                problems.append(f"{side.lower()} hand: {error}")
                print(f"[gui] {side} hand motor gains NOT saved: {error}", flush=True)
        if self.dev_page is not None:
            message = ("Not saved: " + "; ".join(problems)) if problems else ("Saved as default: " + "; ".join(done))
            self.dev_page.set_gains_saved(self.saved_gains, message, error=bool(problems))

    def save_adapters(self):
        """Dev Mode's Save: the chosen adapters become the hands' defaults. An adapter is written into the hand's
        config files (motor_process.save_default_adapter: also used from the terminal); None is remembered by the GUI
        (the configs keep their adapter)."""
        from motor_process import default_adapter, save_default_adapter
        done, problems = [], []
        for side in SIDES:
            path = self.adapters.get(side)
            if path == self.saved_adapters.get(side):
                continue
            try:
                if path is None:
                    self.settings.setValue(f"adapter/{side}", "none")
                    done.append(f"{side.lower()} hand: not used")
                else:
                    files = save_default_adapter(side, path)
                    self.settings.remove(f"adapter/{side}")       # the config is the default now
                    from dev_page import adapter_serial
                    done.append(f"{side.lower()} hand: {adapter_serial(path)}"
                                + (f" ({', '.join(Path(f).name for f in files)})" if files else ""))
                    print(f"[gui] {side} hand default adapter saved: {path} -> {', '.join(files) or 'already set'}",
                          flush=True)
                self.saved_adapters[side] = path
            except Exception as error:
                problems.append(f"{side.lower()} hand: {error}")
                print(f"[gui] {side} hand default adapter NOT saved: {error}", flush=True)
        self.settings.sync()
        defaults = {}
        for side in SIDES:
            try:
                defaults[side] = default_adapter(side)
            except Exception:
                defaults[side] = None
        if self.dev_page is not None:
            message = ("Not saved: " + "; ".join(problems)) if problems else ("Saved as default: " + "; ".join(done))
            self.dev_page.set_saved(self.saved_adapters, defaults, message, error=bool(problems))

    def _switch(self, mode, pipeline):
        """Change the tracking source: the service is restarted on the new one (both sides disconnect). A running
        motor stays on (idle) and is driven again once the hand is found on the new source."""
        before = self.source
        self.mode, self.pipeline = mode, pipeline
        if self.source == before:                               # e.g. a new pipeline chosen in glove mode
            return
        for panel in self.panels.values():                       # no pose left over from the other source
            panel.view.pose([0.0] * 16)
            panel.view.show_skeleton(None)
        for side, controls in self.controls.items():
            controls.set_label(GLOVE, SOURCE_TEXT[self.source]["connect"])
            controls.status.setText("")
        print(f"[gui] tracking source: {MODES[mode].lower()}"
              + (f" ({PIPELINES[pipeline]})" if mode == "camera" else ""), flush=True)
        link = self.link
        if link is not None and link.running():                  # the old service must release the source first
            self.pending_source = self.source
            self.snapshot = None
            for side in self.panels:
                self._refresh_glove(side)
            link.stop()
        else:
            self.start_search()

    def _link_stopped(self, link, code):
        if link is not self.link:
            return
        self.link = None
        if self.pending_source is not None:
            self.pending_source = None
            self.start_search()
            return
        self.link_error = f"tracking service exited (code {code})"
        print(f"[gui] {self.link_error}", flush=True)
        self.snapshot = None
        for side in self.panels:
            self._refresh_glove(side)

    @staticmethod
    def _glove_problem():
        try:
            from teleop.manus.stream.manus_sdk_check import glove_unavailable
            return glove_unavailable()
        except Exception as error:                               # the check itself must never stop the GUI
            print(f"[gui] glove check failed ({error}); offering Glove mode", flush=True)
            return None

    def _glove_failed(self, error):
        """The glove source failed while running: grey Glove out with the reason (until the GUI restarts)."""
        reason = self._glove_problem() or ("The glove could not be started: " + (error or "").splitlines()[0][:160]
                                           + "<br>See the terminal or ~/.local/state/artly-os/gui.log.")
        if reason != self.glove_problem:
            self.glove_problem = reason
            self.mode_bar.set_unavailable("glove", reason)

    def _state(self, link, snapshot):
        """A snapshot from the service: the models, the status lines, the labels, the notes."""
        if link is not self.link or self.pending_source is not None:   # the old service, on its way out
            return
        previous, self.snapshot = self.snapshot, snapshot
        if snapshot.get("source") == "glove" and snapshot.get("phase") == "failed":
            self._glove_failed(snapshot.get("error"))
        for side in self.panels:
            state = snapshot["sides"][side]
            view = self.panels[side].view
            if state["pose"] is not None:
                view.pose(state["pose"])
            overlay = state["skeleton"]
            view.show_skeleton(None if overlay is None else
                               (np.asarray(overlay["points"], dtype=float), np.asarray(overlay["segments"], dtype=float)))
            controls = self.controls[side]
            if state["error"]:
                controls.status.setText(f"{MODES[self.mode]} stopped: {state['error']}")
            elif state["status"]:
                controls.status.setText(state["status"])
            if state["connected"]:
                controls.set_label(GLOVE, "Disconnecting ..." if self.disconnecting[side]
                                   else SOURCE_TEXT[self.source]["disconnect"])
            else:
                self.disconnecting[side] = False
                controls.set_label(GLOVE, SOURCE_TEXT[self.source]["connect"])
            if previous is None or self._refresh_key(previous, side) != self._refresh_key(snapshot, side):
                self._refresh_glove(side)                       # notes, ghost and buttons: only on a change

    @staticmethod
    def _refresh_key(snapshot, side):
        state = snapshot["sides"][side]
        return (snapshot["phase"], snapshot["error"], snapshot["found"].get(side) is not None, state["connected"],
                state["alignment"])

    def _connected(self, side):
        return bool(self.snapshot is not None and self.snapshot["sides"][side]["connected"])

    def _refresh_glove(self, side):
        """The hand solid with the skeleton button (hand found and connected), or see-through with a note."""
        panel = self.panels[side]
        snapshot = self.snapshot
        found = snapshot is not None and snapshot["found"].get(side) is not None
        connected = self._connected(side)
        phase = snapshot["phase"] if snapshot is not None else "starting"
        error = self.link_error or (snapshot["error"] if snapshot is not None else None)
        text = SOURCE_TEXT[self.source]
        name = side.lower()
        self.controls[side].set_unavailable(GLOVE, self._connect_problem(side, found, connected, phase, error))
        if not self.used(side):
            panel.note.setText(f"The {name} hand is not used.\nChoose its adapter in Dev Mode to use it.")
            panel.set_ghost(True)
            fade_to(panel.note, True)
            for widget in (panel.skeleton_button, panel.align_button, panel.hint):
                fade_to(widget, False)
            return
        if error is not None:
            first = (error.splitlines() or [""])[0]
            panel.note.setText(text["failed"].format(error=first[:200]))
        elif phase in ("starting", "searching"):
            panel.note.setText(text["searching"].format(side=name))
        else:
            panel.note.setText(text["missing"].format(side=name))
        panel.set_ghost(not found)
        fade_to(panel.note, not found)
        live = found and connected
        fade_to(panel.skeleton_button, live)
        camera = self.mode == "camera"
        fade_to(panel.align_button, live and camera)
        alignment = snapshot["sides"][side]["alignment"] if snapshot is not None else None
        hint = ALIGN_HINTS.get(alignment or "ok", "").format(side=name) if live and camera else ""
        if hint:
            panel.hint.setText(hint)                       # kept while fading out, so it does not blank at once
        fade_to(panel.hint, bool(hint))

    def _connect_problem(self, side, found, connected, phase, error):
        """Why Connect to glove / camera cannot be used for this side now, or None. Disconnect (while connected)
        always works."""
        name = side.lower()
        if connected:
            return None
        if not self.used(side):
            return f"The {name} hand is not used. Choose its adapter in Dev Mode to use it."
        if error is not None:
            return f"The {MODES[self.mode].lower()} stopped: {(error.splitlines() or [''])[0][:160]}"
        if found:
            return None
        if self.mode == "camera":
            if phase in ("starting", "searching"):
                return "The camera is still starting."
            return f"No {name} hand in view. Show your {name} hand to the camera."
        if phase in ("starting", "searching"):
            return f"Still searching for the {name} glove."
        return (f"No {name} glove found. Switch the {name} MANUS glove on and check that its dongle is plugged in; "
                f"this becomes clickable as soon as the glove is found.")

    def _send(self, **command):
        if self.link is not None:
            self.link.send(**command)

    # --- glove -> GeoRT -> model ---------------------------------------------------------------------------------
    def toggle_glove(self, side):
        if not self.used(side):
            self.controls[side].warn(f"The {side.lower()} hand is not used: choose its adapter in Dev Mode.")
            return
        if self._connected(side):
            self.disconnecting[side] = True                 # stays disconnected until Connect is clicked again
            self.controls[side].set_label(GLOVE, "Disconnecting ...")
            self._send(cmd="disconnect", side=side)
            return
        self._send(cmd="connect", side=side)

    def align(self, side):
        """Camera mode: capture and save a new camera alignment for this side."""
        self._send(cmd="align", side=side)

    def toggle_skeleton(self, side):
        view = self.panels[side].view
        view.set_skeleton_enabled(not view.skeleton_enabled)
        self.panels[side].skeleton_button.setText(SKELETON_LABELS[view.skeleton_enabled])
        self._send(cmd="skeleton", side=side, enabled=view.skeleton_enabled)   # not computed while hidden

    # --- the hand controller -------------------------------------------------------------------------------------
    def toggle_motor(self, side):
        """Start <side>: start this hand's motor, or stop it while it runs. Its messages show under the hand."""
        from motor_process import MotorProcess
        motor = self.motors.get(side)
        panel = self.panels[side]
        if motor is not None and motor.running():
            self._send(cmd="detach_motor", side=side)       # idle the controller first, then stop it
            QTimer.singleShot(300, motor.stop)
            panel.set_motor_state("stopping")
            return
        other = "Right" if side == "Left" else "Left"
        if self.adapters.get(other) == self.adapters.get(side) and any(
                p is not None and p.running() for p in (self.motors.get(other), self.demos.get(other))):
            panel.show_message(f"The {other.lower()} hand is using the same adapter: stop it first, or give each "
                               "hand its own adapter in Dev Mode.", error=True)
            return
        demo = self.demos.get(side)
        if demo is not None and demo.running():                  # the demo has the hand's adapter
            panel.show_message(f"Stop the {demo.label.lower()} first (⋯): it is using the {side.lower()} hand.",
                               error=True)
            return
        if not self._adapter_ready(side, where="hand"):
            return
        gains = self.gains.get(side)
        motor = MotorProcess(side, self, adapter=self.adapters[side],
                             gains=gains if gains != self.saved_gains.get(side) else None)   # None: the config's
        motor.output.connect(lambda line: print(line, flush=True))
        motor.stopped.connect(lambda code, side=side: self._motor_stopped(side, code))
        self.motors[side] = motor
        motor.start()
        panel.set_motor_state("running")
        self._attach(side)                                  # the service drives it as soon as the hand is connected
        source = "glove" if self.mode == "glove" else "camera"
        panel.show_message("Motor started." if self._connected(side) else
                           f"Motor started (idle): it moves once the {source} sees your {side.lower()} hand.")

    def _say(self, side, text, error=True, where="controls"):
        """A message for this hand: under the hand ("hand", the motor's Start button) or in its "..." panel."""
        if where == "hand":
            self.panels[side].show_message(text, error)
        else:
            self.controls[side].notice(text, error)

    def _adapter_ready(self, side, where="controls"):
        """Whether this hand's adapter (Dev Mode's choice) is plugged in; if not, warn (under the hand or in its
        panel: `where`) and keep looking for it."""
        from motor_process import adapter_for, adapter_name
        self.adapter_warned_at = getattr(self, "adapter_warned_at", {})
        self.adapter_warned_at[side] = where
        warn = lambda text: self._say(side, text, True, where)   # noqa: E731
        if not self.used(side):
            warn(f"The {side.lower()} hand is not used (adapter: None).\n"
                 "Choose its adapter in Dev Mode to use it.")
            return False
        try:
            path, present, attached = adapter_for(side, self.adapters[side])
        except Exception as error:                       # an unreadable config: say so instead of starting
            warn(f"Cannot read the {side.lower()} hand's config: {error}")
            return False
        if present and not os.access(path, os.R_OK | os.W_OK):
            # Plugged in but this login may not open it: not in the adapter's group (dialout), or added to it and not
            # logged in again since. The controller would only fail with a PermissionError traceback.
            import grp
            try:
                group = grp.getgrgid(os.stat(path).st_gid).gr_name
            except (KeyError, OSError):
                group = "dialout"
            warn(f"No permission to open the {side.lower()} hand's adapter.\n"
                 f"Run ./setup_envs.sh hand (adds you to '{group}'), then log out and back in.")
            print(f"[gui] {side} motor not started: no read/write permission on {path} (group {group}); "
                  f"fix: sudo usermod -aG {group} $USER, then log in again", flush=True)
            return False
        if present:
            self._clear_adapter_warning(side)
            return True
        others = ", ".join(adapter_name(a) for a in attached) or "none"
        warn(f"Please connect the {side.lower()} hand's adapter ({adapter_name(path)}),\n"
             f"or choose another one in Dev Mode. Plugged in now: {others}.")
        print(f"[gui] {side} motor not started: adapter {path} is not plugged in (attached: {others})", flush=True)
        self.adapter_missing.add(side)
        self.adapter_watch.start()
        return False

    def _clear_adapter_warning(self, side):
        if side in self.adapter_missing:
            self.adapter_missing.discard(side)
            self._say(side, "", where=getattr(self, "adapter_warned_at", {}).get(side, "controls"))
        if not self.adapter_missing:
            self.adapter_watch.stop()

    def _check_adapters(self):
        """While a warning is up: clear it as soon as that adapter appears (Start Left / Start Right works again)."""
        from motor_process import adapter_for
        for side in list(self.adapter_missing):
            try:
                if self.used(side) and adapter_for(side, self.adapters[side])[1]:
                    where = getattr(self, "adapter_warned_at", {}).get(side, "controls")
                    self._clear_adapter_warning(side)
                    self._say(side, f"{side} hand adapter connected" + (f": click Start {side}." if where == "hand"
                                                                        else "."), error=False, where=where)
            except Exception:
                pass

    # --- the demos (their own processes, talking to the hand directly) ---------------------------------------------
    def toggle_demo(self, side, name):
        """Start the demo for this hand, looping until the same button is clicked again (then it eases back to rest
        and, for the torque demo, leaves torque mode first)."""
        from demo_process import DEMOS, DemoProcess
        controls = self.controls[side]
        demo = self.demos.get(side)
        if demo is not None and demo.running():
            if demo.label != name:
                controls.warn(f"Stop {demo.noun} first.")
                return
            demo.stop()
            controls.set_label(name, "Stopping ...")
            return
        if name == TORQUE_DEMO:
            from demo_process import torque_demo_config
            calibrated = torque_demo_config(side).startswith("demo/")
        motor = self.motors.get(side)
        if motor is not None and motor.running():                # the controller has the hand's adapter
            controls.warn(f"Click Stop {side} first: {DEMOS[DEMO_KINDS[name]][3]} drives the {side.lower()} hand "
                          "itself.")
            return
        if not self._adapter_ready(side):
            return
        controls.warn("")
        other = "Right" if side == "Left" else "Left"
        if self.adapters.get(other) == self.adapters.get(side) and any(
                p is not None and p.running() for p in (self.motors.get(other), self.demos.get(other))):
            controls.warn(f"The {other.lower()} hand is using the same adapter: stop it first, or give each hand its "
                          "own adapter in Dev Mode.")
            return
        again = None
        if name == FRICTION:
            again = self._ask_friction_checkpoint(side)
            if again == "cancel":
                return
        demo = DemoProcess(side, DEMO_KINDS[name], self, adapter=self.adapters[side], tip_stop=self.tip_stop,
                           fine=self.friction_fine, again=again)
        demo.output.connect(lambda line: print(line, flush=True))
        demo.stopped.connect(lambda code, side=side, demo=demo: self._demo_stopped(side, demo, code))
        self.demos[side] = demo
        demo.start()
        controls.set_label(name, demo.running_label)
        controls.set_active(name, True)
        if name == CALIBRATE:
            controls.notice(f"Calibrating the {side.lower()} hand for the torque demo (about 10 minutes). "
                            "Keep clear of the fingers; click again to stop.")
        elif name == FRICTION and again == "resume":
            controls.notice(f"Continuing the {side.lower()} hand's friction calibration where it stopped (progress in "
                            "the terminal). Keep clear of the fingers; click again to stop (progress is kept).")
        elif name == FRICTION:
            controls.notice(f"Calibrating the {side.lower()} hand's friction for torque control"
                            + (" (FINE, Dev Mode: several hours for all joints)" if self.friction_fine else "")
                            + ": every joint, one at a time (this takes a long time; progress in the terminal). Keep "
                            "clear of the fingers; click again to stop (progress is kept: you can continue later).")
        elif name == TORQUE_DEMO:
            controls.notice(("Torque demo with this hand's calibration. Click again to stop." if calibrated else
                             f"Torque demo without a calibration (click {CALIBRATE} first for a better one). "
                             "Click again to stop.")
                            + ("" if self.tip_stop else "\nFingertip safety stop is OFF (Dev Mode)."))
        else:
            controls.notice(f"{name} running. Click again to stop.")
        controls.status.setText(f"{name} running on the {side.lower()} hand: click again to stop.")

    def _ask_friction_checkpoint(self, side):
        """A stopped friction calibration of this hand: "resume", "restart" or "cancel"; None when there is none."""
        from demo_process import friction_checkpoint
        saved = friction_checkpoint(side)
        if saved is None:
            return None
        from PyQt5.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Friction calibration")
        box.setIcon(QMessageBox.Question)
        box.setText(f"The {side.lower()} hand has a stopped friction calibration.")
        box.setInformativeText(f"{saved}.\n\nContinue: carry on where it stopped, with its own settings (normal or "
                               "fine as it was started; the Dev Mode choice does not change it).\nStart over: discard "
                               "it and calibrate again from the first joint (joints already measured keep their "
                               "results until measured again).")
        resume = box.addButton("Continue", QMessageBox.AcceptRole)
        restart = box.addButton("Start over", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(resume)
        box.exec_()
        clicked = box.clickedButton()
        return "resume" if clicked is resume else "restart" if clicked is restart else "cancel"

    def _demo_stopped(self, side, demo, code):
        if self.demos.get(side) is not demo:
            return
        controls = self.controls[side]
        controls.set_label(demo.label, demo.label)
        controls.set_active(demo.label, False)
        if demo.kind == "friction" and code != 0:
            from demo_process import friction_checkpoint
            saved = friction_checkpoint(side)
            if saved:
                text = ("Friction calibration stopped" if demo.stop_requested else
                        "Friction calibration ended with an error (see the terminal)")
                controls.notice(f"{text}. Progress saved: {saved}. Click {FRICTION} to continue it.",
                                error=not demo.stop_requested)
                if not demo.stop_requested:
                    self._adapter_ready(side)
                return
        if code == 0 and demo.kind == "calibrate":
            controls.notice("Calibration done: the torque demo now uses it.")
        elif code == 0 and demo.kind == "friction":
            controls.notice("Friction calibration done and written to the hand's torque config: Start the motor to "
                            "use it.")
        elif code == 0 or demo.stop_requested:
            controls.notice(f"{demo.noun[0].upper()}{demo.noun[1:]} stopped.")
        else:
            controls.warn(f"{demo.noun[0].upper()}{demo.noun[1:]} ended with an error (exit code {code}): "
                          "see the terminal.")
            self._adapter_ready(side)                    # e.g. the adapter was unplugged

    def _attach(self, side):
        from motor_process import sockets_for
        self._send(cmd="attach_motor", side=side, **sockets_for(side))

    def _motor_stopped(self, side, code):
        self._send(cmd="detach_motor", side=side)
        panel = self.panels[side]
        panel.set_motor_state("idle")
        if code == 0:
            panel.show_message("Motor stopped.")
        else:                                            # e.g. the adapter was unplugged while it ran
            panel.show_message(f"The motor stopped with an error (exit code {code}): see the terminal or "
                               "~/.local/state/artly-os/gui.log.", error=True)
            self._adapter_ready(side, where="hand")

    def shutdown(self):
        """Window closing: the service idles every controller session and exits, then the controllers are stopped."""
        for demo in self.demos.values():                 # back to rest (and out of torque mode) first
            demo.stop(wait=True)
        if self.link is not None:
            self.link.stop(wait=True)
        for motor in self.motors.values():
            motor.stop(wait=True)

    def select(self, side):
        """Show only `side`'s hand, with its controls in the other hand's place; the same side again shows both.
        Animated: what leaves fades out quickly, then what arrives fades in."""
        if self.animating:                                       # a click mid-transition is ignored
            self.panels[side].more_button.setChecked(self.selected == side)
            return
        before = self.selected
        self.selected = side if before is None else None         # from one hand, any click goes back to both
        for name, panel in self.panels.items():
            panel.more_button.setChecked(self.selected == name)
        if before is None:                                       # both hands -> one: other hand out, controls in
            other = "Right" if side == "Left" else "Left"
            leaving, arriving = [self.panels[other]], [self.controls[side]]
        else:                                                    # one hand -> both: controls out, other hand in
            other = "Right" if before == "Left" else "Left"
            leaving, arriving = [self.controls[before]], [self.panels[other]]
        self._transition(leaving, arriving)

    def _transition(self, leaving, arriving):
        self.animating = True

        def fade_in():
            for widget in leaving:
                widget.hide()                                    # a hand keeps its snapshot for its next fade-in
            for widget in arriving:                              # same layout pass: the chosen hand never moves
                widget.show()
            for i, widget in enumerate(arriving):
                last = i == len(arriving) - 1
                fade(widget, 0.0, 1.0, done=(lambda w=widget, last=last: self._arrived(w, last)))

        for i, widget in enumerate(leaving):
            if isinstance(widget, HandPanel):
                widget.freeze()
            fade(widget, 1.0, 0.0, done=fade_in if i == len(leaving) - 1 else None)

    def _arrived(self, widget, last):
        if isinstance(widget, HandPanel):
            widget.thaw()
        if last:
            self.animating = False


def _build_hands(page):
    """Create the left and right hand views (about half a second: meshes are loaded from the URDF) and their control
    panels, between empty outer space so the pair sits toward the middle of the window. Row order:

        [space | right controls | left hand | right hand | left controls | space]

    Both hands: the control panels are hidden. Left selected: the right hand hides and the left controls take its
    place (right of the left hand); right selected: the mirror. The visible pair always splits the middle in the
    proportions HAND_STRETCH : HAND_STRETCH, so the chosen hand does not move."""
    for side in ("Left", "Right"):
        page.panels[side] = HandPanel(side)
        page.panels[side].selected.connect(page.select)
        page.panels[side].start.connect(page.toggle_motor)
        page.panels[side].skeleton_button.clicked.connect(lambda _=False, side=side: page.toggle_skeleton(side))
        page.panels[side].align_button.clicked.connect(lambda _=False, side=side: page.align(side))
        page.controls[side] = ControlPanel(side)
        page.controls[side].action.connect(page.on_action)
        page.controls[side].hide()
    page.hands.addStretch(OUTER_STRETCH)
    page.hands.addWidget(page.controls["Right"], HAND_STRETCH)
    page.hands.addWidget(page.panels["Left"], HAND_STRETCH)
    page.hands.addWidget(page.panels["Right"], HAND_STRETCH)
    page.hands.addWidget(page.controls["Left"], HAND_STRETCH)
    page.hands.addStretch(OUTER_STRETCH)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)                      # still used by the taskbar and alt-tab
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.resize(1200, 800)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {BACKGROUND.name()}; }}")   # fade in over cream, not grey
        self.setCentralWidget(self.stack)
        self.splash = Splash()
        self.page = MainPage(self)
        self.stack.addWidget(self.splash)
        self.stack.addWidget(self.page)
        self.dev_page = self._build_dev_page()
        self.stack.addWidget(self.dev_page)

        # The main page fades in after the splash has turned cream.
        self._opacity = QGraphicsOpacityEffect(self.page)
        self._opacity.setOpacity(0.0)
        self.page.setGraphicsEffect(self._opacity)
        self._fade_in = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade_in.setDuration(400)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.finished.connect(lambda: self.page.setGraphicsEffect(None))   # crisp text afterwards

        self.splash.finished.connect(self._enter_main)
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_full_screen)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

    def _build_dev_page(self):
        from dev_page import DevPage
        from motor_process import default_adapter, list_adapters
        defaults = {}
        for side in SIDES:
            try:
                defaults[side] = default_adapter(side)
            except Exception:
                defaults[side] = None
        from teleop.manus.geort.checkpoints import list_runs
        page = DevPage(WindowButtons(self), PIPELINES, self.page.pipeline, self.page.adapters, defaults, list_adapters,
                       tip_stop=self.page.tip_stop, pins=self.page.pins, list_runs=list_runs,
                       gains=self.page.gains, gain_overrides=self.page.gain_overrides,
                       friction_fine=self.page.friction_fine)
        page.setAutoFillBackground(True)
        page.setStyleSheet(f"background: {BACKGROUND.name()};")
        page.back.connect(self.show_main)
        page.pipeline_changed.connect(self.page.set_pipeline)
        page.adapter_changed.connect(self.page.set_adapter)
        page.tip_stop_changed.connect(self.page.set_tip_stop)
        page.friction_fine_changed.connect(self.page.set_friction_fine)
        page.save_requested.connect(self.page.save_adapters)
        page.checkpoint_changed.connect(self.page.set_checkpoint)
        page.checkpoints_save_requested.connect(self.page.save_checkpoints)
        page.gains_changed.connect(self.page.set_gains)
        page.gains_save_requested.connect(self.page.save_gains)
        self.page.dev_page = page
        return page

    def show_dev(self):
        self.stack.setCurrentWidget(self.dev_page)

    def show_main(self):
        self.stack.setCurrentWidget(self.page)

    def showEvent(self, event):
        super().showEvent(event)
        if self.stack.currentWidget() is self.splash and not getattr(self, "_started", False):
            self._started = True
            QTimer.singleShot(0, self._prepare)          # after the yellow has been painted

    def _prepare(self):
        """Load the hand models while the yellow shows, then start the fade."""
        _build_hands(self.page)
        self.page.start_search()
        self.splash.start()

    def _enter_main(self):
        self.stack.setCurrentWidget(self.page)
        self._fade_in.start()

    # --- window handling without a title bar -------------------------------------------------------------------
    def mousePressEvent(self, event):
        """Drag the empty background to move the window (the window manager does the move)."""
        if event.button() == Qt.LeftButton and not self.isMaximized() and not self.isFullScreen():
            handle = self.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_from = event.globalPos() - self.frameGeometry().topLeft()   # fallback: move it ourselves
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        drag = getattr(self, "_drag_from", None)
        if drag is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - drag)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_from = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximised()
        super().mouseDoubleClickEvent(event)

    def toggle_maximised(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def toggle_full_screen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def closeEvent(self, event):
        self.page.shutdown()
        super().closeEvent(event)
