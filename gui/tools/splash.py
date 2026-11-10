"""Start-up screen: the window opens in yellow, the colour fades to the background cream, then `finished` is emitted."""
from PyQt5.QtCore import QEasingCurve, QTimer, QVariantAnimation, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

YELLOW = QColor("#F9E076")
BACKGROUND = QColor("#FAF8EF")   # the GUI's background (cream); the fade ends here


class Splash(QWidget):
    """Fills its area with a colour that goes from yellow to the background cream.

    hold_ms: how long the yellow stays before fading; fade_ms: the fade itself."""

    finished = pyqtSignal()

    def __init__(self, hold_ms=500, fade_ms=1200, parent=None):
        super().__init__(parent)
        self._color = QColor(YELLOW)
        self._fade = QVariantAnimation(self)
        self._fade.setStartValue(QColor(YELLOW))
        self._fade.setEndValue(QColor(BACKGROUND))
        self._fade.setDuration(fade_ms)
        self._fade.setEasingCurve(QEasingCurve.InOutCubic)
        self._fade.valueChanged.connect(self._set_color)
        self._fade.finished.connect(self.finished)
        self._hold_ms = hold_ms

    def start(self):
        QTimer.singleShot(self._hold_ms, self._fade.start)

    def _set_color(self, color):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._color)
