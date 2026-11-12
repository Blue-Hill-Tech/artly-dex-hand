"""AAC hand demo GUI. Run with the hand environment:

    hand/.hand/bin/python gui/main.py

or from the app menu / desktop icon that gui/install_launcher.sh adds (./setup_envs.sh hand runs it).
"""
import sys
from pathlib import Path

from PyQt5.QtGui import QFontDatabase, QIcon
from PyQt5.QtWidgets import QApplication

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))          # the GUI's modules (gui/tools/)
from main_window import MainWindow  # noqa: E402

FONT_FILE = HERE / "assets" / "AnthropicSerif-Display-MediumItalic-Static.otf"
FONT_SIZE_PT = 11
ICON_FILE = HERE / "assets" / "aac_hand_icon.png"
# The name also sets the window's WM_CLASS ("Artly-os"), which the launcher's StartupWMClass matches, so the dock
# shows the running window under the launcher's icon. QSettings use their own explicit names (main_window.SETTINGS).
APP_NAME = "Artly-os"


def use_font(app):
    """Make the bundled font the application font, so every widget uses it. Falls back to the system font (with a
    message) if the file is missing or unreadable."""
    font_id = QFontDatabase.addApplicationFont(str(FONT_FILE))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if not families:
        print(f"[gui] could not load {FONT_FILE.name}; using the system font", file=sys.stderr)
        return
    database = QFontDatabase()
    family = families[0]
    style = database.styles(family)[0]               # the file has one face: "Medium Italic"
    app.setFont(database.font(family, style, FONT_SIZE_PT))


def main():
    QApplication.setApplicationName(APP_NAME)
    QApplication.setDesktopFileName("artly-os")     # the launcher's file name (Wayland app id)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(ICON_FILE)))
    use_font(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
