#!/usr/bin/env bash
# Add the AAC hand demo GUI to the app menu (Activities / Show Applications, where it can be pinned to the dock) and put
# an icon on the desktop. Both run gui/launch.sh of THIS checkout; run again after moving the repo.
#   gui/install_launcher.sh            menu entry + desktop icon
#   gui/install_launcher.sh --remove   take both away
set -euo pipefail
GUI="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT="$(cd "$GUI/.." && pwd)"
NAME=artly-os
OLD_NAMES=(aac-hand-demo)          # earlier names of this launcher: removed, so only one entry shows
MENU="${XDG_DATA_HOME:-$HOME/.local/share}/applications/$NAME.desktop"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
ON_DESKTOP="$DESKTOP_DIR/$NAME.desktop"

# GNOME Shell keeps an icon in memory by its file path: a changed picture at the same path is never shown. The icon is
# therefore installed under a name with a fingerprint of its content, so a new picture gets a new path.
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons"

for old in "${OLD_NAMES[@]}"; do
    rm -f "$(dirname "$MENU")/$old.desktop" "$DESKTOP_DIR/$old.desktop" "$ICON_DIR"/$old-*.png
done

if [[ ${1:-} == --remove ]]; then
    rm -f "$MENU" "$ON_DESKTOP" "$ICON_DIR"/$NAME-*.png
    update-desktop-database "$(dirname "$MENU")" >/dev/null 2>&1 || true
    echo "== Artly-os launcher removed"
    exit 0
fi

mkdir -p "$(dirname "$MENU")" "$ICON_DIR"
HASH=$(sha1sum "$GUI/assets/aac_hand_icon.png" | cut -c1-10)
ICON="$ICON_DIR/$NAME-$HASH.png"
if [[ ! -f $ICON ]]; then
    rm -f "$ICON_DIR"/$NAME-*.png                 # the previous picture
    cp "$GUI/assets/aac_hand_icon.png" "$ICON"
fi
cat >"$MENU" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=Artly-os
Comment=Track your hand (MANUS glove or camera) and drive the AAC D21 hands
Exec="$GUI/launch.sh"
Path=$ROOT
Icon=$ICON
Terminal=false
Categories=Science;Engineering;
StartupNotify=true
StartupWMClass=Artly-os
DESKTOP
chmod +x "$MENU"
update-desktop-database "$(dirname "$MENU")" >/dev/null 2>&1 || true
echo "== app menu entry: $MENU"

if [[ -d $DESKTOP_DIR ]]; then
    cp "$MENU" "$ON_DESKTOP"
    chmod +x "$ON_DESKTOP"
    # GNOME's desktop icons only start a launcher marked trusted ("Allow Launching")
    gio set "$ON_DESKTOP" metadata::trusted true 2>/dev/null || true
    echo "== desktop icon: $ON_DESKTOP"
fi
echo "   output of each run: ${XDG_STATE_HOME:-$HOME/.local/state}/artly-os/gui.log"
