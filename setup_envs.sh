#!/usr/bin/env bash
# Build the virtual environment of this repo from its pinned package list:
#   hand/.hand             Python 3.12  teleop, GUI, demos                  (hand/requirements.txt)
#
#   ./setup_envs.sh            the environment, the C++ library, the system setup and the launcher (= hand)
#   ./setup_envs.sh hand       the same
#   ./setup_envs.sh launcher   only the GUI's app-menu / desktop icon: name "Artly-os" and gui/assets/aac_hand_icon.png
#                              (also done by hand); old entries are removed, a changed picture shows at once
#   ./setup_envs.sh --recreate [hand]   delete the venv and rebuild
#
# Needs uv (https://docs.astral.sh/uv/): it downloads the Python version if it is missing. The camera mode tracks the
# hand with MediaPipe (teleop/manus/hand_landmarker.task, in this repo). The trained GeoRT models the GUI uses are in
# retargetor/GeoRT_checkpoint/left_hand and right_hand (training them is done in agi-os).
# With hand, the C++ library the hand controller needs is built (retargetor/dex-retargetor-cpp/build/libdexrt.so:
# the torque/impedance controller and the dexrt solver) and the system is set up for the hardware (sudo;
# SETUP_NO_SUDO=1 only prints the commands): build tools if missing, the Qt X11 libraries the GUI needs, your user in
# the group that may open the hand adapters (dialout), and the MANUS dongle's udev rule; the GUI is added to the
# app menu and the desktop (gui/install_launcher.sh). Afterwards, things that are not in git and must be put in place by hand:
#   MANUS SDK      copy MANUS_Core_3.2.0_SDK_Linux/ beside this repo (../MANUS_Core_3.2.0_SDK_Linux/Python is loaded by
#                  teleop/manus/stream/stream_gloves.py); checked here
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RECREATE=0
WHICH=""
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        hand|launcher) WHICH="$WHICH $arg" ;;
        all) WHICH="$WHICH hand" ;;
        -h|--help) sed -n '2,/^#                  teleop\/manus\/stream\/stream_gloves.py/p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done
[[ -n $WHICH ]] || WHICH="hand"
wants() { [[ " $WHICH " == *" $1 "* ]]; }

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is needed: curl -LsSf https://astral.sh/uv/install.sh | sh   (then open a new shell)" >&2
    exit 1
fi

# uv's default index strategy takes a package from the first index that has it, so torch's "+cpu" / "+cu126" builds
# on the PyTorch extra index would never be considered; this makes it compare all indexes, as pip does.
UV_INSTALL=(uv pip install --index-strategy unsafe-best-match)

make_venv() {   # make_venv <dir> <python version>
    local dir=$1 version=$2
    if [[ $RECREATE -eq 1 && -d $dir ]]; then
        echo "== removing $dir"
        rm -rf "$dir"
    fi
    if [[ -x $dir/bin/python ]]; then
        echo "== $dir exists ($("$dir/bin/python" --version)); installing into it (use --recreate to rebuild)"
    else
        echo "== creating $dir with Python $version"
        uv venv --python "$version" "$dir"
    fi
}

# PyQt5's wheel brings Qt but not the system X11 libraries its xcb plugin loads; a fresh Ubuntu lacks several of them and
# the GUI then aborts with 'Could not load the Qt platform plugin "xcb"'. Lists what is missing and installs it with apt
# (asks for the sudo password; set SETUP_NO_SUDO=1 to only print the command).
may_sudo() { [[ ${SETUP_NO_SUDO:-${SETUP_NO_APT:-0}} != 1 ]] && command -v sudo >/dev/null 2>&1; }

check_qt_libs() {
    command -v ldd >/dev/null 2>&1 || return 0
    local plugin missing packages
    plugin=$(hand/.hand/bin/python -c 'import os, PyQt5; print(os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms", "libqxcb.so"))')
    [[ -f $plugin ]] || return 0
    missing=$(ldd "$plugin" | awk '/not found/ {print $1}' | sort -u)
    if [[ -z $missing ]]; then
        echo "== Qt X11 libraries present"
        return 0
    fi
    # Debian/Ubuntu package names: libxcb-icccm.so.4 -> libxcb-icccm4, libxkbcommon-x11.so.0 -> libxkbcommon-x11-0
    packages=$(echo "$missing" | sed -E 's/\.so\.([0-9]+).*$/ \1/' |
        awk '{name = tolower($1); sep = (name ~ /[0-9]$/) ? "-" : ""; print name sep $2}' | sort -u | tr '\n' ' ')
    echo "== the GUI needs system libraries that are missing: $(echo $missing)"
    if may_sudo && command -v apt-get >/dev/null 2>&1; then
        echo "== installing them: sudo apt-get install -y $packages"
        if sudo apt-get install -y $packages; then
            missing=$(ldd "$plugin" | awk '/not found/ {print $1}')
            [[ -z $missing ]] && { echo "== Qt X11 libraries present"; return 0; }
            echo "   still missing: $(echo $missing)"
        fi
    fi
    echo "   install them with:  sudo apt install $packages"
}

# libdexrt.so (retargetor/dex-retargetor-cpp): the hand controller's torque mode loads it (python/dexrt.py) and fails
# with "libdexrt.so not found" without it. Needs cmake >= 3.22, a C++17 compiler and git; the first build downloads
# Eigen, OSQP and tinyxml2. Rebuilding is incremental (quick when nothing changed).
DEXRT=retargetor/dex-retargetor-cpp
build_dexrt() {
    local missing=() cmake_ok=0
    command -v g++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || missing+=(build-essential)
    command -v git >/dev/null 2>&1 || missing+=(git)
    if command -v cmake >/dev/null 2>&1; then
        local version
        version=$(cmake --version | awk 'NR == 1 {print $3}')
        [[ $(printf '%s\n3.22\n' "$version" | sort -V | head -1) == 3.22 ]] && cmake_ok=1
    fi
    [[ $cmake_ok -eq 1 ]] || missing+=(cmake)
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "== building libdexrt.so needs: ${missing[*]}"
        if may_sudo && command -v apt-get >/dev/null 2>&1; then
            sudo apt-get install -y "${missing[@]}" || true
        else
            echo "   install them with:  sudo apt install ${missing[*]}"
        fi
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        echo "== libdexrt.so NOT built (no cmake): the motor cannot start in torque mode"
        return 0
    fi
    echo "== building $DEXRT (libdexrt.so)"
    if cmake -S "$DEXRT" -B "$DEXRT/build" -DCMAKE_BUILD_TYPE=Release -DDEXRT_BUILD_TESTS=OFF -Wno-dev >/dev/null \
            && cmake --build "$DEXRT/build" -j "$(nproc 2>/dev/null || echo 2)" >/dev/null; then
        echo "== $DEXRT/build/libdexrt.so ready"
    else
        echo "== libdexrt.so build FAILED (output above): the motor cannot start in torque mode"
        echo "   retry: cmake -S $DEXRT -B $DEXRT/build -DCMAKE_BUILD_TYPE=Release && cmake --build $DEXRT/build -j"
    fi
}

# The CAN FD / RS-485 adapters (/dev/ttyACM*, /dev/ttyUSB*) belong to the group dialout: without it the motor and
# the demos fail with "Permission denied" opening the adapter. Takes effect at the next login.
check_adapter_group() {
    [[ $(uname -s) == Linux ]] || return 0
    local group=dialout device
    for device in /dev/serial/by-id/*; do          # the group the adapters plugged in now actually have
        [[ -e $device ]] && { group=$(stat -L -c %G "$device"); break; }
    done
    getent group "$group" >/dev/null 2>&1 || { echo "== no group $group on this system: adapter access not checked"; return 0; }
    if id -nG "$USER" | tr ' ' '\n' | grep -qx "$group"; then
        echo "== $USER may open the hand adapters (group $group)"
        return 0
    fi
    echo "== $USER is not in the group $group, which the hand adapters belong to"
    if may_sudo && sudo usermod -aG "$group" "$USER"; then
        echo "   added: log out and back in (or reboot) before starting the motor"
    else
        echo "   run:  sudo usermod -aG $group $USER   then log out and back in"
    fi
}

# The MANUS glove dongle is a USB HID device that only root may open without this rule (the SDK then finds no glove).
MANUS_UDEV_RULE=/etc/udev/rules.d/70-manus-hid.rules
MANUS_UDEV_TEXT='# MANUS glove dongle (HIDAPI/libusb, hidraw): readable without root. Installed by agi-os/setup_envs.sh.
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"'
check_manus_udev() {
    [[ $(uname -s) == Linux && -d /etc/udev/rules.d ]] || return 0
    if [[ -f $MANUS_UDEV_RULE ]] && grep -q 'idVendor}=="3325"' "$MANUS_UDEV_RULE"; then
        echo "== MANUS dongle udev rule present"
        return 0
    fi
    echo "== the MANUS dongle udev rule is missing ($MANUS_UDEV_RULE)"
    if may_sudo && printf '%s\n' "$MANUS_UDEV_TEXT" | sudo tee "$MANUS_UDEV_RULE" >/dev/null \
            && sudo udevadm control --reload-rules && sudo udevadm trigger; then
        echo "   installed: unplug and replug the dongle"
    else
        echo "   install it: put these lines into $MANUS_UDEV_RULE (sudo), then"
        echo "   sudo udevadm control --reload-rules && sudo udevadm trigger   and replug the dongle"
        printf '%s\n' "$MANUS_UDEV_TEXT" | sed 's/^/     /'
    fi
}

# The MANUS SDK is not in git (licensed download): stream_gloves.py loads it from beside the repo.
check_manus_sdk() {
    local sdk
    sdk="$(cd .. && pwd)/MANUS_Core_3.2.0_SDK_Linux/Python"
    if hand/.hand/bin/python -c "import sys; sys.path.insert(0, sys.argv[1]); import manus_sdk" "$sdk" >/dev/null 2>&1; then
        echo "== MANUS SDK found ($sdk)"
    else
        echo "== MANUS SDK not found: the glove mode will not work (the camera mode does)"
        echo "   copy the MANUS_Core_3.2.0_SDK_Linux folder (its Python/ subfolder is enough) to $(cd .. && pwd)/"
    fi
}

if wants hand; then
    make_venv hand/.hand 3.12
    "${UV_INSTALL[@]}" --python hand/.hand/bin/python -r hand/requirements.txt
    echo "== hand/.hand ready: $(hand/.hand/bin/python -c 'import torch, mediapipe, PyQt5; print("torch", torch.__version__, "mediapipe", mediapipe.__version__)')"
    check_qt_libs
    build_dexrt
    check_adapter_group
    check_manus_udev
    check_manus_sdk
    # the GUI in the app menu and on the desktop (gui/install_launcher.sh; --remove takes them away)
    if [[ $(uname -s) == Linux ]]; then
        gui/install_launcher.sh || echo "== the GUI launcher could not be installed (run gui/install_launcher.sh)"
    fi
fi

# Only the launcher (hand also installs it, after its environment)
if wants launcher && ! wants hand && [[ $(uname -s) == Linux ]]; then
    gui/install_launcher.sh
fi

echo "done"
