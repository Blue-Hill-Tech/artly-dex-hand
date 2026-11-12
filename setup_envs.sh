#!/usr/bin/env bash
# Build the virtual environments of this repo from their pinned package lists:
#   hand/.hand             Python 3.12  teleop, GUI, demos                  (hand/requirements.txt)
#   WiLoR/.venv            Python 3.10  WiLoR hand tracking, CUDA torch    (requirements-wilor.txt + two git packages);
#                                       optional: only for the camera mode's WiLoR / WiLoR-mini pipelines (Dev Mode)
#
#   ./setup_envs.sh            the environment, the C++ library, the system setup and the launcher (= hand)
#   ./setup_envs.sh hand       the same
#   ./setup_envs.sh wilor      only WiLoR/.venv (about 7 GB) and the WiLoR weights (2.7 GB); all = hand + wilor
#   ./setup_envs.sh launcher   only the GUI's app-menu / desktop icon: name "Artly-os" and gui/assets/aac_hand_icon.png
#                              (also done by hand); old entries are removed, a changed picture shows at once
#   ./setup_envs.sh --recreate [hand|wilor]   delete the venv(s) and rebuild
#
# Needs uv (https://docs.astral.sh/uv/): it downloads the Python versions if they are missing. The camera mode tracks the
# hand with MediaPipe (teleop/manus/hand_landmarker.task, in this repo) by default; with wilor, WiLoR (a MANO hand fit
# on an NVIDIA GPU; the upstream checkout is cloned into WiLoR/, not part of this repo; its licence is CC BY-NC-ND 4.0,
# non-commercial) can be picked instead under Dev Mode. The trained GeoRT models the GUI uses are in
# retargetor/GeoRT_checkpoint/left_hand and right_hand (training them is done in agi-os).
# With hand, the C++ library the hand controller needs is built (retargetor/dex-retargetor-cpp/build/libdexrt.so:
# the torque/impedance controller and the dexrt solver) and the system is set up for the hardware (sudo;
# SETUP_NO_SUDO=1 only prints the commands): build tools if missing, the Qt X11 libraries the GUI needs, your user in
# the group that may open the hand adapters (dialout), and the MANUS dongle's udev rule; the GUI is added to the
# app menu and the desktop (gui/install_launcher.sh). Afterwards, things that are not in git and must be put in place by hand:
#   MANUS SDK      copy MANUS_Core_3.2.0_SDK_Linux/ beside this repo (../MANUS_Core_3.2.0_SDK_Linux/Python is loaded by
#                  teleop/manus/stream/stream_gloves.py); checked here
#   MANO model     mano_v1_2.zip from https://mano.is.tue.mpg.de (licensed), unzipped into WiLoR/MANO/ (its
#                  models/MANO_RIGHT.pkl is linked into WiLoR/mano_data/), for the WiLoR pipeline;
#                  WiLoR-mini fetches its own copy on first use unless run with --no-mano-download
# The WiLoR weights (WiLoR/pretrained_models/detector.pt 54 MB, wilor_final.ckpt 2.6 GB) are downloaded by wilor, once.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RECREATE=0
WHICH=""
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        hand|wilor|launcher) WHICH="$WHICH $arg" ;;
        all) WHICH="$WHICH hand wilor" ;;
        -h|--help) sed -n '2,/^# The WiLoR weights/p' "$0"; exit 0 ;;
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
    # Eigen: always the pinned copy the build downloads (cmake/Dependencies.cmake), never a system one: an Eigen CMake
    # config without its headers (libeigen3-dev missing) otherwise fails with "Eigen/Core: No such file or directory"
    if cmake -S "$DEXRT" -B "$DEXRT/build" -DCMAKE_BUILD_TYPE=Release -DDEXRT_BUILD_TESTS=OFF \
            -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE -Wno-dev >/dev/null \
            && cmake --build "$DEXRT/build" -j "$(nproc 2>/dev/null || echo 2)" >/dev/null; then
        echo "== $DEXRT/build/libdexrt.so ready"
    else
        echo "== libdexrt.so build FAILED (output above): the motor cannot start in torque mode"
        echo "   retry: cmake -S $DEXRT -B $DEXRT/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE && cmake --build $DEXRT/build -j"
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

# The camera mode's webcam (/dev/video*, group video). A desktop login usually gets access to it anyway; a user who does
# not (a remote or second session, some distributions) needs the group, or OpenCV cannot open the camera.
check_camera_access() {
    [[ $(uname -s) == Linux ]] || return 0
    local device found=0 blocked=0
    for device in /dev/video*; do
        [[ -e $device ]] || continue
        found=1
        [[ -r $device && -w $device ]] || blocked=1
    done
    if [[ $found -eq 0 ]]; then
        echo "== no camera (/dev/video*) found now: the camera mode needs a webcam (camera key / privacy shutter?)"
        return 0
    fi
    if [[ $blocked -eq 0 ]] || id -nG "$USER" | tr ' ' '\n' | grep -qx video; then
        echo "== $USER may open the camera"
        return 0
    fi
    echo "== $USER may not open the camera (not in the group video)"
    if may_sudo && sudo usermod -aG video "$USER"; then
        echo "   added: log out and back in (or reboot) before using the camera mode"
    else
        echo "   run:  sudo usermod -aG video $USER   then log out and back in"
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
    check_camera_access
    check_manus_udev
    check_manus_sdk
    # the GUI in the app menu and on the desktop (gui/install_launcher.sh; --remove takes them away)
    if [[ $(uname -s) == Linux ]]; then
        gui/install_launcher.sh || echo "== the GUI launcher could not be installed (run gui/install_launcher.sh)"
    fi
fi

WILOR_REPO=https://github.com/rolpotamias/WiLoR.git
WILOR_COMMIT=fcb9113      # the upstream checkout this was built against (WiLoR/ itself is not part of this repo)

if wants wilor; then
    if [[ ! -d WiLoR/wilor ]]; then
        echo "== cloning WiLoR ($WILOR_COMMIT) into WiLoR/"
        git clone "$WILOR_REPO" WiLoR
        git -C WiLoR checkout --quiet "$WILOR_COMMIT"
    fi
    make_venv WiLoR/.venv 3.10
    "${UV_INSTALL[@]}" --python WiLoR/.venv/bin/python -r requirements-wilor.txt
    # WiLoR-mini without its dependencies: its own pins (torch<=2.5) would replace the CUDA torch above
    uv pip install --python WiLoR/.venv/bin/python --no-deps \
        "wilor_mini @ git+https://github.com/warmshao/WiLoR-mini@ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
    # chumpy (MANO's loader) imports numpy and pip in its setup.py: no build isolation, and pip in the venv (uv's venvs
    # have none); setuptools too, which the build needs without isolation
    "${UV_INSTALL[@]}" --python WiLoR/.venv/bin/python pip setuptools wheel
    uv pip install --python WiLoR/.venv/bin/python --no-build-isolation \
        "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
    echo "== WiLoR/.venv ready: $(WiLoR/.venv/bin/python -c 'import torch, wilor_mini, chumpy; print("torch", torch.__version__, "cuda", torch.cuda.is_available())')"
    # the public WiLoR weights (WiLoR's README); skipped when present, a partial download is resumed
    WEIGHTS_URL=https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models
    mkdir -p WiLoR/pretrained_models
    for name in detector.pt wilor_final.ckpt; do
        target=WiLoR/pretrained_models/$name
        if [[ -s $target ]]; then
            echo "== $target present"
            continue
        fi
        echo "== downloading $target"
        curl -L --fail --retry 3 -C - -o "$target.part" "$WEIGHTS_URL/$name"
        mv "$target.part" "$target"
    done
    # the official MANO download unzipped into WiLoR/MANO/: linked to where WiLoR reads it (WiLoR/mano_data/)
    WiLoR/.venv/bin/python -c 'import sys; sys.path.insert(0, "."); from teleop.manus.wilor.wilor_hands import link_mano; link_mano()'
    for f in WiLoR/pretrained_models/detector.pt WiLoR/pretrained_models/wilor_final.ckpt WiLoR/mano_data/MANO_RIGHT.pkl; do
        [[ -e $f ]] || echo "   note: $f is missing (see the header of this script)"
    done
fi
# Only the launcher (hand also installs it, after its environment)
if wants launcher && ! wants hand && [[ $(uname -s) == Linux ]]; then
    gui/install_launcher.sh
fi

echo "done"
