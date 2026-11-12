"""Finding and opening the webcam (the camera mode's MediaPipe source and the WiLoR service).

The camera is not always /dev/video0: a laptop with face login has an infrared camera too (it opens, but its image is
grey and no hand is found in it), every camera also has a metadata node that cannot stream, and a USB webcam may come
after the built-in ones. open_camera() therefore tries, in order:

  1. the camera asked for (index N = /dev/videoN): the --camera-index option, or the AAC_CAMERA_INDEX environment
     variable (e.g. AAC_CAMERA_INDEX=2 before starting the GUI); used as it is, even when it is grey;
  2. otherwise every /dev/videoN that streams (sysfs index 0), lowest first, and takes the first that delivers colour
     frames; infrared (grey) cameras are only taken when there is nothing else.

When none works, the error says what each one did (no permission, busy, no frames, grey ...).

    python teleop/manus/tools/camera_device.py      # lists the cameras and which one would be used
"""
import os
import time
from pathlib import Path

ENV = "AAC_CAMERA_INDEX"
FIRST_FRAME_S = 3.0          # a camera that has not delivered a frame by then is skipped


def requested_index(index=None):
    """The camera asked for: `index` if given, else AAC_CAMERA_INDEX, else None (= find one)."""
    if index is not None:
        return int(index)
    value = os.environ.get(ENV, "").strip()
    if value:
        try:
            return int(value.removeprefix("/dev/video"))
        except ValueError:
            raise RuntimeError(f"{ENV}={value!r} is not a camera number (e.g. 0 for /dev/video0)") from None
    return None


def camera_name(index):
    try:
        return Path(f"/sys/class/video4linux/video{index}/name").read_text().strip()
    except OSError:
        return ""


def candidates():
    """/dev/videoN that can stream, lowest N first (sysfs 'index' 0; 1 is the metadata node of the same camera)."""
    found = []
    for node in Path("/sys/class/video4linux").glob("video*"):
        try:
            n = int(node.name[5:])
        except ValueError:
            continue
        try:
            if int((node / "index").read_text().strip() or 0) != 0:
                continue
        except (OSError, ValueError):
            pass
        found.append(n)
    if not found:                                      # no sysfs (not Linux): the usual first indexes
        found = list(range(4))
    return sorted(found)


def is_grey(bgr):
    """An infrared camera: its three colour channels are (nearly) the same everywhere."""
    if bgr is None or bgr.ndim < 3 or bgr.shape[2] < 3:
        return True
    sample = bgr[::8, ::8].astype("int16")
    return int(abs(sample[..., 0] - sample[..., 1]).max()) <= 2 and int(abs(sample[..., 1] - sample[..., 2]).max()) <= 2


def _open(index, width, height, fps):
    """(capture, first frame, None) or (None, None, why not)."""
    import cv2
    path = f"/dev/video{index}"
    if os.path.exists(path) and not os.access(path, os.R_OK | os.W_OK):
        return None, None, (f"no permission to open {path}: add yourself to the video group "
                            f"(sudo usermod -aG video $USER, then log out and in)")
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened() and not Path("/sys/class/video4linux").is_dir():
        # other systems only: on Linux the default backend counts cameras its own way and would open a different one
        # than /dev/videoN (seen: N = a metadata node opened the infrared camera)
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        return None, None, "cannot be opened (in use by another program, or not a camera)"
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    deadline = time.monotonic() + FIRST_FRAME_S
    while time.monotonic() < deadline:
        ok, bgr = capture.read()
        if ok and bgr is not None:
            return capture, bgr, None
        time.sleep(0.05)
    capture.release()
    return None, None, f"opens but gives no image within {FIRST_FRAME_S:g} s (in use by another program?)"


def describe(index):
    name = camera_name(index)
    return f"/dev/video{index}" + (f" ({name})" if name else "")


def open_camera(index=None, width=640, height=480, fps=30.0, log=print):
    """(cv2.VideoCapture, N) for the camera to use, already delivering frames; RuntimeError naming what was tried."""
    wanted = requested_index(index)
    if wanted is not None:
        capture, _, why = _open(wanted, width, height, fps)
        if capture is None:
            raise RuntimeError(f"camera {describe(wanted)} {why}"
                               + (f" (set by {ENV})" if index is None else ""))
        return capture, wanted
    tried, grey = [], None
    for n in candidates():
        capture, bgr, why = _open(n, width, height, fps)
        if capture is None:
            tried.append(f"{describe(n)}: {why}")
            continue
        if is_grey(bgr):
            tried.append(f"{describe(n)}: infrared (grey image), skipped")
            if grey is None:
                grey = (capture, n)
            else:
                capture.release()
            continue
        if grey is not None:
            grey[0].release()
        if tried and log:
            log("camera: " + "; ".join(tried) + f"; using {describe(n)}")
        return capture, n
    if grey is not None:                               # only infrared cameras: better than nothing
        if log:
            log(f"camera: only an infrared camera found, using {describe(grey[1])}")
        return grey
    if not tried:
        raise RuntimeError("no camera found (no /dev/video* device): is the webcam connected and switched on? "
                           "(some laptops have a camera key or a privacy shutter)")
    raise RuntimeError("no usable camera: " + "; ".join(tried)
                       + f". To pick one yourself, start the GUI with {ENV}=N (N of /dev/videoN).")


def main():
    wanted = requested_index()
    print(f"{ENV}: {wanted if wanted is not None else '(not set: the first colour camera is used)'}")
    for n in candidates():
        capture, bgr, why = _open(n, 640, 480, 30.0)
        if capture is None:
            print(f"  {describe(n)}: {why}")
            continue
        h, w = bgr.shape[:2]
        print(f"  {describe(n)}: OK, {w}x{h}" + (", infrared (grey)" if is_grey(bgr) else ", colour"))
        capture.release()
    try:
        capture, n = open_camera(log=None)
        capture.release()
        print(f"used: {describe(n)}")
    except RuntimeError as error:
        print(f"used: none -- {error}")


if __name__ == "__main__":
    main()
