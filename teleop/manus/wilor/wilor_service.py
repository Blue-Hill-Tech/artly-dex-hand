"""WiLoR webcam hand tracking as a small service: camera -> WiLoR -> hands published on a local ZMQ socket.

Runs in WiLoR's own environment (CUDA torch and WiLoR's pinned packages stay out of the teleop venv):

    WiLoR/.venv/bin/python teleop/manus/wilor/wilor_service.py [--preview]

teleop/manus/wilor/wilor_source.py starts it and reads it (the GUI's WiLoR mode). Each message (JSON, PUB):
    {type: "wilor_hands", seq, t (time.time() of the camera frame), width, height,
     hands: [{side, score, joints (21 x 3, m, wrist at 0, camera axes), image (21 x 2 px)}]}
Prints "WILOR DEVICE {device, fast, name | reason}" (GPU or CPU, JSON) and then "WILOR READY <address>" once the
model is loaded and the camera is open.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # the repo root

DEFAULT_ADDRESS = "tcp://127.0.0.1:6420"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class LatestFrame:
    """Reads the camera in its own thread and keeps only the newest frame, so the model never waits for the next
    exposure after it finishes (reading and inference one after the other halved the rate: 15 of 30 fps)."""

    def __init__(self, capture):
        import threading
        self.capture, self.lock = capture, threading.Lock()
        self.frame, self.stamp, self.seq = None, 0.0, 0
        self.new = threading.Condition(self.lock)
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, bgr = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.new:
                self.frame, self.stamp, self.seq = bgr, time.time(), self.seq + 1
                self.new.notify_all()

    def next(self, after, timeout=1.0):
        """The newest frame newer than `after` (a seq): (seq, stamp, bgr), or None on timeout."""
        with self.new:
            if not self.new.wait_for(lambda: self.seq > after, timeout):
                return None
            return self.seq, self.stamp, self.frame

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)


def device_info(tracker):
    """Where the model runs, for the GUI (Dev Mode shows it): {device: "cuda"|"cpu", fast, name | reason}."""
    import torch
    info = dict(device=tracker.device.type, fast=bool(tracker.fast))
    if tracker.device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(tracker.device)
    elif torch.version.cuda is None:
        info["reason"] = "this torch build has no CUDA support"
    else:
        info["reason"] = (f"no usable NVIDIA GPU (none found, or its driver is too old for CUDA {torch.version.cuda})")
    return info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="ZMQ PUB address to bind")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="/dev/videoN (default: AAC_CAMERA_INDEX, else the first colour camera)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-fast", action="store_true", help="full precision, full backbone (slower)")
    parser.add_argument("--detection-confidence", type=float, default=0.3)
    parser.add_argument("--preview", action="store_true", help="show the camera with the tracked joints")
    parser.add_argument("--wilor-root", type=Path, default=None, help="WiLoR folder (default: WiLoR/ in the repo)")
    parser.add_argument("--tracker", choices=("wilor", "wilor_mini"), default="wilor",
                        help="WiLoR's own code, or the WiLoR-mini package (same weights)")
    parser.add_argument("--no-mano-download", action="store_true",
                        help="wilor_mini only: never fetch MANO_RIGHT.pkl from its author's Hugging Face copy (by default "
                             "it does when WiLoR/mano_data has none; that copy is not the official MANO release)")
    parser.add_argument("--allow-mano-download", action="store_true", help=argparse.SUPPRESS)   # the default now
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import cv2
    import zmq
    from teleop.manus.wilor.wilor_hands import TRACKERS, WILOR_ROOT
    options = dict(root=args.wilor_root or WILOR_ROOT, fast=not args.no_fast,
                   detection_confidence=args.detection_confidence)
    if args.tracker == "wilor_mini":
        options["allow_mano_download"] = not args.no_mano_download
    tracker = TRACKERS[args.tracker](**options)
    from teleop.manus.tools.camera_device import open_camera
    try:
        capture, _ = open_camera(args.camera_index, args.width, args.height, args.fps,
                                 log=lambda text: print(text, flush=True))
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.bind(args.address)
    ok, bgr = capture.read()
    if ok:
        tracker(bgr)                                     # warm-up (CUDA kernels), not published
    camera = LatestFrame(capture)
    last_frame = 0
    print("WILOR DEVICE " + json.dumps(device_info(tracker)), flush=True)
    print(f"WILOR READY {args.address}: {args.tracker} on {tracker.device} ({'fast' if tracker.fast else 'full'})",
          flush=True)
    seq, last_report, frames, spent = 0, time.perf_counter(), 0, 0.0
    parent = os.getppid()                                # the tracking service (or terminal) that started this
    try:
        while True:
            if os.getppid() != parent:                   # it is gone: free the camera and the GPU
                print("wilor: the process that started this service is gone: shutting down", flush=True)
                break
            got = camera.next(last_frame)
            if got is None:
                continue
            last_frame, stamp, bgr = got
            bgr = bgr.copy()                             # the preview draws on it
            start = time.perf_counter()
            hands = tracker(bgr)
            spent += time.perf_counter() - start
            frames += 1
            seq += 1
            h, w = bgr.shape[:2]
            socket.send_string(json.dumps(dict(
                type="wilor_hands", seq=seq, t=stamp, width=w, height=h,
                hands=[dict(side=x["side"], score=x["score"], joints=x["joints"].tolist(), image=x["image"].tolist())
                       for x in hands])))
            if args.preview:
                for x in hands:
                    points = [tuple(int(v) for v in p) for p in x["image"]]
                    color = (0, 200, 255) if x["side"] == "Right" else (255, 160, 0)
                    for a, b in BONES:
                        cv2.line(bgr, points[a], points[b], color, 2)
                view = cv2.flip(bgr, 1)                  # shown like a mirror; the data is not mirrored
                for k, x in enumerate(hands):
                    cv2.putText(view, f"{x['side']} {x['score']:.2f}", (10, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                cv2.imshow("WiLoR hand tracking", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            now = time.perf_counter()
            if now - last_report >= 10.0:
                print(f"wilor: {frames / (now - last_report):.1f} fps, model {1000 * spent / max(frames, 1):.0f} ms "
                      f"per frame", flush=True)
                last_report, frames, spent = now, 0, 0.0
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        capture.release()
        socket.close(0)
        context.term()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
