"""WiLoR webcam hand tracking as a drop-in skeleton source (same interface as skeleton_source.Source / CameraSource).

connect() starts teleop/manus/wilor/wilor_service.py in WiLoR's own environment (WiLoR/.venv: CUDA torch, WiLoR's pinned
packages) and waits until its model is loaded; a thread reads the hands it publishes and turns each into the glove-style
skeleton with camera_source's skeleton_from_landmarks (WiLoR's 21 joints are in MediaPipe's order) after the same palm
stabiliser. No MediaPipe depth fix: WiLoR fits the MANO hand, whose bones keep their length.

Hand size: WiLoR's MANO hand has average proportions (middle finger 81 mm) while GeoRT maps fingertip POSITIONS to
joint angles and was trained on the glove hand (middle finger 100 mm), so a flat MANO hand came out curled (middle
joints 23-63 deg). Every bone is therefore resized to the glove's length for that bone (teleop/manus/
glove_hand_lengths.json), direction kept, before the skeleton is built: the retargeter sees the hand size it was
trained on, and the joint angles WiLoR measured are untouched. fit_to_glove=False gives the raw MANO hand.
"""
import json
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from teleop.manus.tools.camera_source import (PalmStabilizer, _NoSdk, fit_bone_lengths, glove_bone_lengths, metadata,
                                        skeleton_from_landmarks)

ROOT = Path(__file__).resolve().parents[3]   # the repo root
WILOR_PYTHON = ROOT / "WiLoR" / ".venv" / "bin" / "python"
SERVICE = Path(__file__).with_name("wilor_service.py")
DEFAULT_ADDRESS = "tcp://127.0.0.1:6420"
WILOR_GLOVE_ID = {"Left": 0xCA4E1001, "Right": 0xCA4E1002}   # fixed ids, so alignment files can name them
START_TIMEOUT_S = 180.0      # loading the 2.6 GB checkpoint and the first CUDA run
DOWNLOAD_TIMEOUT_S = 1800.0  # WiLoR-mini's first start: it downloads the 2.6 GB checkpoint (and the detector, MANO)


class WilorSource:
    """connect() -> consume() -> (frames, devices), metadata(glove_id), close()."""

    def __init__(self, sides, address=DEFAULT_ADDRESS, camera_index=0, preview=False, fast=True, palm=None,
                 min_score=0.5, python=WILOR_PYTHON, wilor_root=None, tracker="wilor", allow_mano_download=True,
                 fit_to_glove=True):
        self.backend = _NoSdk()
        self.keeps_alignment = True
        self.sides = list(sides)
        self.address, self.camera_index, self.preview, self.fast = address, camera_index, preview, fast
        self.min_score = min_score
        self.python = Path(python)
        self.wilor_root = wilor_root
        self.tracker = tracker                           # "wilor" (WiLoR's code) or "wilor_mini" (the WiLoR-mini package)
        self.allow_mano_download = allow_mano_download    # WiLoR-mini: fetch its MANO copy when none is here
        self.palm = {s: PalmStabilizer(**(palm or {})) for s in self.sides}
        self.bone_lengths = {s: glove_bone_lengths(s) for s in self.sides} if fit_to_glove else None
        self.lock = threading.Lock()
        self.latest, self.error = {}, None
        self.frame_id = 0
        self.stop = threading.Event()
        self.process = self.thread = self.reader = None
        self.output = deque(maxlen=40)
        self.ready = threading.Event()
        self.stats = dict(messages=0, hands=0)

    # --- interface ----------------------------------------------------------------------------------------------
    def start_check(self):
        """Before starting the service: RuntimeError naming what it cannot do without; otherwise whether WiLoR-mini will
        have to download something first (its start then takes longer). WiLoR needs its weights and MANO in place;
        WiLoR-mini downloads what is missing itself, MANO only while allow_mano_download is on."""
        from teleop.manus.wilor.wilor_hands import WILOR_ROOT, missing_files
        if not self.python.exists():
            raise RuntimeError(f"WiLoR environment not found: {self.python} (run ./setup_envs.sh wilor)")
        root = Path(self.wilor_root or WILOR_ROOT)
        missing = missing_files(root)
        if self.tracker == "wilor_mini":
            own = root / "wilor_mini" / "pretrained_models"
            missing = [p for p in missing if not (own / p.name).exists()]     # its own (earlier downloaded) copies
            mano = [p for p in missing if p.name == "MANO_RIGHT.pkl"]
            if mano and not self.allow_mano_download:
                raise RuntimeError("WiLoR files missing: MANO_RIGHT.pkl (download from mano.is.tue.mpg.de into "
                                   "WiLoR/mano_data/, or allow WiLoR-mini's download)")
            return bool(missing)                         # it downloads these on its first start
        if missing:
            names = ", ".join(p.name for p in missing)
            hint = " (MANO_RIGHT.pkl: download from mano.is.tue.mpg.de into WiLoR/mano_data/)" \
                if any(p.name == "MANO_RIGHT.pkl" for p in missing) else " (run ./setup_envs.sh wilor)"
            raise RuntimeError(f"WiLoR files missing: {names}{hint}")
        return False

    def connect(self):
        downloads = self.start_check()
        timeout = DOWNLOAD_TIMEOUT_S if downloads else START_TIMEOUT_S
        if downloads:
            print("[wilor] WiLoR-mini downloads its model files on this first start (about 2.6 GB)", flush=True)
        self.process = subprocess.Popen(self.command(), cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read_output, name="wilor-output", daemon=True)
        self.reader.start()
        start = time.perf_counter()
        while not self.ready.wait(0.2):
            if self.stop.is_set():                       # cancelled while the model was loading
                self.close()
                raise RuntimeError("WiLoR start cancelled")
            if self.process.poll() is not None:
                raise RuntimeError("WiLoR service stopped while starting:\n" + "\n".join(list(self.output)[-12:]))
            if time.perf_counter() - start > timeout:
                self.close()
                raise RuntimeError(f"WiLoR service not ready after {timeout:.0f} s")
        self._subscribe()

    def _subscribe(self):
        """Read the service's hands in a thread (also used by the tests with a stand-in publisher)."""
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)          # only the newest message
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(self.address)
        self.thread = threading.Thread(target=self._loop, name="wilor-hands", daemon=True)
        self.thread.start()

    def command(self):
        """The service's command line."""
        command = [str(self.python), "-u", str(SERVICE), "--address", self.address,
                   "--camera-index", str(self.camera_index), "--tracker", self.tracker]
        if self.preview:
            command.append("--preview")
        if not self.fast:
            command.append("--no-fast")
        if self.wilor_root:
            command += ["--wilor-root", str(self.wilor_root)]
        if self.tracker == "wilor_mini" and not self.allow_mano_download:
            command.append("--no-mano-download")
        return command

    def consume(self):
        with self.lock:
            frames, self.latest = self.latest, {}
            error = self.error
        if error is None and self.process is not None and self.process.poll() is not None:
            error = RuntimeError("WiLoR service stopped:\n" + "\n".join(list(self.output)[-12:]))
        if error is not None:
            raise error
        devices = {WILOR_GLOVE_ID[s]: dict(side=s, family="Camera", device_class="WiLoR") for s in self.sides}
        return frames, devices

    def metadata(self, glove_id):
        side = next(s for s, gid in WILOR_GLOVE_ID.items() if gid == glove_id)
        return metadata(side)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        socket = getattr(self, "socket", None)
        if socket is not None:
            socket.close(0)
            self.context.term()
            self.socket = None

    # --- threads ------------------------------------------------------------------------------------------------
    def _read_output(self):
        for line in self.process.stdout:
            line = line.rstrip()
            self.output.append(line)
            print(f"[wilor] {line}", flush=True)
            if line.startswith("WILOR READY"):
                self.ready.set()

    def _loop(self):
        import zmq
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        try:
            while not self.stop.is_set():
                if not dict(poller.poll(50)):
                    continue
                message = json.loads(self.socket.recv_string())
                # age of the camera frame, carried over to this process's clock
                received = time.perf_counter() - max(0.0, time.time() - float(message["t"]))
                self.stats["messages"] += 1
                best = {}
                for hand in message["hands"]:
                    if hand["side"] in self.sides and hand["score"] >= self.min_score:
                        if hand["side"] not in best or hand["score"] > best[hand["side"]]["score"]:
                            best[hand["side"]] = hand
                if not best:
                    continue
                self.stats["hands"] += 1
                self.frame_id += 1
                frame = {}
                for side, hand in best.items():
                    gid = WILOR_GLOVE_ID[side]
                    points = np.asarray(hand["joints"], dtype=float)
                    if self.bone_lengths is not None:
                        points = fit_bone_lengths(points, self.bone_lengths[side])   # the glove's hand size
                    points = self.palm[side](points, received)
                    frame[gid] = skeleton_from_landmarks(points, gid, self.frame_id, received)
                with self.lock:
                    self.latest.update(frame)
        except Exception as error:                        # surfaced on the next consume()
            with self.lock:
                self.error = error


def load_wilor_source(sides, **options):
    return WilorSource(sides, **options)
