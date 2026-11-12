"""Measure how often the D21 firmware refreshes its feedback registers (positions, currents).

The protocol spec states 50 Hz for joint positions (section 3.2.1) and nothing for motor currents
(section 3.2.3). This reads one register group back to back as fast as the bus allows (~1 ms per
read), records when the values change, and reports the interval between changes.

By default it only reads: no mode change, no speed or position writes. A still hand can report
identical values across real refreshes (the histogram then shows multiples of the true interval);
press on a finger during the test to keep the current changing. --move instead switches to position
mode and moves the index finger slowly up and down (keep it clear). Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/feedback_rate.py              # read only
    hand/.hand/bin/python dex-retargetor-cpp/python/feedback_rate.py --move       # with index motion
"""
import argparse
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from hand.backends.aac import AACHand, JOINT_NAMES  # noqa: E402
from d21_torque import zero_torques  # noqa: E402

INDEX_MCP, INDEX_PIP = JOINT_NAMES.index('index_mcp'), JOINT_NAMES.index('index_pip')


def measure(read, seconds, stop):
    """Back-to-back reads; returns (read timestamps, timestamps where any value changed, all values)."""
    stamps, changes, values_log = [], [], []
    last = None
    end = time.monotonic() + seconds
    while time.monotonic() < end and not stop:
        values = read()
        now = time.monotonic()
        stamps.append(now)
        values_log.append(list(values))
        if last is not None and values != last:
            changes.append(now)
        last = values
    return np.array(stamps), np.array(changes), np.array(values_log, dtype=float)


def report_joints(stamps, values, names, top=4):
    """Per channel: interval between changes of that channel alone (the firmware may refresh channels
    at different moments, which makes the whole vector change more often than any one channel)."""
    if len(stamps) < 3:
        return
    changed = np.diff(values, axis=0) != 0
    order = np.argsort(changed.sum(axis=0))[::-1][:top]
    print('  per channel (most active first):')
    for j in order:
        times = stamps[1:][changed[:, j]]
        if len(times) < 3:
            continue
        iv = np.diff(times) * 1000
        print(f'    {names[j]:>16}: {len(times):4d} changes, interval median {np.median(iv):5.1f} ms, '
              f'p10 {np.percentile(iv, 10):5.1f} ms, min {iv.min():5.1f} ms')


def report(name, stamps, changes, values=None, names=None):
    if len(stamps) < 2:
        print(f'{name}: no data')
        return
    duration = stamps[-1] - stamps[0]
    read_hz = (len(stamps) - 1) / duration
    print(f'\n{name}: {len(stamps)} reads in {duration:.2f} s = {read_hz:.0f} reads/s '
          f'({1000 / read_hz:.2f} ms per read)')
    if len(changes) < 3:
        print('  values almost never changed; press on a finger during the test, or use --move')
        return
    intervals = np.diff(changes) * 1000
    print(f'  values changed {len(changes)} times = {len(changes) / duration:.1f} changes/s')
    print(f'  interval between changes: median {np.median(intervals):.1f} ms, '
          f'p10 {np.percentile(intervals, 10):.1f} ms, p90 {np.percentile(intervals, 90):.1f} ms')
    edges = [0, 3, 7, 12, 17, 23, 30, 45, 60, math.inf]
    counts, _ = np.histogram(intervals, bins=edges)
    labels = ['<3', '3-7', '7-12', '12-17', '17-23', '23-30', '30-45', '45-60', '>60']
    print('  histogram (ms): ' + '  '.join(f'{lab}:{c}' for lab, c in zip(labels, counts)))
    print(f'  -> refresh rate ~ {1000 / np.median(intervals):.0f} Hz '
          '(an interval that is a multiple of the median means a refresh with identical values)')
    if values is not None:
        report_joints(stamps, values, names)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=4.0, help='measurement time per register group')
    parser.add_argument('--move', action='store_true',
                        help='switch to position mode and move the index finger during the test (writes commands)')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()

    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    hand = AACHand(role['serial_device'], role['device_address'], 0.1)
    hand.open()
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    lock = threading.Lock()  # the mover and the reader share one bus: serialize their requests

    def locked(fn):
        def call(*a):
            with lock:
                return fn(*a)
        return call

    mover_done = threading.Event()

    def mover():
        t0 = time.monotonic()
        while not mover_done.is_set():
            phase = 0.5 - 0.5 * math.cos(2 * math.pi * 0.25 * (time.monotonic() - t0))  # 4 s period
            pose = [0.0] * 16
            pose[INDEX_MCP], pose[INDEX_PIP] = 40.0 * phase, 60.0 * phase
            with lock:
                hand.write_positions(pose)
            time.sleep(0.05)  # 20 writes/s: leaves the bus mostly to the reader

    try:
        thread = None
        if args.move:
            if hand.sdk.read_control_mode() != 1:
                print('hand is not in position mode: zeroing torque and switching (keep the hand clear)')
                zero_torques(hand.sdk)
            hand.set_position_mode()
            hand.set_speeds([30] * 5)
            hand.write_positions([0.0] * 16)
            time.sleep(1.0)
            print('index finger moves slowly up and down during the test')
            thread = threading.Thread(target=mover, daemon=True)
            thread.start()
            time.sleep(0.5)
        else:
            print(f'read only (control mode {hand.sdk.read_control_mode()}); press on a finger now and then '
                  'during the test so the current keeps changing')
        print(f'reading motor currents back to back for {args.seconds:.0f} s ...')
        report('motor currents (input registers 32-47)', *measure(locked(hand.read_currents), args.seconds, stop),
               names=[f'motor {m}' for m in range(1, 17)])
        print(f'\nreading joint positions back to back for {args.seconds:.0f} s ...')
        report('joint positions (input registers 0-15)', *measure(locked(hand.read_positions), args.seconds, stop),
               names=list(JOINT_NAMES))
        mover_done.set()
        if thread is not None:
            thread.join()
            with lock:
                hand.write_positions([0.0] * 16)
            time.sleep(0.5)
    finally:
        mover_done.set()
        hand.close()
        print('\ndone' + ('; hand in position mode at the flat pose' if args.move else '; nothing was written'))


if __name__ == '__main__':
    main()
