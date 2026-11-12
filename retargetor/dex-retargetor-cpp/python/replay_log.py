"""Replay a MANUS/AAC JSONL log through the C++ retargeter and compare with the
dex-retargeting (SLSQP) solution recorded in the same log.

    hand/.hand/bin/python dex-retargetor-cpp/python/replay_log.py /tmp/manus_aac_live_02.jsonl

Both solvers minimise the same objective (Huber mean + norm_delta*|q - q_last|^2)
along the same warm-start chain, so the per-joint difference should be small; the
fit errors should be comparable or better.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dexrt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)


def percentiles(values):
    arr = np.asarray(values, dtype=float)
    return 'p50 %.3f  p95 %.3f  max %.3f' % (np.percentile(arr, 50), np.percentile(arr, 95), arr.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--side', default='Left')
    parser.add_argument('--max-steps', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0, help='stop after this many solved frames (0 = all)')
    args = parser.parse_args()

    urdf = ROOT / 'hand' / 'aac' / 'description' / f'D21_{args.side}.urdf'
    solver = dexrt.Retargeter.for_aac(urdf, max_steps=args.max_steps)
    first = True
    dq_deg, err_cpp, err_py, times, steps, bad = [], [], [], [], [], 0
    worst_joint = np.zeros(16)
    count = 0
    with open(args.log) as f:
        for line in f:
            record = json.loads(line)
            if record.get('type') != 'sample' or not record.get('result'):
                continue
            result = record['result']
            if first:
                solver.reset(np.asarray(result['solver']['previous_rad']))
                first = False
            vectors = np.asarray(result['input_vectors_root_m'])
            q, errors, info = solver.solve(vectors)
            if not info['converged']:
                bad += 1
            diff = np.rad2deg(q) - np.asarray(result['raw_degrees'])
            worst_joint = np.maximum(worst_joint, np.abs(diff))
            dq_deg.append(np.abs(diff).max())
            err_cpp.append(max(errors) * 1000)
            err_py.append(max(result['raw_errors_m']) * 1000)
            times.append(info['solve_seconds'] * 1000)
            steps.append(info['steps'])
            count += 1
            if args.limit and count >= args.limit:
                break
    if count == 0:
        print('no solved samples in log')
        return 1
    print(f'frames: {count}   OSQP not converged: {bad}')
    print('max |q_cpp - q_slsqp| per frame (deg):', percentiles(dq_deg))
    print('worst per joint (deg):', ' '.join(f'{name.split("_")[0][:2]}{name[-1]}={v:.2f}'
                                              for name, v in zip(dexrt.MODEL_JOINTS, worst_joint)))
    print('worst vector error, C++ (mm):        ', percentiles(err_cpp))
    print('worst vector error, dex-retarget (mm):', percentiles(err_py))
    print('C++ solve time (ms):', percentiles(times), '  mean steps %.2f' % np.mean(steps))
    py_times = []
    with open(args.log) as f:
        for line in f:
            record = json.loads(line)
            if record.get('type') == 'sample' and record.get('result'):
                py_times.append(record['result']['solve_s'] * 1000)
    if py_times:
        print('dex-retargeting solve time (ms):', percentiles(py_times))
    return 0


if __name__ == '__main__':
    sys.exit(main())
