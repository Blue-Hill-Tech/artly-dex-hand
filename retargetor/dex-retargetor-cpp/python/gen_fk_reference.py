"""Write tests/data/fk_reference_<side>.csv from Pinocchio for test_kinematics.

Run with the project venv (it has pinocchio):
    hand/.hand/bin/python dex-retargetor-cpp/python/gen_fk_reference.py
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
LINKS = ['palm_down', 'thumb_pitch', 'thumb_mcp', 'thumb_tip'] + [
    f'{f}_{part}' for f in ('index', 'middle', 'ring', 'pinky') for part in ('mcp', 'pip', 'tip')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', default='Left', choices=('Left', 'Right'))
    parser.add_argument('--rows', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    urdf = ROOT / 'hand' / 'aac' / 'description' / f'D21_{args.side}.urdf'
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    joint_names = [name for name in model.names[1:]]
    q_index = {name: model.joints[model.getJointId(name)].idx_q for name in joint_names}
    rng = np.random.default_rng(args.seed)
    out = ROOT / 'retargetor' / 'dex-retargetor-cpp' / 'tests' / 'data' / f'fk_reference_{args.side.lower()}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'q:{n}' for n in joint_names] + [f'p:{l}:{a}' for l in LINKS for a in 'xyz'])
        for _ in range(args.rows):
            q = pin.neutral(model)
            for name in joint_names:
                lo, hi = model.lowerPositionLimit[q_index[name]], model.upperPositionLimit[q_index[name]]
                q[q_index[name]] = rng.uniform(lo, hi)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            row = [repr(float(q[q_index[n]])) for n in joint_names]
            for link in LINKS:
                p = data.oMf[model.getFrameId(link)].translation
                row.extend(repr(float(v)) for v in p)
            writer.writerow(row)
    print(f'wrote {args.rows} rows to {out}')


if __name__ == '__main__':
    main()
