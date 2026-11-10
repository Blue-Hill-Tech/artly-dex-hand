"""Record MANUS glove data for GeoRT training: an array (frames, 21, 3) in metres, MediaPipe joint order, in GeoRT's
canonical hand frame, saved where GeoRT's trainer looks for it (retargetor/GeoRT/data/<name>.npy; ./setup_envs.sh geort
clones GeoRT there).

    hand/.hand/bin/python teleop/manus/geort/geort_record.py --side right --name manus_right_01
    # then, in GeoRT's environment:
    python ./geort/trainer.py -hand aac_d21_right -human_data manus_right_01 -ckpt_tag d21_right_01

While recording, explore the whole range of each finger: fully stretch and curl every finger, spread them, and
pinch the thumb against each fingertip; stay natural, no strained poses (GeoRT README, step 2). Quit the MANUS SDK
client first (one program holds the gloves).

Joint mapping, MANUS raw skeleton node -> MediaPipe index (a MANUS node is the joint where its bone starts):
  0 wrist <- Hand;  thumb 1 CMC <- Metacarpal, 2 MCP <- Proximal, 3 IP <- Distal, 4 tip <- Tip;
  fingers MCP <- Proximal, PIP <- Intermediate, DIP <- Distal, tip <- Tip (index 5-8, middle 9-12, ring 13-16,
  pinky 17-20).
Canonical frame (GeoRT geort/mocap/*: hand_to_canonical): origin at the wrist, z wrist -> middle MCP, y ~ ring MCP ->
index MCP, x = y cross z. The D21 GeoRT configs (geort/config/aac_d21_*.json, base link geort_base) use the same rule.
"""
import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # agi-os
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / 'retargetor' / 'GeoRT' / 'data'

FINGERS = ('Index', 'Middle', 'Ring', 'Pinky')
MEDIAPIPE_ORDER = ([('Hand', None)]
                   + [('FingerThumb', j) for j in ('Metacarpal', 'Proximal', 'Distal', 'Tip')]
                   + [(f'Finger{f}', j) for f in FINGERS for j in ('Proximal', 'Intermediate', 'Distal', 'Tip')])
TIPS = (4, 8, 12, 16, 20)


def mediapipe_node_ids(metadata, side):
    """MANUS node id for each of the 21 MediaPipe joints, from the glove's skeleton metadata."""
    ids = []
    for chain, joint in MEDIAPIPE_ORDER:
        found = [n['id'] for n in metadata if n['side'] == side and n['chain'] == chain
                 and (joint is None or n['joint'] == joint)]
        if chain == 'Hand' and len(found) > 1:
            found = [n['id'] for n in metadata if n['side'] == side and n['chain'] == 'Hand' and n['parent'] == n['id']] or found
        if len(found) != 1:
            raise ValueError(f'{side} {chain}/{joint}: expected one node, found {found}')
        ids.append(found[0])
    return ids


def hand_to_canonical(points):
    """GeoRT's canonical hand frame (copied from geort/mocap/manus_mocap_core.py hand_to_canonical)."""
    points = np.asarray(points, dtype=float)
    z = points[9] - points[0]
    z /= np.linalg.norm(z)
    y_aux = points[5] - points[13]
    y_aux /= np.linalg.norm(y_aux)
    x = np.cross(y_aux, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack([x, y, z])
    return (points - points[0]) @ rotation


def frame_keypoints(hand, node_ids):
    """(21, 3) MediaPipe-order keypoints of one MANUS frame, in GeoRT's canonical frame."""
    positions = {n['id']: n['position'] for n in hand['nodes']}
    points = np.array([positions[i] for i in node_ids], dtype=float)
    if not np.isfinite(points).all():
        raise ValueError('non-finite skeleton positions')
    return hand_to_canonical(points)


def summary(data):
    """Plausibility of a recording: bone lengths and how far each fingertip travelled."""
    d = np.asarray(data)
    lengths = {name: 1000 * np.median(np.linalg.norm(d[:, b] - d[:, a], axis=1))
               for name, (a, b) in (('thumb MCP->tip', (2, 4)), ('index MCP->tip', (5, 8)),
                                    ('middle MCP->tip', (9, 12)), ('wrist->middle MCP', (0, 9)))}
    spread = {f: 1000 * np.ptp(d[:, t], axis=0).max() for f, t in zip(('thumb', 'index', 'middle', 'ring', 'pinky'), TIPS)}
    return lengths, spread


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--side', choices=('left', 'right'), required=True)
    parser.add_argument('--name', required=True, help='dataset name: GeoRT -human_data <name>')
    parser.add_argument('--seconds', type=float, default=60.0, help='recording length (default 60)')
    parser.add_argument('--rate', type=float, default=100.0, help='frames kept per second (default 100)')
    parser.add_argument('--countdown', type=float, default=3.0)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT, help=f'default {DEFAULT_OUT}')
    parser.add_argument('--glove-id', type=lambda v: int(v, 16), help='pin a physical glove (hex)')
    args = parser.parse_args(argv)
    out = args.out_dir / f'{args.name}.npy'
    if out.exists():
        raise SystemExit(f'{out} exists: choose another --name')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from teleop.manus.tools.skeleton_source import load_source, select_glove
    side = args.side.capitalize()
    source = load_source()
    stopping = []
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    data, node_ids, glove = [], None, None
    try:
        source.backend.check(source.backend.lib.CoreSdk_InitializeIntegrated(), 'Initialize integrated SDK')
        source.connect()
        print(f'waiting for the {side} glove (Ctrl+C stops)...', flush=True)
        started = None
        last_keep = 0.0
        last_print = 0.0
        while not stopping:
            frames, devices = source.consume()
            if glove is None:
                glove = select_glove(devices, side, args.glove_id)
            hand = frames.get(glove) if glove is not None else None
            if node_ids is None and hand is not None:
                # The skeleton layout exists only once the glove's first skeleton frame has arrived (asking earlier
                # returns NullPointer); retry on the next frame until it does.
                try:
                    node_ids = mediapipe_node_ids(source.metadata(glove), side)
                except RuntimeError:
                    time.sleep(0.01)
                    continue
                print(f'{side} glove {glove:08X}; recording starts in {args.countdown:.0f} s: move through the '
                      f'whole range of every finger, pinch the thumb to each fingertip', flush=True)
                started = time.monotonic() + args.countdown
            now = time.monotonic()
            if hand is None or started is None or now < started:
                time.sleep(0.002)
                continue
            if now - started >= args.seconds:
                break
            if now - last_keep >= 1.0 / args.rate:
                try:
                    data.append(frame_keypoints(hand, node_ids))
                    last_keep = now
                except ValueError:
                    pass
            if now - last_print >= 5.0:
                last_print = now
                print(f'  {now - started:4.0f} s, {len(data)} frames', flush=True)
    finally:
        try:
            source.backend.check(source.backend.lib.CoreSdk_ShutDown(), 'Shut down SDK')
        except Exception as error:
            print('SDK shutdown:', error)
    if not data:
        raise SystemExit('no frames recorded: check the glove, the dongle license, and that no other MANUS program runs')
    data = np.asarray(data, dtype=np.float32)
    np.save(out, data)
    lengths, spread = summary(data)
    print(f'saved {out}: {data.shape[0]} frames')
    print('  median lengths (mm): ' + ', '.join(f'{k} {v:.0f}' for k, v in lengths.items()))
    print('  fingertip travel (mm, largest axis range): ' + ', '.join(f'{k} {v:.0f}' for k, v in spread.items()))
    print(f'train: python ./geort/trainer.py -hand aac_d21_{args.side} -human_data {args.name} -ckpt_tag <tag>  '
          f'(in {ROOT / "retargetor" / "GeoRT"})')


if __name__ == '__main__':
    main()
