"""Put a D21 in torque mode with ZERO torque and print its joint positions and motor currents, then restore position mode.

    hand/.hand/bin/python dex-retargetor-cpp/python/torque_monitor.py --rs485 /dev/serial/by-id/usb-wch.cn_USB_Single_Serial_BC4C8140ABCDE959-if00
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_monitor.py --hand_serial AAC-001 --seconds 20
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_monitor.py --rs485 PORT --finger thumb

Sequence: read the pose, write it back as the position command (so the return to position mode does not jump), zero all
16 torque registers, switch to torque mode (control word 21 = 2) and check it reads back 2, then print positions (deg)
and currents (mA) every --period seconds for --seconds. Move the fingers by hand during the window to see whether the
position readback follows in torque mode (the old firmware froze it). Afterwards: zero torque, position command = the
measured pose, position mode. Nothing but zero torque is ever written, so the hand only goes limp.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import ANGLE_LIMITS, AACHand, JOINT_NAMES, load_sdk_module  # noqa: E402
from teleop.hand_control.hand_controller_aac import load_aac_config  # noqa: E402

SHORT = ('tAbd', 'tRot', 'tCMC', 'tMCP') + tuple(
    f'{f}{j}' for f in ('i', 'm', 'r', 'p') for j in ('Abd', 'MCP', 'PIP'))
# Joint indices per finger. Motor m (1-based) drives the same finger as joint m-1, so the same slice selects the
# finger's motors in the 16 current readings (thumb motors 1-4, index 5-7, middle 8-10, ring 11-13, pinky 14-16).
FINGERS = {'all': range(16), 'thumb': range(0, 4), 'index': range(4, 7), 'middle': range(7, 10),
           'ring': range(10, 13), 'pinky': range(13, 16)}


def clip_pose(pose):
    return [min(max(v, ANGLE_LIMITS[i][0]), ANGLE_LIMITS[i][1]) for i, v in enumerate(pose)]


def open_hand(args):
    settings = load_aac_config(SimpleNamespace(hand_serial=args.hand_serial), args.config)
    if args.rs485:
        rs485 = load_sdk_module('d21_rs485').d21_rs485

        def factory(tty_baudrate=None, **kw):
            sdk = rs485(**kw)             # opens the port in the constructor
            sdk.connect = lambda: None
            return sdk
        return AACHand(args.rs485, settings['device_address'], settings['connection_timeout_s'],
                       sdk_factory=factory, bus='rs485')
    return AACHand(settings['serial_device'], settings['device_address'], settings['connection_timeout_s'])


def row(label, values, fmt):
    return f'{label:>9s} ' + ' '.join(format(v, fmt) for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rs485', metavar='PORT', help='RS-485 serial port instead of the CAN FD adapter in the config')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    parser.add_argument('--hand_serial', default='AAC-001', help='role key in the config (address and timeout)')
    parser.add_argument('--seconds', type=float, default=10.0, help='how long to stay in torque mode')
    parser.add_argument('--period', type=float, default=0.5, help='seconds between printed samples')
    parser.add_argument('--finger', choices=sorted(FINGERS), default='all',
                        help='print only this finger\'s joint positions and motor currents')
    args = parser.parse_args()
    sel = list(FINGERS[args.finger])
    motors = ' '.join(f'm{j + 1:<4d}' for j in sel)

    hand = open_hand(args)
    start = hand.open()
    print(f'opened {hand.serial_device} ({hand.bus}, register map {hand.register_map})')
    if hand.register_map == 'v2':
        print('firmware', hand.sdk.read_firmware_info())
    print(row('', [SHORT[j] for j in sel], '>6s'))
    print(row('pos mode', [start[j] for j in sel], '6.1f'))
    samples = []
    try:
        hand.write_positions(clip_pose(start))
        hand.write_motor_torques([0.0] * 16)
        hand.set_torque_mode()
        print(f'torque mode on (control word reads {hand.sdk.read_control_mode()}), zero torque on all 16 motors')
        print(row('t [s]', [SHORT[j] for j in sel], '>6s') + '   | current mA ' + motors)
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.seconds:
            t = time.monotonic() - t0
            pose = hand.read_positions()
            amps = hand.read_currents()
            samples.append((t, pose, amps))
            print(row(f'{t:5.1f}', [pose[j] for j in sel], '6.1f') + '   |            ' +
                  ' '.join(f'{1000 * amps[j]:5.0f}' for j in sel))
            time.sleep(max(0.0, args.period - (time.monotonic() - t0 - t)))
    finally:
        try:
            hand.write_motor_torques([0.0] * 16)
            end = hand.read_positions()
            hand.write_positions(clip_pose(end))
            hand.set_position_mode()
            print(f'restored: zero torque, position mode at the measured pose (control word '
                  f'{hand.sdk.read_control_mode()})')
        finally:
            hand.close()
    if samples:
        poses = [s[1] for s in samples]
        moved = [max(p[j] for p in poses) - min(p[j] for p in poses) for j in range(16)]
        mean_ma = [1000 * sum(s[2][m] for s in samples) / len(samples) for m in range(16)]
        print(row('range deg', [moved[j] for j in sel], '6.2f'))
        print(f'{"mean mA":>9s} ' + ' '.join(f'{mean_ma[j]:6.0f}' for j in sel) + f'   ({motors.strip()})')
        still = [JOINT_NAMES[j] for j in sel if moved[j] < 0.05]
        print(f'joints whose readback never changed in torque mode: {", ".join(still) or "none"}')


if __name__ == '__main__':
    main()
