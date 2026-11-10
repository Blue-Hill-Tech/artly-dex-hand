"""Move one D21 joint to a given angle, print its raw readback, then put it back. Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/jog_joint.py --hand_serial AAC-002 index_abduction 15
    hand/.hand/bin/python dex-retargetor-cpp/python/jog_joint.py --hand_serial AAC-002 thumb_rotation 20 --hold 4
    hand/.hand/bin/python dex-retargetor-cpp/python/jog_joint.py index_mcp 20 --rs485 /dev/ttyACM0

Use it to check a sign convention on a hand: watch which way the joint moves for a positive command, and whether
the raw readback follows the command with the same sign (then that hand's readback_sign is +1 for the joint).
Reference, left hand: index_abduction + moves the index toward the thumb; its raw abduction readback has the
opposite sign to the command (READBACK_SIGN). The pose is read in the hand's command frame, one joint is
replaced by ANGLE, the readback is printed every 0.1 s for --hold seconds, then the start pose is written back.
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import ANGLE_LIMITS, AACHand, JOINT_NAMES, load_sdk_module  # noqa: E402
from teleop.hand_control.hand_controller_aac import load_aac_config, readback_sign  # noqa: E402


def joint_limits():
    """(lo, hi) degrees per SDK joint; the SDK table is keyed by joint index."""
    return [tuple(ANGLE_LIMITS[i]) for i in range(len(JOINT_NAMES))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('joint', choices=JOINT_NAMES)
    parser.add_argument('angle', type=float, help='target in degrees (command frame)')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    parser.add_argument('--hand_serial', default='AAC-001', help='role key in the config (AAC-001 left, AAC-002 right)')
    parser.add_argument('--hold', type=float, default=3.0, help='seconds to hold and print the readback')
    parser.add_argument('--speed', type=int, default=30, help='finger speed percent for the move')
    parser.add_argument('--keep', action='store_true', help='leave the joint at ANGLE instead of moving it back')
    parser.add_argument('--rs485', metavar='PORT', help='talk over this RS-485 serial port instead of the CAN FD '
                        'adapter in the config (baud from the register map: v2 4000000, v1 500000)')
    args = parser.parse_args()

    settings = load_aac_config(SimpleNamespace(hand_serial=args.hand_serial), args.config)
    j = JOINT_NAMES.index(args.joint)
    lo, hi = joint_limits()[j]
    if not lo <= args.angle <= hi:
        raise SystemExit(f'{args.joint} must be within [{lo}, {hi}] deg')
    if args.rs485:
        rs485 = load_sdk_module('d21_rs485').d21_rs485

        def rs485_factory(tty_baudrate=None, **kw):
            sdk = rs485(**kw)            # opens the port in the constructor; AACHand.open() still calls connect()
            sdk.connect = lambda: None
            return sdk
        hand = AACHand(args.rs485, settings['device_address'], settings['connection_timeout_s'],
                       sdk_factory=rs485_factory, bus='rs485')
    else:
        hand = AACHand(settings['serial_device'], settings['device_address'], settings['connection_timeout_s'])
    start = None
    start_raw = hand.open()
    sign = readback_sign(settings['hand_id'], hand.register_map)   # v2 firmware: no mirror on either hand
    try:
        if hand.register_map == 'v2':
            print('firmware:', hand.sdk.read_firmware_info())
        hand.set_position_mode()
        hand.set_speeds([args.speed] * 5)
        # Command-frame start pose, clipped to the command limits (readback may sit just outside them).
        start = [min(max(s * m, l), h) for (l, h), s, m in zip(joint_limits(), sign, start_raw)]
        print(f"{settings['hand_id']} ({args.hand_serial}): {args.joint} raw readback {start_raw[j]:+.1f}, "
              f"command frame {start[j]:+.1f}; readback_sign for this joint {sign[j]:+d}")
        jog = list(start)
        jog[j] = args.angle
        hand.write_positions(jog)
        print(f"commanded {args.joint} = {args.angle:+.1f}; watch the hand")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.hold:
            time.sleep(0.1)
            raw = hand.read_positions()[j]
            print(f"  t={time.monotonic() - t0:4.1f} s  raw {raw:+6.1f}  command frame {sign[j] * raw:+6.1f}")
        final = hand.read_positions()[j]
        same = (final - start_raw[j]) * (args.angle - start[j]) > 0
        print(f"raw readback moved {'with' if same else 'AGAINST'} the command: readback sign for {args.joint} on "
              f"{settings['hand_id']} is {'+1' if same else '-1'} (configured {sign[j]:+d})")
    finally:
        try:
            if args.keep:
                print('kept (--keep)')
            elif start is not None:
                hand.write_positions(start)
                time.sleep(1.0)
                print(f"back to start: raw {hand.read_positions()[j]:+.1f}")
        finally:
            hand.close()


if __name__ == '__main__':
    main()
