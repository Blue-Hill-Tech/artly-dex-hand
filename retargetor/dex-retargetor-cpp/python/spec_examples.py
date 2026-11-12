"""Send the vendor protocol spec's torque examples as raw bytes, exactly as printed.

D21 Dexterous Hand Communication Protocol Specification, section 4.4:
  Example 9  (FC 06): torque mode on            request 01 06 00 15 00 02 19 CF
  Example 10 (FC 10): 1500 on motors 1 and 2    request 01 10 00 44 00 02 04 05 DC 05 DC 35 93
                      ("the hand performs abduction at the commanded torque")

The raw frames go through the vendor's own CAN FD transport (d21_canfd._transact), so the only
thing this script adds is the byte values from the document. Each response is printed next to
the documented response. After --hold seconds the torque registers are set back to 1000 (zero)
and the hand is left in torque mode. Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/spec_examples.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'hand' / 'aac' / 'Python SDK'))

EXAMPLE_9 = dict(name='Example 9: set torque mode (address 21 = 2)',
                 request=bytes.fromhex('01 06 00 15 00 02 19 CF'),
                 response=bytes.fromhex('01 06 00 15 00 02 19 CF'))
EXAMPLE_10 = dict(name='Example 10: write 1500 to motors 1 and 2 (addresses 68-69)',
                  request=bytes.fromhex('01 10 00 44 00 02 04 05 DC 05 DC 35 93'),
                  response=bytes.fromhex('01 10 00 44 00 02 01 DD'))
ZERO_TORQUE = dict(name='cleanup: write 1000 (zero torque) to motors 1 and 2',
                   request=None, response=None)


def hx(data):
    return ' '.join(f'{b:02X}' for b in data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    parser.add_argument('--hold', type=float, default=3.0, help='seconds to hold the example torque')
    args = parser.parse_args()

    from d21_canfd import d21_canfd
    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    sdk = d21_canfd(hand_id=role['device_address'], modbus_port=role['serial_device'], timeout=0.1,
                    tty_baudrate=10000000)
    # The spec examples assume slave ID 1; the first byte of every frame is that ID.
    if role['device_address'] != 1:
        sys.exit(f"config device_address is {role['device_address']}, but the spec frames are for slave 1")
    sdk.connect()

    def crc_frame(payload):
        crc = sdk._crc16(payload)
        return payload + bytes([crc & 0xFF, crc >> 8])

    ZERO_TORQUE['request'] = crc_frame(bytes.fromhex('01 10 00 44 00 02 04 03 E8 03 E8'))

    def send(example, is_write=True):
        print(f'\n{example["name"]}')
        print(f'  request  (sent)       : {hx(example["request"])}')
        resp = sdk._transact(example['request'], is_write=is_write)
        resp = bytes(resp[:len(example['response'])]) if example['response'] else bytes(resp)
        print(f'  response (received)   : {hx(resp)}')
        if example['response']:
            print(f'  response (documented) : {hx(example["response"])}  ->  '
                  f'{"MATCH" if resp == example["response"] else "DIFFERENT"}')
        return resp

    def status(label):
        q = sdk.read_joint_positions()
        i = sdk.read_motor_currents()
        print(f'  {label:<10} thumb_abduction {q[0]:6.2f} deg  thumb_rotation {q[1]:6.2f} deg  '
              f'| current m1 {i[0]:+.3f} A  m2 {i[1]:+.3f} A')

    try:
        print(f'mode before: {sdk.read_control_mode()}')
        status('before')
        send(EXAMPLE_9)
        print(f'  mode reads: {sdk.read_control_mode()}')
        send(EXAMPLE_10)
        print(f'  torque registers 68-69 read back: {sdk.read_all_torques()[:2]}  (expected [500, 500])')
        print(f'\nholding the documented torque for {args.hold:.1f} s; watch the thumb')
        start = time.monotonic()
        while time.monotonic() - start < args.hold:
            status(f't={time.monotonic() - start:4.1f}s')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            send(ZERO_TORQUE)
            print(f'  torque registers read back: {sdk.read_all_torques()[:2]}  (expected [0, 0])')
            print(f'hand left in mode {sdk.read_control_mode()} (2 = torque); start the AAC controller or '
                  f'power-cycle to return to position mode')
        finally:
            sdk.close()


if __name__ == '__main__':
    main()
