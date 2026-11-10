"""Time each current read on the wire: request written -> first reply byte -> full reply. Read only.

No vendor code: pyserial + slcan text + Modbus RTU built here (same framing as raw_canfd.py). Every
request is Function Code 04 (read input registers); nothing is written to the hand's registers. For
each request it records, with time.perf_counter_ns():

  t0  just before the slcan line is written to the CANable's USB serial port
  t1  write() returned (handed to the USB driver)
  t2  first byte of the reply line arrived on the port
  t3  the reply line (ID 0x020) is complete and its CRC checked

and reports  t1-t0 (USB write), t2-t0 (time until the hand's answer starts arriving), t3-t0 (full
round trip), plus whether the values in that reply differed from the previous reply, so you can see
whether a fresh value costs more time than a repeated one. Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/read_latency.py                  # 16 motor currents
    hand/.hand/bin/python dex-retargetor-cpp/python/read_latency.py --what positions
    hand/.hand/bin/python dex-retargetor-cpp/python/read_latency.py --count 2000 --gap-ms 5 --csv /tmp/lat.csv
    hand/.hand/bin/python dex-retargetor-cpp/python/read_latency.py --motor 7 --count 5000   # one register only

With --motor N only that motor's current register (31 + N) is read, and the report adds how often
that single value changes: changes per second and the interval between changes.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import serial  # pyserial only

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
REQ_ID, RESP_ID = 0x010, 0x020
FD_LENGTHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
GROUPS = {'currents': 32, 'positions': 0}  # input register start; 16 registers each


def crc16_modbus(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(body):
    crc = crc16_modbus(body)
    return body + bytes([crc & 0xFF, crc >> 8])


def open_bus(port):
    ser = serial.Serial(port, baudrate=10000000, timeout=0, write_timeout=0.5)  # non-blocking reads
    time.sleep(2.0)
    ser.reset_input_buffer()
    for cmd in ('C', 'S8', 'Y2', 'O'):  # close, 1 Mbit/s nominal, 2 Mbit/s data, open
        ser.write(cmd.encode() + b'\r')
        time.sleep(0.05)
    time.sleep(0.1)
    ser.reset_input_buffer()
    return ser


def slcan_line(modbus):
    dlc = next(code for code, n in enumerate(FD_LENGTHS) if n >= len(modbus))
    padded = modbus + bytes(FD_LENGTHS[dlc] - len(modbus))
    return (f'b{REQ_ID:03X}{dlc:X}' + padded.hex().upper() + '\r').encode()


def timed_read(ser, line, expect_len, timeout_s=0.1):
    """One request; returns (t0, t1, t2, t3) in ns and the register values."""
    buf = b''
    t0 = time.perf_counter_ns()
    ser.write(line)
    t1 = time.perf_counter_ns()
    t2 = None
    deadline = t0 + int(timeout_s * 1e9)
    while True:
        n = ser.in_waiting
        if n:
            chunk = ser.read(n)
            if t2 is None:
                t2 = time.perf_counter_ns()
            buf += chunk
            while b'\r' in buf:
                raw, buf = buf.split(b'\r', 1)
                text = raw.replace(b'\x07', b'').decode(errors='replace')
                if len(text) >= 5 and text[0] in 'bd' and int(text[1:4], 16) == RESP_ID:
                    t3 = time.perf_counter_ns()
                    data = bytes.fromhex(text[5:5 + 2 * FD_LENGTHS[int(text[4], 16)]])[:expect_len]
                    if data[-2:] != with_crc(data[:-2])[-2:]:
                        raise RuntimeError('bad CRC in reply')
                    if data[1] & 0x80:
                        raise RuntimeError(f'Modbus exception {data[2]}')
                    values = tuple(int.from_bytes(data[3 + 2 * i:5 + 2 * i], 'big') for i in range(data[2] // 2))
                    return (t0, t1, t2, t3), values
        if time.perf_counter_ns() > deadline:
            raise TimeoutError('no reply within 100 ms')


def stats(label, ms):
    ms = np.asarray(ms)
    print(f'  {label:<34} median {np.median(ms):6.3f}  p10 {np.percentile(ms, 10):6.3f}  '
          f'p90 {np.percentile(ms, 90):6.3f}  p99 {np.percentile(ms, 99):6.3f}  max {ms.max():6.3f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--what', choices=sorted(GROUPS), default='currents')
    parser.add_argument('--count', type=int, default=1000, help='number of requests')
    parser.add_argument('--gap-ms', type=float, default=0.0, help='pause between requests (0 = back to back)')
    parser.add_argument('--motor', type=int, default=None, help='read only this motor\'s current register (1-16)')
    parser.add_argument('--csv', default=None, help='optional per-request log')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()
    port = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']['serial_device']

    if args.motor is not None:
        if not 1 <= args.motor <= 16:
            parser.error('--motor must be 1-16')
        start, n_regs, label = 31 + args.motor, 1, f'motor {args.motor} current (input register {31 + args.motor})'
    else:
        start, n_regs = GROUPS[args.what], 16
        label = f'{args.what} (input registers {start}-{start + 15})'
    expect_len = 5 + 2 * n_regs
    request = with_crc(bytes([0x01, 0x04]) + start.to_bytes(2, 'big') + n_regs.to_bytes(2, 'big'))
    line = slcan_line(request)
    print(f'request: modbus {request.hex(" ").upper()}  ->  slcan {line[:-1].decode()}\\r')
    print(f'{args.count} reads of {label}, '
          f'{"back to back" if args.gap_ms <= 0 else f"{args.gap_ms} ms apart"}; read only')

    ser = open_bus(port)
    rows = []
    last = None
    try:
        timed_read(ser, line, expect_len)  # warm-up
        values_log = []
        for _ in range(args.count):
            (t0, t1, t2, t3), values = timed_read(ser, line, expect_len)
            values_log.append(values)
            changed = last is not None and values != last
            last = values
            rows.append((t0, t1, t2, t3, changed))
            if args.gap_ms > 0:
                time.sleep(args.gap_ms / 1000)
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        ser.write(b'C\r')
        ser.close()

    if not rows:
        return
    a = np.array([r[:4] for r in rows], dtype=float)
    changed = np.array([r[4] for r in rows])
    write_ms, first_ms, total_ms = (a[:, 1] - a[:, 0]) / 1e6, (a[:, 2] - a[:, 0]) / 1e6, (a[:, 3] - a[:, 0]) / 1e6
    span = (a[-1, 3] - a[0, 0]) / 1e9
    print(f'\n{len(rows)} requests in {span:.2f} s = {len(rows) / span:.0f} requests/s')
    stats('USB write (t1-t0)', write_ms)
    stats('until reply starts (t2-t0)', first_ms)
    stats('full round trip (t3-t0)', total_ms)
    if changed.any() and (~changed[1:]).any():
        print(f'  round trip, reply with NEW values ({changed.sum()}):      median {np.median(total_ms[changed]):.3f} ms')
        print(f'  round trip, reply with SAME values ({(~changed).sum() - 1}):  median {np.median(total_ms[1:][~changed[1:]]):.3f} ms')
    edges = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf]
    counts, _ = np.histogram(total_ms, bins=edges)
    labels = ['<0.5', '0.5-0.75', '0.75-1', '1-1.25', '1.25-1.5', '1.5-2', '2-3', '3-5', '>5']
    print('  round trip histogram (ms): ' + '  '.join(f'{lab}:{c}' for lab, c in zip(labels, counts)))
    if n_regs == 1:
        t_ms = (a[:, 3] - a[0, 3]) / 1e6
        vals = np.array([v[0] for v in values_log], dtype=float)
        change_t = t_ms[1:][np.diff(vals) != 0]
        print(f'\nmotor {args.motor}: value changed {len(change_t)} times in {span:.2f} s = '
              f'{len(change_t) / span:.1f} changes per second '
              f'(values {int(vals.min()) - 1000:+d}..{int(vals.max()) - 1000:+d} mA)')
        if len(change_t) > 2:
            iv = np.diff(change_t)
            print(f'  interval between changes: median {np.median(iv):.1f} ms, p10 {np.percentile(iv, 10):.1f}, '
                  f'p90 {np.percentile(iv, 90):.1f}, min {iv.min():.1f}, max {iv.max():.1f} ms')
            edges = [0, 3, 7, 12, 17, 23, 30, 45, 60, np.inf]
            counts, _ = np.histogram(iv, bins=edges)
            labels = ['<3', '3-7', '7-12', '12-17', '17-23', '23-30', '30-45', '45-60', '>60']
            print('  interval histogram (ms): ' + '  '.join(f'{lab}:{c}' for lab, c in zip(labels, counts)))
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t0_s', 'usb_write_ms', 'reply_start_ms', 'round_trip_ms', 'values_changed'])
            for r, wm, fm, tm in zip(rows, write_ms, first_ms, total_ms):
                w.writerow([f'{(r[0] - rows[0][0]) / 1e9:.6f}', f'{wm:.4f}', f'{fm:.4f}', f'{tm:.4f}', int(r[4])])
        print(f'log: {args.csv}')


if __name__ == '__main__':
    main()
