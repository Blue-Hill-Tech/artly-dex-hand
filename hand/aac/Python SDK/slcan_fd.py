# -*- coding: utf-8 -*-
"""Lean slcan CAN FD transport for the D21 (pyserial only, no receive thread).

python-can's slcan backend reads the serial port one byte at a time and hands frames to the caller through a second
thread; on the bench that cost ~0.3 ms per Modbus round trip (1.07 ms vs 0.67 ms raw). This transport writes the
request line(s) in one USB write and parses the reply lines from chunked reads, synchronously.

transact() takes a LIST of Modbus RTU frames and returns their replies in order: sending the torque write and the
position read together ("pipelining") completes in ~1.0 ms instead of ~2.1 ms for two round trips, because the
hand answers each as it arrives.

Hard limits (bench 2026-11-04, both wedged the hand's CAN bridge one request behind until a power cycle):
- a reply must fit one CAN FD frame (64 bytes), i.e. at most MAX_READ_REGISTERS (29) registers per read;
- back-to-back requests need a gap (INTER_FRAME_GAP_S) so the hand has finished parsing one before the next
  arrives, as the RS-485 pipelining does; frames sent with no gap at all wedged it within a few hundred ticks.
"""
import select
import time

import serial

CAN_FD_DLC_SIZES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)
DATA_BITRATE_CODES = {2000000: "Y2", 5000000: "Y5"}
NOMINAL_BITRATE_CODES = {1000000: "S8", 500000: "S6"}
MAX_READ_REGISTERS = 29     # 3 + 2*29 + 2 = 63 bytes: the largest Modbus read reply that fits a 64-byte frame
INTER_FRAME_GAP_S = 0.0005  # between back-to-back requests: the hand's reply to the first starts after ~0.25-0.45 ms


class SlcanFD:
    def __init__(self, channel, bitrate=1000000, data_bitrate=5000000, tty_baudrate=10000000, timeout=0.1,
                 request_id=0x010, response_id=0x020, port=None, gap_s=INTER_FRAME_GAP_S):
        if bitrate not in NOMINAL_BITRATE_CODES or data_bitrate not in DATA_BITRATE_CODES:
            raise ValueError(f"unsupported CAN FD rates {bitrate}/{data_bitrate}")
        self.channel, self.bitrate, self.data_bitrate = channel, bitrate, data_bitrate
        self.tty_baudrate, self.timeout = tty_baudrate, timeout
        self.request_prefix = f"b{request_id:03X}".encode()
        self.response_prefix = f"b{response_id:03X}".encode()
        self.port = port            # an open port-like object (tests); None = open the channel on connect()
        self.gap_s = gap_s
        self._buffer = bytearray()
        self.connected = False

    # ---------- channel ----------
    def connect(self):
        if self.connected:
            return
        if self.port is None:
            self.port = serial.Serial(self.channel, self.tty_baudrate, timeout=0, write_timeout=self.timeout)
        self.port.write(b"C\r")                    # close a channel left open by an earlier process
        time.sleep(0.05)
        self.drain()
        self.port.write((NOMINAL_BITRATE_CODES[self.bitrate] + "\r" + DATA_BITRATE_CODES[self.data_bitrate] + "\r"
                         + "O\r").encode())
        time.sleep(0.05)
        self.drain()
        self.connected = True

    def disconnect(self):
        port, self.port = self.port, None
        self.connected = False
        if port is None:
            return
        try:
            port.write(b"C\r")
        finally:
            port.close()

    def drain(self):
        """Drop whatever is pending on the port (stale replies, adapter acknowledgements)."""
        self._buffer.clear()
        reset = getattr(self.port, "reset_input_buffer", None)
        if reset is not None:
            reset()
        else:
            self.port.read(1 << 16)

    # ---------- frames ----------
    def line(self, data):
        """slcan text line for one CAN FD frame (bit-rate switch on), padded to the next DLC size."""
        if len(data) > 64:
            raise ValueError("CAN FD payload exceeds 64 bytes")
        dlc = next(i for i, size in enumerate(CAN_FD_DLC_SIZES) if size >= len(data))
        padded = bytes(data).ljust(CAN_FD_DLC_SIZES[dlc], b"\x00")
        return self.request_prefix + f"{dlc:X}".encode() + padded.hex().upper().encode() + b"\r"

    @staticmethod
    def check_request(req):
        """Refuse a read whose reply would not fit one CAN FD frame."""
        if len(req) >= 6 and req[1] in (0x03, 0x04):
            count = int.from_bytes(req[4:6], "big")
            if count > MAX_READ_REGISTERS:
                raise ValueError(f"CAN FD read of {count} registers: at most {MAX_READ_REGISTERS} per request "
                                 f"(the reply must fit a 64-byte frame)")

    def _decode(self, line):
        """Payload of a response line, or None for anything else (acknowledgements, other IDs, garbage)."""
        if not line.startswith(self.response_prefix) or len(line) < len(self.response_prefix) + 1:
            return None
        try:
            dlc = int(chr(line[len(self.response_prefix)]), 16)
            hexdata = line[len(self.response_prefix) + 1:]
            data = bytes.fromhex(hexdata.decode())
        except ValueError:
            return None
        return data[:CAN_FD_DLC_SIZES[dlc]]

    def _read_chunk(self, remaining):
        port = self.port
        if remaining > 0 and hasattr(port, "fileno"):
            select.select([port], [], [], remaining)
        waiting = getattr(port, "in_waiting", 0)
        return port.read(waiting or 1)

    def transact(self, requests, timeout=None, overlap=False):
        """Send Modbus RTU frames and return their replies in order.

        overlap=False (default): each request goes out once the previous reply is complete; the saving over single
        calls is only the per-call overhead. overlap=True: the next request follows the previous one after gap_s
        without waiting for its reply, so the hand works on one while the adapter carries the other (~1.0 ms for a
        torque write + position read instead of ~1.5 ms). Only safe with a gap the hand tolerates: see the module
        docstring.

        A reply matches when its slave address and function code (or the exception code) are those of the next
        outstanding request; frames that match nothing are skipped. TimeoutError when a reply is missing: the caller
        should drain() and redo the exchange one request at a time."""
        if not self.connected:
            raise RuntimeError("CAN FD transport not connected")
        requests = [bytes(r) for r in requests]
        for req in requests:
            if len(req) < 4:
                raise ValueError("Modbus request too short")
            self.check_request(req)
        timeout = self.timeout if timeout is None else timeout
        self._buffer.clear()
        deadline = time.perf_counter() + timeout
        replies = []
        sent = 0
        while len(replies) < len(requests):
            while sent < len(requests) and (sent == len(replies) or overlap):
                if sent and overlap:    # let the hand finish parsing the previous request first
                    until = time.perf_counter() + self.gap_s
                    while time.perf_counter() < until:
                        pass
                self._write(self.line(requests[sent]))
                sent += 1
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"CAN FD: {len(replies)} of {len(requests)} replies within {1000 * timeout:.0f} ms")
            chunk = self._read_chunk(remaining)
            if not chunk:
                continue
            self._buffer += chunk
            while b"\r" in self._buffer:
                raw, _, rest = self._buffer.partition(b"\r")
                self._buffer = bytearray(rest)
                data = self._decode(bytes(raw))
                if data is None or len(data) < 2:
                    continue
                want = requests[len(replies)]
                if data[0] == want[0] and data[1] in (want[1], want[1] | 0x80):
                    replies.append(data)
        return replies

    def _write(self, line):
        self.port.write(line)
        flush = getattr(self.port, "flush", None)
        if flush is not None:
            flush()
