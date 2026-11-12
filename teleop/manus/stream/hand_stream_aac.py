"""Explicitly armed MANUS/AAC sender. Only the separate controller owns hardware."""

import argparse
import os
from pathlib import Path
import select
import sys
import time

import numpy as np
import zmq

from hand.aac.model import MODEL_JOINTS
from hand.backends.aac import JOINT_NAMES, finite_number, validate_positions, validate_vector
from teleop.manus.tools.skeleton_source import STALE_S
from teleop.manus.tools.preview_dex_retargeting import release


FEEDBACK_MAX_AGE_S = 1.0
SUBMISSION_PERIOD_S = 1/20
# Upper bound of --send-rate. It only caps submissions: the sender never sends faster than glove frames arrive
# (~60-120 Hz), and the controller's torque loop (up to ~360 Hz over RS-485) reuses the latest target between them.
MAX_SEND_RATE_HZ = 1000
CLOCK_TOL_S = .05
RPC_TIMEOUT_MS = 250
IDLE_WARNING = 'Software idle stops new commands; physical motion may continue.'


def joint_map(side):
    if side not in ('Left', 'Right'):
        raise ValueError('AAC model side must be Left or Right')
    # Firmware convention (same on both hands): + abduction is toward the thumb, + flexion toward the palm, + thumb
    # rotation turns the pad toward the fingers. Source: AAC's own host program draws both hands from the SDK angles
    # with per-side sign tables (hand/tests/test_vendor_joint_conventions.py), the left hand on the bench, and the
    # right abduction jog test (2026-10-30: the index moving negative was blocked by the middle finger).
    # The vendor's right URDF is a z-mirror of the left EXCEPT that finger abduction (joint1) and thumb rotation
    # (thumb_joint2) keep the left's joint axes, so for the same sign they move the mirror-opposite way
    # (+abduction = toward the pinky, +thumb_joint2 = thumb tip toward the palm; FK check). Those two get -1 on the
    # right; every other joint keeps the URDF sign. The test above checks all 16 joints on both sides.
    signs = {'thumb_rotation': -1 if side == 'Right' else 1}
    return [('thumb_joint1', 'thumb_abduction', 1), ('thumb_joint2', 'thumb_rotation', signs['thumb_rotation']),
            ('thumb_joint3', 'thumb_cmc', 1), ('thumb_joint4', 'thumb_mcp', 1)] + [
        (f'{finger}_joint{i}', f'{finger}_{name}', -1 if side == 'Right' and i == 1 else 1)
        for finger in ('index', 'middle', 'ring', 'pinky')
        for i, name in enumerate(('abduction', 'mcp', 'pip'), 1)]


def convert_angles(values, bounds, *, to_sdk, side='Left', clip=False):
    """Named inverse mappings, degrees throughout; no display flip.

    Outgoing targets (to_sdk) must lie inside the model bounds. Incoming telemetry may not: the
    controller's tracking correction and the measured pose can reach the wider SDK limits, so
    callers that only display those values pass clip=True.
    """
    mapping = joint_map(side)
    values = validate_vector(values, len(MODEL_JOINTS), 'model angle') if to_sdk else validate_positions(values)
    sdk_values = dict(zip(JOINT_NAMES, values))
    model = dict(zip(MODEL_JOINTS, values)) if to_sdk else {
        model: sdk_values[sdk]*sign for model, sdk, sign in mapping}
    model_values = np.array([model[name] for name in MODEL_JOINTS])
    limits = np.rad2deg(bounds)
    if np.any(model_values < limits[:, 0]) or np.any(model_values > limits[:, 1]):
        if not clip:
            raise ValueError('Angles exceed effective URDF/SDK/coupled model limits')
        model_values = np.clip(model_values, limits[:, 0], limits[:, 1])
    if not to_sdk:
        return model_values.tolist()
    sdk = {sdk: model[name]*sign for name, sdk, sign in mapping}
    return validate_positions([sdk[name] for name in JOINT_NAMES])


# The D21 reads finger abduction (SDK joints 4, 7, 10, 13) back with the opposite sign to the command; the controller's
# tracking loop corrects it with the same table (teleop/hand_control/hand_controller_aac.py READBACK_SIGN) and publishes the raw
# values, so the sender applies it here before comparing with targets or posing the measured hand.
READBACK_SIGN = (1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1)


def readback_sign(side):
    """Left: READBACK_SIGN. Right: the right D21 reads every joint back with the command's sign (teleop logs
    2026-10-30), so no correction; same rule as hand_controller_aac.readback_sign."""
    return (1,)*16 if side == 'Right' else READBACK_SIGN


def validate_telemetry(msg, hand_id, bounds, now, *, side='Left'):
    if (not isinstance(msg, dict) or msg.get('type') != 'hand_telemetry'
            or msg.get('hand_id') != hand_id or msg.get('angle_unit') != 'deg'
            or msg.get('joint_names') != list(JOINT_NAMES)):
        raise ValueError('Invalid AAC telemetry identity/units/joint names')
    try:
        enabled, execution_id = msg['execution_enabled'], msg['execution_id']
        if (type(enabled) is not bool
                or (enabled and (not isinstance(execution_id, str) or not execution_id))
                or (not enabled and execution_id is not None)):
            raise ValueError('Invalid AAC telemetry execution state/ID')
        published = finite_number(msg['t'], 'telemetry t')
        if published > now+CLOCK_TOL_S:
            raise ValueError('Future telemetry timestamp')
        result = dict(msg)
        for kind in ('measured', 'commanded'):
            angles, stamp = msg[kind+'_angle'], msg[kind+'_t']
            if angles is None and stamp is None:
                result[kind+'_model_angle'] = None
                continue
            stamp = finite_number(stamp, kind+'_t')
            if stamp > published+CLOCK_TOL_S:
                raise ValueError('Feedback/acknowledgment time exceeds publication time')
            if kind == 'measured':
                # The controller publishes the sign it uses (per hand AND firmware: the v2 firmware does not mirror
                # the left hand's abduction); the per-side rule is only the fallback for older controllers.
                sign = msg.get('readback_sign', readback_sign(side))
                if not isinstance(sign, (list, tuple)) or len(sign) != 16 or any(s not in (1, -1) for s in sign):
                    raise ValueError('Invalid AAC telemetry readback_sign')
                angles = [s*finite_number(a, 'measured_angle') for s, a in zip(sign, validate_vector(angles, 16, 'measured_angle'))]
                result['measured_angle_raw'], result['measured_angle'] = list(msg['measured_angle']), angles
            result[kind+'_model_angle'] = convert_angles(angles, bounds, to_sdk=False, side=side, clip=True)
        return result
    except KeyError as error:
        raise ValueError('Missing AAC telemetry field') from error


class AACConnection:
    def __init__(self, stream, control, telemetry, hand_id, context=None):
        self.context = context if context is not None else zmq.Context()
        self.owns_context = context is None
        self.control_addr, self.hand_id = control, hand_id
        self.sockets = []
        try:
            self.push = self.socket(zmq.PUSH)
            self.push.setsockopt(zmq.SNDHWM, 1)
            self.push.setsockopt(zmq.IMMEDIATE, 1)
            self.push.connect(stream)
            self.sub = self.socket(zmq.SUB)
            self.sub.setsockopt(zmq.SUBSCRIBE, b'')
            self.sub.setsockopt(zmq.CONFLATE, 1)
            self.sub.connect(telemetry)
        except BaseException as error:
            release([self.close], primary=error)
            raise

    def socket(self, kind):
        sock = self.context.socket(kind)
        self.sockets.append(sock)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RECONNECT_IVL, -1)
        sock.setsockopt(zmq.SNDTIMEO, RPC_TIMEOUT_MS)
        sock.setsockopt(zmq.RCVTIMEO, RPC_TIMEOUT_MS)
        return sock

    def mode(self, mode):
        sock = self.socket(zmq.REQ)
        try:
            sock.connect(self.control_addr)
            sock.send_json(dict(cmd='set_mode', mode=mode, hand_id=self.hand_id))
            reply = sock.recv_json()
        finally:
            sock.close(0)
            self.sockets.remove(sock)
        enabled = mode == 'execute'
        if (not isinstance(reply, dict) or reply.get('ok') is not True
                or reply.get('execution_enabled') is not enabled or 'execution_id' not in reply):
            raise RuntimeError(f'Unconfirmed AAC {mode}: {reply}')
        identifier = reply['execution_id']
        if (enabled and (not isinstance(identifier, str) or not identifier)) or (not enabled and identifier is not None):
            raise RuntimeError(f'Invalid AAC {mode} session ID: {reply}')
        return identifier

    def latest(self):
        try:
            return self.sub.recv_json(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None

    def send(self, message):
        try:
            self.push.send_json(message, flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def close(self):
        actions = [lambda s=s: s.close(0) for s in self.sockets]
        if self.owns_context:
            actions.append(self.context.term)
        release(actions)


class AACSession:
    def __init__(self, adapter, evidence, connection=None, *, hand_id='LeftHand', clock=time,
                 send_rate_hz=1/SUBMISSION_PERIOD_S):
        self.adapter, self.evidence, self.connection = adapter, evidence, connection
        self.hand_id, self.clock = hand_id, clock
        if not 1 <= send_rate_hz <= MAX_SEND_RATE_HZ:
            raise ValueError(f'send_rate_hz must be in [1, {MAX_SEND_RATE_HZ}]')
        self.submission_period = 1/send_rate_hz
        self.dry_run = connection is None
        self.execution_id = self.pending = self.telemetry = None
        self.ack_mono = self.ack_wall = None
        self.offset = clock.time()-clock.perf_counter()
        self.clock_fault = self.closed = False
        self.last_submit = -float('inf')
        self.sent = self.dropped = 0
        self.reason = 'Software idle; explicit E/execute required'
        self.target = self.delta = self.delta_joint = None
        self.last_guard = None

    def start(self):
        print(IDLE_WARNING)
        if not self.dry_run:
            self.idle('Startup confirmed software idle')
        self.evidence.emit('aac_configuration', hand_id=self.hand_id, dry_run=self.dry_run, joint_map=joint_map(self.adapter.side),
            source_max_age_s=STALE_S, measured_max_age_s=FEEDBACK_MAX_AGE_S,
            pose_difference_policy='diagnostic_only', raw_fit_policy='diagnostic_only',
            submission_rate_hz=1/self.submission_period, clock_tolerance_s=CLOCK_TOL_S, rpc_timeout_ms=RPC_TIMEOUT_MS)

    def idle(self, reason):
        # Clear local authorization before any blocking RPC or evidence I/O.
        self.execution_id = self.pending = None
        self.reason = reason
        if self.connection is not None:
            try:
                self.connection.mode('idle')
            except BaseException:
                print('WARNING: COULD NOT CONFIRM CONTROLLER IDLE. '+IDLE_WARNING, file=sys.stderr)
                raise
        self.evidence.emit('session', state='DRY RUN' if self.dry_run else 'SOFTWARE IDLE', reason=reason,
                           execution_id=None, at=self.clock.time())

    def check_clock(self):
        if abs(self.clock.time()-self.clock.perf_counter()-self.offset) > CLOCK_TOL_S and not self.clock_fault:
            self.clock_fault = True
            self.idle('Wall-clock discontinuity; restart sender before executing')
        return not self.clock_fault

    def frame_timestamp(self, hand):
        self.check_clock()
        # Called once at consumption, before metadata/FK/RPC. Never restamp later.
        return self.clock.time()-(self.clock.perf_counter()-hand['received'])

    def guard(self, sample):
        if self.clock_fault:
            return 'Wall-clock discontinuity; restart sender'
        received = sample.get('received')
        if received is None or not 0 <= self.clock.perf_counter()-received < STALE_S:
            return 'Missing/stale tracking (<0.35 s required)'
        result = sample.get('result')
        if sample.get('alignment') is None or sample.get('reason') is not None or result is None:
            return sample.get('reason') or 'Valid identity/alignment and solver output required'
        self.target = convert_angles(result['display_degrees'], self.adapter.bounds, to_sdk=True, side=self.adapter.side)
        errors = np.asarray(result['raw_errors_m'], dtype=float)
        if errors.shape != (len(self.adapter.vector_table),) or not np.isfinite(errors).all() or result['solver']['status'] not in (1,2,3,4):
            raise ValueError('Invalid raw solver result')
        feedback = self.telemetry
        if feedback is not None and feedback['measured_angle'] is not None:
            difference = np.abs(np.asarray(self.target)-feedback['measured_angle'])
            i = int(np.argmax(difference))
            self.delta, self.delta_joint = float(difference[i]), JOINT_NAMES[i]
        if self.dry_run:
            return None
        if (feedback is None or feedback['measured_t'] is None
                or not 0 <= self.clock.time()-feedback['measured_t'] < FEEDBACK_MAX_AGE_S):
            return 'Fresh measured feedback (<0.5 s) required'
        return None

    def report_execute(self, outcome, reason):
        print(f'[aac] E/execute {outcome}: {reason}', flush=True)
        self.evidence.emit('execute_request', outcome=outcome, reason=reason, at=self.clock.time())

    def update(self, sample, frame_stamp=None, execute=False):
        if execute:
            self.report_execute('received', 'Checking execution guards')
        self.check_clock()
        if self.connection is not None and not self.clock_fault:
            incoming = self.connection.latest()
            if incoming is not None:
                received_at = self.clock.time()
                incoming = validate_telemetry(incoming, self.hand_id, self.adapter.bounds, received_at, side=self.adapter.side)
                if self.telemetry is None or incoming['t'] >= self.telemetry['t']:
                    self.telemetry = incoming
                    self.evidence.emit('telemetry', received_unix_s=received_at, message=incoming)
                    if (self.execution_id is not None and incoming['t'] > self.ack_wall
                            and (not incoming['execution_enabled'] or incoming['execution_id'] != self.execution_id)):
                        self.idle('Controller idle/session changed; explicit E required')
                        if execute:
                            self.report_execute('blocked', self.reason)
                        return self.snapshot()
        self.target = self.delta = self.delta_joint = None
        reason = self.guard(sample)
        if reason != self.last_guard:
            self.evidence.emit('guard', reason=reason, eligible=reason is None, at=self.clock.time())
            self.last_guard = reason
        if reason is not None:
            if self.execution_id is not None:
                self.idle(reason)
            self.reason = reason
            self.pending = None
            if execute:
                self.report_execute('blocked', reason)
            return self.snapshot()  # An ineligible E is consumed, never latched.
        self.reason = None
        if execute and (self.dry_run or self.execution_id is not None):
            self.report_execute('ignored', 'Dry-run has no controller connection' if self.dry_run else 'Already executing')
        if execute and self.execution_id is None and not self.dry_run:
            self.execution_id = self.connection.mode('execute')
            self.ack_mono, self.ack_wall = self.clock.perf_counter(), self.clock.time()
            self.pending = None
            self.evidence.emit('session', state='EXECUTING', execution_id=self.execution_id,
                               at=self.ack_wall, receipt_boundary=self.ack_mono)
            self.check_clock()
            reason = self.guard(sample)  # RPC time cannot rejuvenate the sample.
            if reason is not None:
                self.idle(reason)
                self.report_execute('blocked', reason)
            else:
                self.report_execute('accepted', 'Waiting for a fresh post-ACK frame')
            return self.snapshot()  # Always discard the pre-ACK candidate.
        if self.execution_id is not None:
            if frame_stamp is not None and sample['received'] > self.ack_mono:
                if self.pending is not None:
                    self.dropped += 1
                self.pending = dict(t=frame_stamp, received=sample['received'], frame_id=sample['frame_id'], angle=self.target)
            if self.pending is not None and self.clock.perf_counter()-self.last_submit >= self.submission_period:
                candidate, self.pending = self.pending, None
                # Freshness is checked at actual submission, not just before solving.
                self.check_clock()
                reason = self.guard(sample)
                if reason is not None:
                    self.idle(reason)
                    return self.snapshot()
                if (not 0 <= self.clock.perf_counter()-candidate['received'] < STALE_S
                        or not 0 <= self.clock.time()-candidate['t'] < STALE_S):
                    self.dropped += 1
                    self.idle('Pending target expired')
                    return self.snapshot()
                message = dict(type='hand_stream', hand_id=self.hand_id, execution_id=self.execution_id,
                               t=candidate['t'], angle_unit='deg', angle=candidate['angle'])
                self.last_submit = self.clock.perf_counter()
                delivered = self.connection.send(message)
                self.sent += int(delivered)
                self.dropped += int(not delivered)
                self.evidence.emit('submission', frame_id=candidate['frame_id'], submitted=delivered,
                                   message=message, sent=self.sent, dropped=self.dropped)
        return self.snapshot()

    def snapshot(self):
        feedback = self.telemetry
        def age(key):
            return None if feedback is None or feedback[key] is None else self.clock.time()-feedback[key]
        return dict(state='DRY RUN' if self.dry_run else 'EXECUTING' if self.execution_id else
                    'BLOCKED' if self.reason else 'SOFTWARE IDLE', reason=self.reason,
                    execution_id=self.execution_id, sdk_target=self.target,
                    measured_age_s=age('measured_t'), acknowledged_age_s=age('commanded_t'),
                    max_delta_deg=self.delta, delta_joint=self.delta_joint,
                    measured_sdk=None if feedback is None else feedback['measured_angle'],
                    measured_model_deg=None if feedback is None else feedback.get('measured_model_angle'),
                    acknowledged_sdk=None if feedback is None else feedback['commanded_angle'],
                    named_sdk_target=None if self.target is None else dict(zip(JOINT_NAMES, self.target)),
                    sent=self.sent, dropped=self.dropped)

    def close(self):
        if self.closed:
            return
        self.closed = True
        actions = [lambda: self.idle('Sender shutdown')]
        if self.connection is not None:
            actions.append(self.connection.close)
        release(actions)


TERMINAL_KEYS = {'e': 'execute', 'i': 'idle', 'a': 'align', 's': 'save'}


class TerminalCommands:
    """Read available bytes only; partial piped lines cannot block the owner."""
    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.buffer = b''
        self.eof = False

    def poll(self):
        if self.eof:
            return []
        if not select.select([self.stream], [], [], 0)[0]:
            return []
        chunk = os.read(self.stream.fileno(), 4096)
        if not chunk:
            self.eof = True
            return []
        self.buffer += chunk.replace(b'\r', b'\n')  # a raw-mode terminal sends Enter as \r
        lines = self.buffer.split(b'\n')
        self.buffer = lines.pop()
        if len(self.buffer) > 4096:
            raise ValueError('Terminal command too long')
        commands = []
        for line in lines:
            command = line.decode().strip().lower()
            command = TERMINAL_KEYS.get(command, command)  # the viewer's keys, typed and followed by Enter
            if command in ('execute', 'idle', 'align', 'save'):
                commands.append(command)
            elif command:
                print('Commands (then Enter): e = execute, i = idle, a = align, s = save')
        return commands


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glove-side', choices=('left', 'right', 'both'), default='left',
                        help='both: two retargeters and two controller sessions (LeftHand on --stream/--control/'
                             '--telemetry, RightHand on --right-*), solved in parallel threads, one window each')
    parser.add_argument('--glove-id', type=lambda v: int(v, 16), help='pin a physical glove ID (hex); not with both')
    parser.add_argument('--hand-id', default=None, help='controller hand id (default LeftHand/RightHand by side)')
    parser.add_argument('--stream', default='tcp://127.0.0.1:6300', help='controller sockets of the (left) hand')
    parser.add_argument('--control', default='tcp://127.0.0.1:6301')
    parser.add_argument('--telemetry', default='tcp://127.0.0.1:6302')
    parser.add_argument('--right-stream', default='tcp://127.0.0.1:6310',
                        help='right controller sockets (--glove-side right or both; AAC-002 in teleop/configure/acc/torque_control/hand_config_aac_r_torque_canfd.json)')
    parser.add_argument('--right-control', default='tcp://127.0.0.1:6311')
    parser.add_argument('--right-telemetry', default='tcp://127.0.0.1:6312')
    parser.add_argument('--calibration', type=Path, default=None,
                        help='saved alignment JSON (default: teleop/manus/alignment/aac_alignment_<side>.json when it exists)')
    parser.add_argument('--dry-run', action='store_true', help='Retarget without opening any controller sockets')
    parser.add_argument('--no-viz', action='store_true', help='Use execute/idle/align/save terminal commands without a viewer')
    parser.add_argument('--log', type=Path, help='Optional new JSONL file; never overwrites')
    parser.add_argument('--solver', choices=('dex', 'dexrt', 'geort'), default='dex',
                        help='dex: dex-retargeting SLSQP + low-pass; dexrt: dex-retargetor-cpp OSQP, no filter; '
                             'geort: trained GeoRT network (teleop/manus/geort/geort_adapter.py)')
    parser.add_argument('--geort-checkpoint', type=Path, default=None,
                        help='--solver geort: GeoRT checkpoint directory (or a .pth inside one); default '
                             'retargetor/GeoRT_checkpoint/<side>_hand/: the run chosen for the side and source in '
                             'retargetor/GeoRT_checkpoint/selection.json (Dev Mode), else the newest there')
    parser.add_argument('--geort-epoch', type=int, default=None,
                        help='--solver geort: load epoch_N.pth of the checkpoint instead of last.pth')
    parser.add_argument('--retarget-config', type=Path, default=None,
                        help='YAML with the solver tunables: loss shape, smoothing and per-vector weights '
                             '(default: teleop/manus/parameters/retarget_tuning.yaml)')
    parser.add_argument('--filter-config', type=Path, default=None,
                        help='One-Euro YAML for the joint targets (default: teleop/manus/parameters/teleop_filter.yaml if present)')
    parser.add_argument('--no-filter', action='store_true', help='bypass the One-Euro filter')
    parser.add_argument('--method', choices=('vector', 'dexpilot'), default='vector',
                        help='vector: finger base->tip vectors (default); dexpilot: fingertip-to-fingertip vectors '
                             'with contact projection and hand-size scaling (needs --solver dexrt)')
    parser.add_argument('--scale', type=float, default=None,
                        help='human->robot length scale for the target vectors; vector method default 1.0, '
                             'dexpilot default robot palm width / glove palm width')
    parser.add_argument('--thumb-orientation-weight', type=float, default=None,
                        help='vector method, --solver dexrt: match the robot thumb (thumb_mcp link) orientation '
                             'from FK to the glove thumb proximal bone orientation; 0 = off. Try 5 '
                             '(default: vector.thumb_orientation.proximal in the tuning YAML)')
    parser.add_argument('--thumb-tip-orientation-weight', type=float, default=None,
                        help='vector method, --solver dexrt: match the robot thumb tip segment (thumb_ip link) '
                             'orientation to the glove thumb distal bone; 0 = off. Try 3 '
                             '(default: vector.thumb_orientation.distal in the tuning YAML)')
    parser.add_argument('--thumb-cmc-amplify-max-deg', type=float, default=None,
                        help='soft cap for --thumb-cmc-amplify: the amplified CMC swing approaches this many '
                             'degrees instead of growing past what the robot can reach. Try 40')
    parser.add_argument('--thumb-cmc-amplify', type=float, default=1.0,
                        help='multiply the glove thumb metacarpal swing about the CMC by this (vector and dexpilot) '
                             '(relaxed pose = first live frame after A; 1 = off)')
    parser.add_argument('--thumb-cmc-gain', type=float, default=1.0,
                        help='multiply the solved thumb CMC angle (motor 3) by this, clipped to its range (1 = off)')
    parser.add_argument('--thumb-rotation-lock', type=float, default=None, metavar='DEG',
                        help='hold thumb rotation at this angle so motors 1 and 2 move together (abduction only); '
                             'needs --solver dexrt. Overrides vector.thumb_rotation_lock in the tuning YAML')
    parser.add_argument('--thumb-offset-deg', type=float, default=0.0,
                        help='rotate the thumb vectors outward (toward the index side) by this many degrees '
                             'before the solve; the D21 thumb points further out than a human thumb can')
    parser.add_argument('--send-rate', type=float, default=1/SUBMISSION_PERIOD_S,
                        help=f'max target submissions per second to the controller (default 20, up to {MAX_SEND_RATE_HZ}); '
                             'the actual rate is at most the glove frame rate (~60-120)')
    parser.add_argument('--no-loop-cap', action='store_true',
                        help='no effect, kept so old commands still run: the loop always solves every glove frame as it '
                             'arrives (it used to pace itself at 60 Hz unless this was given). The viewer render '
                             '(~24 ms) still limits the loop unless --no-viz is given, and --send-rate caps submissions')
    camera = parser.add_argument_group('camera hand tracking (--source camera, MediaPipe on a webcam instead of the glove)')
    camera.add_argument('--source', choices=('manus', 'camera'), default='manus',
                        help='skeleton source: the MANUS glove (default) or webcam hand tracking')
    camera.add_argument('--camera-index', type=int, default=None,
                        help='/dev/videoN of the webcam (default: AAC_CAMERA_INDEX, else the first colour camera)')
    camera.add_argument('--camera-width', type=int, default=640)
    camera.add_argument('--camera-height', type=int, default=480)
    camera.add_argument('--camera-fps', type=float, default=30.0)
    camera.add_argument('--camera-model', default=None, help='MediaPipe hand_landmarker.task (default: searched in the repo)')
    camera.add_argument('--camera-mirror', action='store_true',
                        help='the camera image is already mirrored (selfie view): do not swap MediaPipe hand labels')
    camera.add_argument('--camera-any-hand', action='store_true',
                        help='use whichever hand is detected, ignoring MediaPipe\'s left/right label (the default '
                             'when only one side is driven)')
    camera.add_argument('--camera-preview', action='store_true', help='show the camera image with the tracked landmarks')
    camera.add_argument('--camera-palm-deadband', type=float, default=4.0,
                        help='deg: the palm frame ignores tilts smaller than this (default 4; 0 = follow every wobble)')
    camera.add_argument('--camera-palm-tau', type=float, default=0.15,
                        help='s: smoothing of the palm frame beyond the deadband (default 0.15)')
    camera.add_argument('--camera-depth-fix', action='store_true',
                        help='rebuild finger curl from the bone lengths (DepthFix; off by default: it made curled poses '
                             'worse than MediaPipe\'s own depth)')
    camera.add_argument('--camera-no-depth-fix', action='store_true', help=argparse.SUPPRESS)   # the default now
    camera.add_argument('--camera-no-flat-pose', action='store_true',
                        help='do not remove the bend MediaPipe reports for a flat hand (the flat pose captured by Align '
                             'in the GUI: teleop/manus/alignment/camera_flat_<side>.json)')
    camera.add_argument('--camera-no-hand-fit', action='store_true',
                        help='keep MediaPipe\'s own hand size instead of resizing every bone to the glove hand GeoRT was '
                             'trained on (teleop/manus/alignment/glove_hand_lengths.json)')
    args = parser.parse_args(argv)
    if args.source == 'camera' and args.glove_id is not None:
        parser.error('--glove-id is for the MANUS glove; the camera source has fixed ids')
    args.calibration_default = True  # the sender loads aac_alignment_<side>.json unless --calibration is given
    if args.hand_id is not None and not args.hand_id.strip():
        parser.error('--hand-id must be nonempty')
    if not 1 <= args.send_rate <= MAX_SEND_RATE_HZ:
        parser.error(f'--send-rate must be in [1, {MAX_SEND_RATE_HZ}]')
    return args


def main(argv=None):
    from teleop.manus.tools.preview_dex_retargeting import launch
    launch(parse_args(argv), aac=True)
