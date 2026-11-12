"""Standalone MANUS -> AAC full-hand dex vector preview; no hardware connection."""

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import sys
import textwrap
import time
import traceback

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # agi-os

from teleop.manus.tools.skeleton_source import load_source, select_glove



def release(actions, primary=None):
    """Attempt all resources, reporting secondary errors without hiding the first."""
    first = primary
    for action in actions:
        try:
            action()
        except BaseException as error:
            if first is None:
                first = error
            else:
                traceback.print_exception(type(error), error, error.__traceback__)
    if primary is None and first is not None:
        raise first.with_traceback(first.__traceback__)


class Evidence:
    """Optional exclusive-create JSONL and bounded logging timings."""
    def __init__(self, path):
        self.file = None if path is None else open(path, "x", encoding="utf-8", buffering=65536)
        self.logging = deque(maxlen=3600)

    def emit(self, record_type, **record):
        start = time.perf_counter()
        if self.file is not None:
            self.file.write(json.dumps(dict(type=record_type, **record), allow_nan=False) + "\n")
        self.logging.append(time.perf_counter() - start)

    def close(self):
        if self.file is not None:
            self.file.close()


def stats(values):
    return None if not len(values) else dict(zip(('p50_ms', 'p95_ms', 'max_ms'),
                                                (np.percentile(values, [50, 95, 100])*1000).tolist()))


JOINT_ORDER = ('Metacarpal', 'Proximal', 'Intermediate', 'Distal', 'Tip')  # MANUS joints, base -> tip


def raw_glove_overlay(engine):
    """The glove skeleton with no knuckle centring and no thumb tip offset: palm-frame positions rotated by the
    alignment and placed with the glove's palm node on the robot's palm frame origin (palm_down). Shows where the
    glove itself puts your hand relative to the robot palm. Returns (points, bone segments) or None."""
    xyz, alignment = engine.geometry, engine.alignment
    if xyz is None or alignment is None or engine.metadata is None:
        return None
    palm = np.asarray(engine.adapter.palm_root)
    rotation = palm[:3, :3] @ np.asarray(alignment['A'])
    placed = {n['id']: palm[:3, 3] + rotation @ np.asarray(xyz[n['id']], dtype=float)
              for n in engine.metadata if n['id'] in xyz}  # metadata order, like glove_overlay
    points = np.array(list(placed.values()), dtype=np.float32)
    segments = np.asarray([[placed[n['parent']], placed[n['id']]] for n in engine.metadata
                           if n['parent'] in placed and n['id'] in placed and n['parent'] != n['id']],
                          dtype=np.float32).reshape(-1, 3)
    return points, segments


def glove_overlay(engine, degrees, scale=1.0, targets=False):
    """The aligned glove skeleton placed on the robot model (root frame, metres), rotated by the alignment like
    the left pane. The alignment fixes only the rotation, so the hand is translated so the glove's four finger
    knuckles (index..pinky MCP) are centred on the robot's: where your hand is, rigidly.

    targets=True draws the hand the solver targets instead. Each finger chain is placed at its robot vector origin
    (knuckle, or the thumb anchor) and scaled from the matching glove origin exactly as the targets are built:
    `scale` times the row scale (retarget_tuning.yaml vector.row_scales). Joints up to the middle target use the
    "-mid" row's scale and the rest the "-tip" row's, so both target rings sit on this hand. Palm nodes scale
    about the knuckle centre by `scale`. Returns (points, bone segments) or None."""
    xyz, alignment = engine.geometry, engine.alignment
    if xyz is None or alignment is None or engine.metadata is None or engine.selected is None:
        return None
    rotation = np.asarray(engine.adapter.palm_root[:3, :3]) @ np.asarray(alignment['A'])
    shown = {i: rotation @ p for i, p in xyz.items()}
    shift = getattr(engine, 'thumb_tip_shift', None)  # glove.thumb_tip_offset, palm frame
    tip_node = engine.selected['local_ids'][1] if 'local_ids' in engine.selected else None
    if shift is not None and tip_node in shown and np.any(shift):
        shown[tip_node] = shown[tip_node] + rotation @ np.asarray(shift)
    poses = engine.adapter.model.transforms(degrees)
    point = getattr(engine.adapter, 'point', None) or (lambda poses, link: poses[link][:3, 3])  # centreline points
    robot_mcp = np.mean([point(poses, f'{finger}_mcp') for finger in ('index', 'middle', 'ring', 'pinky')], axis=0)
    glove_mcp = np.mean([shown[i] for i in engine.selected['mcp']], axis=0)
    chains = {}  # glove chain -> dict(glove origin, robot origin, tip scale, mid scale, mid node, mid joint)
    if targets:
        local_ids = engine.selected['local_ids']
        row_scales = list(getattr(engine.adapter, 'row_scales', None) or [])
        table = engine.adapter.vector_table
        for row, spec in enumerate(table):
            chain, link = str(spec.get('human_chain', '')), str(spec.get('robot_origin', ''))
            if not chain.startswith('Finger') or not str(spec.get('robot_task', '')).endswith('_tip') or chain in chains:
                continue
            node = local_ids[int(engine.adapter.human_indices[0][row])]
            if node not in shown or link not in poses:
                continue
            tip_scale = scale*(row_scales[row] if row < len(row_scales) else 1.0)
            entry = dict(glove=shown[node], robot=point(poses, link), tip=tip_scale, mid=tip_scale, mid_node=None,
                         mid_joint=None)
            for mid_row, mid_spec in enumerate(table):
                if mid_spec.get('human_chain') == chain and not str(mid_spec.get('robot_task', '')).endswith('_tip'):
                    entry.update(mid=scale*(row_scales[mid_row] if mid_row < len(row_scales) else 1.0),
                                 mid_node=local_ids[int(engine.adapter.human_indices[1][mid_row])],
                                 mid_joint=mid_spec.get('human_task'))
                    break
            chains[chain] = entry
    placed = {}
    for n in engine.metadata:
        i = n['id']
        if i not in shown:
            continue
        entry = chains.get(n.get('chain'))
        if entry is None:
            placed[i] = robot_mcp + (scale if targets else 1.0)*(shown[i] - glove_mcp)
            continue
        joint = n.get('joint')
        up_to_mid = i == entry['mid_node'] or (joint in JOINT_ORDER and entry['mid_joint'] in JOINT_ORDER
                                               and JOINT_ORDER.index(joint) < JOINT_ORDER.index(entry['mid_joint']))
        placed[i] = entry['robot'] + (entry['mid'] if up_to_mid else entry['tip'])*(shown[i] - entry['glove'])
    points = np.array(list(placed.values()), dtype=np.float32)
    segments = np.asarray([[placed[n['parent']], placed[n['id']]] for n in engine.metadata
                           if n['parent'] in placed and n['id'] in placed and n['parent'] != n['id']],
                          dtype=np.float32).reshape(-1, 3)
    return points, segments


class Timing:
    def __init__(self):
        self.preprocess = deque(maxlen=3600)
        self.solve = deque(maxlen=3600)
        self.math = deque(maxlen=3600)
        self.ui = deque(maxlen=3600)
        self.session = deque(maxlen=3600)
        self.receipts = deque(maxlen=3600)

    def summary(self, evidence):
        elapsed = self.receipts[-1]-self.receipts[0] if len(self.receipts)>1 else 0.
        return dict(preprocess=stats(self.preprocess), solve=stats(self.solve), math=stats(self.math),
                    ui=stats(self.ui), session=stats(self.session), logging=stats(evidence.logging),
                    processed_hz=(len(self.receipts)-1)/elapsed if elapsed>0 else None)


TOOLBAR_H = 34
TOOLBAR_BUTTONS = (  # (attribute, label, key): click the button or press the key to toggle
    ('show_text', 'Text [T]', 'T'), ('show_raw_glove', 'Glove raw, green [R]', 'R'), ('show_glove', 'Glove, orange [O]', 'O'),
    ('show_targets', 'Target hand, white [W]', 'W'), ('show_target_pose', 'Target pose [P]', 'P'),
    ('show_measured', 'Motor pose, cyan [M]', 'M'), ('show_filtered', 'Filtered vectors, red [F]', 'F'))
MEASURED_TINT = (0.35, 0.8, 1.0, 0.45)


class Preview:
    """One 3D pane: the D21 at the solver's target pose, the glove skeleton (orange) and the target hand (white)
    placed on it, the target/achieved markers, and optionally a translucent cyan D21 at the measured motor pose.
    A toolbar of toggles at the top; the readout text is drawn over the pane and can be hidden too."""

    def __init__(self, side, adapter, *, aac=False, dry_run=False):
        from vispy import app, scene
        from hand.aac.model import HandVisual
        app.use_app('glfw')
        self.app = app
        self.side = side
        self.closed = self.align_requested = False
        self.aac = aac
        self.commands = []
        self.show_text = self.show_glove = self.show_targets = self.show_target_pose = True
        self.show_filtered = True
        self.show_raw_glove = False
        self.show_measured = aac  # only the sender has telemetry
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1100, 900),
            title=(f'MANUS / AAC {side} ' + ('DRY RUN — virtual target' if dry_run else 'sender — virtual target'))
                  if aac else f'MANUS / AAC {side} dex vector preview — no hardware connection',
            bgcolor='#10151c', show=True)
        try:
            self.canvas.events.draw.ignore_callback_errors = False
            grid = self.canvas.central_widget.add_grid(spacing=0)
            spacer = scene.Widget()
            spacer.height_min = spacer.height_max = TOOLBAR_H
            grid.add_widget(spacer, row=0, col=0)
            view = grid.add_view(row=1, col=0, bgcolor='#17212b')
            view.camera = scene.cameras.TurntableCamera(fov=40, azimuth=35, elevation=20, up='+z')
            self.robot = HandVisual(view, side, [0.]*16)
            self.measured = HandVisual(view, side, [0.]*16)  # the real hand's pose, tinted
            for _, node, _ in self.measured.visuals:
                node.color = MEASURED_TINT
                node.set_gl_state('translucent', depth_test=True, cull_face=False)
            self.measured.set_visible(False)
            self.targets = scene.visuals.Markers(parent=view.scene)
            self.achieved = scene.visuals.Markers(parent=view.scene)
            self.errors = scene.visuals.Line(parent=view.scene, connect='segments', color='yellow', width=1)
            # MANUS skeleton placed on the robot model: where your hand is, versus where the robot went.
            self.glove_points = scene.visuals.Markers(parent=view.scene)
            self.glove_bones = scene.visuals.Line(parent=view.scene, connect='segments', color=(1., .6, .2, .8), width=2)
            # The glove skeleton with no centring or offsets, palm node on the robot palm origin (R toggles).
            self.raw_points = scene.visuals.Markers(parent=view.scene)
            self.raw_bones = scene.visuals.Line(parent=view.scene, connect='segments', color=(.55, 1., .35, .8), width=2)
            # The hand the targets are built from (per-finger row_scales x --scale).
            self.scaled_points = scene.visuals.Markers(parent=view.scene)
            self.scaled_bones = scene.visuals.Line(parent=view.scene, connect='segments', color=(1., 1., 1., .6), width=1)
            # The finger vectors at the FILTERED pose (after the one-euro filter, what the hand is sent): each vector
            # row from its robot origin to the achieved point of the filtered joint angles (F toggles).
            self.filtered_bones = scene.visuals.Line(parent=view.scene, connect='segments', color=(1., .15, .15, .9), width=3)
            for visual in (self.targets, self.achieved, self.errors, self.glove_points, self.glove_bones,
                           self.scaled_points, self.scaled_bones, self.raw_points, self.raw_bones, self.filtered_bones):
                visual.set_gl_state(depth_test=False)
            # Overlay in canvas pixels: the readout text and the toolbar buttons.
            self.text = scene.visuals.Text('', color='white', font_size=9, anchor_x='left', anchor_y='top',
                                           pos=(12, TOOLBAR_H + 10), parent=self.canvas.scene)
            self.buttons = []
            x = 8
            for attribute, label, key in TOOLBAR_BUTTONS:
                width = 9*len(label) + 22
                rect = scene.visuals.Rectangle(center=(x + width/2, TOOLBAR_H/2), width=width, height=TOOLBAR_H - 10,
                                               radius=4, color='#3a4a5c', border_color='#8899aa', parent=self.canvas.scene)
                text = scene.visuals.Text(label, pos=(x + width/2, TOOLBAR_H/2), color='white', font_size=8,
                                          anchor_x='center', anchor_y='center', parent=self.canvas.scene)
                self.buttons.append(dict(attribute=attribute, key=key, rect=rect, text=text, x0=x, x1=x + width))
                x += width + 6
            self.paint_buttons()
            self.canvas.events.close.connect(self.on_close)
            self.canvas.events.key_press.connect(self.on_key)
            self.canvas.events.mouse_press.connect(self.on_mouse)
        except BaseException as error:
            release([self.canvas.close], primary=error)
            raise

    def paint_buttons(self):
        for button in self.buttons:
            on = getattr(self, button['attribute'])
            button['rect'].color = '#2f7d4f' if on else '#3a4a5c'
            button['text'].color = 'white' if on else '#aab4bf'

    def toggle(self, attribute):
        setattr(self, attribute, not getattr(self, attribute))
        self.paint_buttons()

    def on_mouse(self, event):
        x, y = event.pos[0], event.pos[1]
        if y > TOOLBAR_H:
            return
        for button in self.buttons:
            if button['x0'] <= x <= button['x1']:
                self.toggle(button['attribute'])
                event.handled = True
                return

    def on_key(self, event):
        if event.key is None:
            return
        name = event.key.name.upper()
        if name == 'A':
            self.align_requested = True
            return
        for button in self.buttons:
            if name == button['key']:
                self.toggle(button['attribute'])
                return
        command = {'E': 'execute', 'I': 'idle', 'S': 'save'}.get(name)
        if command is not None:
            self.commands.append(command)

    def on_close(self, event):
        self.closed = True

    def poll(self):
        self.app.process_events()
        return not self.closed

    def update(self, engine, sample, timing):
        progress = sample['progress']
        progress_text = '' if progress is None else f"{progress['count']} / >=30 frames; {progress['elapsed_s']:.2f} / 1 s"
        age = sample['source_age_s']
        age_text = 'missing' if age is None else f'{age*1000:.1f} ms'
        xyz = engine.geometry
        result = sample['result']
        live = result is not None and sample['reason'] is None
        # A reference is explicit and only shown when a fresh skeleton is available.
        reference = not live and xyz is not None and engine.alignment is None
        self.robot.set_visible((live or reference) and self.show_target_pose)
        for marker in (self.targets, self.achieved, self.errors):
            marker.visible = live and self.show_target_pose
        self.filtered_bones.visible = live and self.show_filtered
        overlay = glove_overlay(engine, result['display_degrees']) if live and self.show_glove else None
        self.glove_points.visible = self.glove_bones.visible = overlay is not None
        if overlay is not None:
            self.glove_points.set_data(overlay[0], face_color=(1., .6, .2, .9), size=5)
            self.glove_bones.set_data(overlay[1])
        raw = raw_glove_overlay(engine) if self.show_raw_glove and engine.geometry is not None else None
        self.raw_points.visible = self.raw_bones.visible = raw is not None
        if raw is not None:
            self.raw_points.set_data(raw[0], face_color=(.55, 1., .35, .9), size=5)
            self.raw_bones.set_data(raw[1])
        scaled = (glove_overlay(engine, result['display_degrees'], engine.adapter.scale, targets=True)
                  if live and self.show_targets else None)
        self.scaled_points.visible = self.scaled_bones.visible = scaled is not None
        if scaled is not None:
            self.scaled_points.set_data(scaled[0], face_color=(1., 1., 1., .8), size=4)
            self.scaled_bones.set_data(scaled[1])
        header = (f'AAC {self.side} | {engine.adapter.method} | scale={engine.adapter.scale:.2f} | '
                  f'glove {sample["state"]} {progress_text} | source age {age_text}\n'
                  + ('RETARGETED VIRTUAL TARGET — NOT MEASURED\n' if self.aac else
                     'VIRTUAL MODEL — NO HARDWARE CONNECTION\n'))
        session = sample.get('session')
        measured = None
        if session is not None:
            def age_text(value):
                return 'missing' if value is None else f'{value*1000:.0f} ms'
            difference = ('unavailable' if session['max_delta_deg'] is None else
                          f"{session['max_delta_deg']:.1f} deg ({session['delta_joint']})")
            status_reason = session['reason'] or ('No controller sockets' if session['state']=='DRY RUN' else
                'Publishing eligible targets' if session['state']=='EXECUTING' else 'Software idle; explicit E required')
            header += (f"{session['state']} | E execute / I idle / A align / S save\n"
                'Idle stops new commands; physical motion may continue\n'
                f"Measured age {age_text(session['measured_age_s'])}; ACK age {age_text(session['acknowledged_age_s'])}\n"
                f"Target/measured max: {difference}\n"
                f"{status_reason}\n")
            measured = session.get('measured_model_deg')
        show_measured = self.show_measured and measured is not None
        self.measured.set_visible(show_measured)
        if show_measured:
            self.measured.update(measured)
        if live:
            self.robot.update(result['display_degrees'])
            rows = engine.adapter.display_rows
            target = np.array(result['target_root_m'], dtype=np.float32)[rows]
            achieved = np.array(result['achieved_root_m'], dtype=np.float32)[rows]
            self.targets.set_data(target, face_color=(0, 0, 0, 0), edge_color='magenta', edge_width=2, size=14)
            self.achieved.set_data(achieved, face_color='lime', size=7)
            self.errors.set_data(np.stack((target, achieved), axis=1).reshape(-1, 3))
            origins_f = np.array(result['origin_root_m'], dtype=np.float32)
            achieved_f = np.array(result['achieved_root_m'], dtype=np.float32)
            self.filtered_bones.set_data(np.stack((origins_f, achieved_f), axis=1).reshape(-1, 3))
            err = np.array(result['display_errors_m'])*1000
            labels = engine.adapter.display_labels
            others = np.delete(err, rows)
            projected = result.get('projected')
            contact = ('' if not projected or not any(projected) else ' | contact: ' + ', '.join(
                lab for lab, on in zip(engine.adapter.vector_labels, projected) if on))
            header += (f"{result['solver']['termination']} | {result['fit_status']} | {result['method']} scale {result['scale']:.2f}\n"
                f'mm: {labels[0]} {err[rows[0]]:.1f}, {labels[1]} {err[rows[1]]:.1f}, {labels[2]} {err[rows[2]]:.1f}, '
                f'{labels[3]} {err[rows[3]]:.1f}, {labels[4]} {err[rows[4]]:.1f} | other vectors max '
                f'{(others.max() if len(others) else 0.0):.1f}{contact}\n'
                f"Raw max {max(result['raw_errors_m'])*1000:.1f} mm\n"
                'Targets (magenta rings) and achieved (green) move with the robot finger origins\n'
                f'Orange: glove skeleton centred on the robot knuckles. White: target hand, '
                f'per-finger row_scales x{result["scale"]:.2f}\n')
        elif reference:
            self.robot.update([0.]*16)
            header += f"REFERENCE POSE — NOT LIVE (all zeros)\n{sample['reason']}\n"
        else:
            header += f"Model hidden / invalid: {sample['reason']}\n"
        if show_measured:
            header += 'Cyan: the hand at its measured motor positions\n'
        if raw is not None:
            header += 'Green: glove skeleton as MANUS reports it, palm node on the robot palm origin (no centring)\n'
        solve = timing.get('solve')
        if solve is not None:
            header += f"Solve p50/p95 {solve['p50_ms']:.2f}/{solve['p95_ms']:.2f} ms | "
        hz = timing.get('processed_hz')
        header += f"Processed Hz: {'waiting' if hz is None else f'{hz:.1f}'}; skipped {sample['skipped_total']}"
        self.text.visible = self.show_text
        if self.show_text:  # one Text entry per line: multi-line strings are centred on pos, not top-anchored
            lines = [wrapped for line in header.splitlines() for wrapped in textwrap.fill(line, width=110).splitlines()]
            self.text.text = lines
            self.text.pos = np.array([(12, TOOLBAR_H + 10 + 15*k) for k in range(len(lines))], dtype=np.float32)

    def close(self):
        self.canvas.close()


class Headless:
    align_requested = False
    def __init__(self):
        self.commands = []
    def poll(self):
        return True
    def update(self, engine, sample, timing):
        pass
    def close(self):
        pass


def run(args, source, lanes, *, terminal=None):
    """The sender loop over one or two sides (Lane objects). With two, each cycle solves both engines in parallel
    worker threads (the dexrt solve runs in C++ with the GIL released); everything else stays per-lane in order."""
    from concurrent.futures import ThreadPoolExecutor
    from teleop.manus.tools.dex_vector_retargeting import VectorDiagnostic, json_value, save_alignment
    for lane in lanes:
        lane.engine = VectorDiagnostic(lane.adapter)
        if getattr(source, 'keeps_alignment', False):
            lane.engine.keep_alignment_on_gap = True
            lane.engine.min_capture_frames = 15
        if getattr(args, 'scale', None) is not None:
            if args.scale <= 0:
                raise SystemExit('--scale must be positive')
            lane.engine.set_fixed_scale(args.scale)
            print(f'{lane.side} scale: {lane.adapter.scale:.3f} (human -> robot vector lengths)')
        if lane.adapter.method == 'dexpilot':
            print(f'{lane.side} method: DexPilot (fingertip pairs, contact projection); scale',
                  f'{lane.adapter.scale:.3f} fixed' if not lane.engine.auto_scale else 'from the glove palm width')
    pool = ThreadPoolExecutor(max_workers=len(lanes), thread_name_prefix='solve') if len(lanes) > 1 else None
    # The loop solves every glove frame as it arrives (~95-120 Hz); it no longer paces itself at 60 Hz (the old
    # viewer pace, which --no-loop-cap used to drop). The viewer still renders every cycle (~24 ms), so the full rate
    # needs --no-viz, and --send-rate caps the submissions to the controller separately.
    stopping = False
    handlers = {}
    initialize_attempted = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, stop)
        for lane in lanes:
            lane.evidence.emit('configuration', requested_glove_id=lane.spec['glove_id'], glove_id=None, **lane.adapter.configuration())
            if lane.calibration is not None:
                lane.evidence.emit('alignment_source', path=str(lane.spec['calibration']), calibration=lane.calibration)
            print(lane.side, 'effective native URDF/model degree bounds:',
                  dict(zip(lane.adapter.optimizer.target_joint_names, np.rad2deg(lane.adapter.bounds).tolist())))
            if lane.session is not None:
                lane.session.start()
        initialize_attempted = True
        source.backend.check(source.backend.lib.CoreSdk_InitializeIntegrated(), 'Initialize integrated SDK')
        source.connect()
        while not stopping:
            alive = True
            for lane in lanes:
                ui_start = time.perf_counter()
                alive = lane.preview.poll() and alive
                lane.ui_s = time.perf_counter()-ui_start
            if not alive:
                break
            terminal_commands = terminal.poll() if terminal is not None else []
            for lane in lanes:
                session_start = time.perf_counter()
                commands = list(lane.preview.commands)
                lane.preview.commands.clear()
                commands.extend(terminal_commands)
                if lane.preview.align_requested:
                    lane.preview.align_requested = False
                    commands.append('align')
                # Idle/A cancel E in the same poll; never execute a pending old intent.
                if 'idle' in commands or 'align' in commands:
                    commands = [c for c in commands if c != 'execute']
                    if lane.session is not None:
                        lane.session.idle('Alignment requested' if 'align' in commands else 'User requested idle')
                lane.commands = commands
                lane.session_s = time.perf_counter()-session_start
            frames, devices = source.consume()
            for lane in lanes:
                lane.new = None
                lane.chosen = select_glove(devices, lane.side, lane.spec['glove_id'], lane.locked)
                if lane.locked is None and lane.chosen is not None:
                    lane.locked = lane.chosen
                    lane.evidence.emit('glove', glove_id=lane.locked, device=devices[lane.locked])
                new = frames.get(lane.locked) if lane.chosen is not None else None
                lane.frame_stamp = lane.session.frame_timestamp(new) if lane.session is not None and new is not None else None
                if new is not None:
                    metadata = source.metadata(lane.locked)
                    if lane.engine.set_metadata(metadata):
                        lane.evidence.emit('metadata', glove_id=lane.locked, nodes=metadata, selected=lane.engine.selected)
                        if lane.session is not None and lane.session.execution_id is not None:
                            lane.session.idle('Hierarchy changed')
                    if lane.calibration is not None and not lane.calibration_installed:
                        lane.engine.install_alignment(lane.calibration, lane.locked)
                        lane.calibration_installed = True
                    if new['frame_id'] == lane.engine.last_frame_id:
                        new = None
                        lane.frame_stamp = None
                    else:
                        lane.hand = new
                lane.new = new
            now = time.perf_counter()
            for lane in lanes:
                if 'align' in lane.commands:
                    lane.engine.start_alignment(now)
                lane.engine.check_freshness(None if lane.chosen is None else lane.hand, now)
            def solve(lane):
                return lane.engine.process(lane.new, now) if lane.new is not None else lane.engine.snapshot(now)
            samples = list(pool.map(solve, lanes)) if pool is not None else [solve(lane) for lane in lanes]
            for lane, sample in zip(lanes, samples):
                lane.sample = sample
                if lane.new is not None:
                    if 'math_s' in sample:
                        lane.timing.math.append(sample['math_s'])
                        lane.timing.receipts.append(now)
                    if sample['result'] is not None:
                        lane.timing.solve.append(sample['result']['solve_s'])
                        lane.timing.preprocess.append(sample['result']['preprocess_s'])
            if stopping:
                break
            for lane in lanes:
                sample, engine, adapter = lane.sample, lane.engine, lane.adapter
                if lane.session is not None:
                    session_start = time.perf_counter()
                    sample['session'] = lane.session.update(sample, lane.frame_stamp, execute='execute' in lane.commands)
                    lane.timing.session.append(lane.session_s+time.perf_counter()-session_start)
                if 'save' in lane.commands:
                    path = lane.spec['calibration'] or (Path(__file__).resolve().parents[1] / 'alignment' / f'aac_alignment_{adapter.side.lower()}.json')
                    record = save_alignment(path, engine, lane.locked)
                    lane.evidence.emit('alignment_saved', path=str(path), calibration=record)
                    print(lane.side, 'saved alignment:', path)
                for event in engine.events:
                    lane.evidence.emit('alignment_or_invalidity', at=now, glove_id=lane.locked, **event)
                engine.events.clear()
                ui_start = time.perf_counter()
                lane.preview.update(engine, sample, lane.timing.summary(lane.evidence))
                lane.ui_s += time.perf_counter()-ui_start
                lane.timing.ui.append(lane.ui_s)
                if lane.new is not None:
                    lane.evidence.emit('sample', glove_id=lane.locked, ui_s=lane.ui_s, **sample)
                if now-lane.last_report >= 1:
                    summary = lane.timing.summary(lane.evidence)
                    print(adapter.side, f'{adapter.method} scale={adapter.scale:.2f}', sample['state'],
                          sample['reason'] or sample['result']['fit_status'], summary)
                    if lane.session is not None:
                        print(f'AAC sender ({lane.side}):', sample['session'])
                    lane.evidence.emit('performance', at=now, processed_count=engine.count, skipped_total=engine.skipped, **summary)
                    lane.last_report = now
            if all(lane.new is None for lane in lanes):
                time.sleep(0.001)  # nothing new from the glove: yield instead of spinning a core
    except Exception as error:
        # Failure status is reset before each optimize call; never reuse an old success.
        def failure(lane):
            lane.evidence.emit('failure', at=time.perf_counter(), glove_id=lane.locked, error_type=type(error).__name__,
                               error=str(error), hand=json_value(lane.new if lane.new is not None else lane.hand),
                               alignment=None if lane.engine is None else lane.engine.alignment,
                               attempt=json_value(lane.adapter.optimizer.attempt),
                               previous_rad=json_value(lane.adapter.sequence.last_qpos),
                               filter_rad=json_value(lane.adapter.sequence.filter.y))
        release([lambda lane=lane: failure(lane) for lane in lanes], primary=error)
        raise
    finally:
        primary = sys.exc_info()[1]
        actions = []
        if pool is not None:
            actions.append(lambda: pool.shutdown(wait=True))
        actions.extend(lane.session.close for lane in lanes if lane.session is not None)
        if initialize_attempted:
            actions.append(lambda: source.backend.check(source.backend.lib.CoreSdk_ShutDown(), 'Shut down SDK'))
        if hasattr(source, 'close'):
            actions.append(source.close)
        for lane in lanes:
            actions.extend([lane.preview.close,
                            lambda lane=lane: print(lane.side, 'final host timings (not end-to-end latency):',
                                                    lane.timing.summary(lane.evidence)),
                            lane.evidence.close])
        actions.extend(lambda sig=s, h=h: signal.signal(sig, h) for s, h in handlers.items())
        release(actions, primary=primary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glove-side', choices=('left', 'right', 'both'), default='left',
                        help='both: two retargeters (one window each) solved in parallel threads')
    parser.add_argument('--glove-id', type=lambda value: int(value, 16), help='physical glove ID in hexadecimal')
    parser.add_argument('--log', type=Path, help='optional new JSONL file; never overwrites')
    parser.add_argument('--calibration', type=Path, help='optional saved alignment; default is manual A')
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
    args = parser.parse_args(argv)
    launch(args)

def thumb_orientation_weights(args, tuning, method):
    """(proximal, distal) thumb orientation weights: the flags override vector.thumb_orientation in the tuning YAML.
    A non-zero weight needs --method vector and --solver dexrt: from a flag that is an error, from the YAML it is
    skipped with a note, so one YAML serves every solver."""
    from_yaml = tuning['vector']['thumb_orientation']
    flags = dict(proximal=getattr(args, 'thumb_orientation_weight', None),
                 distal=getattr(args, 'thumb_tip_orientation_weight', None))
    weights = {key: float(from_yaml[key] if flags[key] is None else flags[key]) for key in ('proximal', 'distal')}
    if weights['proximal'] < 0 or weights['distal'] < 0:
        raise SystemExit('--thumb-orientation-weight and --thumb-tip-orientation-weight must be >= 0')
    usable = method == 'vector' and getattr(args, 'solver', 'dex') == 'dexrt'
    if (weights['proximal'] > 0 or weights['distal'] > 0) and not usable:
        if any(flags.values()):
            raise SystemExit('thumb orientation needs --method vector and --solver dexrt')
        print('Thumb orientation from the tuning YAML ignored: it needs --method vector and --solver dexrt')
        return 0.0, 0.0
    return weights['proximal'], weights['distal']


def build_adapter(args, tuning, spec):
    """One side's retargeting adapter from the flags and the tuning, plus its loaded alignment (or None)."""
    from teleop.manus.tools.dex_vector_retargeting import DexAdapter, DexrtAdapter, load_alignment
    from teleop.manus.tools.dex_vector_retargeting import DexPilotAdapter
    method = getattr(args, 'method', 'vector')
    solver = getattr(args, 'solver', 'dex')
    if method == 'dexpilot':
        if solver != 'dexrt':
            raise SystemExit('--method dexpilot needs --solver dexrt (dex-retargetor-cpp)')
        adapter = DexPilotAdapter(spec['side'], tuning=tuning)
    elif solver == 'geort':
        from teleop.manus.geort.geort_adapter import GeortAdapter
        adapter = GeortAdapter(spec['side'], tuning=tuning, checkpoint=getattr(args, 'geort_checkpoint', None),
                               source=getattr(args, 'source', 'glove'),
                               epoch=getattr(args, 'geort_epoch', None))
        print(f"{spec['side']} GeoRT: {adapter.ik.weights} (the alignment only feeds the viewer and the residuals; "
              'row scales and the thumb vector tweaks do not apply)')
    else:
        adapter = (DexrtAdapter if solver == 'dexrt' else DexAdapter)(spec['side'], tuning=tuning)
    adapter.thumb_offset_deg = float(getattr(args, 'thumb_offset_deg', 0.0) or 0.0)
    adapter.thumb_tip_offset = dict(tuning['glove']['thumb_tip_offset'])
    if adapter.thumb_tip_offset['enabled']:
        print('Glove thumb tip offset: last bone turned {toward_index_deg:+.1f} deg toward the index, then '
              '{toward_index_mm:+.1f} mm toward the index and {along_mm:+.1f} mm along it'.format(**adapter.thumb_tip_offset))
    orientation, tip_orientation = thumb_orientation_weights(args, tuning, adapter.method)
    if orientation > 0 or tip_orientation > 0:
        adapter.enable_thumb_orientation(proximal=orientation, distal=tip_orientation)
        print('Thumb orientation:', ', '.join(f'glove {joint.lower()} bone -> robot {link} axes, weight {w:g}'
                                              + (f', roll {adapter.orientation_roll[key]:+g} deg' if adapter.orientation_roll.get(key) else '')
                                              for key, joint, link, w in adapter.orientation_bones),
              '; reference taken on the first live frame after A (hold the thumb relaxed)')
    adapter.thumb_cmc_amplify = float(getattr(args, 'thumb_cmc_amplify', 1.0))
    if adapter.thumb_cmc_amplify <= 0:
        raise SystemExit('--thumb-cmc-amplify must be positive')
    cap = getattr(args, 'thumb_cmc_amplify_max_deg', None)
    if cap is not None:
        if not 0 < cap <= 180:
            raise SystemExit('--thumb-cmc-amplify-max-deg must be in (0, 180]')
        adapter.thumb_cmc_amplify_max_deg = float(cap)
    if adapter.thumb_cmc_amplify != 1.0:
        print(f'Thumb CMC amplify {adapter.thumb_cmc_amplify:g}: glove metacarpal swing x{adapter.thumb_cmc_amplify:g} '
              'about the relaxed pose (first live frame after A)'
              + ('' if adapter.thumb_cmc_amplify_max_deg is None else f', soft cap {adapter.thumb_cmc_amplify_max_deg:g} deg'))
    adapter.thumb_cmc_gain = float(getattr(args, 'thumb_cmc_gain', 1.0))
    if adapter.thumb_cmc_gain <= 0:
        raise SystemExit('--thumb-cmc-gain must be positive')
    lock = getattr(args, 'thumb_rotation_lock', None)  # the flag overrides vector.thumb_rotation_lock (YAML)
    if lock is not None:
        if getattr(args, 'solver', 'dex') != 'dexrt':
            raise SystemExit('--thumb-rotation-lock needs --solver dexrt')
        if not -35 <= lock <= 35:
            raise SystemExit('--thumb-rotation-lock must be within the rotation range, -35..35 deg')
        adapter.set_thumb_rotation_lock(lock)  # left-hand convention; mirrored for the right hand
    elif adapter.thumb_rotation_lock_deg is not None and getattr(args, 'solver', 'dex') != 'dexrt':
        print('Thumb rotation lock from the tuning YAML ignored: it needs --solver dexrt')
        adapter.thumb_rotation_lock_deg = None
    if adapter.thumb_rotation_lock_deg is not None:
        print(f'Thumb rotation held at {adapter.thumb_rotation_lock_deg:+.0f} deg (weight '
              f'{adapter.thumb_rotation_lock_weight:g}): motors 1 and 2 move together (abduction only)'
              + (' [right hand: the left-convention angle is mirrored]' if adapter.side == 'Right' else ''))
    if adapter.thumb_cmc_gain != 1.0:
        print(f'Thumb CMC gain {adapter.thumb_cmc_gain:g} on the solved motor-3 angle')
    if adapter.thumb_offset_deg:
        print(f'Thumb offset: {adapter.thumb_offset_deg:+.1f} deg (outward)')
    if not getattr(args, 'no_filter', False):
        from teleop.manus.tools.one_euro import DEFAULT_CONFIG, load_filter
        from hand.backends.aac import JOINT_NAMES
        camera_default = Path(__file__).resolve().parents[1] / 'parameters' / 'teleop_filter_camera.yaml'
        path = getattr(args, 'filter_config', None) or (
            camera_default if getattr(args, 'source', 'manus') == 'camera' and camera_default.exists() else DEFAULT_CONFIG)
        if Path(path).exists():
            adapter.output_filter = load_filter(path, JOINT_NAMES, side=adapter.side)   # one_euro.left / .right
            print(f'{adapter.side} output filter:', 'disabled in ' + str(path) if adapter.output_filter is None
                  else f'One-Euro from {path}: ' + str(adapter.output_filter.describe()))
        elif getattr(args, 'filter_config', None):
            raise FileNotFoundError(f'filter config not found: {path}')
    calibration = None
    if spec['calibration'] is not None:
        try:
            calibration = load_alignment(spec['calibration'], adapter, spec['glove_id'])
        except ValueError as error:
            if spec.get('calibration_explicit', True):
                raise
            print(f"{spec['side']}: saved alignment {spec['calibration']} not usable ({error}); press A to align, S to save it")
    # Cold numerical initialization can exceed the tracking continuity budget
    # when combined with the first live render. Do it before SDK acquisition.
    adapter.warmup()
    return adapter, calibration


SIDES = ('left', 'right')
RIGHT_DEFAULT_PORTS = dict(stream='tcp://127.0.0.1:6310', control='tcp://127.0.0.1:6311', telemetry='tcp://127.0.0.1:6312')


def side_plan(args):
    """Per-side settings for --glove-side left|right|both: side name, controller hand id and sockets, alignment file,
    log file and pinned glove id. The defaults are LeftHand/RightHand and aac_alignment_<side>.json; the right side
    always uses the right controller's sockets (--right-stream/--right-control/--right-telemetry), and with both the
    logs are <log>_<side>.jsonl."""
    glove_side = getattr(args, 'glove_side', 'left')
    sides = list(SIDES) if glove_side == 'both' else [glove_side]
    both = len(sides) > 1
    hand_id = getattr(args, 'hand_id', None)
    calibration = getattr(args, 'calibration', None)
    glove_id = getattr(args, 'glove_id', None)
    log = getattr(args, 'log', None)
    if both and hand_id:
        raise SystemExit('--hand-id is per side; with --glove-side both the hand ids are LeftHand and RightHand')
    if both and calibration is not None:
        raise SystemExit('--calibration is per side; with --glove-side both the files are aac_alignment_<side>.json')
    if both and glove_id is not None:
        raise SystemExit('--glove-id is per side; with --glove-side both each side locks to its only glove')
    plan = []
    for side in sides:
        title = side.capitalize()
        camera = getattr(args, 'source', 'manus') == 'camera'
        # The camera's skeleton frame differs from the glove's: it keeps its own alignment file.
        default_calibration = Path(__file__).resolve().parents[1] / 'alignment' / f'aac_alignment_{side}{"_camera" if camera else ""}.json'
        if calibration is not None:
            alignment = Path(calibration)
        elif getattr(args, 'calibration_default', False) and default_calibration.exists():
            alignment = default_calibration  # the sender loads the saved alignment by default when there is one
        else:
            if getattr(args, 'calibration_default', False):
                print(f'{title}: no saved alignment at {default_calibration}; press A to align, S to save it')
            alignment = None
        spec = dict(side=title, hand_id=hand_id or f'{title}Hand', calibration=alignment, glove_id=glove_id,
                    calibration_explicit=calibration is not None,
                    log=None if log is None else (Path(log) if not both else
                        Path(log).with_name(f'{Path(log).stem}_{side}{Path(log).suffix or ".jsonl"}')))
        if side == 'right':  # alone or with both: the right controller's role (AAC-002) listens on its own ports
            spec.update(stream=getattr(args, 'right_stream', RIGHT_DEFAULT_PORTS['stream']),
                        control=getattr(args, 'right_control', RIGHT_DEFAULT_PORTS['control']),
                        telemetry=getattr(args, 'right_telemetry', RIGHT_DEFAULT_PORTS['telemetry']))
        else:
            spec.update(stream=getattr(args, 'stream', None), control=getattr(args, 'control', None),
                        telemetry=getattr(args, 'telemetry', None))
        plan.append(spec)
    return plan


class Lane:
    """One glove side: its adapter, engine, controller session, viewer window and evidence log."""

    def __init__(self, spec, adapter, calibration, preview, evidence, session):
        self.spec, self.side, self.adapter, self.calibration = spec, spec['side'], adapter, calibration
        self.preview, self.evidence, self.session = preview, evidence, session
        self.engine, self.timing = None, Timing()
        self.locked = self.hand = self.new = self.chosen = self.frame_stamp = self.sample = None
        self.calibration_installed = False
        self.last_report = -float('inf')
        self.commands, self.ui_s = [], 0.0

    def close_actions(self):
        return ([] if self.session is None else [self.session.close]) + [self.preview.close, self.evidence.close]


def build_lane(args, spec, adapter, calibration, *, aac):
    evidence = Evidence(spec['log'])
    preview = session = None
    try:
        if aac:
            from teleop.manus.stream.hand_stream_aac import AACSession, AACConnection, SUBMISSION_PERIOD_S
            preview = Headless() if args.no_viz else Preview(adapter.side, adapter, aac=True, dry_run=args.dry_run)
            connection = None if args.dry_run else AACConnection(spec['stream'], spec['control'], spec['telemetry'], spec['hand_id'])
            session = AACSession(adapter, evidence, connection, hand_id=spec['hand_id'],
                                 send_rate_hz=getattr(args, 'send_rate', 1/SUBMISSION_PERIOD_S))
        else:
            preview = Preview(adapter.side, adapter)
    except BaseException as error:
        actions = [] if session is None else [session.close]
        if preview is not None:
            actions.append(preview.close)
        release(actions+[evidence.close], primary=error)
        raise
    return Lane(spec, adapter, calibration, preview, evidence, session)


def launch(args, *, aac=False):
    from teleop.manus.tools.dex_vector_retargeting import load_tuning
    solver = getattr(args, 'solver', 'dex')
    retarget_config = getattr(args, 'retarget_config', None)
    camera_tuning = Path(__file__).resolve().parents[1] / 'parameters' / 'retarget_tuning_camera.yaml'
    if retarget_config is None and getattr(args, 'source', 'manus') == 'camera' and camera_tuning.exists():
        retarget_config = camera_tuning   # camera-fitted row scales (MediaPipe's hand proportions)
        print(f'retarget tuning for the camera: {camera_tuning}')
    tuning = load_tuning(retarget_config)
    print('Retarget tuning:', tuning['path'] or 'built-in defaults (teleop/manus/parameters/retarget_tuning.yaml missing)')
    plan = side_plan(args)
    if len(plan) > 1:
        print(f'Both hands: {", ".join(s["side"] for s in plan)}; the two retargeters solve in parallel threads')
    lanes = []
    terminal = None
    try:
        for spec in plan:
            print(f"--- {spec['side']} hand: controller id {spec['hand_id']}" + (f", sockets {spec['stream']} / {spec['control']} / {spec['telemetry']}" if aac else '')
                  + f", alignment {spec['calibration']}" + (f", log {spec['log']}" if spec['log'] else ''))
            adapter, calibration = build_adapter(args, tuning, spec)
            lanes.append(build_lane(args, spec, adapter, calibration, aac=aac))
        if aac:
            from teleop.manus.stream.hand_stream_aac import TerminalCommands
            terminal = TerminalCommands()
        if getattr(args, 'source', 'manus') == 'camera':
            from teleop.manus.tools.camera_source import load_camera_source
            source = load_camera_source(args, [spec['side'] for spec in plan])
        else:
            source = load_source()
    except BaseException as error:
        release([action for lane in lanes for action in lane.close_actions()], primary=error)
        raise
    run(args, source, lanes, terminal=terminal)


if __name__ == '__main__':
    main()
