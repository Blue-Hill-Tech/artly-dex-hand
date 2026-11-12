"""Friction compensation, step 3 of 3: write the tuning into the RS-485 torque config.

Write the per-joint friction compensation results into the RS-485 torque config.

Reads teleop/calibration/acc/friction/aac_friction_tuning_left.json (from ff_tune2.py) and sets, in teleop/configure/acc/torque_control/hand_config_aac_l_torque_rs485.json:
  torque.breakaway_fine      -> aac_breakaway_left_fine.json (angle-dependent breakaway)
  torque.friction_ff_scale   -> 0 globally (per joint below)
  torque.joints[<joint>]     -> friction_scale, friction_ff_scale, and kd / dither_ma where the tuning chose them
Joints without a result keep the global values. Prints a table of what was written.
"""
import json

from friction_side import SIDE  # --side left|right

TUNING = SIDE.tuning
CONFIG = SIDE.config

tuning = json.loads(TUNING.read_text())
cfg = json.loads(CONFIG.read_text())
torque = cfg['torque']
torque['breakaway_fine'] = SIDE.fine.name
torque['friction_ff_scale'] = 0.0
torque['friction_ff_ramp_dps'] = 10.0
torque['target_velocity_window_s'] = 0.03
joints = torque.get('joints') or {}
print(f"{'joint':16s} {'err-friction':>12s} {'feed-forward':>12s} {'kd':>5s} {'dither':>7s} {'validation':>40s}")
for name, r in tuning.items():
    block = {'friction_scale': r['err_share'], 'friction_ff_scale': r['ff_share']}
    if r.get('kd') is not None:
        block['kd'] = r['kd']
    if r.get('dither') is not None:
        block['dither_ma'] = r['dither']
    joints[name] = block
    v = r['validation']
    print(f"{name:16s} {r['err_share']:12.1f} {r['ff_share']:12.1f} {str(r.get('kd') or ''):>5s} {str(r.get('dither') or ''):>7s} "
          f"err {v['err']:.1f} max {v['err_max']:.1f} slip {v['vel_p95']:.0f} {'PASS' if r['passed'] else 'borderline':>10s}")
torque['joints'] = joints
CONFIG.write_text(json.dumps(cfg, indent=2) + '\n')
print(f'written {CONFIG}')
