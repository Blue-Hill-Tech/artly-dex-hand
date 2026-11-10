# dex-retargetor-cpp

C++ replacement for the dex-retargeting solver used by the MANUS → AAC D21 teleop,
plus a joint-space impedance (spring-damper) controller for compliant operation.

| Piece | File | Status |
| --- | --- | --- |
| URDF kinematics (FK, point Jacobians) | `include/dexrt/kinematics.hpp` | validated against Pinocchio to 1e-9 m |
| Vector retargeting: IRLS-weighted QP on OSQP | `include/dexrt/retargeter.hpp` | validated on 7,281 recorded glove frames |
| Impedance controller | `include/dexrt/impedance.hpp` | unit-tested; **not yet run on hardware** |
| CAN FD / Modbus hand link (port of the vendor Python SDK) | `include/dexrt/d21_canfd.hpp` | frame encoding unit-tested; **not yet run on hardware** |
| 300–500 Hz impedance loop tool | `tools/impedance_loop.cpp` | dry-run mode by default |
| C ABI + ctypes bindings | `include/dexrt/c_api.h`, `python/dexrt.py` | used by `python/replay_log.py` |

## Build

```bash
cd dex-retargetor-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # fetches Eigen 3.4, OSQP 1.0, tinyxml2 (network needed once)
cmake --build build -j
(cd build && ctest --output-on-failure)
```

Nothing from the Python venv or conda is needed to build. Pinocchio is only used by
`python/gen_fk_reference.py` to regenerate the FK test data.

## How the retargeter works

Same problem as `teleop/manus/tools/dex_vector_retargeting.py`: ten finger vectors
(MCP→tip and MCP→PIP per finger; thumb from `thumb_pitch`) in the URDF world frame,
sixteen independent joints, five coupled distal joints (0.85× thumb, 0.8× fingers),
bounds = URDF ∩ SDK ∩ coupled limits, Huber loss with δ = 0.02 m, and the
`norm_delta·|q − q_last|²` smoothness term.

Instead of SLSQP each frame does up to `max_steps` (default 3) Gauss-Newton steps:

1. FK + Jacobians at the current `q` (own kinematics, no Pinocchio).
2. Huber weight per vector: `w = 1` if `|e| ≤ δ`, else `δ / |e|` (IRLS).
3. Box-constrained QP in `Δq` with `P = Σ w JᵀJ/(mδ) + 2·norm_delta·I`, solved by OSQP
   with warm start; bounds become `lower − q ≤ Δq ≤ upper − q`.
4. `q += Δq`, stop early when `max|Δq| < stop_step`.

There is **no low-pass filter** on the output. The `norm_delta` term is kept because it is
part of the objective dex-retargeting minimises (it is not a filter; it only ties each frame to
the previous solution the way the original does).

### Validation against the recorded run

`python/replay_log.py /tmp/manus_aac_live_02.jsonl` replays the glove vectors logged by the
Python sender and compares with the SLSQP result stored in the same log:

| | C++ (OSQP, 3 steps) | dex-retargeting (SLSQP) |
| --- | --- | --- |
| worst vector error, p50 / p95 / max | 19.29 / 48.33 / 62.76 mm | 19.30 / 48.30 / 62.82 mm |
| solve time, p50 / p95 / max | 0.17 / 0.32 / 0.59 ms | 1.38 / 2.72 / 10.5 ms |
| OSQP failures | 0 of 7,281 | — |

Fit quality is identical and the solve is ~8× faster. Joint angles differ by 1.2° (p50) and
up to ~19° on the thumb: the recorded poses sit in the Huber-linear regime (20–60 mm residuals,
the known "poor fit" of the current alignment), where several joint configurations give the
same cost. More Gauss-Newton steps do not change this, so it is a property of the objective,
not of the solver.

## DexPilot retargeting (thumb placed relative to the fingertips)

The plain vector method matches each finger's base->tip direction. The D21 thumb base is shaped unlike a
human's (at CMC 0 it points straight out sideways), so that direction can't be reproduced: in teleop the
thumb pinned itself at the CMC's inward stop and could not come back out. DexPilot (Handa et al. 2019, as
in dex-retargeting's `DexPilotOptimizer`) matches **fingertip-to-fingertip** vectors instead, so the thumb
is placed where it matters for grasping: relative to the other fingertips.

`VectorTask` now carries a weight and a projection class, `RetargeterOptions` the DexPilot thresholds, and
`VectorRetargeter::set_scaling` a per-session human->robot length scale. The AAC layout
(`dexrt.dexpilot_table`, `DEXPILOT_TABLE` in `teleop/manus/tools/dex_vector_retargeting.py`): 4 thumb->finger tip
vectors (primary: projected to a contact distance while the human pair is under `project_dist` = 3 cm, released
above `escape_dist` = 5 cm, weight 200 while projected), 6 finger->finger tip vectors (secondary, projected only
while both of their thumb pairs are), each finger's knuckle->tip shape vector (weight 15; DexPilot's wrist->tip,
anchored at a point the glove and the URDF both have), and the thumb's base->knuckle vector at low weight so the
D21's four thumb joints stay determined. Unprojected targets are scaled by robot palm width / glove palm width,
measured from the skeleton at the first live frame (`--scale` fixes it instead).

```bash
# viewer only
PYGLFW_LIBRARY_VARIANT=x11 hand/.hand/bin/python teleop/manus/tools/preview_dex_retargeting.py --glove-side left \
    --calibration teleop/manus/alignment/aac_alignment_left.json --solver dexrt --method dexpilot
# teleop sender (robot terminal unchanged)
PYGLFW_LIBRARY_VARIANT=x11 python teleop/manus/stream/hand_stream_client.py --model aac --glove-side left --hand-id LeftHand \
    --calibration teleop/manus/alignment/aac_alignment_left.json --solver dexrt --method dexpilot --send-rate 100 --log /tmp/manus_dp_01.jsonl
```

Replayed on a logged teleop session, DexPilot's robot thumb-index gap tracks the glove's (scaled) gap within
a few mm and closes to contact on every pinch, where the plain method left a 6-20 mm gap and sat at the CMC
stop half the time. The viewer header shows the method, the scale and which pairs are currently in contact.

## Position-mode compliance (works on today's hand)

Torque mode is not usable on the current D21 firmware (commands produce no current, and entering it can
leave motors unresponsive until a power cycle). `include/dexrt/compliance.hpp` gives compliance in
**position mode** instead: the firmware keeps its position loop, and the PC moves the target.

Per joint, every ~20 ms (the hand's feedback rate):

- **effort** = motor current above that motor's idle level (learned at startup; MCP/abduction pairs share it)
- **push direction** comes from the sign of the motor current: the motor fights the push, and each motor's
  sign convention is learned online from the moves the firmware makes anyway (the hold tool wiggles every
  joint at start; in teleop the glove motion teaches it). The firmware loop is so stiff that a hand push
  displaces a finger by only ~0.2°, which is why displacement alone (the fallback) is a weak signal
- while a joint has effort, is not moving along its own drive, and its target is not being moved, an
  **offset** grows in the push direction at `gain_dps_per_a × effort`, so the command follows the push
- the offset decays at `return_rate`, so the finger returns to the glove target when the push stops
- a finger stalled on an object behind its command backs the command off, which limits grip force;
  `error_clamp_deg` also caps how far the command may lead the real finger
- finger abduction reads back mirrored on this hand; `readback_sign` handles it

Steady-state softness is `gain / return_rate` degrees per amp (default 150 / 2 = 75 deg/A).

```bash
# 1. No glove: hold a pose compliantly; push a finger and it gives way, release and it returns
hand/.hand/bin/python dex-retargetor-cpp/python/compliance_hold.py
hand/.hand/bin/python dex-retargetor-cpp/python/compliance_hold.py --gain 250 --return-rate 1.5 --log /tmp/hold.csv
```

The controller's compliance is opt-in through the `compliance` block in the config; without it the controller
behaves exactly as before. `--solver dexrt` swaps only the solver (`DexrtAdapter`); the alignment, guards,
viewer and transport are unchanged, and the saved alignment stays valid.

## Closed-loop tracking correction (the opposite of compliance)

The retargeter is open loop: the viewer shows the *commanded* pose, and the hand's firmware stops short of
it (speed cap, tracking error). `TrackingCorrection` in `teleop/hand_control/hand_controller_aac.py` closes that loop on
the command: `command = target + kp·e + I`, `e = target − measured`, where `I` is a leaky integrator with a
deadband, bounded by `max_deg`, and frozen while the target itself is moving. Enable it with the `tracking`
block; it cannot be combined with `compliance`, since one pushes through resistance and the other yields to it.

## Current-based admittance (Minimalist Compliance Control)

`teleop/hand_control/admittance.py` makes the position-controlled hand compliant from motor current alone, after Shi et
al., *Minimalist Compliance Control* (RSS 2026). The servo's current is its torque; above the free-motion
baseline `I_free(q)` it is an external torque estimate, which drives a virtual spring-mass-damper on an
offset added to the task target: `M·δq̈ + D·δq̇ + K·δq = τ̄_ext`, `q_cmd = q_des + δq`. The spring is anchored to
the target, so a released finger returns; `K = 0` is hand guiding. Every parameter lives in
`teleop/calibration/acc/admittance/admittance.yaml`; the per-joint `I_free(q)` fit and current sign come from a calibration sweep.

```bash
# 1. Calibrate (position mode, nothing touching the hand; ~40 s per joint, all 16 by default)
hand/.hand/bin/python dex-retargetor-cpp/python/admittance_calibrate.py
hand/.hand/bin/python dex-retargetor-cpp/python/admittance_calibrate.py --joints index_pip index_mcp
# 2. Validate: print tau_ext while pushing each finger closed then open (the sign must flip)
hand/.hand/bin/python dex-retargetor-cpp/python/admittance_calibrate.py --watch
```

Tune one joint first, in `teleop/calibration/acc/admittance/admittance.yaml`: `K` (stiffness felt at the finger), then `M`, raise `D`
before `M` if a joint chatters, `tau_db_nm` from the calibration's `hysteresis_a × kt_nm_per_a`. Without a
calibration file the controller falls back to the idle current sampled at startup and `+1` signs, which is
enough to see motion but not to trust its direction. `admittance` cannot be combined with `compliance`,
`tracking` or `torque`. Telemetry adds `admittance_offset` (deg) and `external_torque_nm`.

## Impedance controller

```
tau = kp .* (q_target − q) − kd .* qdot,   clamped to ±tau_max
```

- Unit-agnostic. The loop tool uses **degrees** (SDK convention): `kp` in torque units/deg,
  `kd` in torque units/(deg/s), torque units = the D21 register scale (−1000..1000).
- Velocity: measured (`--measured-velocity`, firmware registers 16–31, unit undocumented) or a
  finite difference over `velocity_window_s` (default 10 ms; at 500 Hz this spans 5 samples,
  which keeps the 0.05° position quantisation from exploding into noise).
- Watchdog: if no target arrives for `target_timeout_s`, either hold the last target (default)
  or output zero torque (`--zero-on-timeout`).

## Running the inner loop

```bash
# Dry run: reads the hand at 500 Hz, computes torques, writes nothing.
./build/dexrt_impedance_loop --port /dev/serial/by-id/usb-Openlight_Labs_CANable2_...-if00 --rate 500

# Real: enters torque mode. Keep the hand clear and a hand on Ctrl+C.
./build/dexrt_impedance_loop --port ... --rate 500 --kp 2 --kd 0.02 --tau-max 150 --torque-mode --log /tmp/imp.csv
```

Targets are UDP datagrams of 16 little-endian float64 degrees to `127.0.0.1:6400`
(`dexrt.UdpTargetSender` in Python). Until the first target arrives the target is the pose
measured at start, so the spring exerts no force. On exit/error/Ctrl+C the tool writes zero
torque, copies the measured pose into the position targets, and restores position mode so
the hand does not jump when the firmware resumes position control.

Only one process may own the CANable adapter: stop `hand.hand_controller` first.

### Before trusting torque mode on hardware

The vendor SDK exposes torque mode but documents none of the following; measure each with
one finger, small torques (±50), and the hand clear of objects:

1. What one torque unit is, and the sign per joint.
2. Whether torque index 1–16 maps 1:1 to joint 0–15 (the API says "see the device manual").
3. What the firmware does when torque commands stop arriving.
4. Whether `set_control_mode(1)` resumes position control without a jump.

## Using the solver from Python

```python
import sys; sys.path.insert(0, 'dex-retargetor-cpp/python')
import dexrt
solver = dexrt.Retargeter.for_aac('hand/aac/description/D21_Left.urdf')
q_rad, errors_m, info = solver.solve(vectors_10x3)   # same input as DexAdapter.solve
```

Wiring this into `teleop/manus/stream/hand_stream_aac.py` means replacing `DexAdapter.solve` and
bumping the alignment `mapping_schema`, since the saved alignment records the solver version.
