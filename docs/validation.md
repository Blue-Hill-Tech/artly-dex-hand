# Validation and known limitations

Review date: October 21, 2026 (Pacific). All validation was offline. No serial or
CAN connection, controller executable, hardware example, calibration, firmware
write or physical motion was performed. This repository is an engineering source
handoff, not physical or customer-release acceptance.

## Public package and offline review

The public package contains SDK source, CAD, simulation assets, three interface
PDFs, two register workbooks, four technical reference images and a generated
model render. The controller, its manuals and original delivery archives are
not distributed here. The [asset manifest](../provenance/asset-manifest.json)
records the published asset paths, byte sizes and SHA-256 hashes.

The following results come from the offline engineering review. They describe
specific checks and their limits; they do not establish physical performance.

| Check | Result and boundary |
| --- | --- |
| Reference documents | Electrical and mechanical drawings and register values reviewed; conflicting revisions documented. |
| Register workbooks | Both XLSX archives readable and CRC-valid; four sheets each, zero formulas. |
| Reference images | The four published reference images decoded and visually reviewed. |
| Python | All 18 supplied Python source files parse; five standalone SDK files also parse using Python 3.8 grammar. Python 3.8 execution not tested. |
| Driver imports | Eight standalone/ROS2 driver modules imported without transport construction on Python 3.12.3. |
| Protocol functions | Known CRC vector, angle boundaries, valid read parsing and corrupted-CRC rejection checked in both core variants. Write-response defects reproduced separately. |
| CAN FD dependency | Mocked SLCAN constructor probe confirmed a data-bitrate configuration issue with python-can 4.6.1; no serial device opened. |
| Model files | Four URDFs and two MuJoCo XMLs parse; mesh references resolve. |
| Headless simulation | Both standalone hands loaded and stepped 100 times in MuJoCo and PyBullet DIRECT; states finite; MuJoCo warning counts zero. No physical fidelity claim. |
| ROS2 packaging | Setuptools package build passed. ROS2 Jazzy/rclpy unavailable during review; full colcon build, launch, RViz and ROS messaging not tested. |

Dependency versions for that run: Python 3.12.3, pyserial 3.5, python-can 4.6.1,
MuJoCo 3.16.0, PyBullet 3.2.7 and NumPy 2.5.4. These are observed test versions,
not a certified compatibility matrix. Results are recorded in
[offline-validation.json](../provenance/offline-validation.json).

To verify published asset bytes, run `python3 tools/verify_assets.py` from the
repository root. For offline import, protocol and model checks, follow the
[setup guide](setup.md). Known implementation defects are reported distinctly
from unexpected test failures; a successful script exit does not qualify those
defects.

## Confirmed implementation limitations

The hardware-control behavior retains the limitations below.
[The detailed SDK review](sdk-review.md) gives code references and reproduction
details.

1. ROS2 defaults to hardware mode and writes sixteen zero joint targets at startup.
   Explicitly select `use_simulation:=true` for offline use.
2. ROS2 `/joint_states` publishes commanded targets, including after failed writes.
   Hardware connection errors can fall back to simulation. A running node or
   moving visualization is not proof of successful physical operation.
3. Both Modbus cores accept a CRC-valid write acknowledgement with the wrong
   normal function code and do not validate echoed address/value/count. Five-byte
   exception replies are rejected as incomplete before exception decoding.
   The RS485 transport also assumes response lengths that mishandle exceptions.
4. With python-can 4.6.1, the supplied SLCAN construction does not configure
   the requested CAN FD data bitrate: `data_bitrate` is ignored by that backend
   without a `BitTimingFd` timing object. The mocked serial probe emitted no
   2 Mbps data-rate command. `ttyBaudrate` is also deprecated by that dependency.
   Adapter behavior must be resolved before CAN FD hardware qualification.
5. ROS2 imports python-can unconditionally but does not declare it as a dependency.
   Install it in the ROS2 interpreter even for RS485/simulation. A broad `>=4.6.1`
   requirement does not demonstrate compatibility with future versions.

## Document conflicts and missing acceptance evidence

[The product review](product-review.md) records six concrete documentation issues:
thumb-rotation range/formula disagreement; velocity register-table addresses;
tactile XY raw-128 sign/range; connector designation, pin-8 label and diagram
orientation changes; a three-versus-four-register protocol example; and hybrid
control terminology versus the provided position/torque modes. Confirm these
against exact firmware and released drawings before hardware acceptance.

Still open: shipped hand/firmware/adapter identification; physical wiring and
mounting fit; safe bring-up and recovery; functional bus tests; actual feedback;
tactile calibration; forces, precision, cycle time, thermal limits and endurance;
simulator dynamics; full ROS2 runtime; Windows GUI installation; firmware update
and recovery procedures; supported platform/firmware matrix; complete license and
redistribution terms; environment/ingress/food-service claims; measured demos and
videos. Existing SDK presence closes an asset-availability gap only.

No separate firmware release or hardware test report was identified in the
reviewed engineering package. Do not infer certification or production readiness
from file availability or simulation results.
