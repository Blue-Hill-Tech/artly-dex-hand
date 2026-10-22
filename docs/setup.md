# Engineering setup

Read [validation and known limitations](validation.md) before hardware bring-up.
Commands below install dependencies, inspect imports, build packages or run
simulation. Supplied hardware examples are retained for later review and are not
safe smoke tests. No physical-hand test was performed during the offline review.

## Python SDK

Supplied SDK requirement: Python 3.8+, `pyserial`, `python-can>=4.6.1`. Validated
offline with Python 3.12.3, pyserial 3.5 and python-can 4.6.1 on this Linux host.
That is an import check, not a supported-platform or firmware matrix.

Use **Python 3.12** for the documented environment. The SDK README's Python 3.8+
statement does not cover the reviewed dependency pins: python-can 4.6.1 requires
Python 3.9+, MuJoCo 3.16.0 requires 3.10+, and NumPy 2.5.4 requires 3.12+ according
to their installed package metadata. Python 3.8 grammar parsing alone does not
establish compatibility with those dependencies.

From the repository root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'pyserial==3.5' 'python-can==4.6.1'
cd sdk/python
python -B -c 'import D21ModbusCore, d21_rs485, CANFDDevice, d21_canfd'
```

The pins reproduce the offline review environment; CAN FD timing configuration
remains a known defect even with these versions. Quote requirement specifiers:
the original README's unquoted `>=` can be interpreted as shell redirection.
On Windows, use the virtual environment's Windows activation
command and Python executable. Windows hardware behavior was not tested here.

Keep the four modules together on the Python import path. The
[API reference](../sdk/python/API.md) covers angle, speed, torque, tactile, fan and
configuration methods. `d21_rs485(...)` opens a port during construction.
`d21_canfd(...)` opens its bus when `connect()` is called. Both need `close()`
after use. [example.py](../sdk/python/example.py) opens a device, changes fan and
speed settings, and commands motion immediately when executed or imported.

| Interface | Supplied defaults | Prerequisites still to verify |
| --- | --- | --- |
| RS485 | ID 1, 500000 baud, 8N1, 0.1 s timeout | Exact adapter driver, port, cable, hand firmware and bus termination |
| CAN FD | ID 1; CAN IDs 0x010 request / 0x020 reply; 1 Mbps arbitration, 2 Mbps data, 10 Mbps SLCAN serial | CANable firmware supporting the supplied SLCAN/CAN FD conventions |

The RS485 class docstring says 115200, while its constructor and protocol use
500000. The latter is the supplied default. Neither bus was connected during
review. Confirm the actual device ID and firmware before sending commands.

## ROS2 workspace

Documented target: Ubuntu 24.04 and ROS2 Jazzy. These instructions are derived from
the supplied package and have not been run in a Jazzy environment. A setuptools
package build passed; a full `colcon` build and launch remain open.

On a host with ROS2 Jazzy installed, from the cloned repository root:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/artly_d21_ws/src
cp -a sdk/ros2/hand_control ~/artly_d21_ws/src/
cd ~/artly_d21_ws
rosdep install --from-paths src --ignore-src -r -y
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install 'pyserial==3.5' 'python-can==4.6.1' colcon-common-extensions
colcon build --packages-select hand_control --symlink-install
source install/setup.bash
ros2 launch hand_control hand_control.launch.py use_simulation:=true hand_type:=right
```

Use the ROS2 host's matching Python interpreter and the `colcon` entry point from
the active virtual environment. Keep that environment active for the launch. If the package's console
entry point uses a different interpreter, install the required Python dependencies
in that interpreter's environment before launch. `python-can` is required even
for RS485/simulation imports because the node imports CANFD unconditionally, but
the supplied package does not declare that dependency. `rosdep` alone cannot fix it.

Use `hand_type:=left` to select the left model. The launch also starts RViz2 and
requires a graphical session. The source launch defaults to `use_simulation=false`
and initializes hardware with sixteen zero targets; always supply `true` for
offline work. A connection failure can also trigger fallback to simulation, so
inspect logs and mode before interpreting any running node as a hardware result.

`/joint_states` shows commanded targets, not measured hardware feedback, including
after failed writes. Physical readbacks are published on `/hand/joint/raw`.
Configured control/tactile/raw timer rates (50/50/20 Hz) are not measured throughput.
The package's gesture, angle, torque, fan and calibration topics can move or
reconfigure hardware and are not included as copy-and-run installation tests here.

## Standalone simulation

From the repository root, with an isolated Python environment active:

```bash
python -m pip install 'mujoco==3.16.0' 'pybullet==3.2.7' 'numpy==2.5.4' 'pyserial==3.5' 'python-can==4.6.1'
python -B tools/check_offline.py
```

This script imports driver modules without constructing transports, checks pure
protocol functions, and loads/steps both models with no GUI. It does not import
the hardware example scripts. The tested dependency versions are recorded in
[offline-validation.json](../provenance/offline-validation.json).

For later interactive simulation, run the original examples from `simulation/`:

```bash
cd simulation
python pybullet_example.py
# Or, in a separate run after closing the first example:
python mujoco_example.py
```

These GUI examples were not executed in this review. PyBullet defaults to the
right hand; change its `urdf_path` to choose the left. MuJoCo defaults to the left;
change the filename in `MjModel.from_xml_path` to choose the right. The original
README's `xml_path` instruction does not match that script. Modify a working copy
to keep the repository copy reproducible.

The scripts apply a 0.85 thumb coupling and 0.8 coupling for the other four
fingers. Do not assume loading the XML/URDF alone reproduces those script rules.
The models' inertias, collisions, contacts and force behavior remain unqualified.

## Electrical and mechanical preparation

The newer [electrical reference](reference/zh/electrical-interface.pdf) states
12 V rated, 240 W / 20 A maximum, with separate XT30UM power and GHD-10AB
communication connectors. Its ten-pin diagram identifies CANFD_H/L on pins 7/8
and RS485 A/B on pins 9/10. Pins 1–6 are reserved EtherCAT lines and currently
disabled; leave them unconnected. Connect only one bus type at a time and power
off before rewiring or plugging communication cables.

The older drawing uses a different connector designation and pin-8 label, with a
different drawing viewpoint. Match the exact hand/cable revision and orientation
to the released drawing before wiring. Do not derive a power return from signal
GND labels. Check termination, protection and supply capability on the actual
assembly. These descriptions are source references, not a validated wiring job.

Open the [STEP files](../cad) in a STEP-capable tool and compare the
[mechanical reference](reference/zh/mechanical-interface.pdf). The RealMan adapter
is an example; fit, fastener lengths/torques, cable clearance, cooling and mounting
loads require confirmation for the actual arm and hand. Do not fabricate from
mixed dimensions taken from different photographed drawings.

## Windows controller

The controller and its manuals are not distributed in this repository. Contact
Artly for an approved package and matching adapter instructions. The reviewed
manual describes Windows 10/11 and a portable executable; controller installation
and hardware operation have not been qualified here.
