
---

```markdown
# Dexterous Hand SDK API Reference

## Base Class `D21ModbusCore` (Shared by RS485 and CANFD)
All methods listed below are available in both `d21_rs485` and `d21_canfd` classes.  
The only differences are in initialization and connection management (see each class).

---

## Class `d21_rs485` (RS485 Serial)

### Initialization
- `__init__(hand_id=0x01, modbus_port="COM3", baudrate=500000, timeout=0.1)`
  - `hand_id`: device address (default 0x01)
  - `modbus_port`: serial port name (e.g., "COM3" or "/dev/ttyUSB0")
  - `baudrate`: communication baud rate (default 500000)
  - `timeout`: serial read timeout in seconds (default 0.1)

The serial connection is opened automatically upon initialization. Use `close()` to release resources.

---

## Class `d21_canfd` (CANFD via slcan)

### Initialization
- `__init__(hand_id=0x01, modbus_port="COM3", timeout=0.1, tty_baudrate=10000000)`
  - `hand_id`: device address (default 0x01)
  - `modbus_port`: serial port name for CANable device (e.g., "COM3")
  - `timeout`: response timeout in seconds (default 0.1)
  - `tty_baudrate`: baud rate of the CANable USB‑to‑serial adapter (default 10,000,000 bps)

**Important**: After initialization, you **must** call `connect()` to open the CAN bus and start the receiver thread.  
The CANFD interface uses the following fixed CAN IDs:
- Request: `0x010` (PC → Hand)
- Response: `0x020` (Hand → PC)

---

### Connection Management (Both Classes)
- `close()` – Closes the connection and releases resources (serial port / CAN bus).  
  For CANFD, also stops the internal receiver thread.

- `connect()` – **Only available for `d21_canfd`**. Opens the CAN bus and starts the background receiver thread.  
  Must be called before any read/write operation.

---

## Joint Angle Control (Write Operations)
All angles are in **degrees**. Values outside the joint limits (see Appendix) raise `ValueError`.

- `write_single_joint_position(joint_index, angle_deg)`
  - Writes target angle for a single joint.
  - `joint_index`: 0–15 (see joint index table)
  - `angle_deg`: target angle in degrees

- `write_joint_position_commands(angles_deg)`
  - Writes target angles for all 16 joints.
  - `angles_deg`: list of 16 floats, in order of joint indices 0–15

- `write_thumb_position_commands(angles_deg)`
  - Writes the 4 thumb joint angles (order: abduction, rotation, CMC, MCP).
  - `angles_deg`: list of 4 floats

- `write_index_position_commands(angles_deg)`
  - Writes the 3 index finger joint angles (order: abduction, MCP, PIP).
  - `angles_deg`: list of 3 floats

- `write_middle_position_commands(angles_deg)`
  - Writes the 3 middle finger joint angles (abduction, MCP, PIP).

- `write_ring_position_commands(angles_deg)`
  - Writes the 3 ring finger joint angles (abduction, MCP, PIP).

- `write_pinky_position_commands(angles_deg)`
  - Writes the 3 pinky finger joint angles (abduction, MCP, PIP).

---

## Speed Control (Write Operations)
Speed values are given as percentages (0–100%).

- `write_joint_speed_commands(speeds_percent)`
  - Writes speed percentages for all 5 fingers.
  - `speeds_percent`: list of 5 ints/floats, order: thumb, index, middle, ring, pinky

- `write_single_finger_speed(finger_index, speed_percent)`
  - Writes speed percentage for a single finger.
  - `finger_index`: 1–5 (1=thumb, 2=index, …, 5=pinky)
  - `speed_percent`: 0–100

---

## Force/Torque Control Mode

- `set_control_mode(mode)`
  - Sets the control mode.
  - `mode`: `1` = position mode (default), `2` = torque mode

- `set_single_torque(joint_index, torque_reg)`
  - Sets torque command for one joint (only effective in torque mode).
  - `joint_index`: 1–16 (motor index, see motor‑to‑joint mapping in device manual)
  - `torque_reg`: -1000 to 1000 (maps to internal register value 0–2000)

- `set_all_torques(torques_reg)`
  - Sets torque commands for all 16 joints at once.
  - `torques_reg`: list of 16 ints, each in [-1000, 1000]

---

## Calibration and Configuration

- `calibrate_sensor(finger_index)`
  - Starts zero‑calibration for the tactile sensor of the specified finger.
  - `finger_index`: 1–5
  - The calibration completes automatically and the register resets to 0.

- `set_device_id(new_id)`
  - Changes the device ID (1–254).
  - `new_id`: new address (persistent only if `save_config()` is called)

- `set_fan_mode(mode)`
  - Sets the fan speed level.
  - `mode`: 0–5 (0=stop, 5=maximum speed)

- `save_config()`
  - Saves all current register parameters to Flash (persistent across power cycles).

- `set_hand_enable(enable)`
  - Enables or disables the whole hand.
  - `enable`: `0` = disable (all joints stop and hold position), `1` = enable (normal operation, default)

---

## Read Sensor Data (Input Registers)
All read functions return values converted to physical units.

- `read_joint_positions()`
  - Returns a list of 16 current joint angles (degrees).

- `read_joint_velocities()`
  - Returns a list of 16 joint velocities (deg/s)

- `read_motor_currents()`
  - Returns a list of 16 motor currents in amperes (A). 

- `read_hardware_version()`
  - Returns the hardware version number.

- `read_software_version()`
  - Returns the software version number.

---

## Read Tactile Force (Resultant 3‑Axis Forces)
Each finger provides three force components (X, Y, Z) in Newtons (N).

- `read_tactile_forces()`
  - Returns a flat list of 15 values: thumb X,Y,Z, index X,Y,Z, …, pinky X,Y,Z.

- `read_thumb_forces()` – returns [X, Y, Z] for thumb
- `read_index_forces()` – returns [X, Y, Z] for index
- `read_middle_forces()` – returns [X, Y, Z] for middle
- `read_ring_forces()` – returns [X, Y, Z] for ring
- `read_pinky_forces()` – returns [X, Y, Z] for pinky

---

## Read Tactile Pressure Distribution
Each finger has an array of sensing points: thumb has 116 points, the other four have 52 points each.  
The following methods return lists of force values (in Newtons) for the respective axis or magnitude.

### Thumb (116 points)
- `read_thumb_x_pressures()` – X‑axis forces
- `read_thumb_y_pressures()` – Y‑axis forces
- `read_thumb_z_pressures()` – Z‑axis forces
- `read_thumb_norm_pressures()` – magnitude sqrt(X²+Y²+Z²)

### Index (52 points)
- `read_index_x_pressures()`
- `read_index_y_pressures()`
- `read_index_z_pressures()`
- `read_index_norm_pressures()`

### Middle (52 points)
- `read_middle_x_pressures()`
- `read_middle_y_pressures()`
- `read_middle_z_pressures()`
- `read_middle_norm_pressures()`

### Ring (52 points)
- `read_ring_x_pressures()`
- `read_ring_y_pressures()`
- `read_ring_z_pressures()`
- `read_ring_norm_pressures()`

### Pinky (52 points)
- `read_pinky_x_pressures()`
- `read_pinky_y_pressures()`
- `read_pinky_z_pressures()`
- `read_pinky_norm_pressures()`

---

## Read Configuration Parameters (Holding Registers)
These functions read back the current settings written to the holding registers.

- `read_commanded_joint_positions()`
  - Returns a list of 16 target joint angles (degrees) that were last sent.

- `read_single_commanded_joint_position(joint_index)`
  - Returns the target angle for a single joint.
  - `joint_index`: 0–15

- `read_finger_commanded_positions(finger_index)`
  - Returns the target angles for all joints of a specific finger.
  - `finger_index`: 1–5

- `read_single_finger_speed(finger_index)`
  - Returns the speed percentage (0–100%) for a single finger.
  - `finger_index`: 1–5

- `read_all_finger_speeds()`
  - Returns a list of 5 speed percentages (thumb to pinky).

- `read_control_mode()`
  - Returns the current control mode: `1` = position, `2` = torque.

- `read_single_torque(joint_index)`
  - Returns the torque register value (in the range [-1000, 1000]) for a single joint.
  - `joint_index`: 1–16

- `read_all_torques()`
  - Returns a list of 16 torque values (each in [-1000, 1000]).

- `read_device_id()`
  - Returns the current device ID.

- `read_fan_mode()`
  - Returns the current fan speed level (0–5).

- `read_hand_enable()`
  - Returns the hand enable status: `0` = disabled, `1` = enabled.

---

## Appendix: Joint Index and Angle Limits

| Index | Joint Name          | Min (°) | Max (°) |
|-------|---------------------|---------|---------|
| 0     | Thumb abduction     | 0       | 100     |
| 1     | Thumb rotation      | -35     | 35      |
| 2     | Thumb CMC           | 0       | 50      |
| 3     | Thumb MCP           | 0       | 70      |
| 4     | Index abduction     | -20     | 20      |
| 5     | Index MCP           | 0       | 90      |
| 6     | Index PIP           | 0       | 100     |
| 7     | Middle abduction    | -20     | 20      |
| 8     | Middle MCP          | 0       | 90      |
| 9     | Middle PIP          | 0       | 100     |
| 10    | Ring abduction      | -20     | 20      |
| 11    | Ring MCP            | 0       | 90      |
| 12    | Ring PIP            | 0       | 100     |
| 13    | Pinky abduction     | -20     | 20      |
| 14    | Pinky MCP           | 0       | 90      |
| 15    | Pinky PIP           | 0       | 100     |

---

## Important Notes

1. All angle values must be within the specified limits; out‑of‑range values raise `ValueError`.
2. Speed percentages must be between 0 and 100.
3. **Control mode**: `1` = position, `2` = torque.
4. Torque register values range from 0 to 2000 internally, but the API accepts values in [-1000, 1000] for convenience.
5. `set_hand_enable(0)` immediately stops all joint motion; the hand will not respond to new commands until re‑enabled with `set_hand_enable(1)`.
6. `save_config()` writes all current settings to Flash; they persist after power‑off.
7. Tactile sensors: thumb has 116 sensing points; each of the other fingers has 52 points.
8. For CANFD, always call `connect()` after creating the `d21_canfd` object and before any other operation.