# Dexterous Hand RS485 Control SDK

## Introduction
This SDK provides a Python interface for controlling the dexterous hand via **RS485** (Modbus RTU) or **CANFD** (Modbus over CANFD).  
It supports joint angle control, speed setting, force control, sensor reading, fan control, and configuration management.

## System Requirements
- Python 3.8+
- Required libraries:
  - `pyserial` (for RS485)
  - `python-can` >= 4.6.1 (for CANFD support)
  - Install with:
    ```bash
    pip install pyserial python-can>=4.6.1
    ```

## Quick Start
RS485 (Serial)
```python
from d21_rs485 import d21_rs485
hand = d21_rs485(hand_id=0x01, modbus_port="COM3")
angles = [30,0,30,40, 0,45,50, 0,45,50, 0,45,50, 0,45,50]
hand.write_joint_position_commands(angles)
hand.close()
```

CANFD (CANable slcan)
```python
from d21_canfd import d21_canfd
hand = d21_canfd(hand_id=0x01, modbus_port="COM3", tty_baudrate=10000000)
hand.connect()                      # must be called before any operation
angles = [30,0,30,40, 0,45,50, 0,45,50, 0,45,50, 0,45,50]
hand.write_joint_position_commands(angles)
hand.close()
```

## Main Features

Joint angle control (16 joints)

Speed adjustment (5 fingers independently)

Force control modes (position / torque)

Sensor reading (angles, current, tactile forces)

Fan control (manual speed levels)

Configuration management (ID, save/restore)

## Installation
```bash
pip install pyserial python-can>=4.6.1
```

Place all *.py files (d21_rs485.py, d21_canfd.py, D21ModbusCore.py, CANFDDevice.py) into your project directory.

## Example

See example.py for a complete demonstration covering both interfaces.

##Full API Documentation

For detailed interface descriptions, see API.md

## FAQ
Cannot open serial port: Check port number, permissions, and close other programs using the port.

No response from joint movement: Ensure control mode is set to 1 (position mode).

Read data all zeros: Verify the device ID matches.

CANFD not working: Ensure the CANable device is in slcan mode and the correct COM port is used.

## License
MIT