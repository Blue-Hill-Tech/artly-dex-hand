#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dexterous Hand Control Example - Supports both RS485 and CANFD interfaces.
Adjust the interface selection and port parameters according to your setup.
"""

import time

# ===== Choose interface: 'rs485' or 'canfd' =====
INTERFACE = 'rs485'     #change to 'canfd' for CANFD
# =================================================

if INTERFACE == 'rs485':
    from d21_rs485 import d21_rs485
    # RS485 parameters
    hand = d21_rs485(hand_id=0x01, modbus_port="COM8", baudrate=500000, timeout=0.1)
    # No explicit connect needed; serial is opened in __init__

elif INTERFACE == 'canfd':
    from d21_canfd import d21_canfd
    # CANFD parameters (using slcan)
    hand = d21_canfd(hand_id=0x01, modbus_port="COM14", timeout=0.1, tty_baudrate=10000000)
    hand.connect()   # Must call connect() before any operation
else:
    raise ValueError("INTERFACE must be 'rs485' or 'canfd'")

print(f"Connected via {INTERFACE.upper()}")

# ---------- 1. Set fan mode ----------
print("Setting fan to maximum speed...")
hand.set_fan_mode(5)   # 0=stop, 5=max

# ---------- 2. Set speed (percentage) for all fingers ----------
speeds = [80, 80, 80, 80, 80]  # thumb, index, middle, ring, pinky
hand.write_joint_speed_commands(speeds)
print("Speed set to 80% for all fingers.")

time.sleep(0.01)

# ---------- 3. Set all joint positions (unit: degrees) ----------
# Joint indices: 0~3 Thumb, 4~6 Index, 7~9 Middle, 10~12 Ring, 13~15 Pinky
angles = [
    30,    # Thumb abduction (0~100)
    0,     # Thumb rotation (-35~35)
    15,    # Thumb CMC (0~50)
    25,    # Thumb MCP (0~70)
    0,     # Index abduction (-20~20)
    45,    # Index MCP (0~90)
    50,    # Index PIP (0~100)
    0,     # Middle abduction (-20~20)
    45,    # Middle MCP (0~90)
    50,    # Middle PIP (0~100)
    0,     # Ring abduction (-20~20)
    45,    # Ring MCP (0~90)
    50,    # Ring PIP (0~100)
    0,     # Pinky abduction (-20~20)
    45,    # Pinky MCP (0~90)
    50     # Pinky PIP (0~100)
]
hand.write_joint_position_commands(angles)
print("Angle commands sent to all joints.")

time.sleep(0.3)

# ---------- 4. Write a single joint (thumb abduction) ----------
hand.write_single_joint_position(0, 20)
print("Thumb abduction set to 20 degrees.")

time.sleep(0.5)

# ---------- 5. Read current joint angles ----------
positions = hand.read_joint_positions()
print("\nCurrent joint angles (degrees):")
for i, pos in enumerate(positions):
    print(f"  J{i}: {pos:.2f}°")

# ---------- 6. Read tactile forces (3-axis forces) ----------
forces = hand.read_tactile_forces()
print("\nTactile forces (N):")
finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
for f in range(5):
    idx = f * 3
    print(f"  {finger_names[f]}: X={forces[idx]:.2f}, Y={forces[idx+1]:.2f}, Z={forces[idx+2]:.2f}")

# ---------- 7. Disconnect ----------
hand.close()
print("Connection closed.")
