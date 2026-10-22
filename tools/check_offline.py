#!/usr/bin/env python3
"""Offline checks only: no hardware transport construction or vendor examples."""
from datetime import datetime, timezone
from importlib import import_module, metadata
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import math
import struct
import sys
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'sdk/python'), str(ROOT / 'sdk/ros2/hand_control')]
report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
          'python': sys.version, 'packages': {}, 'checks': [],
          'known_supplier_defects': [],
          'scope': 'No hardware connections, controller binaries, GUI examples, or ROS2 node launch.'}


def check(name, operation):
    try:
        details = operation()
        report['checks'].append({'name': name, 'status': 'pass', 'details': details})
    except Exception as exc:
        report['checks'].append({'name': name, 'status': 'fail',
                                 'error': f'{type(exc).__name__}: {exc}'})


def core_checks(module):
    core = import_module(module).D21ModbusCore()
    assert core._crc16(bytes.fromhex('01030000000A')) == 0xCDC5
    for index, (low, high) in core._ANGLE_LIMITS.items():
        assert core._angle_to_register(index, low) == 0
        assert core._angle_to_register(index, high) == 2000
        assert core._register_to_angle(index, 0) == low
        assert core._register_to_angle(index, 2000) == high
        try:
            core._angle_to_register(index, high + 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Joint {index} accepted angle above range')

    def frame(body):
        return body + struct.pack('<H', core._crc16(body))

    response = frame(bytes.fromhex('0103021234'))
    assert core._parse_response(response, 3) == bytes.fromhex('1234')
    try:
        core._parse_response(response[:-1] + bytes([response[-1] ^ 1]), 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Corrupt response CRC accepted')
    wrong_function_accepted = False
    try:
        core._parse_write_response(frame(bytes.fromhex('010300000001')), 6)
        wrong_function_accepted = True
    except RuntimeError:
        pass
    try:
        core._parse_write_response(frame(bytes.fromhex('018602')), 6)
        exception_result = 'accepted without error'
    except RuntimeError as exc:
        exception_result = str(exc)
    report['known_supplier_defects'].append({
        'module': module,
        'write_response_accepts_wrong_function': wrong_function_accepted,
        'five_byte_write_exception_result': exception_result,
        'qualification': 'Recorded defects; passing pure checks do not qualify the transport.'})
    return 'CRC vector, 16 joint endpoint mappings/bounds, valid read and corrupt-CRC rejection'


def canfd_configuration_probe():
    from can.interfaces.slcan import slcanBus
    serial = MagicMock()
    # Substitute the serial factory before constructing the backend: no port opens.
    with patch('can.interfaces.slcan.serial.serial_for_url', return_value=serial), \
            patch('can.interfaces.slcan.time.sleep'), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bus = slcanBus(channel='offline-mock-only', ttyBaudrate=10000000,
                       bitrate=1000000, fd=True, data_bitrate=2000000, brs=True)
        try:
            commands = [call.args[0].decode('ascii') for call in serial.write.call_args_list]
        finally:
            bus.shutdown()
    report['known_supplier_defects'].append({
        'module': 'CANFDDevice / python-can SLCAN',
        'mock_configuration_commands': commands,
        'requested_2mbps_data_rate_not_configured': 'Y2\r' not in commands,
        'warnings': [str(w.message) for w in caught],
        'qualification': 'Mocks only; verify adapter timing/firmware before hardware use.'})
    return 'Serial factory mocked; vendor constructor arguments inspected without hardware'


def mujoco_model(name):
    import mujoco
    import numpy as np
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'simulation' / name))
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    warning_counts = [int(w.number) for w in data.warning]
    assert not any(warning_counts), warning_counts
    return {'nq': model.nq, 'nv': model.nv, 'njnt': model.njnt, 'nu': model.nu,
            'steps': 100, 'finite_state': True, 'warnings': warning_counts}


def pybullet_model(name):
    import pybullet as p
    client = p.connect(p.DIRECT)
    assert client >= 0
    try:
        robot = p.loadURDF(str(ROOT / 'simulation' / name),
                           useFixedBase=True, physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        for _ in range(100):
            p.stepSimulation(physicsClientId=client)
        count = p.getNumJoints(robot, physicsClientId=client)
        states = p.getJointStates(robot, range(count), physicsClientId=client)
        assert all(math.isfinite(s[0]) and math.isfinite(s[1]) for s in states)
        return {'joints': count, 'steps': 100, 'finite_state': True}
    finally:
        p.disconnect(client)


for package in ('pyserial', 'python-can', 'mujoco', 'pybullet', 'numpy'):
    try:
        report['packages'][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        report['packages'][package] = 'not installed'
for prefix in ('', 'hand_control.'):
    for module in ('D21ModbusCore', 'd21_rs485', 'CANFDDevice', 'd21_canfd'):
        name = prefix + module
        check('import ' + name, lambda name=name: (import_module(name), 'no constructors called')[1])
    check('pure core ' + prefix, lambda prefix=prefix: core_checks(prefix + 'D21ModbusCore'))
check('mock SLCAN configuration', canfd_configuration_probe)
for side in ('Left', 'Right'):
    check('MuJoCo ' + side, lambda side=side: mujoco_model(f'D21_{side}.xml'))
    check('PyBullet ' + side, lambda side=side: pybullet_model(f'D21_{side}.urdf'))
failed = any(item['status'] == 'fail' for item in report['checks'])
report['status'] = 'failed' if failed else 'offline checks passed; known supplier defects recorded separately'
report['not_validated'] = ['physical D21 hand', 'RS485/CAN transport', 'ROS2 Jazzy build/launch',
                         'GUI examples', 'script couplings', 'simulation fidelity or contact performance']
output = ROOT / 'provenance/offline-validation.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
sys.exit(1 if failed else 0)
