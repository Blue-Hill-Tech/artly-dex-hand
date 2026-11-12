"""Torque register writes the D21 firmware actually acts on.

Bench results (2026-10-23): one FC10 write covering all 16 torque registers (68-83) is acknowledged
and stored, but the motors do not respond. Writes of up to 15 registers work, wherever they start
(including register 83 / motor 16), written once or repeated. So a full update is two frames:
motors 1-15, then motor 16. Neither frame splits a differential motor pair (1-2, 5-6, 8-9, 11-12,
14-15; spec section 3.5), so both motors of a pair always change together.
"""

MAX_REGISTERS_PER_WRITE = 15
# Contiguous motor groups (1-based), each at most MAX_REGISTERS_PER_WRITE long, pairs kept together.
MOTOR_GROUPS = [tuple(range(1, 16)), (16,)]


def write_torques(sdk, torques, motors=None):
    """Write torque values (16 entries, motor order) in firmware-accepted frames, in the SDK's register units:
    -1000..1000 on old firmware (v1), -20000..20000 on new firmware (v2, see D21ModbusCore.REGISTER_MAPS).

    motors: optional iterable of 1-based motor numbers; only groups containing one of them are
    written (the whole group is written, using the values from `torques`).
    """
    if len(torques) != 16:
        raise ValueError('need 16 torque values')
    wanted = None if motors is None else set(motors)
    for group in MOTOR_GROUPS:
        if wanted is not None and not wanted.intersection(group):
            continue
        values = []
        for m in group:
            values.append(sdk.torque_to_register(int(torques[m - 1])))
        sdk._write_multiple_holding(67 + group[0], values, f'torque motors {group[0]}-{group[-1]}')


def zero_torques(sdk):
    """Zero all 16 motors with firmware-accepted writes."""
    write_torques(sdk, [0] * 16)
