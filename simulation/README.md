# D21 Dexterous Hand – Quick Start

This folder contains URDF/XML models and simulation scripts for the D21 right/left hand with built‑in joint couplings.

## Files
- `D21_Right.urdf`, `D21_Left.urdf` – for PyBullet
- `D21_Right.xml`, `D21_Left.xml` – for MuJoCo
- `right_hand_meshes/`, `left_hand_meshes/` – STL files
- `pybullet_example.py` – PyBullet simulation
- `mujoco_example.py` – MuJoCo simulation

## Dependencies
- PyBullet: `pip install pybullet`
- MuJoCo: `pip install mujoco`

## Run
- PyBullet: `python pybullet_example.py` (edit `urdf_path` to choose left/right)
- MuJoCo: `python mujoco_example.py` (edit `xml_path` to choose left/right)

## Couplings (active by default)
| Master | Slave | Ratio |
|--------|-------|-------|
| thumb_joint4 | thumb_joint5 | 0.85 |
| index_joint3 | index_joint4 | 0.8 |
| middle_joint3 | middle_joint4 | 0.8 |
| ring_joint3 | ring_joint4 | 0.8 |
| pinky_joint3 | pinky_joint4 | 0.8 |

To modify, edit the `coupling_rules` list in the script.

## Control
The scripts do **not** move the hand automatically. You can set target positions for master joints (e.g., `thumb_joint4`). Slaves follow automatically.

Example (PyBullet):
```python
p.setJointMotorControl2(robot_id, joint_indices["thumb_joint4"], p.POSITION_CONTROL, targetPosition=0.5)
```