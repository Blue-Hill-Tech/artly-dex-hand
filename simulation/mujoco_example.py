import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('D21_Left.xml')
data = mujoco.MjData(model)

# ============================================================
# 1. Build joint name → index mapping (correct attribute: nq, not njq)
# ============================================================
joint_name_to_id = {model.joint(i).name: i for i in range(model.nq)}
actuator_name_to_id = {model.actuator(i).name: i for i in range(model.nu)}

# ============================================================
# 2. Define coupling rules: (master_joint_name, slave_joint_name, ratio)
#    slave_position = ratio * master_position
# ============================================================
coupling_rules = [
    ("thumb_joint4", "thumb_joint5", 0.85),
    ("index_joint3", "index_joint4", 0.8),
    ("middle_joint3", "middle_joint4", 0.8),
    ("ring_joint3", "ring_joint4", 0.8),
    ("pinky_joint3", "pinky_joint4", 0.8),
]

# Pre-store joint and actuator indices for efficiency
master_joint_ids = []
slave_actuator_ids = []
for master_name, slave_name, ratio in coupling_rules:
    if master_name not in joint_name_to_id or slave_name not in joint_name_to_id:
        print(f"Warning: joint {master_name} or {slave_name} not found, skipping")
        continue
    master_id = joint_name_to_id[master_name]
    slave_id = joint_name_to_id[slave_name]
    # Assume actuator name is joint_name + "_act"
    act_name = slave_name + "_act"
    if act_name not in actuator_name_to_id:
        print(f"Warning: actuator {act_name} not found, skipping")
        continue
    act_id = actuator_name_to_id[act_name]
    master_joint_ids.append(master_id)
    slave_actuator_ids.append(act_id)

# ============================================================
# 3. Coupling application function
# ============================================================
def apply_couplings():
    """Read master joint positions and set slave actuator targets."""
    for i in range(len(master_joint_ids)):
        master_id = master_joint_ids[i]
        act_id = slave_actuator_ids[i]
        ratio = coupling_rules[i][2]
        master_pos = data.qpos[master_id]
        target = ratio * master_pos
        data.ctrl[act_id] = target

# ============================================================
# 4. Initialize all actuators to 0 (optional)
# ============================================================
for i in range(model.nu):
    data.ctrl[i] = 0.0

# Apply couplings once to match slave joints to initial master positions
apply_couplings()

print("Coupling relationships activated:")
for master, slave, ratio in coupling_rules:
    print(f"  {slave} = {ratio} * {master}")
print("You can control the master joints by modifying data.ctrl[actuator_index].")
print("Slave joints will follow automatically.")

# ============================================================
# 5. Original viewer loop (unchanged except for applying couplings each step)
# ============================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Enable/disable contact point display (keeping your settings)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

    print("Simulation running. Press Ctrl+C to exit...")
    try:
        while viewer.is_running():
            # ---- Key: apply couplings every step ----
            apply_couplings()

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Exiting simulation")
