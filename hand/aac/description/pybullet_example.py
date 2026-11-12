import pybullet as p
import pybullet_data
import time
import os

# 1. Connect to the GUI server
physics_client = p.connect(p.GUI)
if not p.isConnected():
    print("Failed to connect to the physics server.")
    exit()

p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 2. Load the ground plane
plane_id = p.loadURDF("plane.urdf")

# 3. Load the URDF model
urdf_path = "D21_Right.urdf"   # Adjust path if needed
if not os.path.exists(urdf_path):
    print(f"Error: URDF file not found: {urdf_path}")
    p.disconnect()
    exit()

robot_id = p.loadURDF(urdf_path,
                      basePosition=[0, 0, 0.5],
                      useFixedBase=True)

if robot_id == -1:
    print("URDF model loading failed. Please check the file format or path.")
    p.disconnect()
    exit()

print(f"Model loaded successfully! Robot ID: {robot_id}")

# 4. Retrieve joint information and build a name-to-index mapping
num_joints = p.getNumJoints(robot_id)
print(f"Number of joints: {num_joints}")
joint_indices = {}
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    joint_indices[joint_name] = i
    print(f"  [{i}] {joint_name}")

# 5. Define coupling rules: (master_joint, slave_joint, ratio)
#    slave_position = ratio * master_position
coupling_rules = [
    ("thumb_joint4", "thumb_joint5", 0.85),
    ("index_joint3", "index_joint4", 0.8),
    ("middle_joint3", "middle_joint4", 0.8),
    ("ring_joint3", "ring_joint4", 0.8),
    ("pinky_joint3", "pinky_joint4", 0.8),
]

# 6. Function to apply couplings: read master position, set slave target
def apply_couplings():
    for master_name, slave_name, ratio in coupling_rules:
        if master_name not in joint_indices or slave_name not in joint_indices:
            continue
        master_idx = joint_indices[master_name]
        slave_idx = joint_indices[slave_name]
        master_state = p.getJointState(robot_id, master_idx)
        master_pos = master_state[0]
        target_pos = ratio * master_pos
        p.setJointMotorControl2(robot_id, slave_idx, p.POSITION_CONTROL,
                                targetPosition=target_pos,
                                force=1.0)

# 7. Set gravity
p.setGravity(0, 0, -9.81)

# 8. Adjust camera to a closer, frontal view of the hand
#    (fine‑tune these parameters to your liking)
p.resetDebugVisualizerCamera(
    cameraDistance=0.6,          
    cameraYaw=0,                 
    cameraPitch=-20,             
    cameraTargetPosition=[0, 0, 0.4]   
)


print("Coupling relationships activated:")
for master, slave, ratio in coupling_rules:
    print(f"  {slave} = {ratio} * {master}")
print("\nYou can now control the master joints by setting their target positions.")
print("For example, to bend the thumb, set:")
print("  p.setJointMotorControl2(robot_id, joint_indices['thumb_joint4'], p.POSITION_CONTROL, targetPosition=0.5)")
print("The slave joints will follow automatically.")
print("Simulation running. Close the GUI window to exit.")

dt = 1.0 / 240.0

# 9. Main simulation loop – just step and apply couplings
while True:
    if not p.isConnected():
        print("Connection lost. Exiting.")
        break

    # Apply couplings every step (this keeps slaves locked to masters)
    apply_couplings()

    # Step the simulation
    p.stepSimulation()
    time.sleep(dt)

# Clean up on exit
if p.isConnected():
    p.disconnect()
