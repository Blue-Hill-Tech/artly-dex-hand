================================================================================
D21 Hand – ROS2 Control (Ubuntu 24.04 + Jazzy)
================================================================================

Guide for controlling the D21 dexterous hand (left/right) with ROS2 Jazzy.

Supports RS485 (default) and CANFD interfaces, with simulation mode for testing.


================================================================================
1. Folder Structure
================================================================================

Place the hand_control package inside your ROS2 workspace:

    ros2_hand_ws/
    └── src/
        └── hand_control/
            ├── hand_control/
            │   ├── __init__.py
            │   ├── hand_controller_node.py
            │   ├── d21_rs485.py
            │   ├── d21_canfd.py
            │   ├── CANFDDevice.py
            │   └── D21ModbusCore.py
            ├── urdf/
            │   ├── right_hand.urdf
            │   └── left_hand.urdf
            ├── launch/
            │   └── hand_control.launch.py
            ├── rviz/
            │   └── hand.rviz
            ├── right_hand_meshes/          # STL mesh files for right hand
            │   └── *.stl
            ├── left_hand_meshes/           # STL mesh files for left hand
            │   └── *.stl
            ├── setup.py
            └── package.xml

Note: The mesh folders are used by the URDF files for visualization in RViz2.
Make sure the URDF references the correct mesh paths (relative to the package).


================================================================================
2. Install Dependencies (ROS2 Jazzy)
================================================================================

    sudo apt install ros-jazzy-xacro ros-jazzy-joint-state-publisher ros-jazzy-robot-state-publisher

If you haven't sourced ROS2 yet:

    source /opt/ros/jazzy/setup.bash

For CANFD interface, install the Python CAN library:

    pip install python-can>=4.6.1


================================================================================
3. Build
================================================================================

    cd ~/ros2_hand_ws
    colcon build --packages-select hand_control --symlink-install
    source install/setup.bash


================================================================================
4. Run
================================================================================

4.1 Using the Launch File (Recommended)
----------------------------------------

Right hand (default) with RS485:
    ros2 launch hand_control hand_control.launch.py hand_type:=right

Left hand with RS485:
    ros2 launch hand_control hand_control.launch.py hand_type:=left

CANFD interface:
    ros2 launch hand_control hand_control.launch.py interface:=canfd port:=/dev/ttyACM0


4.2 Manual Startup (if launch file not available)
--------------------------------------------------

Terminal 1 – Robot State Publisher:
    ros2 run robot_state_publisher robot_state_publisher $(ros2 pkg prefix hand_control)/share/hand_control/urdf/right_hand.urdf

Terminal 2 – Hand Controller (RS485):
    ros2 run hand_control hand_controller --ros-args -p port:=/dev/ttyACM0

Terminal 2 – Hand Controller (CANFD):
    ros2 run hand_control hand_controller --ros-args -p interface:=canfd -p port:=/dev/ttyACM0 -p tty_baudrate:=10000000

Terminal 3 – RViz2:
    ros2 run rviz2 rviz2 -d $(ros2 pkg prefix hand_control)/share/hand_control/rviz/hand.rviz


4.3 RViz2 Configuration
------------------------

- Set Fixed Frame to "base_link"
- Add RobotModel display
- Set Description Topic to "/robot_description"


================================================================================
5. Control Topics
================================================================================

5.1 Position Control (16 joints, degrees)
------------------------------------------

Send joint angles (16 values, degrees):
    ros2 topic pub /hand/command/joint_angles std_msgs/Float32MultiArray "data: [30,10,20,30, 0,60,60, 0,60,60, 0,60,60, 0,60,60]" --once


5.2 Preset Actions
-------------------

Fist:
    ros2 topic pub /hand/command/action std_msgs/String "data: 'fist'" --once

Open hand:
    ros2 topic pub /hand/command/action std_msgs/String "data: 'open'" --once

Pinch:
    ros2 topic pub /hand/command/action std_msgs/String "data: 'pinch'" --once


5.3 Fan Mode Control
---------------------

Set fan speed (0=stop, 1-5=speeds):
    ros2 topic pub /hand/set_fan_mode std_msgs/Float32MultiArray "data: [5]" --once


5.4 Control Mode Switching
---------------------------

Switch to torque mode:
    ros2 topic pub /hand/set_control_mode std_msgs/Float32MultiArray "data: [2]" --once

Switch back to position mode:
    ros2 topic pub /hand/set_control_mode std_msgs/Float32MultiArray "data: [1]" --once


5.5 Torque Parameters
----------------------

Set torque values for all 16 motors (range: -1000 ~ 1000):

Set all motors to 500:
    ros2 topic pub /hand/set_torques std_msgs/Float32MultiArray "data: [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]" --once

Set individual values per motor:
    ros2 topic pub /hand/set_torques std_msgs/Float32MultiArray "data: [100, -200, 300, -400, 0, 500, -600, 700, -800, 900, -1000, 0, 200, -300, 400, -500]" --once

NOTE: Torque parameters only take effect when the hand is in torque mode (2).


5.6 Sensor Calibration
-----------------------

Calibrate tactile sensors for a specific finger (1=thumb, 2=index, 3=middle, 4=ring, 5=pinky):

Calibrate thumb:
    ros2 topic pub /hand/calibrate_sensor std_msgs/Float32MultiArray "data: [1]" --once

Calibrate index finger:
    ros2 topic pub /hand/calibrate_sensor std_msgs/Float32MultiArray "data: [2]" --once

NOTE: During calibration, ensure the finger is in a load-free (unloaded) state.


================================================================================
6. Parameters
================================================================================

Override at launch, e.g.:
    ros2 run hand_control hand_controller --ros-args -p port:=/dev/ttyACM1 -p use_simulation:=true -p interface:=canfd

Parameter           Default         Description
----------------------------------------------------------------------------------------------------
port                /dev/ttyACM0    Serial port (RS485 or CANable device)
baudrate            500000          RS485 baud rate (RS485 only)
hand_id             1               Modbus slave ID (1-254)
interface           rs485           "rs485" or "canfd"
tty_baudrate        10000000        TTY baud rate for CANable (CANFD only)
use_simulation      false           Enable simulation mode (no hardware)
coupling_ratio      0.8             Coupling ratio for URDF joints
control_rate        50              Control + joint state publish rate (Hz)
tactile_rate        50              Tactile data publish rate (Hz)
raw_rate            20              Raw joint data publish rate (Hz)


================================================================================
7. Published Topics
================================================================================

Topic                       Type                               Rate    Description
----------------------------------------------------------------------------------------------------
/joint_states               sensor_msgs/JointState             50 Hz   Joint states for RViz2
/hand/tactile/forces        std_msgs/Float32MultiArray         50 Hz   Triaxial forces (X/Y/Z) for all 5 fingers
/hand/tactile/pressure      std_msgs/Float32MultiArray         50 Hz   Normal pressure distribution
/hand/joint/raw             std_msgs/Float32MultiArray         20 Hz   Raw sensor data: positions + velocities + currents

Monitor sensor data:
    ros2 topic echo /hand/tactile/forces
    ros2 topic echo /hand/tactile/pressure
    ros2 topic echo /hand/joint/raw


================================================================================
8. Troubleshooting
================================================================================

8.1 Model Not Visible in RViz2
-------------------------------
- Check robot_state_publisher is running
- Verify /robot_description topic exists: ros2 topic list | grep description
- Check Fixed Frame is set to "base_link"
- Verify Description Topic in RobotModel is "/robot_description"


8.2 Serial Port Permission Error
---------------------------------
    sudo chmod 666 /dev/ttyACM0


8.3 CANFD Connection Failure
-----------------------------
- Check tty baud rate: Try common values like 115200, 921600, or 10000000
- Check CANable firmware: Ensure it's in SLCAN mode
- Verify port: Use ls /dev/ttyUSB* or /dev/ttyACM* to confirm correct port

Example with different baud rate:
    ros2 run hand_control hand_controller --ros-args -p interface:=canfd -p port:=/dev/ttyACM0 -p tty_baudrate:=115200


8.4 Module Not Found: D21ModbusCore
------------------------------------
- Ensure all .py files are in the hand_control/ directory
- Check __init__.py exists in hand_control/
- Rebuild with: colcon build --packages-select hand_control --symlink-install


8.5 Hardware Write Failed
--------------------------
- Verify hand is powered on
- Check correct port and hand_id
- Try simulation mode: -p use_simulation:=true


8.6 Joint Mismatch
-------------------
Adjust the joint_map inside hand_controller_node.py to match your URDF joint names and coupling configuration.


================================================================================
9. Command Reference (Quick Summary)
================================================================================

Action                                              Command
----------------------------------------------------------------------------------------------------
Launch RS485                                        ros2 launch hand_control hand_control.launch.py
Launch CANFD                                        ros2 launch hand_control hand_control.launch.py interface:=canfd port:=/dev/ttyACM0
Switch to torque mode                               ros2 topic pub /hand/set_control_mode std_msgs/Float32MultiArray "data: [2]" --once
Set torques (all 500)                               ros2 topic pub /hand/set_torques std_msgs/Float32MultiArray "data: [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]" --once
Calibrate thumb sensor                              ros2 topic pub /hand/calibrate_sensor std_msgs/Float32MultiArray "data: [1]" --once
Set fan to full speed                               ros2 topic pub /hand/set_fan_mode std_msgs/Float32MultiArray "data: [5]" --once
Send fist action                                    ros2 topic pub /hand/command/action std_msgs/String "data: 'fist'" --once
Send joint angles                                   ros2 topic pub /hand/command/joint_angles std_msgs/Float32MultiArray "data: [30,10,20,30, 0,60,60, 0,60,60, 0,60,60, 0,60,60]" --once


================================================================================
10. Hardware Interface Comparison
================================================================================

Feature                     RS485                           CANFD
----------------------------------------------------------------------------------------------------
Default baud rate           500000 bps                      Arbitration: 1 Mbps / Data: 2 Mbps
Serial settings             8N1 (fixed)                     N/A (uses CAN frames)
Protocol                    Modbus RTU                      Modbus over CANFD
Max response length         250 bytes                      59 bytes (auto-split)
Parameter "interface"       rs485                           canfd


================================================================================
Document version: 2026-09-02
================================================================================
