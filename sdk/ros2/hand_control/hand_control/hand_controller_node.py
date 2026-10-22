#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import JointState
import time
import threading
import math

# Import drivers for RS485 and CANFD
from .d21_rs485 import d21_rs485
from .d21_canfd import d21_canfd


class HandControllerNode(Node):
    def __init__(self):
        super().__init__('hand_controller_node')
        
        # ---------- Parameters ----------
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 500000)          # RS485 baud rate
        self.declare_parameter('hand_id', 1)
        self.declare_parameter('use_simulation', False)
        self.declare_parameter('coupling_ratio', 0.8)
        
        # Interface selection: 'rs485' (default) or 'canfd'
        self.declare_parameter('interface', 'rs485')
        self.declare_parameter('tty_baudrate', 10000000)     # CANFD serial baud rate (e.g., CANable)
        
        # Update frequencies
        self.declare_parameter('control_rate', 50)      # Control write + joint state publish (Hz)
        self.declare_parameter('tactile_rate', 50)      # Tactile data publish (Hz)
        self.declare_parameter('raw_rate', 20)          # Raw joint data publish (Hz)
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baudrate').value
        hand_id = self.get_parameter('hand_id').value
        self.use_simulation = self.get_parameter('use_simulation').value
        self.coupling_ratio = self.get_parameter('coupling_ratio').value
        
        interface = self.get_parameter('interface').value
        tty_baudrate = self.get_parameter('tty_baudrate').value
        
        self.control_rate = self.get_parameter('control_rate').value
        self.tactile_rate = self.get_parameter('tactile_rate').value
        self.raw_rate = self.get_parameter('raw_rate').value
        
        # ---------- URDF joint names (26 joints) ----------
        self.urdf_joint_names = [
            'thumb_joint1', 'thumb_joint2', 'thumb_joint3', 
            'thumb_joint4', 'thumb_joint5', 'thumb_joint6',
            'index_joint1', 'index_joint2', 'index_joint3', 
            'index_joint4', 'index_joint5',
            'middle_joint1', 'middle_joint2', 'middle_joint3', 
            'middle_joint4', 'middle_joint5',
            'ring_joint1', 'ring_joint2', 'ring_joint3', 
            'ring_joint4', 'ring_joint5',
            'pinky_joint1', 'pinky_joint2', 'pinky_joint3', 
            'pinky_joint4', 'pinky_joint5'
        ]
        
        # ---------- Hardware-to-URDF mapping table ----------
        self.joint_map = [
            {'hw_idx': 0, 'urdf_idx': 0, 'coupled': False},
            {'hw_idx': 1, 'urdf_idx': 1, 'coupled': False},
            {'hw_idx': 2, 'urdf_idx': 2, 'coupled': False},
            {'hw_idx': 3, 'urdf_idx': 3, 'coupled': False},
            {'hw_idx': 3, 'urdf_idx': 4, 'coupled': True, 'source_idx': 3, 'ratio': 0.85}, 
            {'hw_idx': 3, 'urdf_idx': 5, 'coupled': True, 'source_idx': 3, 'ratio': 0.85},
            {'hw_idx': 4, 'urdf_idx': 6, 'coupled': False},
            {'hw_idx': 5, 'urdf_idx': 7, 'coupled': False},
            {'hw_idx': 6, 'urdf_idx': 8, 'coupled': False},
            {'hw_idx': 6, 'urdf_idx': 9, 'coupled': True, 'source_idx': 6, 'ratio': 0.8},
            {'hw_idx': 6, 'urdf_idx': 10, 'coupled': True, 'source_idx': 6, 'ratio': 0.8},
            {'hw_idx': 7, 'urdf_idx': 11, 'coupled': False},
            {'hw_idx': 8, 'urdf_idx': 12, 'coupled': False},
            {'hw_idx': 9, 'urdf_idx': 13, 'coupled': False},
            {'hw_idx': 9, 'urdf_idx': 14, 'coupled': True, 'source_idx': 9, 'ratio': 0.8},
            {'hw_idx': 9, 'urdf_idx': 15, 'coupled': True, 'source_idx': 9, 'ratio': 0.8},
            {'hw_idx': 10, 'urdf_idx': 16, 'coupled': False},
            {'hw_idx': 11, 'urdf_idx': 17, 'coupled': False},
            {'hw_idx': 12, 'urdf_idx': 18, 'coupled': False},
            {'hw_idx': 12, 'urdf_idx': 19, 'coupled': True, 'source_idx': 12, 'ratio': 0.8},
            {'hw_idx': 12, 'urdf_idx': 20, 'coupled': True, 'source_idx': 12, 'ratio': 0.8},
            {'hw_idx': 13, 'urdf_idx': 21, 'coupled': False},
            {'hw_idx': 14, 'urdf_idx': 22, 'coupled': False},
            {'hw_idx': 15, 'urdf_idx': 23, 'coupled': False},
            {'hw_idx': 15, 'urdf_idx': 24, 'coupled': True, 'source_idx': 15, 'ratio': 0.8},
            {'hw_idx': 15, 'urdf_idx': 25, 'coupled': True, 'source_idx': 15, 'ratio': 0.8},
        ]
        
        # ---------- Hardware initialization (RS485 or CANFD) ----------
        self.hand = None
        if not self.use_simulation:
            try:
                if interface.lower() == 'canfd':
                    self.get_logger().info(f'Using CANFD interface on port {port}, tty baud {tty_baudrate}')
                    self.hand = d21_canfd(
                        hand_id=hand_id,
                        modbus_port=port,
                        timeout=0.1,
                        tty_baudrate=tty_baudrate
                    )
                    self.hand.connect()   # Start CANFD receive thread
                    self.get_logger().info('CANFD device connected and receive thread started')
                else:
                    self.get_logger().info(f'Using RS485 interface on port {port}, baud {baud}')
                    self.hand = d21_rs485(
                        hand_id=hand_id,
                        modbus_port=port,
                        baudrate=baud,
                        timeout=0.1
                    )
                    self.get_logger().info('RS485 serial port opened')
            except Exception as e:
                self.get_logger().error(f'Hardware connection failed: {e}')
                self.use_simulation = True
                self.get_logger().info('Switched to simulation mode')
        else:
            self.get_logger().info('Simulation mode (no hardware)')
        
        if self.hand is None and not self.use_simulation:
            self.use_simulation = True
            self.get_logger().warn('Forced simulation mode')
        
        self.lock = threading.Lock()
        self.target_hw_angles = [0.0] * 16
        
        # ---------- Publishers ----------
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        self.tactile_force_pub = self.create_publisher(
            Float32MultiArray,
            '/hand/tactile/forces',
            10
        )
        self.tactile_pressure_pub = self.create_publisher(
            Float32MultiArray,
            '/hand/tactile/pressure',
            10
        )
        self.joint_raw_pub = self.create_publisher(
            Float32MultiArray,
            '/hand/joint/raw',
            10
        )
        
        # ---------- Subscribers ----------
        # Joint angle command
        self.joint_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/command/joint_angles',
            self.joint_callback,
            10
        )
        # Action command (fist/open/pinch)
        self.action_sub = self.create_subscription(
            String,
            '/hand/command/action',
            self.action_callback,
            10
        )
        # Fan mode setting
        self.fan_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/set_fan_mode',
            self.set_fan_callback,
            10
        )
        # Control mode: 1=position, 2=torque
        self.control_mode_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/set_control_mode',
            self.set_control_mode_callback,
            10
        )
        # Torque parameters for all 16 motors (values in -1000..1000)
        self.torques_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/set_torques',
            self.set_torques_callback,
            10
        )
        # ========== NEW: Sensor calibration ==========
        self.calibrate_sub = self.create_subscription(
            Float32MultiArray,
            '/hand/calibrate_sensor',
            self.calibrate_callback,
            10
        )
        
        # ---------- Timers with independent frequencies ----------
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_and_joint_publish)
        self.tactile_timer = self.create_timer(1.0 / self.tactile_rate, self.publish_tactile)
        self.raw_timer = self.create_timer(1.0 / self.raw_rate, self.publish_raw_joint_data)
        
        # ---------- Zero initialization ----------
        self.get_logger().info('Initializing angles to zero...')
        self.target_hw_angles = [0.0] * 16
        if not self.use_simulation and self.hand is not None:
            try:
                self.hand.write_joint_position_commands([0.0]*16)
                self.get_logger().info('Hardware zeroed')
            except Exception as e:
                self.get_logger().error(f'Hardware zeroing failed: {e}')
        time.sleep(0.5)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Dexterous Hand Controller Node Started')
        self.get_logger().info(f'   Interface: {"Simulation" if self.use_simulation else interface.upper()}')
        self.get_logger().info(f'   Control+Joint State Rate: {self.control_rate} Hz')
        self.get_logger().info(f'   Tactile Data Rate: {self.tactile_rate} Hz')
        self.get_logger().info(f'   Raw Joint Data Rate: {self.raw_rate} Hz')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Control Topics:')
        self.get_logger().info('   /hand/command/joint_angles  - 16 angles (degrees)')
        self.get_logger().info('   /hand/command/action        - fist / open / pinch')
        self.get_logger().info('   /hand/set_fan_mode          - 0..5 (0=stop, 1-5 speeds)')
        self.get_logger().info('   /hand/set_control_mode      - 1 (position) or 2 (torque)')
        self.get_logger().info('   /hand/set_torques           - 16 values in [-1000, 1000]')
        # ========== NEW: Print calibration topic ==========
        self.get_logger().info('   /hand/calibrate_sensor      - 1..5 (finger index to calibrate)')
        self.get_logger().info('Published Topics:')
        self.get_logger().info('   /joint_states               - joint states (50 Hz)')
        self.get_logger().info('   /hand/tactile/forces        - triaxial forces (50 Hz)')
        self.get_logger().info('   /hand/tactile/pressure      - pressure distribution (50 Hz)')
        self.get_logger().info('   /hand/joint/raw             - raw joint data (20 Hz)')
        self.get_logger().info('=' * 60)
    
    # ---------- Mapping function ----------
    def hw_to_urdf(self, hw_angles):
        urdf = [0.0] * 26
        for m in self.joint_map:
            if m['coupled']:
                src = m['source_idx']
                ratio = m.get('ratio', self.coupling_ratio)   
                urdf[m['urdf_idx']] = hw_angles[src] * ratio
            else:
                urdf[m['urdf_idx']] = hw_angles[m['hw_idx']]
        return urdf
    
    # ---------- Callbacks ----------
    def joint_callback(self, msg):
        if len(msg.data) != 16:
            self.get_logger().warn(f'Need 16 angles, got {len(msg.data)}')
            return
        with self.lock:
            self.target_hw_angles = list(msg.data)
        self.get_logger().info(f'Received joint command: {msg.data[:4]}...')
    
    def action_callback(self, msg):
        action = msg.data.lower()
        preset = {
            'fist':  [80,0,30,40, 0,80,90, 0,80,90, 0,80,90, 0,80,90],
            'open':  [0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
            'pinch': [60,0,30,20, 0,30,40, 0,0,0, 0,0,0, 0,0,0]
        }
        if action in preset:
            with self.lock:
                self.target_hw_angles = preset[action]
            self.get_logger().info(f'Executing action: {action}')
        else:
            self.get_logger().warn(f'Unknown action: {action}')
    
    # ---------- Control + joint state publish (50 Hz) ----------
    def control_and_joint_publish(self):
        with self.lock:
            target = self.target_hw_angles.copy()
        
        # Write to hardware if changed
        if not self.use_simulation and self.hand is not None:
            if not hasattr(self, '_last_sent_angles'):
                self._last_sent_angles = [0.0] * 16
            
            changed = False
            for i in range(16):
                if abs(target[i] - self._last_sent_angles[i]) > 0.3:
                    changed = True
                    break
            
            if changed:
                try:
                    self.hand.write_joint_position_commands(target)
                    self._last_sent_angles = target.copy()
                except Exception as e:
                    self.get_logger().error(f'Hardware write failed: {e}')
        
        # Publish joint states for RViz2
        urdf_angles_deg = self.hw_to_urdf(target)
        urdf_angles_rad = [math.radians(a) for a in urdf_angles_deg]
        now = self.get_clock().now()
        joint_msg = JointState()
        joint_msg.header.stamp = now.to_msg()
        joint_msg.name = self.urdf_joint_names
        joint_msg.position = urdf_angles_rad
        self.joint_state_pub.publish(joint_msg)
    
    # ---------- Tactile data publish (50 Hz) ----------
    def publish_tactile(self):
        if self.use_simulation or self.hand is None:
            return
        try:
            forces = self.hand.read_tactile_forces()
            if forces:
                self.tactile_force_pub.publish(Float32MultiArray(data=forces))
            
            thumb = self.hand.read_thumb_norm_pressures()
            index = self.hand.read_index_norm_pressures()
            middle = self.hand.read_middle_norm_pressures()
            ring = self.hand.read_ring_norm_pressures()
            pinky = self.hand.read_pinky_norm_pressures()
            pressures = thumb + index + middle + ring + pinky
            if pressures:
                self.tactile_pressure_pub.publish(Float32MultiArray(data=pressures))
        except Exception as e:
            self.get_logger().debug(f'Tactile read failed: {e}')
    
    # ---------- Raw joint data publish (20 Hz) ----------
    def publish_raw_joint_data(self):
        if self.use_simulation or self.hand is None:
            return
        try:
            positions = self.hand.read_joint_positions()
            velocities = self.hand.read_joint_velocities()
            currents = self.hand.read_motor_currents()
            raw_data = positions + velocities + currents
            self.joint_raw_pub.publish(Float32MultiArray(data=raw_data))
        except Exception as e:
            self.get_logger().debug(f'Raw data read failed: {e}')
    
    # ---------- Fan mode setting ----------
    def set_fan_callback(self, msg):
        if len(msg.data) != 1:
            self.get_logger().warn('Fan mode requires one value')
            return
        mode = int(msg.data[0])
        if mode not in range(6):
            self.get_logger().warn(f'Invalid fan mode {mode}, should be 0-5')
            return
        if self.use_simulation or self.hand is None:
            self.get_logger().info(f'Simulation: set fan mode to {mode}')
            return
        try:
            self.hand.set_fan_mode(mode)
            self.get_logger().info(f'Fan mode set to {mode}')
        except Exception as e:
            self.get_logger().error(f'Failed to set fan mode: {e}')
    
    # ---------- Control mode setting (1=position, 2=torque) ----------
    def set_control_mode_callback(self, msg):
        if len(msg.data) != 1:
            self.get_logger().warn('Control mode requires one value (1 or 2)')
            return
        mode = int(msg.data[0])
        if mode not in (1, 2):
            self.get_logger().warn(f'Invalid control mode {mode}, should be 1 (position) or 2 (torque)')
            return
        if self.use_simulation or self.hand is None:
            self.get_logger().info(f'Simulation: set control mode to {mode}')
            return
        try:
            self.hand.set_control_mode(mode)
            self.get_logger().info(f'Control mode set to {mode}')
        except Exception as e:
            self.get_logger().error(f'Failed to set control mode: {e}')
    
    # ---------- Torque parameters setting (16 values in -1000..1000) ----------
    def set_torques_callback(self, msg):
        if len(msg.data) != 16:
            self.get_logger().warn(f'Need 16 torque values, got {len(msg.data)}')
            return
        # Clamp values to [-1000, 1000]
        torques = [max(-1000.0, min(1000.0, v)) for v in msg.data]
        if self.use_simulation or self.hand is None:
            self.get_logger().info(f'Simulation: set torques to {torques}')
            return
        try:
            self.hand.set_all_torques(torques)
            self.get_logger().info(f'Torques set to {torques[:4]}...')
        except Exception as e:
            self.get_logger().error(f'Failed to set torques: {e}')
    
    # ========== NEW: Sensor calibration callback ==========
    def calibrate_callback(self, msg):
        """
        Calibrate tactile sensor for a specific finger.
        Expects a Float32MultiArray with one element: finger index (1~5).
        """
        if len(msg.data) != 1:
            self.get_logger().warn(f'Calibration requires one finger index, got {len(msg.data)}')
            return
        finger = int(msg.data[0])
        if finger < 1 or finger > 5:
            self.get_logger().warn(f'Invalid finger index {finger}, should be 1~5')
            return
        if self.use_simulation or self.hand is None:
            self.get_logger().info(f'Simulation: calibrate finger {finger}')
            return
        try:
            self.hand.calibrate_sensor(finger)
            self.get_logger().info(f'Calibration triggered for finger {finger}')
        except Exception as e:
            self.get_logger().error(f'Calibration failed: {e}')
    
    # ---------- Node destruction ----------
    def destroy_node(self):
        if self.hand is not None and not self.use_simulation:
            try:
                self.hand.close()
                self.get_logger().info('Hardware interface closed')
            except Exception as e:
                self.get_logger().error(f'Error closing hardware: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
