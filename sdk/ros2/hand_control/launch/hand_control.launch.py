import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def get_robot_description(context):
    """Read the URDF file content based on the hand_type parameter."""
    hand_type = LaunchConfiguration('hand_type').perform(context)
    if hand_type == 'left':
        urdf_file = 'left_hand.urdf'
    else:
        urdf_file = 'right_hand.urdf'   # default to right hand
    pkg_dir = get_package_share_directory('hand_control')
    urdf_path = os.path.join(pkg_dir, 'urdf', urdf_file)
    with open(urdf_path, 'r') as f:
        return f.read()


def generate_launch_description():
    return LaunchDescription([
        # ========== Hand type (left/right) ==========
        DeclareLaunchArgument(
            'hand_type',
            default_value='right',
            choices=['left', 'right'],
            description='Hand type: left or right'
        ),
        # ========== Serial/communication parameters ==========
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0',
                              description='Serial port (RS485) or CANable device (CANFD)'),
        DeclareLaunchArgument('hand_id', default_value='1',
                              description='Modbus slave ID (1-254)'),
        DeclareLaunchArgument('baudrate', default_value='500000',
                              description='RS485 baud rate (only for RS485)'),
        DeclareLaunchArgument('use_simulation', default_value='false',
                              description='Enable simulation mode (no hardware)'),
        DeclareLaunchArgument('coupling_ratio', default_value='0.8',
                              description='Coupling ratio for URDF joints'),
        # ========== New parameters for interface selection ==========
        DeclareLaunchArgument('interface', default_value='rs485',
                              choices=['rs485', 'canfd'],
                              description='Communication interface: rs485 or canfd'),
        DeclareLaunchArgument('tty_baudrate', default_value='10000000',
                              description='TTY baud rate for CANable (CANFD mode)'),
        # ========== Update frequencies ==========
        DeclareLaunchArgument('control_rate', default_value='50',
                              description='Control + joint state publish rate (Hz)'),
        DeclareLaunchArgument('tactile_rate', default_value='50',
                              description='Tactile data publish rate (Hz)'),
        DeclareLaunchArgument('raw_rate', default_value='20',
                              description='Raw joint data publish rate (Hz)'),

        # ========== OpaqueFunction to launch nodes dynamically ==========
        OpaqueFunction(function=lambda context: [
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                output='screen',
                parameters=[{'robot_description': get_robot_description(context)}]
            ),
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                parameters=[{'source_list': ['/joint_states']}],
                output='screen'
            ),
            Node(
                package='hand_control',
                executable='hand_controller',
                name='hand_controller_node',
                parameters=[{
                    'port': LaunchConfiguration('port'),
                    'baudrate': LaunchConfiguration('baudrate'),
                    'hand_id': LaunchConfiguration('hand_id'),
                    'use_simulation': LaunchConfiguration('use_simulation'),
                    'coupling_ratio': LaunchConfiguration('coupling_ratio'),
                    'control_rate': LaunchConfiguration('control_rate'),
                    'tactile_rate': LaunchConfiguration('tactile_rate'),
                    'raw_rate': LaunchConfiguration('raw_rate'),
                    # ----- newly added parameters -----
                    'interface': LaunchConfiguration('interface'),
                    'tty_baudrate': LaunchConfiguration('tty_baudrate'),
                }],
                output='screen'
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                arguments=['-d', os.path.join(get_package_share_directory('hand_control'), 'rviz', 'hand.rviz')],
                output='screen'
            )
        ])
    ])
