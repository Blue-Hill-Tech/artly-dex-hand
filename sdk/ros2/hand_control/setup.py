import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hand_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        # ==== 修改：分别安装左右手 mesh ====
        (os.path.join('share', package_name, 'right_hand_meshes'), glob('right_hand_meshes/*.stl')),
        (os.path.join('share', package_name, 'left_hand_meshes'), glob('left_hand_meshes/*.stl')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),  # 如果有配置文件
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='zouxixi',
    maintainer_email='zouxixi@example.com',
    description='ROS2 control for Artly_D21 hand over RS485',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_controller = hand_control.hand_controller_node:main',
        ],
    },
)
