from setuptools import find_packages, setup

package_name = 'robot_arm_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='zyh',
    maintainer_email='zyh@todo.todo',
    description='Robot arm control: TCP communication, gripper, and motion state machine',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'io_node=robot_arm_control.io_node:main',
        'motion_node=robot_arm_control.motion_node:main',
        ],
    },
)
"""
colcon build
source install/setup.bash
ros2 run robot_arm_control io_node
ros2 run robot_arm_control motion_node

"""
