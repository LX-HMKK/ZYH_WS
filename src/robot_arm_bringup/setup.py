from setuptools import setup
from glob import glob
import os

package_name = 'robot_arm_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zyh',
    maintainer_email='zyh@todo.todo',
    description='Launch files for the robot arm vision system',
    license='Apache-2.0',
)
