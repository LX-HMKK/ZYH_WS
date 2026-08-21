from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vision_grasp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/python3.10/site-packages/' + package_name + '/msg',
            glob(package_name + '/msg/*.py')),
    ],
    install_requires=['setuptools','pyrealsense2'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='RealSense vision and grasp detection node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grasp_node = vision_grasp.grasp_node:main',
        ],
    },
)
