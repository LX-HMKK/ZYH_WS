from setuptools import find_packages, setup
# from rosidl_setup_py import generate_distutils_setup
from glob import glob
import os

package_name = 'grasp_publisher'

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
        # (os.path.join('share', package_name, 'msg'),
        #  glob('msg/*.msg')),
    ],
    install_requires=['setuptools','pyrealsense2'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Grasp result publisher for ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grasp_node = grasp_publisher.grasp_node:main',
        ],
    },
)