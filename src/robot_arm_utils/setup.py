from setuptools import find_packages, setup

package_name = 'robot_arm_utils'

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
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Common utilities for robot arm packages',
    license='Apache-2.0',
    tests_require=['pytest'],
)
