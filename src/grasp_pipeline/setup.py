from glob import glob
from setuptools import find_packages, setup

package_name = 'grasp_pipeline'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'ultralytics',
        'segment-anything',
        'open3d',
        'Pillow',
    ],
    zip_safe=True,
    maintainer='lx_hmkk',
    maintainer_email='lx_hmkk@qq.com',
    description='Grasp detection pipeline library (YOLO + SAM + GraspNet)',
    license='Apache-2.0',
    tests_require=['pytest'],
)
