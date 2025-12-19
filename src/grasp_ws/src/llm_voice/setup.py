from setuptools import setup
from glob import glob
import os

package_name = 'llm_voice'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'srv'),
            glob('srv/*.srv')),
        # ↓↓↓ 关键：让 ament 知道这里有可执行文件 ↓↓↓
        (os.path.join('share', package_name), ['resource/llm_voice']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@todo.todo',
    description='TODO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_node = llm_voice.llm_node:main',
        ],
    },
)
