from setuptools import find_packages
from distutils.core import setup

setup(
    name='LeggedLab',
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        # 'isaacsim',
        'IsaacLab',
        'rsl-rl-lib',
    ]
)
