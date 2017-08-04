import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This module is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='pposgd_mpi',
    version='0.0.1',
    install_requires=[
        'mujoco-py ~=0.5.7',
        'gym>=0.9.1[mujoco]',
        'mujoco-py',
        'scipy',
        'tqdm',
        'joblib',
        'zmq',
        'dill',
        'progressbar2',
        'mpi4py ~= 2.0.0'
    ],
    # https://github.com/tensorflow/tensorflow/issues/7166#issuecomment-280881808
    extras_require={
        "tf": ["tensorflow ~= 1.2"],
        "tf_gpu": ["tensorflow-gpu >= 1.1"],
    }
)
