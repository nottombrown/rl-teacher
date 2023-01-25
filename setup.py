import sys

from setuptools import setup

if sys.version_info.major != 3:
    print("This module is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='rl_teacher',
    version='0.0.1',
    py_modules=['rl_teacher'],
    install_requires=[
        'mujoco-py ~=2.1.2.14',
        'gym[mujoco] ~=0.26.2',
        'tqdm',
        'matplotlib',
        'ipython',
        'scipy',
        'ipdb',
        'keras',
        'glfw'
    ],
    # https://github.com/tensorflow/tensorflow/issues/7166#issuecomment-280881808
    extras_require={
        "tf": ["tensorflow >=2.11.0"],
        "tf_gpu": ["tensorflow-gpu >=2.11.0"],
    }
)