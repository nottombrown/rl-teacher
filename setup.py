from setuptools import setup

setup(name='rl_teacher',
    version='0.0.1',
    install_requires=[
        'mujoco-py ~=0.5.7',
        'gym[mujoco]',
        'mujoco-py',
        'tqdm',
        'tensorflow', # Tested on Tensorflow 1.2
        'matplotlib',
        'ipython',
        'scipy',
        'ipdb',
        'keras',
    ]
)
