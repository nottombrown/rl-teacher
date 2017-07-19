from setuptools import setup

setup(name='parallel_trpo',
    version='0.0.1',
    install_requires=[
        'gym[mujoco] ~= 0.9.2',
        'mujoco-py ~= 0.5.7',
        'tensorflow',
        'multiprocess ~= 0.70.5'
    ]
)
