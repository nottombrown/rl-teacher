from setuptools import setup

setup(name='rl_teacher',
    version='0.0.1',
    install_requires=[
        'mujoco-py ~=0.5.7',
        'gym[mujoco]',
        'mujoco-py',
        'tqdm',
        'matplotlib',
        'ipython',
        'scipy',
        'ipdb',
        'keras',
    ],
    # https://github.com/tensorflow/tensorflow/issues/7166#issuecomment-280881808
    extras_require={
        "tf": ["tensorflow ~= 1.2"],
        "tf_gpu": ["tensorflow-gpu >= 1.1"],
    }
)
