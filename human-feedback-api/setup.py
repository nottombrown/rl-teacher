from setuptools import setup

setup(name='human_feedback_api',
    version='0.0.1',
    py_modules=['human_feedback_api'],
    install_requires=[
        'Django >=4.0.0',
        'dj_database_url ~=1.0.0',
        'gunicorn',
        'whitenoise ~=6.2.0', # version check
        'ipython',
    ]
)
