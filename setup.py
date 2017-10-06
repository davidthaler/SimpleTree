from setuptools import setup

setup(
    name='simple_tree',
    version='0.2.0',
    description='An all-python decision tree with key sections accelerated using Numba.',
    url='https://github.com/davidthaler/SimpleTree',
    author='David Thaler',
    author_email='davidthaler@gmail.com',
    license='MIT',
    packages=['simple_tree', 'simple_tree.tests']
)
