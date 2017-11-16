from setuptools import setup

setup(
    name='simple_tree',
    version='0.4.0',
    description='Decision tree, random forest and gradient boosting models in python.',
    url='https://github.com/davidthaler/SimpleTree',
    author='David Thaler',
    author_email='davidthaler@gmail.com',
    license='MIT',
    packages=['simple_tree', 'simple_tree.tests', 'simple_tree.datasets']
)
