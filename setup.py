from setuptools import setup

setup(
    name='simple_tree',
    version='0.3.0',
    description='A simple all-python decision tree library',
    url='https://github.com/davidthaler/SimpleTree',
    author='David Thaler',
    author_email='davidthaler@gmail.com',
    license='MIT',
    packages=['simple_tree', 'simple_tree.tests', 'simple_tree.datasets']
)
