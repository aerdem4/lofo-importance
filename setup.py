from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lofo-importance',
    version='0.0.2',
    packages=['lofo'],
    install_requires=requirements
)
