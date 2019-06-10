from setuptools import find_packages
from setuptools import setup

setup(
    name='ml-framework',
    version='0.1',
    install_requires=[
        'tensorflow',
        'apache-beam[gcp]',
        'tensorflow-transform',
        'pyyaml',
    ],
    packages=find_packages(),
    include_package_data=True,
    description='ML framework.'
)
