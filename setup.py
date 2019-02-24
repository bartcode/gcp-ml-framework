from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow',
    'apache-beam[gcp]',
    'tensorflow_transform',
]

setup(
    name='ml-framework',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='ML framework.'
)
