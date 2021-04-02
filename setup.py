
import os
import sys
import subprocess
from setuptools import setup, find_packages

with open("requirements.txt", mode='r') as f:
    install_requires = f.read().split('\n')

install_requires = [e for e in install_requires if len(e) > 0]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


install('requests')


setup(
    name="moseq2-ephys-sync", # Replace with your own username
    version="0.0.1",
    author="Grigori Guitchounts",
    author_email="guitchounts@fas.harvard.edu",
    description="Tools to sync Open Ephys data with video data using IR LEDs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
)