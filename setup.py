#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="tfcv",
    version="0.2.0",
    python_requires=">=3",
    description="Computer vision for tensorflow",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    url="https://github.com/fferflo/tfcv",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=[
        "pyunpack",
        "imageio",
        "numpy",
        "h5py",
        "googledrivedownloader",
        "torch",
        "distinctipy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
