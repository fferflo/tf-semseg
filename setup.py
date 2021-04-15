#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="tf-semseg",
    version="0.1",
    python_requires=">=3",
    description="Semantic segmentation utilities for tensorflow",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    url="https://github.com/fferflo/tf-semseg",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=[
        "pyunpack",
        "imageio",
        "numpy",
        "h5py",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
  ],
)
