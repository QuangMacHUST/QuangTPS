#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import re
from pathlib import Path

# Get version from package __init__.py
init_file = Path("quangtps/__init__.py").read_text()
version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", init_file)
version = version_match.group(1) if version_match else "0.1.0"

# Read README.md for long description
readme = Path("README.md").read_text() if Path("README.md").exists() else ""

setup(
    name="quangtps",
    version=version,
    author="QuangTPS Team",
    author_email="quangtps@example.com",
    description="A Modern Radiotherapy Treatment Planning System",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/quangtps/quangtps",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "pydicom>=2.0.0",
        "PyQt5>=5.12.0",
        "matplotlib>=3.0.0",
        "scikit-image>=0.16.0",
    ],
    entry_points={
        "console_scripts": [
            "quangtps=quangtps.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
