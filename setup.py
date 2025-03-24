#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="quangtps",
    version="1.0.0",
    description="Hệ thống lập kế hoạch xạ trị mã nguồn mở",
    author="QuangTPS Developers",
    author_email="quangtps@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'pydicom>=2.2.0',
        'PyQt5>=5.15.0',
        'matplotlib>=3.4.0',
        'scikit-image>=0.18.0',
        'dicompyler-core>=0.5.5',
    ],
    entry_points={
        'console_scripts': [
            'quangtps=quangtps.__main__:main',
        ],
    },
)
