#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script thiết lập môi trường ban đầu cho QuangTPS.
Tạo các thư mục cần thiết và kiểm tra dependencies.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

def setup_directories():
    """Tạo cấu trúc thư mục cần thiết"""
    root_dir = Path(__file__).parent.parent
    
    # Các thư mục cần tạo
    directories = {
        'data': {
            'beam_data': None,
            'clinical_protocols': None,
            'database': None,
            'dicom': None,
            'images': None,
            'machine_data': None,
            'models': None,
            'structures': None,
            'templates': None
        },
        'logs': None,
        'temp': None
    }
    
    # Tạo các thư mục
    for dir_name, subdirs in directories.items():
        dir_path = root_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")
        
        if subdirs:
            for subdir in subdirs:
                subdir_path = dir_path / subdir
                subdir_path.mkdir(exist_ok=True)
                print(f"Created subdirectory: {subdir_path}")

def check_dependencies():
    """Kiểm tra các thư viện phụ thuộc"""
    required_packages = [
        'numpy',
        'scipy',
        'pydicom',
        'PyQt5',
        'matplotlib',
        'pandas',
        'scikit-image',
        'tensorflow',
        'torch',
        'vtk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"Found {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"Missing {package}")
    
    return missing_packages

def setup_database():
    """Thiết lập cơ sở dữ liệu SQLite"""
    from quangtps.database.db_connector import DBConnector
    
    db = DBConnector()
    db.initialize_database()
    print("Database initialized")

def main():
    """Hàm chính để thiết lập môi trường"""
    print("Setting up QuangTPS environment...")
    
    # Tạo cấu trúc thư mục
    setup_directories()
    
    # Kiểm tra dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print("\nMissing packages:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return 1
    
    # Thiết lập cơ sở dữ liệu
    try:
        setup_database()
    except Exception as e:
        print(f"Error setting up database: {e}")
        return 1
    
    print("\nEnvironment setup completed successfully!")
    return 0

if __name__ == "__main__":
 