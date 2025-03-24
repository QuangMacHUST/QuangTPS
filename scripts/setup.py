#!/usr/bin/env python3
"""Script thiết lập môi trường QuangTPS"""

import os
import sys
from pathlib import Path

def setup():
    """Thiết lập môi trường cơ bản"""
    root = Path(__file__).parent.parent
    
    # Tạo thư mục cần thiết
    dirs = [
        'data/beam_data',
        'data/dicom',
        'data/database',
        'logs',
        'temp'
    ]
    
    for d in dirs:
        path = root / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {path}")
    
    # Kiểm tra dependencies
    deps = ['numpy', 'scipy', 'pydicom', 'PyQt5']
    missing = []
    
    for pkg in deps:
        try:
            __import__(pkg)
            print(f"Found {pkg}")
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("\nMissing packages:")
        print(" ".join(missing))
        return 1
        
    print("\nSetup completed!")
    return 0

if __name__ == "__main__":
    sys.exit(setup()) 