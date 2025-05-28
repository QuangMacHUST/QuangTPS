#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script đơn giản để kiểm tra khởi động MainWindow
"""

import sys
import os
import traceback

# Thêm path cho QuangTPS
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def test_basic_imports():
    """Test các import cơ bản"""
    print("Testing basic imports...")

    try:
        from PyQt5.QtWidgets import QApplication

        print("✓ PyQt5.QtWidgets imported successfully")
    except ImportError as e:
        print(f"✗ PyQt5.QtWidgets import failed: {e}")
        return False

    try:
        from PyQt5.QtCore import Qt

        print("✓ PyQt5.QtCore imported successfully")
    except ImportError as e:
        print(f"✗ PyQt5.QtCore import failed: {e}")
        return False

    return True


def test_mainwindow_import():
    """Test import MainWindow"""
    print("\nTesting MainWindow import...")

    try:
        from quangtps.ui.main_window import MainWindow

        print("✓ MainWindow imported successfully")
        return True
    except ImportError as e:
        print(f"✗ MainWindow import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ MainWindow import error: {e}")
        traceback.print_exc()
        return False


def test_mainwindow_creation():
    """Test tạo MainWindow instance"""
    print("\nTesting MainWindow creation...")

    try:
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.main_window import MainWindow

        # Tạo QApplication nếu chưa có
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Tạo MainWindow
        window = MainWindow()
        print("✓ MainWindow created successfully")

        # Test hiển thị
        window.show()
        print("✓ MainWindow shown successfully")

        return True

    except Exception as e:
        print(f"✗ MainWindow creation failed: {e}")
        traceback.print_exc()
        return False


def test_launch_application():
    """Test launch_application function"""
    print("\nTesting launch_application function...")

    try:
        from quangtps.ui.main_window import launch_application

        print("✓ launch_application imported successfully")
        return True
    except ImportError as e:
        print(f"✗ launch_application import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ launch_application error: {e}")
        traceback.print_exc()
        return False


def main():
    """Chạy tất cả tests"""
    print("QuangTPS Simple Startup Test")
    print("=" * 40)

    tests = [
        test_basic_imports,
        test_mainwindow_import,
        test_launch_application,
        test_mainwindow_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! MainWindow should start successfully.")
        return 0
    else:
        print("✗ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
