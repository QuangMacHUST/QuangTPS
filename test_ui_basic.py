#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic UI test script - Test các component UI cơ bản
"""

import sys
import os

# Thêm path cho QuangTPS
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def test_pyqt5_availability():
    """Test xem PyQt5 có khả dụng không"""
    print("Testing PyQt5 availability...")
    try:
        from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
        from PyQt5.QtCore import Qt, pyqtSignal
        from PyQt5.QtGui import QIcon

        print("✓ PyQt5 is available")
        return True
    except ImportError as e:
        print(f"✗ PyQt5 not available: {e}")
        return False


def test_main_window_creation():
    """Test tạo MainWindow"""
    print("\nTesting MainWindow creation...")
    try:
        # Import với cách an toàn
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.main_window import MainWindow

        # Tạo QApplication nếu chưa có
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        # Test tạo object (không show)
        main_window = MainWindow()
        print(f"✓ MainWindow created successfully: {type(main_window).__name__}")

        # Test một số methods cơ bản
        if hasattr(main_window, "setWindowTitle"):
            print("✓ Basic window methods available")

        return True
    except Exception as e:
        print(f"✗ MainWindow creation failed: {e}")
        return False


def test_launch_function():
    """Test launch function without actually launching"""
    print("\nTesting launch function import...")
    try:
        from quangtps.ui.main_window import launch_application

        print(f"✓ launch_application function imported: {callable(launch_application)}")
        return True
    except Exception as e:
        print(f"✗ launch_application import failed: {e}")
        return False


def test_basic_ui_components():
    """Test các UI components cơ bản"""
    print("\nTesting basic UI components...")

    try:
        from PyQt5.QtWidgets import QApplication

        # Tạo QApplication nếu chưa có
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
    except Exception as e:
        print(f"✗ Cannot create QApplication: {e}")
        return False

    passed = 0
    total = 0

    components = [
        "quangtps.ui.visualization_3d.StructureViewer3D",
        "quangtps.ui.isodose_selector.IsodoseSelector",
        "quangtps.ui.structure_visibility_panel.StructureVisibilityPanel",
    ]

    for component_path in components:
        total += 1
        module_path, class_name = component_path.rsplit(".", 1)
        try:
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)

            # Test tạo instance
            if class_name in ["IsodoseSelector", "StructureVisibilityPanel"]:
                # Các widget này cần QApplication
                instance = component_class()
                print(f"✓ {class_name} imported and created successfully")
            else:
                print(f"✓ {class_name} imported successfully")
            passed += 1
        except Exception as e:
            print(f"✗ {class_name} import/creation failed: {e}")

    print(f"Components: {passed}/{total} passed")
    return passed == total


def main():
    """Chạy basic UI tests"""
    print("QuangTPS Basic UI Test")
    print("=" * 30)

    if not test_pyqt5_availability():
        print("PyQt5 không khả dụng - không thể test UI")
        return 1

    tests = [
        test_main_window_creation,
        test_launch_function,
        test_basic_ui_components,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print(f"\nResults: {passed}/{total} UI tests passed")

    if passed == total:
        print("✓ Basic UI functionality working!")
        return 0
    else:
        print("✗ Some UI issues remain.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
