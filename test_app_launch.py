#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick App Launch Test - Test khởi động ứng dụng nhanh
"""

import sys
import os
import time

# Thêm path cho QuangTPS
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_app_launch():
    """Test khởi động ứng dụng"""
    print("Testing QuangTPS Application Launch...")
    print("=" * 40)

    try:
        # Test 1: Import PyQt5
        print("1. Testing PyQt5...")
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        print("   ✓ PyQt5 available")

        # Test 2: Import MainWindow
        print("2. Testing MainWindow import...")
        from quangtps.ui.main_window import MainWindow, launch_application
        print("   ✓ MainWindow imported successfully")

        # Test 3: Create QApplication
        print("3. Testing QApplication creation...")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        print("   ✓ QApplication created")

        # Test 4: Create MainWindow instance
        print("4. Testing MainWindow creation...")
        main_window = MainWindow()
        print(f"   ✓ MainWindow created: {type(main_window).__name__}")

        # Test 5: Test basic window properties
        print("5. Testing window properties...")
        main_window.setWindowTitle("QuangTPS Test")
        title = main_window.windowTitle()
        print(f"   ✓ Window title: {title}")

        # Test 6: Test launch function
        print("6. Testing launch function...")
        if callable(launch_application):
            print("   ✓ launch_application function is callable")
        else:
            print("   ✗ launch_application function not callable")
            return False

        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("QuangTPS application is ready to launch!")
        print("=" * 40)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_modules():
    """Test các core modules cần thiết"""
    print("\nTesting Core Modules...")
    print("-" * 30)

    modules = [
        "quangtps.core.types",
        "quangtps.core.patient",
        "quangtps.dose.dose_calculator",
        "quangtps.evaluation.metrics.gamma_analysis",
    ]

    passed = 0
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
            passed += 1
        except Exception as e:
            print(f"✗ {module}: {e}")

    print(f"\nCore modules: {passed}/{len(modules)} passed")
    return passed == len(modules)

def main():
    """Main test function"""
    start_time = time.time()

    print("QuangTPS Quick Launch Test")
    print("=" * 50)

    # Test core modules first
    core_ok = test_core_modules()

    # Test app launch
    app_ok = test_app_launch()

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nTest completed in {duration:.2f} seconds")

    if core_ok and app_ok:
        print("🚀 QuangTPS is ready for launch!")
        return 0
    else:
        print("⚠️  Some issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())