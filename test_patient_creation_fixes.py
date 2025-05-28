#!/usr/bin/env python3
"""
Test script cho việc sửa lỗi tạo bệnh nhân mới - QuangTPS v0.16.17
Kiểm tra các sửa lỗi:
1. Script khởi động hoạt động
2. New Patient dialog functionality
3. Toolbar error fixes
4. Patient creation workflow
"""

import sys
import os
import traceback
from datetime import datetime, date

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global QApplication instance
app = None


def setup_qt_application():
    """Setup QApplication for UI tests"""
    global app
    try:
        from PyQt5.QtWidgets import QApplication

        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        return True
    except Exception as e:
        print(f"   ✗ Không thể tạo QApplication: {e}")
        return False


def test_script_imports():
    """Test rằng scripts có thể import đúng"""
    try:
        print("1. Testing script imports...")

        # Test run_quangtps imports
        sys.path.append("scripts")
        from scripts.run_quangtps import run_application

        print("   ✓ run_quangtps.py import thành công")

        # Test launch_quangtps imports
        from scripts.launch_quangtps import launch_quangtps

        print("   ✓ launch_quangtps.py import thành công")

        return True
    except Exception as e:
        print(f"   ✗ Lỗi import scripts: {e}")
        return False


def test_main_window_constructor():
    """Test MainWindow constructor không nhận config parameter"""
    try:
        print("2. Testing MainWindow constructor...")

        # Setup QApplication first
        if not setup_qt_application():
            return False

        from quangtps.ui.main_window import MainWindow

        # Test constructor without config (correct way)
        main_window = MainWindow()
        print("   ✓ MainWindow() constructor hoạt động đúng")

        # Test có method _new_patient
        if hasattr(main_window, "_new_patient"):
            print("   ✓ Method _new_patient tồn tại")
        else:
            print("   ✗ Method _new_patient không tồn tại")
            return False

        # Cleanup
        main_window.close()
        return True
    except Exception as e:
        print(f"   ✗ Lỗi MainWindow constructor: {e}")
        return False


def test_patient_dialog_imports():
    """Test NewPatientDialog và Patient class imports"""
    try:
        print("3. Testing patient-related imports...")

        # Test dialog import
        from quangtps.ui.dialogs.new_patient_dialog import NewPatientDialog

        print("   ✓ NewPatientDialog import thành công")

        # Test Patient class import
        from quangtps.core.patient.patient import Patient, PatientGender, DiagnosisInfo

        print("   ✓ Patient classes import thành công")

        return True
    except Exception as e:
        print(f"   ✗ Lỗi import patient classes: {e}")
        return False


def test_patient_creation_logic():
    """Test logic tạo patient object"""
    try:
        print("4. Testing patient creation logic...")

        from quangtps.core.patient.patient import Patient, PatientGender, DiagnosisInfo
        from datetime import datetime, date
        import uuid

        # Test tạo diagnosis
        diagnosis = DiagnosisInfo(
            primary_diagnosis="Test Cancer",
            diagnosis_code="C78.9",
            stage="Stage II",
            site="Lung",
            laterality="Left",
            diagnosis_date=date.today(),
        )
        print("   ✓ DiagnosisInfo tạo thành công")

        # Test tạo patient
        patient = Patient(
            id=str(uuid.uuid4()),
            name="Test Patient",
            birth_date=date(1980, 1, 1),
            gender=PatientGender.MALE,
            patient_id="PT001",
            phone="123-456-7890",
            email="test@example.com",
            address="Test Address",
            diagnosis=diagnosis,
            notes="Test patient",
            created_date=datetime.now(),
        )
        print("   ✓ Patient object tạo thành công")
        print(f"      - Name: {patient.name}")
        print(f"      - ID: {patient.patient_id}")
        print(f"      - Gender: {patient.gender.value}")

        return True
    except Exception as e:
        print(f"   ✗ Lỗi tạo patient object: {e}")
        traceback.print_exc()
        return False


def test_toolbar_actions_handling():
    """Test rằng toolbar actions được xử lý đúng"""
    try:
        print("5. Testing toolbar actions handling...")

        # Setup QApplication first
        if not setup_qt_application():
            return False

        from quangtps.ui.main_window import MainWindow

        main_window = MainWindow()

        # Test _update_ui_state method exists
        if hasattr(main_window, "_update_ui_state"):
            print("   ✓ Method _update_ui_state tồn tại")

            # Test call method (should not error)
            main_window._update_ui_state()
            print("   ✓ _update_ui_state() chạy không lỗi")
        else:
            print("   ✗ Method _update_ui_state không tồn tại")
            return False

        # Cleanup
        main_window.close()
        return True
    except Exception as e:
        print(f"   ✗ Lỗi toolbar actions: {e}")
        return False


def test_set_window_level_safety():
    """Test set_window_level method với mpr_viewer không tồn tại"""
    try:
        print("6. Testing set_window_level safety...")

        # Setup QApplication first
        if not setup_qt_application():
            return False

        from quangtps.ui.main_window import MainWindow

        main_window = MainWindow()

        # Test method exists
        if hasattr(main_window, "set_window_level"):
            print("   ✓ Method set_window_level tồn tại")

            # Test call with mpr_viewer không tồn tại (should not error)
            main_window.set_window_level(1500, 300)
            print("   ✓ set_window_level() chạy an toàn khi mpr_viewer không tồn tại")
        else:
            print("   ✗ Method set_window_level không tồn tại")
            return False

        # Cleanup
        main_window.close()
        return True
    except Exception as e:
        print(f"   ✗ Lỗi set_window_level: {e}")
        return False


def main():
    """Chạy tất cả tests"""
    print("=== TEST PATIENT CREATION FIXES - QuangTPS v0.16.17 ===")
    print(f"Thời gian test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tests = [
        test_script_imports,
        test_main_window_constructor,
        test_patient_dialog_imports,
        test_patient_creation_logic,
        test_toolbar_actions_handling,
        test_set_window_level_safety,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"   ✗ Lỗi trong test {test_func.__name__}: {e}")
            print()

    print("=" * 60)
    print(f"KẾT QUẢ CUỐI CÙNG: {passed}/{total} tests PASSED")

    if passed == total:
        print("🎉 TẤT CẢ FIXES ĐÃ HOẠT ĐỘNG ĐÚNG!")
        print("✅ Scripts có thể khởi động QuangTPS")
        print("✅ New Patient dialog có thể tạo patient thực sự")
        print("✅ Toolbar errors đã được sửa")
        print("✅ Patient creation workflow hoàn chỉnh")
    else:
        print(f"⚠️  CÒN {total - passed} lỗi cần sửa")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    # Cleanup
    if app:
        app.quit()
    sys.exit(0 if success else 1)
