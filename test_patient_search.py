#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script test chức năng search patients và sửa issues
"""

import os
import sys
import traceback

# Add project root to path
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)


def test_patient_search():
    """Test chức năng search patients"""
    try:
        from quangtps.database.patient_db import PatientDB

        print("🔍 Test PatientDB Search Functionality")
        print("=" * 50)

        # Khởi tạo PatientDB
        db = PatientDB()

        # Test 1: Lấy tất cả patients
        print("\n1. Lấy tất cả patients:")
        all_patients = db.get_all_patients()
        print(f"   Số lượng patients: {len(all_patients)}")

        if all_patients:
            print("   Patients trong database:")
            for i, patient in enumerate(all_patients[:3]):  # Chỉ hiển thị 3 đầu tiên
                print(
                    f"     {i + 1}. ID: {patient.get('id')}, Name: {patient.get('name')}"
                )

        # Test 2: Search không có query
        print("\n2. Search với query rỗng:")
        empty_search = db.search_patients("")
        print(f"   Kết quả: {len(empty_search)} patients")

        # Test 3: Search với query cụ thể
        print("\n3. Search với query 'huy':")
        huy_search = db.search_patients("huy")
        print(f"   Kết quả: {len(huy_search)} patients")
        for patient in huy_search:
            print(f"     - ID: {patient.get('id')}, Name: {patient.get('name')}")

        # Test 4: Search với query không tồn tại
        print("\n4. Search với query 'không_tồn_tại':")
        no_result = db.search_patients("không_tồn_tại")
        print(f"   Kết quả: {len(no_result)} patients")

        # Test 5: Kiểm tra cấu trúc dữ liệu patient
        if all_patients:
            print("\n5. Cấu trúc dữ liệu patient đầu tiên:")
            first_patient = all_patients[0]
            print("   Các trường có trong patient:")
            for key, value in first_patient.items():
                print(f"     {key}: {value}")

        db.close()
        print("\n✅ Test PatientDB hoàn thành")

    except Exception as e:
        print(f"❌ Lỗi trong test PatientDB: {e}")
        traceback.print_exc()


def test_ui_patient_search():
    """Test UI patient search component"""
    try:
        print("\n🖥️ Test UI Patient Search")
        print("=" * 50)

        # Import UI components
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.patient_tab import PatientTab

        # Tạo QApplication nếu chưa có
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        # Tạo PatientTab
        patient_tab = PatientTab()

        # Test search functionality
        print("   ✅ PatientTab tạo thành công")

        # Test PatientDB connection trong UI
        if hasattr(patient_tab, "patient_db") and patient_tab.patient_db:
            print("   ✅ PatientDB connection thành công")

            # Test search method
            if hasattr(patient_tab, "_search_patients"):
                print("   ✅ _search_patients method tồn tại")
            else:
                print("   ❌ _search_patients method không tồn tại")
        else:
            print("   ❌ PatientDB connection thất bại")

    except Exception as e:
        print(f"❌ Lỗi trong test UI: {e}")
        traceback.print_exc()


def test_dose_implementers():
    """Test xem dose implementers đã được tạo đúng chưa"""
    try:
        print("\n🧮 Test Dose Implementers")
        print("=" * 50)

        # Test PencilBeamImplementer
        try:
            from quangtps.dose.algorithms.pencil_beam import PencilBeamImplementer

            pencil_implementer = PencilBeamImplementer()
            print("   ✅ PencilBeamImplementer tạo thành công")
            print(f"   📝 Description: {pencil_implementer.get_description()}")
        except Exception as e:
            print(f"   ❌ PencilBeamImplementer lỗi: {e}")

        # Test AAAImplementer
        try:
            from quangtps.dose.algorithms.aaa import AAAImplementer

            aaa_implementer = AAAImplementer()
            print("   ✅ AAAImplementer tạo thành công")
            print(f"   📝 Description: {aaa_implementer.get_description()}")
        except Exception as e:
            print(f"   ❌ AAAImplementer lỗi: {e}")

        # Test DoseEngine registration
        try:
            from quangtps.dose.dose_engine import DoseEngine, DoseCalculationAlgorithm

            engine = DoseEngine(DoseCalculationAlgorithm.CCC)
            print("   ✅ DoseEngine tạo thành công")

            # Test xem có implementers nào được đăng ký không
            available_algorithms = engine.get_available_algorithms()
            print(f"   📋 Algorithms available: {len(available_algorithms)}")
            for algo in available_algorithms:
                print(f"      - {algo.value}")

        except Exception as e:
            print(f"   ❌ DoseEngine lỗi: {e}")

    except Exception as e:
        print(f"❌ Lỗi trong test dose implementers: {e}")
        traceback.print_exc()


def main():
    """Chạy tất cả tests"""
    print("🧪 QuangTPS Issue Debug Script")
    print("=" * 60)

    # Test 1: Patient Search
    test_patient_search()

    # Test 2: UI Patient Search
    test_ui_patient_search()

    # Test 3: Dose Implementers
    test_dose_implementers()

    print("\n" + "=" * 60)
    print("✅ Hoàn thành tất cả tests")


if __name__ == "__main__":
    main()
