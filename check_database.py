#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm tra database và các module dose calculation
"""

import os
import sys
import sqlite3
import logging

# Add project root to path
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)


def check_database():
    """Kiểm tra database và dữ liệu bệnh nhân"""
    db_path = "data/database/quangtps.db"

    if not os.path.exists(db_path):
        print(f"❌ Database không tồn tại: {db_path}")
        return

    print(f"✅ Database tồn tại: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Kiểm tra các bảng
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print(f"\n📊 Các bảng trong database ({len(tables)} bảng):")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} bản ghi")

            # Hiển thị chi tiết cho bảng patients
            if table_name == "patients" and count > 0:
                cursor.execute("SELECT id, name, dob, gender FROM patients LIMIT 5")
                patients = cursor.fetchall()
                print(f"    Mẫu dữ liệu:")
                for patient in patients:
                    print(
                        f"      ID: {patient[0]}, Name: {patient[1]}, DOB: {patient[2]}, Gender: {patient[3]}"
                    )

        conn.close()

    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra database: {e}")


def check_dose_algorithms():
    """Kiểm tra các module dose calculation"""
    print("\n🧮 Kiểm tra các module dose calculation:")

    try:
        from quangtps.dose.dose_engine import DoseEngine, DoseCalculationAlgorithm

        # Khởi tạo dose engine
        engine = DoseEngine()
        available_algorithms = engine.get_available_algorithms()

        print(f"  ✅ DoseEngine khởi tạo thành công")
        print(f"  📋 Các thuật toán có sẵn ({len(available_algorithms)}):")

        for algo in available_algorithms:
            print(f"    - {algo.name}: {algo.value}")

        # Kiểm tra các module cụ thể
        modules_to_check = [
            ("quangtps.dose.algorithms.pencil_beam", "PencilBeamImplementer"),
            ("quangtps.dose.algorithms.aaa", "AAAImplementer"),
            ("quangtps.dose.algorithms.pencil_beam", "PencilBeamAlgorithm"),
            ("quangtps.dose.algorithms.collapsed_cone", "CollapsedConeAlgorithm"),
            ("quangtps.dose.algorithms.monte_carlo", "MonteCarloAlgorithm"),
        ]

        print(f"\n  🔍 Kiểm tra các class cụ thể:")
        for module_name, class_name in modules_to_check:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if hasattr(module, class_name):
                    print(f"    ✅ {module_name}.{class_name}")
                else:
                    print(f"    ❌ {module_name}.{class_name} - Class không tồn tại")
            except ImportError as e:
                print(f"    ❌ {module_name}.{class_name} - Import error: {e}")

    except Exception as e:
        print(f"  ❌ Lỗi khi kiểm tra dose algorithms: {e}")


def check_patient_service():
    """Kiểm tra patient service"""
    print("\n👤 Kiểm tra Patient Service:")

    try:
        from quangtps.database.patient_db import PatientDB

        patient_db = PatientDB()
        print(f"  ✅ PatientDB khởi tạo thành công")

        # Thử lấy danh sách bệnh nhân
        try:
            patients = patient_db.get_all_patients()
            print(f"  📋 Số lượng bệnh nhân trong database: {len(patients)}")

            if len(patients) > 0:
                print(f"  📝 Mẫu bệnh nhân:")
                for i, patient in enumerate(patients[:3]):  # Hiển thị 3 bệnh nhân đầu
                    patient_id = patient.get("id", "N/A")
                    patient_name = patient.get("name", "N/A")
                    print(f"    {i + 1}. ID: {patient_id}, Name: {patient_name}")
            else:
                print(f"  ⚠️  Không có bệnh nhân nào trong database")

        except Exception as e:
            print(f"  ❌ Lỗi khi lấy danh sách bệnh nhân: {e}")

    except Exception as e:
        print(f"  ❌ Lỗi khi khởi tạo PatientDB: {e}")


def main():
    """Hàm chính"""
    print("🔍 Kiểm tra hệ thống QuangTPS\n" + "=" * 50)

    # Kiểm tra database
    check_database()

    # Kiểm tra dose algorithms
    check_dose_algorithms()

    # Kiểm tra patient service
    check_patient_service()

    print("\n" + "=" * 50)
    print("✅ Hoàn tất kiểm tra hệ thống")


if __name__ == "__main__":
    main()
