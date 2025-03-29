#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module ví dụ cho việc sử dụng tính năng kiểm tra va chạm trong QuangTPS.

Mô-đun này minh họa cách sử dụng các lớp CollisionSimulator và CollisionDetector
để kiểm tra va chạm trong kế hoạch xạ trị.
"""

import os
import sys
import json
import numpy as np
import traceback
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Thêm thư mục gốc vào đường dẫn Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from quangtps.evaluation.qa.collision_check import (
        CollisionSimulator, CollisionDetector, CollisionType, 
        CollisionSeverity, CollisionEvent, MachinePart
    )
    from quangtps.treatment.machine.linac import Linac
except ImportError as e:
    print(f"Lỗi khi import: {e}")
    traceback.print_exc()
    sys.exit(1)


def create_sample_plan():
    """
    Tạo một kế hoạch điều trị mẫu với nhiều trường xạ trị ở các góc khác nhau.
    
    Returns
    -------
    dict
        Kế hoạch điều trị mẫu
    """
    plan = {
        "id": "SAMPLE001",
        "name": "Sample Plan for Collision Testing",
        "machine": "TrueBeam",
        "isocenter": [0, -100, 0],  # [x, y, z] in mm
        "patient_data": {
            "id": "PT001",
            "body_contour": {
                "type": "cylinder",
                "radius": 150,
                "height": 1800
            }
        },
        "fields": [
            # Trường thẳng góc - an toàn
            {
                "id": "Field1",
                "gantry_angle": 0.0,
                "couch_angle": 0.0,
                "collimator_angle": 0.0,
                "field_size": [100, 100],  # 10x10cm
                "table_position": {
                    "vertical": 0.0,
                    "lateral": 0.0,
                    "longitudinal": 0.0
                }
            },
            # Trường với bàn xoay - có thể va chạm
            {
                "id": "Field2",
                "gantry_angle": 30.0,
                "couch_angle": 90.0,
                "collimator_angle": 0.0,
                "field_size": [100, 100],
                "table_position": {
                    "vertical": 0.0,
                    "lateral": 0.0,
                    "longitudinal": 0.0
                }
            },
            # Trường với gantry gần ngang - có thể va chạm với bàn
            {
                "id": "Field3",
                "gantry_angle": 95.0,
                "couch_angle": 0.0,
                "collimator_angle": 0.0,
                "field_size": [150, 150],
                "table_position": {
                    "vertical": 50.0,
                    "lateral": 0.0,
                    "longitudinal": 0.0
                }
            },
            # Trường với gantry và bàn đều xoay - khả năng va chạm cao
            {
                "id": "Field4",
                "gantry_angle": 135.0,
                "couch_angle": 45.0,
                "collimator_angle": 90.0,
                "field_size": [120, 120],
                "table_position": {
                    "vertical": 0.0,
                    "lateral": 50.0,
                    "longitudinal": 0.0
                }
            },
            # Trường xoay ngược - ít khả năng va chạm
            {
                "id": "Field5",
                "gantry_angle": 180.0,
                "couch_angle": 0.0,
                "collimator_angle": 0.0,
                "field_size": [100, 100],
                "table_position": {
                    "vertical": 0.0,
                    "lateral": 0.0,
                    "longitudinal": 0.0
                }
            },
            # Trường với bàn và gantry ở vị trí có nguy cơ va chạm cao
            {
                "id": "Field6",
                "gantry_angle": 80.0,
                "couch_angle": 70.0,
                "collimator_angle": 0.0,
                "field_size": [100, 100],
                "table_position": {
                    "vertical": 100.0,
                    "lateral": 150.0,
                    "longitudinal": 200.0
                }
            }
        ]
    }
    
    return plan


def save_sample_plan(plan, filename="sample_plan.json"):
    """
    Lưu kế hoạch mẫu vào file JSON.
    
    Parameters
    ----------
    plan : dict
        Kế hoạch điều trị
    filename : str, optional
        Tên file, mặc định là "sample_plan.json"
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2)
        
        print(f"Đã lưu kế hoạch mẫu vào file: {filename}")
    except Exception as e:
        print(f"Lỗi khi lưu kế hoạch: {e}")


def visualize_collision_check(detector, field):
    """
    Hiển thị mô phỏng kiểm tra va chạm cho một trường xạ trị.
    
    Parameters
    ----------
    detector : CollisionDetector
        Bộ phát hiện va chạm
    field : dict
        Thông tin trường xạ trị
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Lấy thông tin trường xạ trị
        gantry_angle = field.get("gantry_angle", 0.0)
        couch_angle = field.get("couch_angle", 0.0)
        
        # Vẽ hệ tọa độ
        ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1)
        
        # Vẽ isocenter
        ax.scatter([0], [0], [0], color='k', s=50)
        
        # Lấy vị trí đầu gantry
        gantry_head_pos = detector.simulator.get_gantry_head_position(gantry_angle)
        
        # Vẽ đầu gantry
        ax.scatter(
            gantry_head_pos[0], 
            gantry_head_pos[1], 
            gantry_head_pos[2], 
            color='blue', 
            s=100, 
            label=f'Gantry Head ({gantry_angle}°)'
        )
        
        # Vẽ vector từ isocenter đến đầu gantry
        ax.plot(
            [0, gantry_head_pos[0]], 
            [0, gantry_head_pos[1]], 
            [0, gantry_head_pos[2]], 
            'b--'
        )
        
        # Lấy thông tin bàn điều trị
        table_position = field.get("table_position", {})
        vertical = table_position.get("vertical", 0.0)
        lateral = table_position.get("lateral", 0.0)
        longitudinal = table_position.get("longitudinal", 0.0)
        
        # Lấy vị trí bàn điều trị
        couch_pos = detector.simulator.get_couch_position(
            couch_angle, vertical, lateral, longitudinal
        )
        
        # Vẽ bàn điều trị
        ax.scatter(
            couch_pos[0], 
            couch_pos[1], 
            couch_pos[2], 
            color='green', 
            s=100, 
            label=f'Couch ({couch_angle}°)'
        )
        
        # Vẽ vector từ isocenter đến bàn điều trị
        ax.plot(
            [0, couch_pos[0]], 
            [0, couch_pos[1]], 
            [0, couch_pos[2]], 
            'g--'
        )
        
        # Lấy vị trí bệnh nhân
        patient_pos = detector.simulator.patient_model["body"]["center"].copy()
        patient_pos[0] += lateral
        patient_pos[1] += vertical
        patient_pos[2] += longitudinal
        
        # Điều chỉnh vị trí bệnh nhân nếu bàn quay
        if couch_angle != 0:
            angle_rad = np.radians(couch_angle)
            x, z = patient_pos[0], patient_pos[2]
            patient_pos[0] = x * np.cos(angle_rad) - z * np.sin(angle_rad)
            patient_pos[2] = x * np.sin(angle_rad) + z * np.cos(angle_rad)
        
        # Vẽ bệnh nhân (đơn giản hóa là một điểm)
        ax.scatter(
            patient_pos[0], 
            patient_pos[1], 
            patient_pos[2], 
            color='red', 
            s=100, 
            label='Patient'
        )
        
        # Tính khoảng cách giữa đầu gantry và bàn điều trị
        distance_gantry_couch = np.linalg.norm(gantry_head_pos - couch_pos)
        
        # Tính khoảng cách giữa đầu gantry và bệnh nhân
        distance_gantry_patient = np.linalg.norm(gantry_head_pos - patient_pos)
        
        # Thiết lập giới hạn đồ thị
        ax.set_xlim(-700, 700)
        ax.set_ylim(-700, 700)
        ax.set_zlim(-700, 700)
        
        # Thiết lập nhãn
        ax.set_xlabel('X axis (mm)')
        ax.set_ylabel('Y axis (mm)')
        ax.set_zlabel('Z axis (mm)')
        
        # Tiêu đề
        field_id = field.get("id", "Unknown")
        ax.set_title(f'Collision Check Visualization - Field {field_id}\n'
                    f'Gantry: {gantry_angle}°, Couch: {couch_angle}°\n'
                    f'Distance Gantry-Couch: {distance_gantry_couch:.2f}mm, '
                    f'Gantry-Patient: {distance_gantry_patient:.2f}mm')
        
        # Thêm chú thích
        ax.legend()
        
        # Hiển thị lưới
        ax.grid(True)
        
        plt.tight_layout()
        
        # Hiển thị đồ thị
        plt.show()
    except Exception as e:
        print(f"Lỗi khi hiển thị mô phỏng: {e}")
        traceback.print_exc()


def create_linac_machine():
    """
    Tạo đối tượng máy Linac để sử dụng trong kiểm tra va chạm.
    
    Returns
    -------
    Linac
        Đối tượng máy Linac
    """
    try:
        from quangtps.treatment.machine.machine_status import MachineStatus
        import datetime
        
        # Tạo một máy Linac mẫu (TrueBeam của Varian)
        linac = Linac(
            name="TrueBeam", 
            machine_id="TB001",
            manufacturer="Varian",
            model="TrueBeam",
            installation_date=datetime.date(2022, 1, 1),
            status=MachineStatus.OPERATIONAL
        )
        
        return linac
    except Exception as e:
        print(f"Lỗi khi tạo máy Linac: {e}")
        traceback.print_exc()
        return None


def run_example():
    """
    Chạy ví dụ kiểm tra va chạm.
    """
    try:
        print("=== Ví dụ kiểm tra va chạm (Collision Check) trong QuangTPS ===")
        
        # Tạo kế hoạch mẫu
        print("\nTạo kế hoạch mẫu...")
        plan = create_sample_plan()
        save_sample_plan(plan)
        
        # Tạo máy Linac
        linac = create_linac_machine()
        if not linac:
            print("Không thể tiếp tục do lỗi khi tạo máy Linac")
            return
        
        # Tạo bộ phát hiện va chạm
        print("\nThiết lập bộ phát hiện va chạm...")
        detector = CollisionDetector(linac)
        
        # Kiểm tra từng trường xạ trị
        print("\nKiểm tra va chạm cho từng trường xạ trị:")
        for i, field in enumerate(plan["fields"]):
            print(f"\nTrường {i+1} ({field['id']}):")
            print(f"  Góc gantry: {field['gantry_angle']}°")
            print(f"  Góc bàn điều trị: {field['couch_angle']}°")
            print(f"  Kích thước trường: {field['field_size'][0]/10} x {field['field_size'][1]/10} cm")
            
            # Lấy thông tin vị trí bàn
            table_pos = field["table_position"]
            
            # Kiểm tra va chạm
            collisions = detector.check_field(
                field["gantry_angle"],
                field["couch_angle"],
                field["collimator_angle"],
                field["field_size"],
                table_pos["vertical"],
                table_pos["lateral"],
                table_pos["longitudinal"]
            )
            
            # Hiển thị kết quả
            if not collisions:
                print("  → Không phát hiện va chạm")
            else:
                print(f"  → Phát hiện {len(collisions)} va chạm:")
                for j, collision in enumerate(collisions):
                    print(f"    Va chạm {j+1}:")
                    print(f"      Loại: {collision.collision_type.name}")
                    print(f"      Khoảng cách: {collision.distance:.2f} mm")
                    print(f"      Mức độ: {collision.severity.value}")
                    print(f"      Mô tả: {collision.description}")
        
        # Kiểm tra toàn bộ kế hoạch
        print("\nKiểm tra va chạm cho toàn bộ kế hoạch...")
        collisions = detector.check_plan(plan)
        
        # Lưu báo cáo
        print("\nTạo báo cáo va chạm...")
        summary = detector.generate_collision_report(plan, "collision_report.json")
        
        # Hiển thị tổng kết
        print("\nTổng kết:")
        print(f"  Tổng số trường xạ trị: {summary['total_fields']}")
        print(f"  Tổng số va chạm: {summary['total_collisions']}")
        print(f"  Số va chạm nghiêm trọng: {summary['critical_collisions']}")
        print(f"  Số va chạm cảnh báo: {summary['warning_collisions']}")
        print(f"  Số va chạm thông tin: {summary['info_collisions']}")
        print(f"  Báo cáo đã được lưu vào: {summary['report_file']}")
        
        # Hiển thị mô phỏng cho một số trường
        print("\nHiển thị mô phỏng cho một số trường xạ trị...")
        for field in [plan["fields"][0], plan["fields"][2], plan["fields"][5]]:
            print(f"\nMô phỏng cho trường {field['id']}...")
            visualize_collision_check(detector, field)
            
    except Exception as e:
        print(f"Lỗi trong quá trình chạy ví dụ: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    run_example() 