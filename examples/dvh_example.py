#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ví dụ minh họa cách sử dụng module DVH trong hệ thống QuangTPS.

File này bao gồm các ví dụ về cách:
- Tính toán DVH từ dữ liệu 3D
- Phân tích DVH với các chỉ số đánh giá
- Vẽ biểu đồ DVH đơn và nhiều DVH
- Tạo báo cáo DVH đầy đủ
- Xuất dữ liệu DVH
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Import các module cần thiết từ QuangTPS
from quangtps.evaluation.dvh import (
    calculate_dvh,
    calculate_dvh_metrics,
    calculate_dvh_from_dose_grid,
    DVHAnalysis,
    plot_dvh,
    plot_multiple_dvh,
    create_dvh_report,
    export_dvh_to_csv
)
from quangtps.dose.dose_grid import DoseGrid

def create_sample_data():
    """Tạo dữ liệu mẫu để minh họa."""
    # Tạo mảng dose 3D mẫu (60x60x60 voxels)
    # Giả lập một phân bố liều chùm tia đơn giản (giảm theo khoảng cách)
    size = 60
    dose_array = np.zeros((size, size, size), dtype=np.float32)
    
    # Tạo một phân bố liều với gradient từ trung tâm
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size)
    )
    
    # Tính khoảng cách từ trung tâm
    dist = np.sqrt(x**2 + y**2 + z**2)
    
    # Tạo phân bố liều theo công thức: 50 * exp(-5*dist)
    # Cho liều trung tâm là 50 Gy và giảm dần theo đường cong mũ
    dose_array = 50 * np.exp(-5 * dist)
    
    # Tạo các cấu trúc mẫu:
    # 1. PTV: hình cầu ở trung tâm
    ptv_mask = dist < 0.3
    
    # 2. OAR 1: hình cầu gần PTV
    oar1_center = np.array([0.4, 0.4, 0.4])
    oar1_x = x - oar1_center[0]
    oar1_y = y - oar1_center[1]
    oar1_z = z - oar1_center[2]
    oar1_dist = np.sqrt(oar1_x**2 + oar1_y**2 + oar1_z**2)
    oar1_mask = oar1_dist < 0.2
    
    # 3. OAR 2: hình cầu xa PTV
    oar2_center = np.array([-0.5, -0.5, -0.5])
    oar2_x = x - oar2_center[0]
    oar2_y = y - oar2_center[1]
    oar2_z = z - oar2_center[2]
    oar2_dist = np.sqrt(oar2_x**2 + oar2_y**2 + oar2_z**2)
    oar2_mask = oar2_dist < 0.15
    
    # Đảm bảo không có phần giao nhau
    oar1_mask = oar1_mask & (~ptv_mask)
    oar2_mask = oar2_mask & (~ptv_mask) & (~oar1_mask)
    
    return dose_array, ptv_mask, oar1_mask, oar2_mask

def example_basic_dvh_calculation():
    """Ví dụ về cách tính toán DVH cơ bản."""
    print("\n=== Ví dụ 1: Tính toán DVH cơ bản ===")
    
    # Tạo dữ liệu mẫu
    dose_array, ptv_mask, oar1_mask, oar2_mask = create_sample_data()
    
    # Tính DVH cho PTV
    ptv_dvh = calculate_dvh(
        dose_array=dose_array,
        structure_mask=ptv_mask,
        num_bins=100,
        dose_unit='Gy',
        volume_type='relative',
        verbose=True
    )
    
    # In thông tin chính
    print(f"PTV DVH Info:")
    print(f"  - Min dose: {ptv_dvh['min_dose']:.2f} Gy")
    print(f"  - Max dose: {ptv_dvh['max_dose']:.2f} Gy")
    print(f"  - Mean dose: {ptv_dvh['mean_dose']:.2f} Gy")
    
    # Tính chỉ số DVH
    prescription_dose = 50.0  # Gy
    metrics = ['D98', 'D95', 'D50', 'D2', 'V95', 'V100', 'V105']
    ptv_metrics = calculate_dvh_metrics(ptv_dvh, metrics, prescription_dose)
    
    # In chỉ số DVH
    print("\nPTV Metrics:")
    for metric, value in ptv_metrics.items():
        if metric.startswith('D'):
            print(f"  - {metric}: {value:.2f} Gy")
        elif metric.startswith('V'):
            print(f"  - {metric}: {value:.2f} %")
    
    return ptv_dvh, oar1_mask, oar2_mask, dose_array

def example_dvh_analysis():
    """Ví dụ về cách phân tích DVH nâng cao."""
    print("\n=== Ví dụ 2: Phân tích DVH nâng cao ===")
    
    ptv_dvh, oar1_mask, oar2_mask, dose_array = example_basic_dvh_calculation()
    
    # Tính DVH cho OAR1 và OAR2
    oar1_dvh = calculate_dvh(dose_array, oar1_mask, num_bins=100)
    oar2_dvh = calculate_dvh(dose_array, oar2_mask, num_bins=100)
    
    # Sử dụng DVHAnalysis để phân tích nâng cao
    prescription_dose = 50.0  # Gy
    ptv_analyzer = DVHAnalysis(ptv_dvh, "PTV")
    oar1_analyzer = DVHAnalysis(oar1_dvh, "OAR1")
    
    # Tính chỉ số đồng nhất HI
    hi = ptv_analyzer.get_homogeneity_index(prescription_dose, method='icru83')
    print(f"PTV Homogeneity Index (ICRU83): {hi:.4f}")
    
    # Tính chỉ số phù hợp CI
    ci = ptv_analyzer.get_conformity_index(prescription_dose, method='paddick')
    print(f"PTV Conformity Index (Paddick): {ci:.4f}")
    
    # Tính chỉ số gradient GI
    gi = ptv_analyzer.get_gradient_index(prescription_dose, prescription_dose * 0.5)
    print(f"PTV Gradient Index (R50%): {gi:.4f}")
    
    # Tính EUD (Equivalent Uniform Dose) cho PTV và OAR1
    ptv_eud = ptv_analyzer.get_equivalent_uniform_dose(parameter_a=1.0)
    oar1_eud = oar1_analyzer.get_equivalent_uniform_dose(parameter_a=-5.0)
    print(f"PTV EUD (a=1.0): {ptv_eud:.2f} Gy")
    print(f"OAR1 EUD (a=-5.0): {oar1_eud:.2f} Gy")
    
    # Tính liều tích phân (Integral Dose)
    ptv_integral = ptv_analyzer.get_integral_dose(density=1.0)
    print(f"PTV Integral Dose: {ptv_integral:.2f} Gy*cc")
    
    return ptv_dvh, oar1_dvh, oar2_dvh

def example_plot_dvh():
    """Ví dụ về cách vẽ biểu đồ DVH."""
    print("\n=== Ví dụ 3: Vẽ biểu đồ DVH ===")
    
    ptv_dvh, oar1_dvh, oar2_dvh = example_dvh_analysis()
    
    # Vẽ DVH đơn
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_dvh(
        dvh_data=ptv_dvh,
        structure_name="PTV",
        dvh_type='cumulative',
        ax=ax,
        color='red',
        show_metrics=True,
        metrics_to_show=['D95', 'D50'],
        prescription_dose=50.0
    )
    plt.tight_layout()
    plt.savefig("ptv_dvh.png", dpi=300)
    print("Đã lưu biểu đồ DVH đơn vào 'ptv_dvh.png'")
    
    # Vẽ nhiều DVH trên cùng một biểu đồ
    dvh_list = [ptv_dvh, oar1_dvh, oar2_dvh]
    structure_names = ["PTV", "OAR1", "OAR2"]
    fig, ax = plot_multiple_dvh(
        dvh_list=dvh_list,
        structure_names=structure_names,
        structure_colors={"PTV": "red", "OAR1": "blue", "OAR2": "green"},
        title="Cumulative DVH",
        legend_loc="lower left",
        save_path="multiple_dvh.png"
    )
    print("Đã lưu biểu đồ nhiều DVH vào 'multiple_dvh.png'")
    
    return dvh_list, structure_names

def example_create_report():
    """Ví dụ về cách tạo báo cáo DVH đầy đủ."""
    print("\n=== Ví dụ 4: Tạo báo cáo DVH đầy đủ ===")
    
    dvh_list, structure_names = example_plot_dvh()
    
    # Tạo thư mục cho output nếu chưa tồn tại
    output_dir = "dvh_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo báo cáo DVH
    prescription_doses = {"PTV": 50.0}
    structure_types = {"PTV": "target", "OAR1": "oar", "OAR2": "oar"}
    
    report = create_dvh_report(
        dvh_list=dvh_list,
        structure_names=structure_names,
        prescription_doses=prescription_doses,
        structure_types=structure_types,
        output_path=output_dir,
        plot_differential=True,
        show_statistics=True
    )
    
    # Xuất dữ liệu DVH ra file CSV
    csv_path = os.path.join(output_dir, "dvh_data.csv")
    export_dvh_to_csv(
        dvh_list=dvh_list,
        structure_names=structure_names,
        output_path=csv_path,
        include_metrics=True,
        prescription_doses=prescription_doses
    )
    
    print(f"Đã tạo báo cáo DVH trong thư mục '{output_dir}'")
    print(f"Đã xuất dữ liệu DVH ra file '{csv_path}'")
    
    # In các chỉ số chính trong báo cáo
    if 'metrics_df' in report:
        print("\nChỉ số đánh giá DVH:")
        print(report['metrics_df'])
    
    return report

def example_dvh_from_dose_grid():
    """Ví dụ về cách tính DVH từ đối tượng DoseGrid."""
    print("\n=== Ví dụ 5: Tính DVH từ DoseGrid ===")
    
    # Tạo dữ liệu mẫu
    dose_array, ptv_mask, _, _ = create_sample_data()
    
    # Tạo đối tượng DoseGrid
    dose_grid = DoseGrid.from_array(
        dose_array=dose_array,
        voxel_size=[2.0, 2.0, 2.0],  # mm
        origin=[-60.0, -60.0, -60.0]  # mm
    )
    
    # Tính DVH từ DoseGrid
    ptv_dvh_from_grid = calculate_dvh_from_dose_grid(
        dose_grid=dose_grid,
        structure_mask=ptv_mask,
        num_bins=100,
        volume_type='relative'
    )
    
    # In thông tin
    print(f"PTV DVH từ DoseGrid:")
    print(f"  - Min dose: {ptv_dvh_from_grid['min_dose']:.2f} Gy")
    print(f"  - Max dose: {ptv_dvh_from_grid['max_dose']:.2f} Gy")
    print(f"  - Mean dose: {ptv_dvh_from_grid['mean_dose']:.2f} Gy")
    print(f"  - Thể tích PTV: {ptv_dvh_from_grid['structure_volume_cc']:.2f} cc")
    
    return ptv_dvh_from_grid

def run_all_examples():
    """Chạy tất cả các ví dụ."""
    print("=== Chạy các ví dụ về module DVH ===")
    
    example_basic_dvh_calculation()
    example_dvh_analysis()
    example_plot_dvh()
    example_create_report()
    example_dvh_from_dose_grid()
    
    print("\n=== Hoàn thành tất cả các ví dụ ===")

if __name__ == "__main__":
    run_all_examples() 