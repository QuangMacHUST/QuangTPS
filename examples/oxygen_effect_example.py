#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ví dụ về cách sử dụng module hiệu ứng oxy (Oxygen Effect) trong phân tích kế hoạch xạ trị.

Module này minh họa các phương pháp đánh giá ảnh hưởng của nồng độ oxy lên hiệu quả
của kế hoạch xạ trị và cách đánh giá các kế hoạch xạ trị dựa trên thông tin về oxy hóa.
"""

import numpy as np
import matplotlib.pyplot as plt
from quangtps.evaluation.biological import OxygenEffect, calculate_oxygen_effect


def plot_oer_curve():
    """
    Vẽ đồ thị minh họa mối quan hệ giữa nồng độ oxy và OER.
    """
    # Tạo mảng các giá trị nồng độ oxy từ 0 đến 60 mmHg
    oxygen_concentrations = np.linspace(0, 60, 100)
    
    # Tính OER cho mỗi nồng độ oxy
    oer_values = [OxygenEffect.calculate_oer(p) for p in oxygen_concentrations]
    
    # Tạo đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(oxygen_concentrations, oer_values, 'b-', linewidth=2)
    plt.xlabel('Nồng độ oxy (mmHg)')
    plt.ylabel('Tỷ số tăng cường oxy (OER)')
    plt.title('Mối quan hệ giữa nồng độ oxy và OER')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # Thêm nhãn cho vùng thiếu oxy
    plt.axvspan(0, 10, alpha=0.2, color='red', label='Thiếu oxy')
    plt.axvspan(10, 30, alpha=0.2, color='yellow', label='Chuyển tiếp')
    plt.axvspan(30, 60, alpha=0.2, color='green', label='Đủ oxy')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('oer_curve.png')
    plt.show()


def compare_dose_effectiveness():
    """
    So sánh hiệu quả của liều bức xạ ở các nồng độ oxy khác nhau.
    """
    # Tạo mảng các giá trị liều
    doses = np.linspace(0, 10, 50)
    
    # Các nồng độ oxy khác nhau để so sánh
    oxygen_concentrations = [2, 5, 10, 20, 50]
    colors = ['r', 'orange', 'g', 'b', 'purple']
    
    plt.figure(figsize=(10, 6))
    
    for i, p_o2 in enumerate(oxygen_concentrations):
        # Tính phân số sống sót tế bào cho mỗi liều
        survival = [OxygenEffect.calculate_oxygen_modified_survival(d, p_o2) for d in doses]
        plt.semilogy(doses, survival, color=colors[i], linewidth=2, 
                   label=f'pO₂ = {p_o2} mmHg')
    
    plt.xlabel('Liều (Gy)')
    plt.ylabel('Phân số sống sót')
    plt.title('Ảnh hưởng của oxy lên hiệu quả của liều bức xạ')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.savefig('oxygen_dose_effectiveness.png')
    plt.show()


def analyze_hypoxic_tumor():
    """
    Phân tích ảnh hưởng của phân đoạn thiếu oxy trong khối u.
    """
    # Các giá trị phân đoạn thiếu oxy khác nhau
    hypoxic_fractions = [0, 0.1, 0.2, 0.5, 0.8]
    # Liều bức xạ
    doses = np.linspace(0, 15, 50)
    
    plt.figure(figsize=(10, 6))
    
    for hf in hypoxic_fractions:
        survival = [OxygenEffect.calculate_hypoxic_fraction_effect(d, hf) for d in doses]
        plt.semilogy(doses, survival, linewidth=2, label=f'HF = {hf}')
    
    # Tìm liều cần để đạt được sống sót 10^-6 ở các phân đoạn thiếu oxy khác nhau
    plt.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='SF = 10⁻⁶')
    
    plt.xlabel('Liều (Gy)')
    plt.ylabel('Phân số sống sót')
    plt.title('Ảnh hưởng của phân đoạn thiếu oxy lên hiệu quả xạ trị')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(1e-7, 1)
    plt.tight_layout()
    plt.savefig('hypoxic_fraction_effect.png')
    plt.show()


def display_reoxygenation_dynamics():
    """
    Hiển thị động học tái oxy hóa trong quá trình xạ trị phân liều.
    """
    # Thời gian điều trị (ngày)
    treatment_days = np.linspace(0, 42, 100)
    
    # Các thời gian bán rã tái oxy hóa khác nhau (ngày)
    half_times = [3, 7, 14, 21]
    initial_hf = 0.3  # Phân đoạn thiếu oxy ban đầu
    
    plt.figure(figsize=(10, 6))
    
    for t_half in half_times:
        hypoxic_fractions = OxygenEffect.calculate_reoxygenation_dynamics(
            initial_hf, treatment_days, t_half)
        plt.plot(treatment_days, hypoxic_fractions, linewidth=2, 
               label=f'T₁/₂ = {t_half} ngày')
    
    # Thêm đánh dấu cho lịch phân liều điển hình (5 phân liều mỗi tuần)
    fraction_days = np.arange(1, 41, 7/5)
    fraction_days = np.delete(fraction_days, np.where((fraction_days % 7) >= 5)[0])
    plt.scatter(fraction_days, [0.02] * len(fraction_days), marker='|', color='r', s=50,
              label='Các phân liều')
    
    plt.xlabel('Thời gian (ngày)')
    plt.ylabel('Phân đoạn thiếu oxy')
    plt.title('Động học tái oxy hóa trong quá trình xạ trị')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(0, initial_hf * 1.1)
    plt.tight_layout()
    plt.savefig('reoxygenation_dynamics.png')
    plt.show()


def evaluate_treatment_plan():
    """
    Đánh giá kế hoạch xạ trị có xét đến hiệu ứng oxy.
    """
    # Thông tin kế hoạch
    dose = 60.0  # Tổng liều (Gy)
    fractions = 30  # Số phân liều
    dose_per_fraction = dose / fractions
    
    # Các vùng khối u có nồng độ oxy khác nhau
    regions = [
        {"name": "Vùng đủ oxy", "oxygen": 40, "volume": 0.5},
        {"name": "Vùng thiếu oxy trung bình", "oxygen": 10, "volume": 0.3},
        {"name": "Vùng thiếu oxy nặng", "oxygen": 2, "volume": 0.2}
    ]
    
    print("ĐÁNH GIÁ KẾ HOẠCH XẠ TRỊ CÓ XÉT ĐẾN HIỆU ỨNG OXY")
    print("=" * 50)
    print(f"Tổng liều: {dose} Gy")
    print(f"Số phân liều: {fractions}")
    print(f"Liều mỗi phân liều: {dose_per_fraction:.2f} Gy")
    print("-" * 50)
    
    weighted_survival = 0
    
    for region in regions:
        print(f"\nVùng: {region['name']}")
        print(f"Nồng độ oxy: {region['oxygen']} mmHg")
        
        # Đánh giá trạng thái oxy hóa
        status = OxygenEffect.evaluate_oxygen_status(region['oxygen'])
        print(f"Trạng thái: {status['status']} ({status['description']})")
        print(f"Độ nhạy cảm tương đối: {status['relative_radiosensitivity']:.2f}")
        print(f"OER: {status['oer']:.2f}")
        
        # Tính toán liều hiệu quả
        effective_dose = OxygenEffect.calculate_oer_effective_dose(dose, region['oxygen'])
        print(f"Liều hiệu quả: {effective_dose:.2f} Gy (so với {dose} Gy liều vật lý)")
        
        # Tính toán sự sống sót tế bào
        survival = OxygenEffect.calculate_oxygen_modified_survival(dose, region['oxygen'])
        print(f"Phân số sống sót tế bào: {survival:.2e}")
        
        weighted_survival += survival * region['volume']
    
    print("\n" + "-" * 50)
    print(f"Phân số sống sót tế bào trung bình (có trọng số): {weighted_survival:.2e}")
    
    # Ước tính TCP đơn giản dựa trên phân số sống sót
    tcp = np.exp(-weighted_survival * 10**7) if weighted_survival > 0 else 1.0
    print(f"Ước tính TCP: {tcp:.4f}")
    
    # Tạo báo cáo tổng hợp
    oxygen_report = OxygenEffect.generate_oxygen_effect_report(
        sum(r['oxygen'] * r['volume'] for r in regions) / sum(r['volume'] for r in regions),
        dose,
        sum(r['volume'] for r in regions if r['oxygen'] < 10) / sum(r['volume'] for r in regions)
    )
    
    print("\nBÁO CÁO TỔNG HỢP HIỆU ỨNG OXY")
    print("=" * 50)
    print(f"Nồng độ oxy trung bình: {oxygen_report['oxygen_concentration']:.2f} mmHg")
    print(f"Phân đoạn thiếu oxy: {oxygen_report['hypoxic_fraction']:.2f}")
    print(f"OER trung bình: {oxygen_report['oer']:.2f}")
    print(f"Liều hiệu quả: {oxygen_report['effective_dose']:.2f} Gy")
    print(f"Diễn giải: {oxygen_report['interpretation']}")


def main():
    """
    Hàm chính thực thi các ví dụ hiệu ứng oxy.
    """
    print("MINH HỌA HIỆU ỨNG OXY TRONG XẠ TRỊ")
    print("=" * 50)
    
    # 1. Vẽ đồ thị mối quan hệ giữa nồng độ oxy và OER
    print("\n1. Mối quan hệ giữa nồng độ oxy và OER")
    plot_oer_curve()
    
    # 2. So sánh hiệu quả của liều ở các nồng độ oxy khác nhau
    print("\n2. So sánh hiệu quả của liều ở các nồng độ oxy khác nhau")
    compare_dose_effectiveness()
    
    # 3. Phân tích ảnh hưởng của phân đoạn thiếu oxy
    print("\n3. Phân tích ảnh hưởng của phân đoạn thiếu oxy")
    analyze_hypoxic_tumor()
    
    # 4. Hiển thị động học tái oxy hóa
    print("\n4. Động học tái oxy hóa trong quá trình xạ trị")
    display_reoxygenation_dynamics()
    
    # 5. Đánh giá kế hoạch xạ trị có xét đến hiệu ứng oxy
    print("\n5. Đánh giá kế hoạch xạ trị có xét đến hiệu ứng oxy")
    evaluate_treatment_plan()


if __name__ == "__main__":
    main() 