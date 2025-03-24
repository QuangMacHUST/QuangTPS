#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo báo cáo kế hoạch điều trị.

Module này cung cấp các lớp và hàm để tạo báo cáo chi tiết về kế hoạch
xạ trị, bao gồm thông tin bệnh nhân, kế hoạch, liều lượng, DVH, và các
thông số đánh giá khác.
"""

import os
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Lớp tạo báo cáo kế hoạch điều trị.
    
    Lớp này cung cấp các phương thức để tạo báo cáo chi tiết về kế hoạch xạ trị,
    bao gồm thông tin bệnh nhân, thông tin kế hoạch, biểu đồ DVH, và các
    thông số đánh giá.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Khởi tạo lớp ReportGenerator.
        
        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra để lưu báo cáo
        """
        self.output_dir = output_dir or os.path.expanduser("~/Documents/QuangTPS/Reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Khởi tạo các thành phần phụ
        self.dvh_calculator = DVHCalculator()
        
    def generate_report(self, patient: Patient, plan: Plan, output_file: Optional[str] = None) -> str:
        """
        Tạo báo cáo kế hoạch điều trị.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        output_file : str, optional
            Tên file đầu ra, mặc định là 'PatientID_PlanName_Report.pdf'
            
        Returns
        -------
        str
            Đường dẫn đến file báo cáo đã tạo
        """
        # Xác định tên file đầu ra
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{patient.patient_id}_{plan.name}_Report_{timestamp}.pdf"
            
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # Tạo file PDF
            with PdfPages(output_path) as pdf:
                # Trang bìa
                self._create_cover_page(pdf, patient, plan)
                
                # Trang thông tin bệnh nhân
                self._create_patient_info_page(pdf, patient)
                
                # Trang thông tin kế hoạch
                self._create_plan_info_page(pdf, plan)
                
                # Trang biểu đồ DVH
                self._create_dvh_page(pdf, plan)
                
                # Trang thông số đánh giá
                self._create_evaluation_page(pdf, plan)
                
                # Trang liều theo từng cấu trúc
                self._create_structure_dose_page(pdf, plan)
                
                # Trang thông tin chùm tia
                self._create_beam_info_page(pdf, plan)
            
            logger.info(f"Đã tạo báo cáo: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}", exc_info=True)
            raise
    
    def _create_cover_page(self, pdf: PdfPages, patient: Patient, plan: Plan):
        """
        Tạo trang bìa báo cáo.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.9, "BÁO CÁO KẾ HOẠCH XẠ TRỊ", 
                 fontsize=24, fontweight='bold', ha='center')
        
        # Logo hoặc hình ảnh
        # plt.figimage(...)
        
        # Thông tin bệnh nhân
        plt.text(0.5, 0.7, f"Bệnh nhân: {patient.name}", fontsize=14, ha='center')
        plt.text(0.5, 0.65, f"ID: {patient.patient_id}", fontsize=14, ha='center')
        
        # Thông tin kế hoạch
        plt.text(0.5, 0.55, f"Kế hoạch: {plan.name}", fontsize=14, ha='center')
        plt.text(0.5, 0.5, f"Ngày tạo: {plan.creation_date}", fontsize=14, ha='center')
        
        # Thông tin cơ sở
        plt.text(0.5, 0.3, "TRUNG TÂM XẠ TRỊ", fontsize=18, ha='center')
        plt.text(0.5, 0.25, "BỆNH VIỆN QUANG TPS", fontsize=16, ha='center')
        
        # Ngày tạo báo cáo
        current_date = datetime.datetime.now().strftime("%d/%m/%Y")
        plt.text(0.5, 0.1, f"Ngày báo cáo: {current_date}", fontsize=12, ha='center')
        
        # Tắt trục
        plt.axis('off')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _create_patient_info_page(self, pdf: PdfPages, patient: Patient):
        """
        Tạo trang thông tin bệnh nhân.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        patient : Patient
            Đối tượng bệnh nhân
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.95, "THÔNG TIN BỆNH NHÂN", 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Thông tin cơ bản
        y_pos = 0.85
        for label, value in [
            ("Họ và tên", patient.name),
            ("ID", patient.patient_id),
            ("Ngày sinh", patient.date_of_birth),
            ("Giới tính", patient.gender),
            ("Địa chỉ", patient.address),
            ("Chẩn đoán", patient.diagnosis),
            ("Bác sĩ phụ trách", patient.physician),
            ("Ngày nhập viện", patient.admission_date)
        ]:
            plt.text(0.2, y_pos, f"{label}:", fontsize=12, ha='right')
            plt.text(0.25, y_pos, f"{value}", fontsize=12, ha='left')
            y_pos -= 0.06
        
        # Tắt trục
        plt.axis('off')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _create_plan_info_page(self, pdf: PdfPages, plan: Plan):
        """
        Tạo trang thông tin kế hoạch.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        plan : Plan
            Kế hoạch điều trị
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.95, "THÔNG TIN KẾ HOẠCH", 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Thông tin cơ bản
        y_pos = 0.85
        for label, value in [
            ("Tên kế hoạch", plan.name),
            ("Ngày tạo", plan.creation_date),
            ("Mô tả", plan.description),
            ("Loại kế hoạch", plan.plan_type),
            ("Tổng liều", f"{plan.total_dose:.2f} Gy"),
            ("Số phân liều", plan.number_of_fractions),
            ("Liều mỗi phân liều", f"{plan.dose_per_fraction:.2f} Gy"),
            ("Thuật toán tính liều", plan.dose_algorithm),
            ("Thời gian điều trị", plan.treatment_time),
            ("Trạng thái", plan.status)
        ]:
            plt.text(0.2, y_pos, f"{label}:", fontsize=12, ha='right')
            plt.text(0.25, y_pos, f"{value}", fontsize=12, ha='left')
            y_pos -= 0.05
        
        # Kê đơn
        plt.text(0.5, 0.3, "THÔNG TIN KÊ ĐƠN", 
                 fontsize=14, fontweight='bold', ha='center')
        
        # Bảng kê đơn
        if hasattr(plan, 'prescriptions') and plan.prescriptions:
            headers = ["Cấu trúc", "Liều (Gy)", "Thể tích (%)", "Ưu tiên"]
            cell_text = []
            
            for p in plan.prescriptions:
                cell_text.append([
                    p.structure_name,
                    f"{p.dose:.2f}",
                    f"{p.volume:.1f}",
                    str(p.priority)
                ])
            
            plt.table(cellText=cell_text, colLabels=headers, 
                     loc='center', cellLoc='center',
                     bbox=[0.1, 0.05, 0.8, 0.2])
        
        # Tắt trục
        plt.axis('off')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _create_dvh_page(self, pdf: PdfPages, plan: Plan):
        """
        Tạo trang biểu đồ DVH.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        plan : Plan
            Kế hoạch điều trị
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.95, "BIỂU ĐỒ HISTOGRAM LIỀU THỂ TÍCH (DVH)", 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Lấy dữ liệu DVH
        dvh_data = plan.dvh_data if hasattr(plan, 'dvh_data') else None
        
        if dvh_data:
            # Vẽ biểu đồ DVH
            ax = plt.subplot(111)
            
            # Màu cho từng cấu trúc
            colors = plt.cm.tab10.colors
            
            # Vẽ đường DVH cho từng cấu trúc
            for i, (struct_name, data) in enumerate(dvh_data.items()):
                color = colors[i % len(colors)]
                ax.plot(data['dose'], data['volume'], 
                        label=struct_name, color=color, linewidth=2)
            
            # Thiết lập trục
            ax.set_xlabel('Liều (Gy)')
            ax.set_ylabel('Thể tích (%)')
            ax.set_xlim(0, max(data['dose'][-1] for name, data in dvh_data.items()) * 1.1)
            ax.set_ylim(0, 105)
            ax.grid(True)
            
            # Chú thích
            ax.legend(loc='upper right')
            
        else:
            plt.text(0.5, 0.5, "Không có dữ liệu DVH", 
                     fontsize=14, ha='center')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _create_evaluation_page(self, pdf: PdfPages, plan: Plan):
        """
        Tạo trang đánh giá kế hoạch.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        plan : Plan
            Kế hoạch điều trị
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.95, "CHỈ SỐ ĐÁNH GIÁ KẾ HOẠCH", 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Lấy dữ liệu đánh giá
        evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
        
        if evaluation_data:
            # Tạo bảng chỉ số đánh giá cho PTV
            plt.text(0.5, 0.85, "CHỈ SỐ PTV", 
                     fontsize=14, fontweight='bold', ha='center')
            
            ptv_headers = ["Chỉ số", "Giá trị", "Bình thường"]
            ptv_metrics = []
            
            for metric, value in evaluation_data.get('PTV', {}).items():
                normal_range = self._get_normal_range(metric)
                ptv_metrics.append([metric, f"{value:.3f}", normal_range])
            
            if ptv_metrics:
                plt.table(cellText=ptv_metrics, colLabels=ptv_headers, 
                         loc='center', cellLoc='center',
                         bbox=[0.1, 0.6, 0.8, 0.2])
            
            # Tạo bảng chỉ số đánh giá cho OAR
            plt.text(0.5, 0.55, "CHỈ SỐ CƠ QUAN NGUY CẤP (OAR)", 
                     fontsize=14, fontweight='bold', ha='center')
            
            oar_metrics = []
            oar_headers = ["Cơ quan", "Chỉ số", "Giá trị", "Giới hạn"]
            
            for organ, metrics in evaluation_data.items():
                if organ != 'PTV':
                    for metric, value in metrics.items():
                        limit = self._get_organ_limit(organ, metric)
                        oar_metrics.append([organ, metric, f"{value:.3f}", limit])
            
            if oar_metrics:
                plt.table(cellText=oar_metrics, colLabels=oar_headers, 
                         loc='center', cellLoc='center',
                         bbox=[0.1, 0.2, 0.8, 0.3])
        else:
            plt.text(0.5, 0.5, "Không có dữ liệu đánh giá", 
                     fontsize=14, ha='center')
        
        # Tắt trục
        plt.axis('off')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _create_structure_dose_page(self, pdf: PdfPages, plan: Plan):
        """
        Tạo trang thông tin liều theo cấu trúc.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        plan : Plan
            Kế hoạch điều trị
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.95, "LIỀU THEO CẤU TRÚC", 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Lấy dữ liệu liều cấu trúc
        structure_doses = plan.structure_doses if hasattr(plan, 'structure_doses') else {}
        
        if structure_doses:
            # Tạo bảng liều cấu trúc
            headers = ["Cấu trúc", "D_mean (Gy)", "D_min (Gy)", "D_max (Gy)", "D95 (Gy)", "V95 (%)"]
            cell_text = []
            
            for struct_name, dose_data in structure_doses.items():
                cell_text.append([
                    struct_name,
                    f"{dose_data.get('D_mean', 0):.2f}",
                    f"{dose_data.get('D_min', 0):.2f}",
                    f"{dose_data.get('D_max', 0):.2f}",
                    f"{dose_data.get('D95', 0):.2f}",
                    f"{dose_data.get('V95', 0):.1f}"
                ])
            
            plt.table(cellText=cell_text, colLabels=headers, 
                     loc='center', cellLoc='center',
                     bbox=[0.05, 0.3, 0.9, 0.6])
        else:
            plt.text(0.5, 0.5, "Không có dữ liệu liều cấu trúc", 
                     fontsize=14, ha='center')
        
        # Tắt trục
        plt.axis('off')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _create_beam_info_page(self, pdf: PdfPages, plan: Plan):
        """
        Tạo trang thông tin chùm tia.
        
        Parameters
        ----------
        pdf : PdfPages
            Đối tượng PDF
        plan : Plan
            Kế hoạch điều trị
        """
        plt.figure(figsize=(8.5, 11))
        
        # Tiêu đề
        plt.text(0.5, 0.95, "THÔNG TIN CHÙM TIA", 
                 fontsize=16, fontweight='bold', ha='center')
        
        # Lấy dữ liệu chùm tia
        beams = plan.beams if hasattr(plan, 'beams') else []
        
        if beams:
            # Tạo bảng thông tin chùm tia
            headers = ["Tên", "Năng lượng", "Góc cánh tay", "Góc bàn", "MU", "Trọng số"]
            cell_text = []
            
            for beam in beams:
                cell_text.append([
                    beam.name,
                    beam.energy,
                    f"{beam.gantry_angle:.1f}°",
                    f"{beam.couch_angle:.1f}°",
                    f"{beam.monitor_units:.1f}",
                    f"{beam.weight:.2f}"
                ])
            
            plt.table(cellText=cell_text, colLabels=headers, 
                     loc='center', cellLoc='center',
                     bbox=[0.05, 0.5, 0.9, 0.4])
            
            # Vẽ biểu đồ góc chùm tia
            ax = plt.subplot(111, polar=True)
            angles = np.radians([beam.gantry_angle for beam in beams])
            weights = [beam.weight for beam in beams]
            
            # Vẽ các điểm trên biểu đồ cực
            ax.scatter(angles, [0.8] * len(angles), s=[w * 500 for w in weights], 
                      alpha=0.5, edgecolors='none')
            
            # Đặt nhãn cho các điểm
            for i, beam in enumerate(beams):
                angle = np.radians(beam.gantry_angle)
                ax.text(angle, 0.85, beam.name, 
                        ha='center', va='center', fontsize=8)
            
            # Thiết lập trục
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
            
            # Di chuyển biểu đồ xuống dưới
            ax.set_position([0.1, 0.1, 0.8, 0.35])
            
        else:
            plt.text(0.5, 0.5, "Không có thông tin chùm tia", 
                     fontsize=14, ha='center')
        
        # Thêm vào PDF
        pdf.savefig()
        plt.close()
    
    def _get_normal_range(self, metric: str) -> str:
        """
        Lấy phạm vi bình thường cho chỉ số đánh giá.
        
        Parameters
        ----------
        metric : str
            Tên chỉ số
            
        Returns
        -------
        str
            Phạm vi bình thường
        """
        normal_ranges = {
            'CI': '0.9 - 1.0',
            'HI': '<0.2',
            'GI': '<3.0',
            'CN': '>0.6',
            'PITV': '0.95 - 1.05',
            'TCov': '>0.95'
        }
        
        return normal_ranges.get(metric, 'N/A')
    
    def _get_organ_limit(self, organ: str, metric: str) -> str:
        """
        Lấy giới hạn liều cho cơ quan nguy cấp.
        
        Parameters
        ----------
        organ : str
            Tên cơ quan
        metric : str
            Tên chỉ số
            
        Returns
        -------
        str
            Giới hạn liều
        """
        # Các giới hạn liều theo QUANTEC hoặc các hướng dẫn khác
        organ_limits = {
            'Brain': {'D_max': '<60 Gy', 'V60Gy': '<3%'},
            'Brainstem': {'D_max': '<54 Gy', 'D1cc': '<59 Gy'},
            'Spinal Cord': {'D_max': '<50 Gy', 'D0.03cc': '<50 Gy'},
            'Optic Chiasm': {'D_max': '<55 Gy'},
            'Optic Nerve': {'D_max': '<55 Gy'},
            'Eye': {'D_mean': '<35 Gy'},
            'Lens': {'D_max': '<10 Gy'},
            'Cochlea': {'D_mean': '<45 Gy'},
            'Parotid': {'D_mean': '<26 Gy'},
            'Larynx': {'D_mean': '<45 Gy'},
            'Esophagus': {'D_mean': '<34 Gy', 'V60Gy': '<17%'},
            'Heart': {'D_mean': '<26 Gy', 'V25Gy': '<10%'},
            'Lung': {'V20Gy': '<30%', 'D_mean': '<20 Gy'},
            'Liver': {'D_mean': '<30 Gy', 'V30Gy': '<30%'},
            'Kidney': {'D_mean': '<18 Gy', 'V20Gy': '<32%'},
            'Bowel': {'V45Gy': '<195cc'},
            'Rectum': {'V50Gy': '<50%', 'V70Gy': '<20%'},
            'Bladder': {'V65Gy': '<50%', 'V70Gy': '<35%'},
            'Femoral Head': {'V50Gy': '<5%'}
        }
        
        if organ in organ_limits and metric in organ_limits[organ]:
            return organ_limits[organ][metric]
        
        return 'N/A'
