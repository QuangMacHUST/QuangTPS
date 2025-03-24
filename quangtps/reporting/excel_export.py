#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module xuất dữ liệu ra file Excel.

Module này cung cấp các lớp và hàm để xuất dữ liệu từ hệ thống
QuangTPS ra định dạng Excel cho các báo cáo và phân tích.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator

logger = logging.getLogger(__name__)


class ExcelExporter:
    """
    Lớp xuất dữ liệu ra file Excel.
    
    Lớp này cung cấp các phương thức để xuất dữ liệu từ QuangTPS ra
    định dạng Excel, bao gồm thông tin bệnh nhân, kế hoạch, liều lượng,
    DVH, và các thông số đánh giá.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Khởi tạo lớp ExcelExporter.
        
        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra để lưu file Excel
        """
        self.output_dir = output_dir or os.path.expanduser("~/Documents/QuangTPS/Reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_plan_evaluation(self, patient: Patient, plan: Plan, output_file: Optional[str] = None) -> str:
        """
        Xuất dữ liệu đánh giá kế hoạch ra file Excel.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        output_file : str, optional
            Tên file đầu ra, mặc định là 'PatientID_PlanName_Evaluation.xlsx'
            
        Returns
        -------
        str
            Đường dẫn đến file Excel đã tạo
        """
        # Xác định tên file đầu ra
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{patient.patient_id}_{plan.name}_Evaluation_{timestamp}.xlsx"
            
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # Tạo file Excel writer với pandas
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Xuất thông tin bệnh nhân
                self._export_patient_info(writer, patient)
                
                # Xuất thông tin kế hoạch
                self._export_plan_info(writer, plan)
                
                # Xuất dữ liệu liều theo cấu trúc
                self._export_structure_doses(writer, plan)
                
                # Xuất dữ liệu DVH
                self._export_dvh_data(writer, plan)
                
                # Xuất các chỉ số đánh giá
                self._export_evaluation_metrics(writer, plan)
                
                # Xuất thông tin chùm tia
                self._export_beam_info(writer, plan)
                
            logger.info(f"Đã xuất dữ liệu ra file Excel: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất file Excel: {str(e)}", exc_info=True)
            raise
    
    def export_dvh_data(self, plan: Plan, output_file: Optional[str] = None) -> str:
        """
        Xuất dữ liệu DVH ra file Excel.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch điều trị
        output_file : str, optional
            Tên file đầu ra, mặc định là 'PlanName_DVH.xlsx'
            
        Returns
        -------
        str
            Đường dẫn đến file Excel đã tạo
        """
        # Xác định tên file đầu ra
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{plan.name}_DVH_{timestamp}.xlsx"
            
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # Lấy dữ liệu DVH
            dvh_data = plan.dvh_data if hasattr(plan, 'dvh_data') else {}
            
            if not dvh_data:
                logger.warning("Không có dữ liệu DVH để xuất")
                return ""
            
            # Tạo dataframe cho từng cấu trúc
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet tổng hợp DVH
                summary_data = {}
                max_points = max([len(data['dose']) for name, data in dvh_data.items()], default=0)
                
                for struct_name, data in dvh_data.items():
                    # Đảm bảo tất cả các mảng có cùng kích thước
                    dose = np.pad(data['dose'], (0, max_points - len(data['dose'])), 'constant', constant_values=np.nan)
                    volume = np.pad(data['volume'], (0, max_points - len(data['volume'])), 'constant', constant_values=np.nan)
                    
                    summary_data[f"{struct_name}_Dose"] = dose
                    summary_data[f"{struct_name}_Volume"] = volume
                
                # Tạo dataframe và xuất
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='DVH Summary', index=False)
                
                # Xuất từng cấu trúc vào sheet riêng
                for struct_name, data in dvh_data.items():
                    # Tạo dataframe
                    struct_df = pd.DataFrame({
                        'Dose (Gy)': data['dose'],
                        'Volume (%)': data['volume']
                    })
                    
                    # Xuất ra sheet
                    sheet_name = struct_name[:31]  # Excel giới hạn tên sheet 31 ký tự
                    struct_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Đã xuất dữ liệu DVH ra file Excel: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất dữ liệu DVH ra Excel: {str(e)}", exc_info=True)
            raise
    
    def export_multiple_plan_comparison(self, patient: Patient, plans: List[Plan], 
                                      output_file: Optional[str] = None) -> str:
        """
        Xuất so sánh nhiều kế hoạch ra file Excel.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plans : List[Plan]
            Danh sách các kế hoạch cần so sánh
        output_file : str, optional
            Tên file đầu ra
            
        Returns
        -------
        str
            Đường dẫn đến file Excel đã tạo
        """
        # Xác định tên file đầu ra
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{patient.patient_id}_PlanComparison_{timestamp}.xlsx"
            
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # Tạo Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Xuất thông tin bệnh nhân
                self._export_patient_info(writer, patient)
                
                # Xuất thông tin so sánh các kế hoạch
                self._export_plan_comparison(writer, plans)
                
                # Xuất so sánh chỉ số đánh giá
                self._export_evaluation_comparison(writer, plans)
                
                # Xuất so sánh liều theo cấu trúc
                self._export_structure_dose_comparison(writer, plans)
            
            logger.info(f"Đã xuất so sánh kế hoạch ra file Excel: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất so sánh kế hoạch ra Excel: {str(e)}", exc_info=True)
            raise
    
    def _export_patient_info(self, writer: pd.ExcelWriter, patient: Patient):
        """
        Xuất thông tin bệnh nhân ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        patient : Patient
            Đối tượng bệnh nhân
        """
        # Tạo dataframe từ thông tin bệnh nhân
        data = {
            'Thuộc tính': [
                'ID', 'Họ và tên', 'Ngày sinh', 'Giới tính', 
                'Địa chỉ', 'Chẩn đoán', 'Bác sĩ phụ trách', 'Ngày nhập viện'
            ],
            'Giá trị': [
                patient.patient_id, patient.name, patient.date_of_birth, patient.gender,
                patient.address, patient.diagnosis, patient.physician, patient.admission_date
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Xuất ra sheet
        df.to_excel(writer, sheet_name='Thông tin bệnh nhân', index=False)
    
    def _export_plan_info(self, writer: pd.ExcelWriter, plan: Plan):
        """
        Xuất thông tin kế hoạch ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plan : Plan
            Kế hoạch điều trị
        """
        # Tạo dataframe từ thông tin kế hoạch
        data = {
            'Thuộc tính': [
                'Tên kế hoạch', 'Ngày tạo', 'Mô tả', 'Loại kế hoạch',
                'Tổng liều (Gy)', 'Số phân liều', 'Liều mỗi phân liều (Gy)',
                'Thuật toán tính liều', 'Thời gian điều trị', 'Trạng thái'
            ],
            'Giá trị': [
                plan.name, plan.creation_date, plan.description, plan.plan_type,
                plan.total_dose, plan.number_of_fractions, plan.dose_per_fraction,
                plan.dose_algorithm, plan.treatment_time, plan.status
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Xuất ra sheet
        df.to_excel(writer, sheet_name='Thông tin kế hoạch', index=False)
        
        # Xuất thông tin kê đơn nếu có
        if hasattr(plan, 'prescriptions') and plan.prescriptions:
            prescription_data = []
            
            for p in plan.prescriptions:
                prescription_data.append({
                    'Cấu trúc': p.structure_name,
                    'Liều (Gy)': p.dose,
                    'Thể tích (%)': p.volume,
                    'Ưu tiên': p.priority,
                    'Mô tả': p.description
                })
            
            if prescription_data:
                prescription_df = pd.DataFrame(prescription_data)
                prescription_df.to_excel(writer, sheet_name='Kê đơn', index=False)
    
    def _export_structure_doses(self, writer: pd.ExcelWriter, plan: Plan):
        """
        Xuất dữ liệu liều theo cấu trúc ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plan : Plan
            Kế hoạch điều trị
        """
        # Lấy dữ liệu liều cấu trúc
        structure_doses = plan.structure_doses if hasattr(plan, 'structure_doses') else {}
        
        if not structure_doses:
            return
        
        # Tạo dataframe
        data = []
        
        for struct_name, dose_data in structure_doses.items():
            row = {
                'Cấu trúc': struct_name,
                'D_mean (Gy)': dose_data.get('D_mean', 0),
                'D_min (Gy)': dose_data.get('D_min', 0),
                'D_max (Gy)': dose_data.get('D_max', 0),
                'D95 (Gy)': dose_data.get('D95', 0),
                'D98 (Gy)': dose_data.get('D98', 0),
                'D2 (Gy)': dose_data.get('D2', 0),
                'V95 (%)': dose_data.get('V95', 0),
                'V100 (%)': dose_data.get('V100', 0),
                'V107 (%)': dose_data.get('V107', 0)
            }
            
            # Thêm các chỉ số đặc biệt cho từng cấu trúc
            for key, value in dose_data.items():
                if key not in row:
                    row[key] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Xuất ra sheet
        df.to_excel(writer, sheet_name='Liều theo cấu trúc', index=False)
    
    def _export_dvh_data(self, writer: pd.ExcelWriter, plan: Plan):
        """
        Xuất dữ liệu DVH ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plan : Plan
            Kế hoạch điều trị
        """
        # Lấy dữ liệu DVH
        dvh_data = plan.dvh_data if hasattr(plan, 'dvh_data') else {}
        
        if not dvh_data:
            return
        
        # Xuất các chỉ số DVH
        dvh_stats = []
        
        for struct_name, data in dvh_data.items():
            if 'statistics' in data:
                stats = data['statistics']
                
                row = {
                    'Cấu trúc': struct_name,
                    'Thể tích (cc)': stats.get('volume_cc', 0),
                    'Liều min (Gy)': stats.get('min_dose', 0),
                    'Liều max (Gy)': stats.get('max_dose', 0),
                    'Liều trung bình (Gy)': stats.get('mean_dose', 0),
                    'Liều trung vị (Gy)': stats.get('median_dose', 0)
                }
                
                # Thêm các chỉ số D_x và V_x
                for key, value in stats.items():
                    if key.startswith('D') or key.startswith('V'):
                        row[key] = value
                
                dvh_stats.append(row)
        
        if dvh_stats:
            dvh_stats_df = pd.DataFrame(dvh_stats)
            dvh_stats_df.to_excel(writer, sheet_name='DVH Statistics', index=False)
    
    def _export_evaluation_metrics(self, writer: pd.ExcelWriter, plan: Plan):
        """
        Xuất các chỉ số đánh giá ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plan : Plan
            Kế hoạch điều trị
        """
        # Lấy dữ liệu đánh giá
        evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
        
        if not evaluation_data:
            return
        
        # Xuất chỉ số PTV
        ptv_metrics = []
        
        for metric, value in evaluation_data.get('PTV', {}).items():
            normal_range = self._get_normal_range(metric)
            ptv_metrics.append({
                'Chỉ số': metric,
                'Giá trị': value,
                'Bình thường': normal_range
            })
        
        if ptv_metrics:
            ptv_df = pd.DataFrame(ptv_metrics)
            ptv_df.to_excel(writer, sheet_name='Chỉ số PTV', index=False)
        
        # Xuất chỉ số OAR
        oar_metrics = []
        
        for organ, metrics in evaluation_data.items():
            if organ != 'PTV':
                for metric, value in metrics.items():
                    limit = self._get_organ_limit(organ, metric)
                    oar_metrics.append({
                        'Cơ quan': organ,
                        'Chỉ số': metric,
                        'Giá trị': value,
                        'Giới hạn': limit
                    })
        
        if oar_metrics:
            oar_df = pd.DataFrame(oar_metrics)
            oar_df.to_excel(writer, sheet_name='Chỉ số OAR', index=False)
    
    def _export_beam_info(self, writer: pd.ExcelWriter, plan: Plan):
        """
        Xuất thông tin chùm tia ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plan : Plan
            Kế hoạch điều trị
        """
        # Lấy dữ liệu chùm tia
        beams = plan.beams if hasattr(plan, 'beams') else []
        
        if not beams:
            return
        
        # Tạo dataframe
        data = []
        
        for beam in beams:
            row = {
                'Tên': beam.name,
                'Năng lượng': beam.energy,
                'Góc cánh tay (°)': beam.gantry_angle,
                'Góc bàn (°)': beam.couch_angle,
                'MU': beam.monitor_units,
                'Trọng số': beam.weight,
                'SSD (cm)': beam.ssd,
                'Kích thước trường (cm²)': beam.field_size,
                'Loại chùm tia': beam.beam_type,
                'Wedge': beam.wedge,
                'Bộ lọc': beam.filter,
                'Khoảng cách nguồn-trục (cm)': beam.sad
            }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Xuất ra sheet
        df.to_excel(writer, sheet_name='Thông tin chùm tia', index=False)
    
    def _export_plan_comparison(self, writer: pd.ExcelWriter, plans: List[Plan]):
        """
        Xuất so sánh kế hoạch ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plans : List[Plan]
            Danh sách các kế hoạch
        """
        # Tạo dataframe so sánh các thông số chung
        data = []
        
        for plan in plans:
            row = {
                'Tên kế hoạch': plan.name,
                'Ngày tạo': plan.creation_date,
                'Tổng liều (Gy)': plan.total_dose,
                'Số phân liều': plan.number_of_fractions,
                'Liều mỗi phân liều (Gy)': plan.dose_per_fraction,
                'Thuật toán tính liều': plan.dose_algorithm,
                'Số chùm tia': len(plan.beams) if hasattr(plan, 'beams') else 0,
                'Tổng MU': sum(beam.monitor_units for beam in plan.beams) if hasattr(plan, 'beams') else 0,
                'Thời gian điều trị': plan.treatment_time
            }
            
            data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name='So sánh kế hoạch', index=False)
    
    def _export_evaluation_comparison(self, writer: pd.ExcelWriter, plans: List[Plan]):
        """
        Xuất so sánh chỉ số đánh giá ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plans : List[Plan]
            Danh sách các kế hoạch
        """
        # Danh sách tất cả các chỉ số đánh giá PTV
        all_ptv_metrics = set()
        for plan in plans:
            evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
            for metric in evaluation_data.get('PTV', {}):
                all_ptv_metrics.add(metric)
        
        # Tạo dataframe so sánh chỉ số PTV
        ptv_data = []
        
        for metric in sorted(all_ptv_metrics):
            row = {'Chỉ số': metric, 'Bình thường': self._get_normal_range(metric)}
            
            for plan in plans:
                evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
                row[plan.name] = evaluation_data.get('PTV', {}).get(metric, 'N/A')
            
            ptv_data.append(row)
        
        if ptv_data:
            ptv_df = pd.DataFrame(ptv_data)
            ptv_df.to_excel(writer, sheet_name='So sánh chỉ số PTV', index=False)
        
        # Danh sách tất cả các cấu trúc OAR trong tất cả các kế hoạch
        all_oars = set()
        for plan in plans:
            evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
            for organ in evaluation_data:
                if organ != 'PTV':
                    all_oars.add(organ)
        
        # Tạo dataframe so sánh chỉ số OAR
        for organ in sorted(all_oars):
            # Danh sách tất cả các chỉ số cho cơ quan này
            all_metrics = set()
            for plan in plans:
                evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
                for metric in evaluation_data.get(organ, {}):
                    all_metrics.add(metric)
            
            # Tạo dataframe so sánh
            organ_data = []
            
            for metric in sorted(all_metrics):
                row = {'Chỉ số': metric, 'Giới hạn': self._get_organ_limit(organ, metric)}
                
                for plan in plans:
                    evaluation_data = plan.evaluation_metrics if hasattr(plan, 'evaluation_metrics') else {}
                    row[plan.name] = evaluation_data.get(organ, {}).get(metric, 'N/A')
                
                organ_data.append(row)
            
            if organ_data:
                organ_df = pd.DataFrame(organ_data)
                sheet_name = f"{organ} (OAR)"[:31]  # Excel giới hạn tên sheet 31 ký tự
                organ_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _export_structure_dose_comparison(self, writer: pd.ExcelWriter, plans: List[Plan]):
        """
        Xuất so sánh liều theo cấu trúc ra sheet Excel.
        
        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer
        plans : List[Plan]
            Danh sách các kế hoạch
        """
        # Danh sách tất cả các cấu trúc trong tất cả các kế hoạch
        all_structures = set()
        for plan in plans:
            structure_doses = plan.structure_doses if hasattr(plan, 'structure_doses') else {}
            for struct_name in structure_doses:
                all_structures.add(struct_name)
        
        # Các chỉ số liều cơ bản cần so sánh
        basic_metrics = ['D_mean', 'D_min', 'D_max', 'D95', 'V95']
        
        # Tạo dataframe so sánh cho từng cấu trúc
        for metric in basic_metrics:
            metric_data = []
            
            for struct_name in sorted(all_structures):
                row = {'Cấu trúc': struct_name}
                
                for plan in plans:
                    structure_doses = plan.structure_doses if hasattr(plan, 'structure_doses') else {}
                    dose_data = structure_doses.get(struct_name, {})
                    row[plan.name] = dose_data.get(metric, 'N/A')
                
                metric_data.append(row)
            
            if metric_data:
                metric_df = pd.DataFrame(metric_data)
                metric_df.to_excel(writer, sheet_name=f"So sánh {metric}", index=False)
    
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
