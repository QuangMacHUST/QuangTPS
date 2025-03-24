#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để nhập dữ liệu kế hoạch từ các nguồn bên ngoài vào cơ sở dữ liệu QuangTPS
để sử dụng cho huấn luyện mô hình KBP.

Script này cho phép nhập dữ liệu kế hoạch đã có từ các định dạng khác nhau như
DICOM-RT, CSV hoặc JSON vào cơ sở dữ liệu của QuangTPS.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time
from pathlib import Path
import pydicom
import glob

# Thêm thư mục cha vào đường dẫn để có thể import các module QuangTPS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quangtps.database.patient_db import PatientDatabase
from quangtps.database.plan_db import PlanDatabase
from quangtps.database.structure_db import StructureDatabase
from quangtps.database.dose_db import DoseDatabase
from quangtps.database.db_connector import DBConnector
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.rt_structure import RTStructureParser
from quangtps.dicom.rt_plan import RTPlanParser
from quangtps.dicom.rt_dose import RTDoseParser

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kbp_data_import.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Nhập dữ liệu kế hoạch từ các nguồn bên ngoài cho KBP"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Đường dẫn đến thư mục chứa dữ liệu đầu vào"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["dicom", "csv", "json"],
        default="dicom",
        help="Định dạng dữ liệu đầu vào"
    )
    
    parser.add_argument(
        "--site", "-s",
        required=True,
        help="Vị trí điều trị của các kế hoạch (ví dụ: Prostate, H&N)"
    )
    
    parser.add_argument(
        "--mapping", "-m",
        help="Đường dẫn đến file định nghĩa ánh xạ tên cấu trúc (JSON)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Tìm kiếm đệ quy trong các thư mục con"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy thử mà không ghi vào cơ sở dữ liệu"
    )
    
    return parser.parse_args()

def load_structure_mapping(mapping_file):
    """Tải ánh xạ tên cấu trúc từ file JSON."""
    if not mapping_file or not os.path.exists(mapping_file):
        return {}
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        logger.error(f"Lỗi khi tải file ánh xạ: {str(e)}")
        return {}

def normalize_structure_name(name, mapping):
    """Chuẩn hóa tên cấu trúc sử dụng ánh xạ."""
    # Loại bỏ các ký tự đặc biệt và chuyển thành chữ thường
    normalized = name.lower().strip()
    
    # Áp dụng ánh xạ
    for pattern, replacement in mapping.items():
        if pattern.lower() in normalized:
            return replacement
    
    return name

def import_dicom_data(input_dir, site, structure_mapping, recursive=False, dry_run=False):
    """
    Nhập dữ liệu kế hoạch từ các file DICOM.
    
    Args:
        input_dir: Thư mục chứa dữ liệu DICOM
        site: Vị trí điều trị
        structure_mapping: Ánh xạ tên cấu trúc
        recursive: Có tìm kiếm đệ quy không
        dry_run: Có ghi vào cơ sở dữ liệu không
    
    Returns:
        Tuple: (Số bệnh nhân đã nhập, số kế hoạch đã nhập)
    """
    # Khởi tạo các đối tượng cơ sở dữ liệu
    patient_db = PatientDatabase()
    plan_db = PlanDatabase()
    structure_db = StructureDatabase()
    dose_db = DoseDatabase()
    
    # Tìm tất cả thư mục bệnh nhân
    if recursive:
        patient_dirs = [d for d in glob.glob(os.path.join(input_dir, "**"), recursive=True) 
                       if os.path.isdir(d)]
    else:
        patient_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) 
                       if os.path.isdir(d)]
    
    patient_count = 0
    plan_count = 0
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        logger.info(f"Xử lý bệnh nhân: {patient_id}")
        
        # Tìm tất cả file DICOM trong thư mục bệnh nhân
        dicom_files = []
        
        if recursive:
            dicom_files = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
        else:
            dicom_files = glob.glob(os.path.join(patient_dir, "*.dcm"))
        
        if not dicom_files:
            logger.warning(f"Không tìm thấy file DICOM nào cho bệnh nhân {patient_id}")
            continue
        
        # Phân loại các file DICOM
        ct_files = []
        structure_files = []
        plan_files = []
        dose_files = []
        
        dicom_reader = DicomReader()
        
        for file_path in dicom_files:
            try:
                # Đọc thông tin DICOM
                dataset = pydicom.dcmread(file_path, force=True)
                
                # Phân loại theo modality
                if hasattr(dataset, 'Modality'):
                    if dataset.Modality == 'CT':
                        ct_files.append(file_path)
                    elif dataset.Modality == 'RTSTRUCT':
                        structure_files.append(file_path)
                    elif dataset.Modality == 'RTPLAN':
                        plan_files.append(file_path)
                    elif dataset.Modality == 'RTDOSE':
                        dose_files.append(file_path)
            except Exception as e:
                logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        
        logger.info(f"Đã tìm thấy: {len(ct_files)} CT, {len(structure_files)} RTSTRUCT, "
                  f"{len(plan_files)} RTPLAN, {len(dose_files)} RTDOSE")
        
        # Nếu không tìm thấy đủ dữ liệu, bỏ qua bệnh nhân này
        if not ct_files or not structure_files or not plan_files:
            logger.warning(f"Không đủ dữ liệu cho bệnh nhân {patient_id}")
            continue
        
        # Tạo hoặc cập nhật bệnh nhân trong cơ sở dữ liệu
        if not dry_run:
            # Đọc thông tin bệnh nhân từ file CT đầu tiên
            try:
                ct_dataset = pydicom.dcmread(ct_files[0], force=True)
                
                patient_name = ""
                if hasattr(ct_dataset, 'PatientName'):
                    patient_name = str(ct_dataset.PatientName)
                
                patient_gender = ""
                if hasattr(ct_dataset, 'PatientSex'):
                    patient_gender = ct_dataset.PatientSex
                
                patient_birthdate = ""
                if hasattr(ct_dataset, 'PatientBirthDate'):
                    patient_birthdate = ct_dataset.PatientBirthDate
                
                # Thêm bệnh nhân vào cơ sở dữ liệu
                patient_record = patient_db.add_patient(
                    patient_id=patient_id,
                    name=patient_name,
                    gender=patient_gender,
                    birth_date=patient_birthdate,
                    site=site
                )
                
                patient_count += 1
            except Exception as e:
                logger.error(f"Lỗi khi thêm bệnh nhân {patient_id}: {str(e)}")
                continue
        
        # Xử lý từng kế hoạch
        for plan_file in plan_files:
            logger.info(f"Xử lý kế hoạch từ file: {os.path.basename(plan_file)}")
            
            try:
                # Đọc dữ liệu kế hoạch
                plan_parser = RTPlanParser()
                plan_data = plan_parser.read(plan_file)
                
                plan_name = plan_data.get('name', f"Plan_{time.time()}")
                
                # Tìm file cấu trúc phù hợp với kế hoạch này
                structure_file = None
                if structure_files:
                    structure_file = structure_files[0]  # Dùng file đầu tiên
                
                if not structure_file:
                    logger.warning(f"Không tìm thấy file cấu trúc phù hợp cho kế hoạch {plan_name}")
                    continue
                
                # Tìm file liều phù hợp với kế hoạch này
                dose_file = None
                if dose_files:
                    dose_file = dose_files[0]  # Dùng file đầu tiên
                
                # Đọc dữ liệu cấu trúc
                structure_parser = RTStructureParser()
                structure_data = structure_parser.read(structure_file)
                
                # Đọc dữ liệu liều nếu có
                dose_data = None
                if dose_file:
                    dose_parser = RTDoseParser()
                    dose_data = dose_parser.read(dose_file)
                
                # Lấy thông tin kê đơn từ kế hoạch
                prescription = {}
                fractions = plan_data.get('fractions', 0)
                dose_value = 0
                
                if 'prescription' in plan_data:
                    prescription = plan_data['prescription']
                    dose_value = prescription.get('dose', 0)
                
                # Chuẩn hóa các tên cấu trúc
                normalized_structures = {}
                
                for struct_id, struct_info in structure_data.get('structures', {}).items():
                    original_name = struct_info.get('name', '')
                    normalized_name = normalize_structure_name(original_name, structure_mapping)
                    
                    normalized_structures[struct_id] = {
                        'original_name': original_name,
                        'normalized_name': normalized_name,
                        'data': struct_info
                    }
                
                # Phân loại cấu trúc thành PTV và OAR
                ptvs = {}
                oars = {}
                
                for struct_id, struct_info in normalized_structures.items():
                    name = struct_info['normalized_name']
                    
                    if name.lower().startswith('ptv'):
                        ptvs[struct_id] = struct_info
                    else:
                        oars[struct_id] = struct_info
                
                logger.info(f"Đã phân loại: {len(ptvs)} PTV, {len(oars)} OAR")
                
                if not ptvs:
                    logger.warning(f"Không tìm thấy cấu trúc PTV nào cho kế hoạch {plan_name}")
                    continue
                
                # Thêm kế hoạch và cấu trúc vào cơ sở dữ liệu nếu không phải dry run
                if not dry_run:
                    try:
                        # Tạo bản ghi kế hoạch
                        plan_record = plan_db.add_plan(
                            patient_id=patient_id,
                            name=plan_name,
                            description=f"Imported plan from {os.path.basename(plan_file)}",
                            prescribed_dose=dose_value,
                            fractions=fractions,
                            site=site,
                            structure_set_id=str(time.time()),  # ID tạm thời
                            technique=plan_data.get('technique', ''),
                            status="APPROVED"
                        )
                        
                        plan_id = plan_record["id"]
                        
                        # Thêm các cấu trúc
                        for struct_id, struct_info in normalized_structures.items():
                            struct_data = struct_info['data']
                            name = struct_info['normalized_name']
                            
                            # Thêm cấu trúc
                            structure_db.add_structure(
                                plan_id=plan_id,
                                name=name,
                                type="PTV" if name.lower().startswith('ptv') else "OAR",
                                color=struct_data.get('color', [255, 0, 0]),
                                data=struct_data.get('contour_data', {})
                            )
                        
                        # Thêm thông tin liều nếu có
                        if dose_data:
                            dose_db.add_dose(
                                plan_id=plan_id,
                                dose_matrix=dose_data.get('dose_matrix', []),
                                grid=dose_data.get('grid', {}),
                                scaling=dose_data.get('scaling', 1.0)
                            )
                        
                        plan_count += 1
                        logger.info(f"Đã thêm kế hoạch {plan_name} vào cơ sở dữ liệu")
                        
                    except Exception as e:
                        logger.error(f"Lỗi khi thêm kế hoạch {plan_name}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Lỗi khi xử lý kế hoạch từ file {plan_file}: {str(e)}")
    
    return patient_count, plan_count

def import_csv_data(input_file, site, structure_mapping, dry_run=False):
    """
    Nhập dữ liệu kế hoạch từ file CSV.
    
    Args:
        input_file: Đường dẫn đến file CSV
        site: Vị trí điều trị
        structure_mapping: Ánh xạ tên cấu trúc
        dry_run: Có ghi vào cơ sở dữ liệu không
    
    Returns:
        Tuple: (Số bệnh nhân đã nhập, số kế hoạch đã nhập)
    """
    logger.info(f"Nhập dữ liệu từ file CSV: {input_file}")
    
    # Đọc dữ liệu từ file CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file CSV: {str(e)}")
        return 0, 0
    
    # Kiểm tra các cột bắt buộc
    required_columns = ['patient_id', 'plan_name', 'prescribed_dose', 'fractions']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Thiếu cột bắt buộc: {col}")
            return 0, 0
    
    # Khởi tạo các đối tượng cơ sở dữ liệu
    patient_db = PatientDatabase()
    plan_db = PlanDatabase()
    
    patient_count = 0
    plan_count = 0
    
    # Xử lý từng hàng trong DataFrame
    for i, row in df.iterrows():
        patient_id = row['patient_id']
        plan_name = row['plan_name']
        prescribed_dose = row['prescribed_dose']
        fractions = row['fractions']
        
        logger.info(f"Xử lý kế hoạch {plan_name} cho bệnh nhân {patient_id}")
        
        # Thêm bệnh nhân vào cơ sở dữ liệu
        if not dry_run:
            try:
                # Kiểm tra xem bệnh nhân đã tồn tại chưa
                existing_patient = patient_db.get_patient(patient_id)
                
                if not existing_patient:
                    # Thêm bệnh nhân mới
                    patient_name = row.get('patient_name', f"Patient_{patient_id}")
                    patient_gender = row.get('patient_gender', '')
                    patient_birthdate = row.get('patient_birthdate', '')
                    
                    patient_db.add_patient(
                        patient_id=patient_id,
                        name=patient_name,
                        gender=patient_gender,
                        birth_date=patient_birthdate,
                        site=site
                    )
                    
                    patient_count += 1
                
                # Thêm kế hoạch
                plan_description = row.get('plan_description', f"Imported from CSV")
                plan_technique = row.get('technique', '')
                
                plan_record = plan_db.add_plan(
                    patient_id=patient_id,
                    name=plan_name,
                    description=plan_description,
                    prescribed_dose=prescribed_dose,
                    fractions=fractions,
                    site=site,
                    structure_set_id=str(time.time()),  # ID tạm thời
                    technique=plan_technique,
                    status="APPROVED"
                )
                
                plan_count += 1
                logger.info(f"Đã thêm kế hoạch {plan_name} vào cơ sở dữ liệu")
                
            except Exception as e:
                logger.error(f"Lỗi khi thêm kế hoạch {plan_name}: {str(e)}")
    
    return patient_count, plan_count

def import_json_data(input_file, site, structure_mapping, dry_run=False):
    """
    Nhập dữ liệu kế hoạch từ file JSON.
    
    Args:
        input_file: Đường dẫn đến file JSON
        site: Vị trí điều trị
        structure_mapping: Ánh xạ tên cấu trúc
        dry_run: Có ghi vào cơ sở dữ liệu không
    
    Returns:
        Tuple: (Số bệnh nhân đã nhập, số kế hoạch đã nhập)
    """
    logger.info(f"Nhập dữ liệu từ file JSON: {input_file}")
    
    # Đọc dữ liệu từ file JSON
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file JSON: {str(e)}")
        return 0, 0
    
    # Kiểm tra cấu trúc dữ liệu
    if not isinstance(data, list):
        logger.error("Dữ liệu JSON phải là một danh sách các kế hoạch")
        return 0, 0
    
    # Khởi tạo các đối tượng cơ sở dữ liệu
    patient_db = PatientDatabase()
    plan_db = PlanDatabase()
    structure_db = StructureDatabase()
    
    patient_count = 0
    plan_count = 0
    
    # Xử lý từng kế hoạch trong danh sách
    for plan_data in data:
        # Kiểm tra các trường bắt buộc
        if 'patient_id' not in plan_data or 'plan_name' not in plan_data:
            logger.warning("Thiếu trường bắt buộc, bỏ qua kế hoạch này")
            continue
        
        patient_id = plan_data['patient_id']
        plan_name = plan_data['plan_name']
        
        logger.info(f"Xử lý kế hoạch {plan_name} cho bệnh nhân {patient_id}")
        
        # Thêm bệnh nhân vào cơ sở dữ liệu
        if not dry_run:
            try:
                # Kiểm tra xem bệnh nhân đã tồn tại chưa
                existing_patient = patient_db.get_patient(patient_id)
                
                if not existing_patient:
                    # Thêm bệnh nhân mới
                    patient_name = plan_data.get('patient_name', f"Patient_{patient_id}")
                    patient_gender = plan_data.get('patient_gender', '')
                    patient_birthdate = plan_data.get('patient_birthdate', '')
                    
                    patient_db.add_patient(
                        patient_id=patient_id,
                        name=patient_name,
                        gender=patient_gender,
                        birth_date=patient_birthdate,
                        site=site
                    )
                    
                    patient_count += 1
                
                # Lấy thông tin kế hoạch
                prescribed_dose = plan_data.get('prescribed_dose', 0)
                fractions = plan_data.get('fractions', 0)
                plan_description = plan_data.get('plan_description', f"Imported from JSON")
                plan_technique = plan_data.get('technique', '')
                
                # Thêm kế hoạch
                plan_record = plan_db.add_plan(
                    patient_id=patient_id,
                    name=plan_name,
                    description=plan_description,
                    prescribed_dose=prescribed_dose,
                    fractions=fractions,
                    site=site,
                    structure_set_id=str(time.time()),  # ID tạm thời
                    technique=plan_technique,
                    status="APPROVED"
                )
                
                plan_id = plan_record["id"]
                
                # Thêm các cấu trúc nếu có
                if 'structures' in plan_data:
                    for struct in plan_data['structures']:
                        struct_name = struct.get('name', '')
                        struct_type = struct.get('type', '')
                        
                        # Chuẩn hóa tên cấu trúc
                        normalized_name = normalize_structure_name(struct_name, structure_mapping)
                        
                        # Xác định loại cấu trúc nếu không có
                        if not struct_type:
                            struct_type = "PTV" if normalized_name.lower().startswith('ptv') else "OAR"
                        
                        # Thêm cấu trúc
                        structure_db.add_structure(
                            plan_id=plan_id,
                            name=normalized_name,
                            type=struct_type,
                            color=struct.get('color', [255, 0, 0]),
                            data=struct.get('contour_data', {})
                        )
                
                plan_count += 1
                logger.info(f"Đã thêm kế hoạch {plan_name} vào cơ sở dữ liệu")
                
            except Exception as e:
                logger.error(f"Lỗi khi thêm kế hoạch {plan_name}: {str(e)}")
    
    return patient_count, plan_count

def main():
    """Hàm chính của script."""
    args = parse_args()
    
    # Tải ánh xạ tên cấu trúc
    structure_mapping = load_structure_mapping(args.mapping)
    
    # Kiểm tra đường dẫn đầu vào
    if not os.path.exists(args.input):
        logger.error(f"Đường dẫn đầu vào không tồn tại: {args.input}")
        return 1
    
    logger.info(f"Bắt đầu nhập dữ liệu từ {args.input} (định dạng: {args.format})")
    
    if args.dry_run:
        logger.info("Chế độ thử nghiệm: Không ghi vào cơ sở dữ liệu")
    
    start_time = time.time()
    
    try:
        # Thực hiện nhập dữ liệu theo định dạng
        if args.format == "dicom":
            patient_count, plan_count = import_dicom_data(
                args.input, args.site, structure_mapping, args.recursive, args.dry_run
            )
        elif args.format == "csv":
            patient_count, plan_count = import_csv_data(
                args.input, args.site, structure_mapping, args.dry_run
            )
        elif args.format == "json":
            patient_count, plan_count = import_json_data(
                args.input, args.site, structure_mapping, args.dry_run
            )
        
        process_time = time.time() - start_time
        
        # Hiển thị kết quả
        logger.info(f"Đã nhập {patient_count} bệnh nhân và {plan_count} kế hoạch trong {process_time:.2f} giây")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Lỗi không mong đợi: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 