#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý tập dữ liệu DICOM.

Module này cung cấp các lớp và phương thức để tổ chức, lưu trữ và truy xuất
các tập dữ liệu DICOM theo bệnh nhân, nghiên cứu và series.
"""

import os
import json
import shutil
import datetime
import logging
from collections import defaultdict
from typing import List, Dict, Any, Union, Optional, Tuple, Set

import pydicom

from quangtps.core.config import Config
from quangtps.core.exceptions import DicomError, IOError, ValidationError
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.dicom_writer import DicomWriter
from quangtps.dicom.dicom_validator import DicomValidator
from quangtps.dicom.rt_structure import RTStructure
from quangtps.dicom.rt_dose import RTDose
from quangtps.dicom.rt_plan import RTPlan
from quangtps.dicom.rt_image import RTImage

logger = logging.getLogger(__name__)

class DicomDataset:
    """
    Lớp đại diện cho một tập dữ liệu DICOM.
    
    Lớp này chứa thông tin về một tập dữ liệu DICOM, bao gồm thông tin bệnh nhân,
    nghiên cứu và series. Nó cũng chứa các dataset DICOM thực sự.
    """
    
    def __init__(self, patient_id: str, patient_name: str):
        """
        Khởi tạo DicomDataset.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        patient_name : str
            Tên bệnh nhân
        """
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.studies = {}  # study_uid -> study_info
        
    def add_dataset(self, dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Thêm một dataset DICOM vào tập dữ liệu.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM cần thêm
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu không
        """
        try:
            # Kiểm tra dataset có hợp lệ không
            DicomValidator.validate_dataset(dataset)
            
            # Kiểm tra dataset có thuộc về bệnh nhân này không
            if dataset.PatientID != self.patient_id:
                logger.warning(f"Dataset không thuộc về bệnh nhân {self.patient_id}")
                return False
            
            # Lấy Study UID
            study_uid = dataset.StudyInstanceUID
            
            # Lấy Series UID
            series_uid = dataset.SeriesInstanceUID
            
            # Thêm study nếu chưa tồn tại
            if study_uid not in self.studies:
                study_date = getattr(dataset, 'StudyDate', '')
                study_time = getattr(dataset, 'StudyTime', '')
                study_description = getattr(dataset, 'StudyDescription', '')
                
                self.studies[study_uid] = {
                    'study_date': study_date,
                    'study_time': study_time,
                    'study_description': study_description,
                    'series': {}
                }
            
            # Thêm series nếu chưa tồn tại
            if series_uid not in self.studies[study_uid]['series']:
                modality = getattr(dataset, 'Modality', '')
                series_description = getattr(dataset, 'SeriesDescription', '')
                
                self.studies[study_uid]['series'][series_uid] = {
                    'modality': modality,
                    'series_description': series_description,
                    'datasets': []
                }
            
            # Thêm dataset vào series
            self.studies[study_uid]['series'][series_uid]['datasets'].append(dataset)
            
            logger.info(f"Đã thêm dataset {dataset.SOPInstanceUID} vào tập dữ liệu")
            return True
        
        except (ValidationError, DicomError) as e:
            logger.error(f"Lỗi khi thêm dataset: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Lỗi không xác định khi thêm dataset: {str(e)}")
            return False
    
    def get_studies(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các nghiên cứu.
        
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các nghiên cứu
        """
        result = []
        
        for study_uid, study_info in self.studies.items():
            result.append({
                'study_uid': study_uid,
                'study_date': study_info['study_date'],
                'study_time': study_info['study_time'],
                'study_description': study_info['study_description'],
                'series_count': len(study_info['series'])
            })
        
        return sorted(result, key=lambda x: x['study_date'], reverse=True)
    
    def get_series(self, study_uid: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các series trong một nghiên cứu.
        
        Parameters
        ----------
        study_uid : str
            UID của nghiên cứu
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các series
        """
        if study_uid not in self.studies:
            return []
        
        result = []
        
        for series_uid, series_info in self.studies[study_uid]['series'].items():
            result.append({
                'series_uid': series_uid,
                'modality': series_info['modality'],
                'series_description': series_info['series_description'],
                'instance_count': len(series_info['datasets'])
            })
        
        return result
    
    def get_datasets(self, study_uid: str, series_uid: str) -> List[pydicom.dataset.FileDataset]:
        """
        Lấy danh sách các dataset trong một series.
        
        Parameters
        ----------
        study_uid : str
            UID của nghiên cứu
        series_uid : str
            UID của series
            
        Returns
        -------
        List[pydicom.dataset.FileDataset]
            Danh sách các dataset
        """
        if study_uid not in self.studies or series_uid not in self.studies[study_uid]['series']:
            return []
        
        return self.studies[study_uid]['series'][series_uid]['datasets']
    
    def get_datasets_by_modality(self, modality: str) -> List[pydicom.dataset.FileDataset]:
        """
        Lấy danh sách các dataset theo modality.
        
        Parameters
        ----------
        modality : str
            Loại modality (CT, MR, RTSTRUCT, ...)
            
        Returns
        -------
        List[pydicom.dataset.FileDataset]
            Danh sách các dataset
        """
        result = []
        
        for study_info in self.studies.values():
            for series_info in study_info['series'].values():
                if series_info['modality'] == modality:
                    result.extend(series_info['datasets'])
        
        return result
    
    def get_rt_structures(self) -> List[RTStructure]:
        """
        Lấy danh sách các cấu trúc RT.
        
        Returns
        -------
        List[RTStructure]
            Danh sách các cấu trúc RT
        """
        rt_struct_datasets = self.get_datasets_by_modality('RTSTRUCT')
        return [RTStructure(dataset) for dataset in rt_struct_datasets]
    
    def get_rt_doses(self) -> List[RTDose]:
        """
        Lấy danh sách các liều RT.
        
        Returns
        -------
        List[RTDose]
            Danh sách các liều RT
        """
        rt_dose_datasets = self.get_datasets_by_modality('RTDOSE')
        return [RTDose(dataset) for dataset in rt_dose_datasets]
    
    def get_rt_plans(self) -> List[RTPlan]:
        """
        Lấy danh sách các kế hoạch RT.
        
        Returns
        -------
        List[RTPlan]
            Danh sách các kế hoạch RT
        """
        rt_plan_datasets = self.get_datasets_by_modality('RTPLAN')
        return [RTPlan(dataset) for dataset in rt_plan_datasets]
    
    def get_rt_images(self) -> List[RTImage]:
        """
        Lấy danh sách các hình ảnh RT.
        
        Returns
        -------
        List[RTImage]
            Danh sách các hình ảnh RT
        """
        rt_image_datasets = self.get_datasets_by_modality('RTIMAGE')
        return [RTImage(dataset) for dataset in rt_image_datasets]
    
    def save_to_directory(self, directory: str) -> bool:
        """
        Lưu tất cả dataset vào thư mục.
        
        Parameters
        ----------
        directory : str
            Đường dẫn đến thư mục
            
        Returns
        -------
        bool
            True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(directory, exist_ok=True)
            
            # Tạo thư mục cho bệnh nhân
            patient_dir = os.path.join(directory, f"{self.patient_id}")
            os.makedirs(patient_dir, exist_ok=True)
            
            # Lưu thông tin bệnh nhân
            with open(os.path.join(patient_dir, "patient_info.json"), 'w') as f:
                json.dump({
                    'patient_id': self.patient_id,
                    'patient_name': self.patient_name
                }, f, indent=4)
            
            # Lưu từng nghiên cứu
            for study_uid, study_info in self.studies.items():
                # Tạo thư mục cho nghiên cứu
                study_dir = os.path.join(patient_dir, study_uid)
                os.makedirs(study_dir, exist_ok=True)
                
                # Lưu thông tin nghiên cứu
                with open(os.path.join(study_dir, "study_info.json"), 'w') as f:
                    json.dump({
                        'study_uid': study_uid,
                        'study_date': study_info['study_date'],
                        'study_time': study_info['study_time'],
                        'study_description': study_info['study_description']
                    }, f, indent=4)
                
                # Lưu từng series
                for series_uid, series_info in study_info['series'].items():
                    # Tạo thư mục cho series
                    series_dir = os.path.join(study_dir, series_uid)
                    os.makedirs(series_dir, exist_ok=True)
                    
                    # Lưu thông tin series
                    with open(os.path.join(series_dir, "series_info.json"), 'w') as f:
                        json.dump({
                            'series_uid': series_uid,
                            'modality': series_info['modality'],
                            'series_description': series_info['series_description']
                        }, f, indent=4)
                    
                    # Lưu từng dataset
                    for i, dataset in enumerate(series_info['datasets']):
                        # Tạo tên file
                        filename = f"{i+1:04d}.dcm"
                        file_path = os.path.join(series_dir, filename)
                        
                        # Lưu dataset
                        DicomWriter.save_file(dataset, file_path)
            
            logger.info(f"Đã lưu tập dữ liệu vào {directory}")
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi lưu tập dữ liệu: {str(e)}")
            return False
    
    @classmethod
    def load_from_directory(cls, directory: str) -> Optional['DicomDataset']:
        """
        Tải tập dữ liệu từ thư mục.
        
        Parameters
        ----------
        directory : str
            Đường dẫn đến thư mục
            
        Returns
        -------
        Optional[DicomDataset]
            Tập dữ liệu nếu tải thành công, None nếu không
        """
        try:
            # Kiểm tra thư mục có tồn tại không
            if not os.path.exists(directory) or not os.path.isdir(directory):
                logger.error(f"Thư mục {directory} không tồn tại")
                return None
            
            # Tìm thư mục bệnh nhân
            patient_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
            
            if not patient_dirs:
                logger.error(f"Không tìm thấy thư mục bệnh nhân trong {directory}")
                return None
            
            # Giả định chỉ có một bệnh nhân trong thư mục
            patient_dir = os.path.join(directory, patient_dirs[0])
            
            # Tải thông tin bệnh nhân
            patient_info_file = os.path.join(patient_dir, "patient_info.json")
            if not os.path.exists(patient_info_file):
                logger.error(f"Không tìm thấy file thông tin bệnh nhân {patient_info_file}")
                return None
            
            with open(patient_info_file, 'r') as f:
                patient_info = json.load(f)
            
            # Tạo đối tượng DicomDataset
            dicom_dataset = cls(
                patient_id=patient_info['patient_id'],
                patient_name=patient_info['patient_name']
            )
            
            # Tìm tất cả các thư mục nghiên cứu
            study_dirs = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d)) and d != "__pycache__"]
            
            for study_dir_name in study_dirs:
                study_dir = os.path.join(patient_dir, study_dir_name)
                
                # Tìm tất cả các thư mục series
                series_dirs = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
                
                for series_dir_name in series_dirs:
                    series_dir = os.path.join(study_dir, series_dir_name)
                    
                    # Tìm tất cả các file DICOM
                    dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
                    
                    for dicom_file in dicom_files:
                        file_path = os.path.join(series_dir, dicom_file)
                        
                        try:
                            # Đọc file DICOM
                            dataset = DicomReader.read_file(file_path)
                            
                            # Thêm vào tập dữ liệu
                            dicom_dataset.add_dataset(dataset)
                        except Exception as e:
                            logger.warning(f"Lỗi khi đọc file {file_path}: {str(e)}")
            
            logger.info(f"Đã tải tập dữ liệu từ {directory}")
            return dicom_dataset
        
        except Exception as e:
            logger.error(f"Lỗi khi tải tập dữ liệu: {str(e)}")
            return None


class DicomDatasetManager:
    """
    Lớp quản lý các tập dữ liệu DICOM.
    
    Lớp này quản lý việc lưu trữ và truy xuất các tập dữ liệu DICOM.
    """
    
    def __init__(self):
        """
        Khởi tạo DicomDatasetManager.
        """
        self.config = Config.get_instance()
        self.dicom_dir = self.config.get('dicom_dir')
        self.datasets = {}  # patient_id -> DicomDataset
        
        # Tạo thư mục DICOM nếu chưa tồn tại
        os.makedirs(self.dicom_dir, exist_ok=True)
    
    def get_all_patients(self) -> List[Dict[str, str]]:
        """
        Lấy danh sách tất cả bệnh nhân.
        
        Returns
        -------
        List[Dict[str, str]]
            Danh sách các bệnh nhân
        """
        result = []
        
        # Kiểm tra các thư mục trong thư mục DICOM
        for patient_id in os.listdir(self.dicom_dir):
            patient_dir = os.path.join(self.dicom_dir, patient_id)
            
            if os.path.isdir(patient_dir):
                # Tìm file thông tin bệnh nhân
                patient_info_file = os.path.join(patient_dir, "patient_info.json")
                
                if os.path.exists(patient_info_file):
                    try:
                        with open(patient_info_file, 'r') as f:
                            patient_info = json.load(f)
                        
                        result.append({
                            'patient_id': patient_info['patient_id'],
                            'patient_name': patient_info['patient_name']
                        })
                    except Exception as e:
                        logger.warning(f"Lỗi khi đọc thông tin bệnh nhân {patient_id}: {str(e)}")
        
        return result
    
    def load_patient(self, patient_id: str) -> Optional[DicomDataset]:
        """
        Tải dữ liệu bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        Optional[DicomDataset]
            Tập dữ liệu nếu tải thành công, None nếu không
        """
        # Kiểm tra xem đã tải chưa
        if patient_id in self.datasets:
            return self.datasets[patient_id]
        
        # Tạo đường dẫn đến thư mục bệnh nhân
        patient_dir = os.path.join(self.dicom_dir, patient_id)
        
        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(patient_dir) or not os.path.isdir(patient_dir):
            logger.error(f"Thư mục bệnh nhân {patient_dir} không tồn tại")
            return None
        
        # Tải dữ liệu
        dicom_dataset = DicomDataset.load_from_directory(self.dicom_dir)
        
        if dicom_dataset is not None:
            # Lưu vào cache
            self.datasets[patient_id] = dicom_dataset
        
        return dicom_dataset
    
    def import_dicom_data(self, dicom_files: List[str], patient_id: Optional[str] = None) -> Optional[DicomDataset]:
        """
        Nhập dữ liệu DICOM từ danh sách file.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách đường dẫn đến các file DICOM
        patient_id : Optional[str]
            ID của bệnh nhân, nếu None thì sẽ lấy từ file DICOM đầu tiên
            
        Returns
        -------
        Optional[DicomDataset]
            Tập dữ liệu nếu nhập thành công, None nếu không
        """
        try:
            if not dicom_files:
                logger.error("Danh sách file DICOM trống")
                return None
            
            # Đọc file DICOM đầu tiên để lấy thông tin bệnh nhân
            first_dataset = None
            for file_path in dicom_files:
                try:
                    first_dataset = DicomReader.read_file(file_path)
                    break
                except Exception:
                    continue
            
            if first_dataset is None:
                logger.error("Không thể đọc bất kỳ file DICOM nào")
                return None
            
            # Lấy thông tin bệnh nhân
            if patient_id is None:
                patient_id = getattr(first_dataset, 'PatientID', 'unknown')
            
            patient_name = getattr(first_dataset, 'PatientName', 'Unknown')
            
            # Tạo tập dữ liệu
            dicom_dataset = DicomDataset(patient_id, str(patient_name))
            
            # Thêm các dataset
            for file_path in dicom_files:
                try:
                    dataset = DicomReader.read_file(file_path)
                    dicom_dataset.add_dataset(dataset)
                except Exception as e:
                    logger.warning(f"Lỗi khi đọc file {file_path}: {str(e)}")
            
            # Lưu vào thư mục
            patient_dir = os.path.join(self.dicom_dir, patient_id)
            dicom_dataset.save_to_directory(self.dicom_dir)
            
            # Lưu vào cache
            self.datasets[patient_id] = dicom_dataset
            
            return dicom_dataset
        
        except Exception as e:
            logger.error(f"Lỗi khi nhập dữ liệu DICOM: {str(e)}")
            return None
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Xóa dữ liệu bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không
        """
        try:
            # Tạo đường dẫn đến thư mục bệnh nhân
            patient_dir = os.path.join(self.dicom_dir, patient_id)
            
            # Kiểm tra thư mục có tồn tại không
            if not os.path.exists(patient_dir) or not os.path.isdir(patient_dir):
                logger.error(f"Thư mục bệnh nhân {patient_dir} không tồn tại")
                return False
            
            # Xóa thư mục
            shutil.rmtree(patient_dir)
            
            # Xóa khỏi cache
            if patient_id in self.datasets:
                del self.datasets[patient_id]
            
            logger.info(f"Đã xóa dữ liệu bệnh nhân {patient_id}")
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi xóa dữ liệu bệnh nhân {patient_id}: {str(e)}")
            return False
    
    def get_patient_directory(self, patient_id: str) -> str:
        """
        Lấy đường dẫn đến thư mục bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        str
            Đường dẫn đến thư mục bệnh nhân
        """
        return os.path.join(self.dicom_dir, patient_id) 