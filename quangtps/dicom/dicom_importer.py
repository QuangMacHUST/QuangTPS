#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module xử lý việc nhập dữ liệu DICOM từ nhiều nguồn khác nhau.

Moduel này cung cấp các phương thức để nhập dữ liệu DICOM từ:
- Thư mục trên đĩa cục bộ
- Tệp tin nén (ZIP, TAR)
- Các máy chủ PACS qua giao thức DICOM
- Nhập trực tiếp từ máy quét (scanner)
- Các API web
"""

import os
import logging
import tempfile
import zipfile
import tarfile
import shutil
import urllib.request
import pydicom
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.pacs_client import PACSClient
from quangtps.database.patient_db import PatientDatabase, Patient, Study, Series
from quangtps.core.exceptions import DicomError, IOError, NetworkError, AuthenticationError

logger = logging.getLogger(__name__)

class DicomImporter:
    """
    Lớp xử lý việc nhập dữ liệu DICOM từ nhiều nguồn khác nhau.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Khởi tạo DicomImporter.
        
        Parameters
        ----------
        temp_dir : str, optional
            Thư mục tạm để lưu các file DICOM tạm thời khi giải nén
            hoặc tải về. Nếu None, sẽ sử dụng thư mục tạm của hệ thống.
        """
        self.temp_dir = temp_dir if temp_dir else tempfile.gettempdir()
        self.reader = DicomReader()
        self.pacs_client = PACSClient()
        self.patient_db = PatientDatabase()
        
        # Tạo thư mục tạm nếu không tồn tại
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def import_from_directory(self, directory_path: str) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Nhập tất cả file DICOM từ một thư mục.
        
        Parameters
        ----------
        directory_path : str
            Đường dẫn đến thư mục chứa file DICOM
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là loại DICOM (CT, RTSTRUCT, RTDOSE, RTPLAN) 
            và giá trị là danh sách các dataset tương ứng
            
        Raises
        ------
        IOError
            Nếu thư mục không tồn tại
        """
        if not os.path.exists(directory_path):
            raise IOError(f"Directory not found", file_path=directory_path)
            
        logger.info(f"Importing DICOM files from directory: {directory_path}")
        
        # Đọc tất cả file DICOM trong thư mục
        dicom_datasets = self.reader.read_directory(directory_path)
        
        # Phân loại các file DICOM
        categorized_datasets = self._categorize_datasets(dicom_datasets)
        
        logger.info(f"Imported {len(dicom_datasets)} DICOM files from {directory_path}")
        for modality, datasets in categorized_datasets.items():
            logger.info(f"Found {len(datasets)} {modality} datasets")
            
        return categorized_datasets
    
    def import_from_zip(self, zip_file_path: str) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Nhập tất cả file DICOM từ một file ZIP.
        
        Parameters
        ----------
        zip_file_path : str
            Đường dẫn đến file ZIP chứa file DICOM
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là loại DICOM (CT, RTSTRUCT, RTDOSE, RTPLAN) 
            và giá trị là danh sách các dataset tương ứng
            
        Raises
        ------
        IOError
            Nếu file ZIP không tồn tại hoặc không thể giải nén
        """
        if not os.path.exists(zip_file_path):
            raise IOError(f"ZIP file not found", file_path=zip_file_path)
            
        logger.info(f"Importing DICOM files from ZIP: {zip_file_path}")
        
        # Tạo thư mục tạm để giải nén
        extract_dir = os.path.join(self.temp_dir, f"dicom_extract_{os.path.basename(zip_file_path)}")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Giải nén file ZIP
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Nhập DICOM từ thư mục đã giải nén
            return self.import_from_directory(extract_dir)
            
        except zipfile.BadZipFile:
            raise IOError(f"Invalid ZIP file", file_path=zip_file_path)
        except Exception as e:
            raise IOError(f"Error extracting ZIP file: {str(e)}", file_path=zip_file_path)
        finally:
            # Xóa thư mục tạm
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
    
    def import_from_tar(self, tar_file_path: str) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Nhập tất cả file DICOM từ một file TAR (có thể nén bằng gzip).
        
        Parameters
        ----------
        tar_file_path : str
            Đường dẫn đến file TAR chứa file DICOM
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là loại DICOM (CT, RTSTRUCT, RTDOSE, RTPLAN) 
            và giá trị là danh sách các dataset tương ứng
            
        Raises
        ------
        IOError
            Nếu file TAR không tồn tại hoặc không thể giải nén
        """
        if not os.path.exists(tar_file_path):
            raise IOError(f"TAR file not found", file_path=tar_file_path)
            
        logger.info(f"Importing DICOM files from TAR: {tar_file_path}")
        
        # Tạo thư mục tạm để giải nén
        extract_dir = os.path.join(self.temp_dir, f"dicom_extract_{os.path.basename(tar_file_path)}")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Giải nén file TAR
            with tarfile.open(tar_file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
            
            # Nhập DICOM từ thư mục đã giải nén
            return self.import_from_directory(extract_dir)
            
        except tarfile.ReadError:
            raise IOError(f"Invalid TAR file", file_path=tar_file_path)
        except Exception as e:
            raise IOError(f"Error extracting TAR file: {str(e)}", file_path=tar_file_path)
        finally:
            # Xóa thư mục tạm
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
    
    def import_from_url(self, url: str) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Tải và nhập file DICOM từ một URL.
        
        Parameters
        ----------
        url : str
            URL tới file DICOM hoặc file nén chứa DICOM
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là loại DICOM (CT, RTSTRUCT, RTDOSE, RTPLAN) 
            và giá trị là danh sách các dataset tương ứng
            
        Raises
        ------
        NetworkError
            Nếu không thể tải file từ URL
        """
        logger.info(f"Importing DICOM from URL: {url}")
        
        # Tạo tên file tạm để lưu file tải về
        temp_file = os.path.join(self.temp_dir, f"dicom_download_{os.path.basename(url)}")
        
        try:
            # Tải file từ URL
            urllib.request.urlretrieve(url, temp_file)
            
            # Kiểm tra loại file và nhập phù hợp
            if temp_file.lower().endswith('.zip'):
                return self.import_from_zip(temp_file)
            elif temp_file.lower().endswith(('.tar', '.tar.gz', '.tgz')):
                return self.import_from_tar(temp_file)
            else:
                # Giả sử đây là một file DICOM
                dataset = self.reader.read_file(temp_file)
                modality = self._get_modality(dataset)
                return {modality: [dataset]}
                
        except urllib.error.URLError as e:
            raise NetworkError(f"Error downloading file from URL: {str(e)}", url=url)
        except Exception as e:
            raise NetworkError(f"Error processing file from URL: {str(e)}", url=url)
        finally:
            # Xóa file tạm
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def import_from_pacs(self, query_params: Dict[str, Any]) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Truy vấn và nhập dữ liệu DICOM từ máy chủ PACS.
        
        Parameters
        ----------
        query_params : Dict[str, Any]
            Tham số truy vấn PACS (PatientID, StudyInstanceUID, etc.)
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là loại DICOM (CT, RTSTRUCT, RTDOSE, RTPLAN) 
            và giá trị là danh sách các dataset tương ứng
            
        Raises
        ------
        NetworkError
            Nếu không thể kết nối đến máy chủ PACS
        """
        logger.info(f"Importing DICOM from PACS with query: {query_params}")
        
        try:
            # Sử dụng PACSClient để truy vấn và tải dữ liệu
            datasets = self.pacs_client.query_and_retrieve(query_params)
            
            # Phân loại các dataset
            categorized_datasets = self._categorize_datasets(datasets)
            
            logger.info(f"Imported {len(datasets)} DICOM files from PACS")
            
            return categorized_datasets
            
        except Exception as e:
            raise NetworkError(f"Error importing from PACS: {str(e)}")
    
    def import_from_scanner(self, scanner_name: str, patient_id: str) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Nhập dữ liệu DICOM trực tiếp từ máy quét.
        
        Parameters
        ----------
        scanner_name : str
            Tên hoặc địa chỉ của máy quét
        patient_id : str
            ID bệnh nhân để lọc dữ liệu
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là loại DICOM (CT, RTSTRUCT, RTDOSE, RTPLAN) 
            và giá trị là danh sách các dataset tương ứng
            
        Raises
        ------
        NetworkError
            Nếu không thể kết nối đến máy quét
        NotImplementedError
            Nếu chức năng chưa được triển khai
        """
        logger.info(f"Importing DICOM from scanner {scanner_name} for patient {patient_id}")
        
        # Hiện tại, đây là một tính năng giả định
        # Trong triển khai thực tế, cần có kết nối đến máy quét qua giao thức DICOM
        raise NotImplementedError("Importing from scanner is not yet implemented")
    
    def _categorize_datasets(self, datasets: List[pydicom.dataset.FileDataset]) -> Dict[str, List[pydicom.dataset.FileDataset]]:
        """
        Phân loại các DICOM datasets theo modality.
        
        Parameters
        ----------
        datasets : List[pydicom.dataset.FileDataset]
            Danh sách các datasets cần phân loại
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Dictionary với key là tên modality, value là danh sách các datasets thuộc modality đó
        """
        result = {}
        invalid_datasets = []
        
        if not datasets:
            logger.warning("Không có DICOM datasets để phân loại")
            return {}
        
        logger.info(f"Phân loại {len(datasets)} DICOM datasets")
        
        for ds in datasets:
            try:
                # Validate that the dataset is a FileDataset
                if not isinstance(ds, pydicom.dataset.FileDataset):
                    logger.warning(f"Bỏ qua đối tượng không phải FileDataset: {type(ds)}")
                    invalid_datasets.append(ds)
                    continue
                
                # Kiểm tra xem dataset có thuộc tính cơ bản không
                basic_attrs = ['SOPClassUID', 'InstanceNumber']
                missing_attrs = [attr for attr in basic_attrs if not hasattr(ds, attr)]
                if missing_attrs:
                    logger.warning(f"Dataset thiếu các thuộc tính cơ bản: {missing_attrs}")
                    # Ghi chi tiết hơn để hỗ trợ debug
                    logger.debug(f"Dataset info: {str(ds)[:200]}...")
                    
                    # Nếu thiếu thuộc tính quan trọng nhưng vẫn muốn phân loại
                    # Có thể bỏ qua hoặc thử lấy thông tin từ các thuộc tính khác
                
                # Get modality, with fallbacks if not directly available
                modality = self._get_modality(ds)
                
                # If modality is unknown, try to infer it from SOP Class UID
                if modality == 'UNKNOWN' and hasattr(ds, 'SOPClassUID'):
                    sop_class = ds.SOPClassUID
                    # Map common SOP Class UIDs to modalities
                    sop_to_modality = {
                        '1.2.840.10008.5.1.4.1.1.2': 'CT',        # CT Image Storage
                        '1.2.840.10008.5.1.4.1.1.4': 'MR',        # MR Image Storage
                        '1.2.840.10008.5.1.4.1.1.128': 'PT',      # PET Image Storage
                        '1.2.840.10008.5.1.4.1.1.481.3': 'RTSTRUCT', # RT Structure Set Storage
                        '1.2.840.10008.5.1.4.1.1.481.2': 'RTDOSE',   # RT Dose Storage
                        '1.2.840.10008.5.1.4.1.1.481.5': 'RTPLAN',   # RT Plan Storage
                        '1.2.840.10008.5.1.4.1.1.481.1': 'RTIMAGE',  # RT Image Storage
                        '1.2.840.10008.5.1.4.1.1.7': 'SC',        # Secondary Capture Image Storage
                        '1.2.840.10008.5.1.4.1.1.6.1': 'US',      # Ultrasound Image Storage
                    }
                    
                    # Tìm kiếm tất cả UIDs có thể phù hợp
                    for uid_prefix, mod in sop_to_modality.items():
                        if str(sop_class).startswith(uid_prefix):
                            modality = mod
                            logger.info(f"Suy ra modality '{modality}' từ SOPClassUID: {sop_class}")
                            break
                
                # Kiểm tra thêm nếu vẫn không xác định được
                if modality == 'UNKNOWN':
                    # Thử kiểm tra các thuộc tính khác
                    if hasattr(ds, 'StudyDescription'):
                        study_desc = str(ds.StudyDescription).upper()
                        if 'CT' in study_desc:
                            modality = 'CT'
                        elif 'MR' in study_desc:
                            modality = 'MR'
                        elif 'PET' in study_desc:
                            modality = 'PT'
                        logger.info(f"Suy ra modality '{modality}' từ StudyDescription: {study_desc}")
                
                # Kiểm tra tính hợp lệ của modality với kiểu dữ liệu hình ảnh
                if modality in ['CT', 'MR', 'PT'] and not hasattr(ds, 'PixelData'):
                    logger.warning(f"Dataset có modality '{modality}' nhưng không có PixelData")
                    # Đánh dấu là không hợp lệ nếu đây là vấn đề nghiêm trọng
                    if modality == 'CT' and hasattr(ds, 'PatientID'):  # Giảm nhẹ cảnh báo nếu có thể xác định bệnh nhân
                        logger.warning(f"Dữ liệu CT không có pixel data cho bệnh nhân: {ds.PatientID}")
                        
                    # Vẫn thêm vào danh sách theo modality để có thể xử lý sau
                
                # Initialize result entry if needed
                if modality not in result:
                    result[modality] = []
                
                # Add dataset to result
                result[modality].append(ds)
                
            except Exception as e:
                logger.error(f"Lỗi khi phân loại DICOM dataset: {str(e)}")
                logger.debug(f"Thông tin dataset: {str(ds)[:200]}...")
                
                # Ghi log chi tiết hơn trong debug mode
                import traceback
                logger.debug(traceback.format_exc())
                
                invalid_datasets.append(ds)
        
        # Log information about invalid datasets
        if invalid_datasets:
            logger.warning(f"Tìm thấy {len(invalid_datasets)} DICOM datasets không hợp lệ và bị bỏ qua")
        
        # Log summary of results
        valid_count = sum(len(datasets) for datasets in result.values())
        logger.info(f"Đã phân loại {valid_count} DICOM datasets hợp lệ thành {len(result)} modalities")
        for modality, mod_datasets in result.items():
            logger.info(f"  - {modality}: {len(mod_datasets)} datasets")
        
        return result
    
    def _get_modality(self, dataset: pydicom.dataset.FileDataset) -> str:
        """
        Lấy modality từ dataset DICOM.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset DICOM
            
        Returns
        -------
        str
            Modality của dataset (CT, RTSTRUCT, RTDOSE, RTPLAN, etc.)
            hoặc 'UNKNOWN' nếu không xác định được
        """
        if not hasattr(dataset, 'Modality'):
            return 'UNKNOWN'
            
        modality = dataset.Modality
        
        # Tinh chỉnh modality để phù hợp với các nhóm
        if modality == 'CT' or modality == 'MR' or modality == 'PT':
            return modality
        elif modality == 'RTSTRUCT':
            return 'RTSTRUCT'
        elif modality == 'RTPLAN':
            return 'RTPLAN'
        elif modality == 'RTDOSE':
            return 'RTDOSE'
        elif modality == 'RTIMAGE':
            return 'RTIMAGE'
        else:
            return modality 

    def import_for_patient(self, directory_path: str, patient_id: Optional[str] = None) -> str:
        """
        Nhập dữ liệu DICOM cho một bệnh nhân cụ thể.
        
        Parameters
        ----------
        directory_path : str
            Đường dẫn đến thư mục chứa file DICOM
        patient_id : str, optional
            ID của bệnh nhân cần nhập dữ liệu. Nếu None, sẽ tạo bệnh nhân mới từ thông tin DICOM.
            
        Returns
        -------
        str
            ID của bệnh nhân
            
        Raises
        ------
        IOError
            Nếu thư mục không tồn tại
        DicomError
            Nếu dữ liệu DICOM không hợp lệ hoặc không có thông tin bệnh nhân
        """
        categorized_datasets = self.import_from_directory(directory_path)
        
        # Lấy thông tin bệnh nhân từ dữ liệu DICOM
        patient_info = self._extract_patient_info(categorized_datasets)
        
        if not patient_info:
            raise DicomError("No valid patient information found in DICOM data")
        
        # Nếu không có patient_id, tạo bệnh nhân mới
        if not patient_id:
            try:
                # Kiểm tra xem bệnh nhân đã tồn tại chưa (dựa vào Patient ID trong DICOM)
                if 'dicom_id' in patient_info and patient_info['dicom_id']:
                    existing_patients = self.patient_db.search_patients(
                        query={'dicom_id': patient_info['dicom_id']}
                    )
                    if existing_patients:
                        patient_id = existing_patients[0]['id']
                        logger.info(f"Found existing patient with DICOM ID {patient_info['dicom_id']}, using patient_id: {patient_id}")
                
                # Nếu vẫn không có patient_id, tạo mới
                if not patient_id:
                    metadata = {
                        'dicom_id': patient_info.get('dicom_id', ''),
                        'notes': f"Imported from {directory_path}"
                    }
                    
                    patient_id = self.patient_db.create_patient(
                        name=patient_info.get('name', 'Unknown'),
                        birth_date=patient_info.get('birth_date'),
                        gender=patient_info.get('gender'),
                        metadata=metadata
                    )
                    logger.info(f"Created new patient with ID: {patient_id}")
            except Exception as e:
                logger.error(f"Error creating patient: {str(e)}")
                raise DicomError(f"Failed to create patient: {str(e)}")
        
        # Tạo nghiên cứu mới cho bệnh nhân
        study_info = self._extract_study_info(categorized_datasets)
        study = Study(
            description=study_info.get('description', 'Imported Study'),
            date=study_info.get('date'),
            patient_id=patient_id,
            metadata=study_info.get('metadata', {})
        )
        
        # Tạo các chuỗi cho từng loại dữ liệu
        for modality, datasets in categorized_datasets.items():
            if not datasets:
                continue
                
            # Tạo thư mục lưu trữ
            storage_dir = os.path.join(
                "data", "patients", patient_id, 
                "studies", study.uid, 
                "series", modality.lower()
            )
            os.makedirs(storage_dir, exist_ok=True)
            
            # Lưu file DICOM
            file_paths = []
            for i, dataset in enumerate(datasets):
                try:
                    file_name = f"{modality}_{i+1:04d}.dcm"
                    file_path = os.path.join(storage_dir, file_name)
                    
                    # Ensure the dataset is valid before saving
                    if not hasattr(dataset, 'SOPInstanceUID'):
                        logger.warning(f"Dataset {i+1} missing SOPInstanceUID, generating a placeholder")
                        dataset.SOPInstanceUID = pydicom.uid.generate_uid()
                    
                    # Ensure pixel data is valid for image modalities
                    if modality in ['CT', 'MR', 'PT'] and not hasattr(dataset, 'PixelData'):
                        logger.warning(f"Image dataset {i+1} missing PixelData, skipping")
                        continue
                    
                    # Make sure the dataset can be serialized
                    try:
                        dataset.save_as(file_path)
                        file_paths.append(file_path)
                        logger.debug(f"Saved DICOM file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error saving DICOM dataset {i+1}: {str(e)}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error processing DICOM dataset {i+1}: {str(e)}")
                    # Continue with other datasets
                    continue
            
            # Skip if no valid files were saved
            if not file_paths:
                logger.warning(f"No valid DICOM files saved for modality {modality}, skipping series creation")
                continue
            
            # Tạo chuỗi
            try:
                series = Series(
                    description=f"{modality} Series",
                    modality=modality,
                    study_id=study.uid,
                    metadata={
                        'count': len(file_paths),
                        'first_instance_date': self._get_acquisition_date(datasets[0]) if datasets else ''
                    }
                )
                
                # Thêm các file vào chuỗi
                for file_path in file_paths:
                    series.add_file(file_path)
                    
                # Thêm chuỗi vào nghiên cứu
                study.add_series(series)
                logger.info(f"Created series for modality {modality} with {len(file_paths)} files")
            except Exception as e:
                logger.error(f"Error creating series for modality {modality}: {str(e)}")
                # Continue with other modalities
        
        # Lưu nghiên cứu vào cơ sở dữ liệu
        self.patient_db.add_study_to_patient(patient_id, study)
        
        return patient_id
    
    def _extract_patient_info(self, categorized_datasets: Dict[str, List[pydicom.dataset.FileDataset]]) -> Dict[str, Any]:
        """
        Trích xuất thông tin bệnh nhân từ dữ liệu DICOM.
        
        Parameters
        ----------
        categorized_datasets : Dict[str, List[pydicom.dataset.FileDataset]]
            Dữ liệu DICOM đã phân loại
            
        Returns
        -------
        Dict[str, Any]
            Thông tin bệnh nhân
        """
        # Lấy một dataset đại diện (dùng dataset đầu tiên tìm thấy)
        representative_dataset = None
        for datasets in categorized_datasets.values():
            if datasets:
                representative_dataset = datasets[0]
                break
                
        if not representative_dataset:
            return {}
            
        try:
            # Trích xuất thông tin bệnh nhân
            patient_info = {
                'name': getattr(representative_dataset, 'PatientName', 'Unknown').original_string if hasattr(getattr(representative_dataset, 'PatientName', 'Unknown'), 'original_string') else str(getattr(representative_dataset, 'PatientName', 'Unknown')),
                'dicom_id': getattr(representative_dataset, 'PatientID', ''),
                'birth_date': getattr(representative_dataset, 'PatientBirthDate', None),
                'gender': getattr(representative_dataset, 'PatientSex', None),
            }
            
            # Chuyển đổi giới tính theo quy ước của hệ thống
            gender_map = {'M': 'male', 'F': 'female', 'O': 'other'}
            if patient_info['gender'] in gender_map:
                patient_info['gender'] = gender_map[patient_info['gender']]
                
            return patient_info
            
        except Exception as e:
            logger.warning(f"Error extracting patient info: {str(e)}")
            return {}
    
    def _extract_study_info(self, categorized_datasets: Dict[str, List[pydicom.dataset.FileDataset]]) -> Dict[str, Any]:
        """
        Trích xuất thông tin nghiên cứu từ dữ liệu DICOM.
        
        Parameters
        ----------
        categorized_datasets : Dict[str, List[pydicom.dataset.FileDataset]]
            Dữ liệu DICOM đã phân loại
            
        Returns
        -------
        Dict[str, Any]
            Thông tin nghiên cứu
        """
        # Lấy một dataset đại diện
        representative_dataset = None
        for datasets in categorized_datasets.values():
            if datasets:
                representative_dataset = datasets[0]
                break
                
        if not representative_dataset:
            return {}
            
        try:
            # Trích xuất thông tin nghiên cứu
            study_info = {
                'description': getattr(representative_dataset, 'StudyDescription', 'Imported Study'),
                'date': getattr(representative_dataset, 'StudyDate', None),
                'study_instance_uid': getattr(representative_dataset, 'StudyInstanceUID', ''),
                'metadata': {
                    'accession_number': getattr(representative_dataset, 'AccessionNumber', ''),
                    'study_id': getattr(representative_dataset, 'StudyID', ''),
                    'referring_physician': str(getattr(representative_dataset, 'ReferringPhysicianName', '')),
                }
            }
            
            return study_info
            
        except Exception as e:
            logger.warning(f"Error extracting study info: {str(e)}")
            return {}
    
    def _get_acquisition_date(self, dataset: pydicom.dataset.FileDataset) -> str:
        """
        Lấy ngày thu nhận dữ liệu từ dataset DICOM.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            DICOM dataset
            
        Returns
        -------
        str
            Ngày thu nhận dữ liệu dạng ISO
        """
        try:
            # Thử lấy từ AcquisitionDate nếu có
            if hasattr(dataset, 'AcquisitionDate'):
                return dataset.AcquisitionDate
            # Thử lấy từ SeriesDate
            elif hasattr(dataset, 'SeriesDate'):
                return dataset.SeriesDate
            # Hoặc lấy từ StudyDate nếu không có các trường trên
            elif hasattr(dataset, 'StudyDate'):
                return dataset.StudyDate
            else:
                return ""
        except Exception:
            return "" 