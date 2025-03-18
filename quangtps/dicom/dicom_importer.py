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
        Phân loại các dataset DICOM theo modality.
        
        Parameters
        ----------
        datasets : List[pydicom.dataset.FileDataset]
            Danh sách các dataset DICOM
            
        Returns
        -------
        Dict[str, List[pydicom.dataset.FileDataset]]
            Từ điển với khóa là modality và giá trị là danh sách dataset
        """
        result = {}
        
        for ds in datasets:
            modality = self._get_modality(ds)
            
            if modality not in result:
                result[modality] = []
                
            result[modality].append(ds)
        
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