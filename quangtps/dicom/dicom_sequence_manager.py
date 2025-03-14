"""
Quản lý chuỗi DICOM.

Module này cung cấp các chức năng để quản lý chuỗi các dữ liệu DICOM,
cho phép tìm kiếm, sắp xếp và lọc các dataset DICOM theo các tiêu chí khác nhau.
"""

import os
import logging
from datetime import datetime
import pydicom
from pydicom.dataset import FileDataset
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

from quangtps.core.exceptions import DicomError
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.dicom_utils import get_patient_info, get_study_info, get_series_info

logger = logging.getLogger(__name__)

class DicomSequenceManager:
    """
    Lớp quản lý chuỗi DICOM.
    
    Class này cung cấp các phương thức để quản lý chuỗi DICOM,
    bao gồm việc tìm kiếm, sắp xếp và lọc các dataset theo các tiêu chí khác nhau.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng DicomSequenceManager."""
        self.datasets = []
        self.patient_dict = {}  # Dict[patient_id, List[dataset]]
        self.study_dict = {}    # Dict[study_uid, List[dataset]]
        self.series_dict = {}   # Dict[series_uid, List[dataset]]
        self.reader = DicomReader()
    
    def load_directory(self, directory: str, recursive: bool = True, file_pattern: str = "*.dcm") -> int:
        """
        Tải tất cả các file DICOM từ một thư mục.
        
        Parameters
        ----------
        directory : str
            Đường dẫn đến thư mục chứa file DICOM
        recursive : bool, optional
            Có tìm kiếm đệ quy trong các thư mục con hay không
        file_pattern : str, optional
            Mẫu tên file để lọc
            
        Returns
        -------
        int
            Số lượng file đã tải
            
        Raises
        ------
        DicomError
            Nếu không thể tải thư mục
        """
        try:
            import glob
            
            # Xây dựng mẫu tìm kiếm
            if recursive:
                search_pattern = os.path.join(directory, "**", file_pattern)
                files = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(directory, file_pattern)
                files = glob.glob(search_pattern)
            
            # Tải từng file
            loaded_count = 0
            for file_path in files:
                try:
                    dataset = self.reader.read_file(file_path)
                    self.add_dataset(dataset)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Error loading file {file_path}: {str(e)}")
            
            logger.info(f"Loaded {loaded_count} DICOM files from {directory}")
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading directory {directory}: {str(e)}")
            raise DicomError(f"Error loading directory {directory}: {str(e)}")
    
    def add_dataset(self, dataset: FileDataset) -> None:
        """
        Thêm một dataset DICOM vào manager.
        
        Parameters
        ----------
        dataset : FileDataset
            Dataset DICOM cần thêm
        """
        self.datasets.append(dataset)
        
        # Cập nhật từ điển bệnh nhân
        patient_id = dataset.PatientID if hasattr(dataset, 'PatientID') else 'unknown'
        if patient_id not in self.patient_dict:
            self.patient_dict[patient_id] = []
        self.patient_dict[patient_id].append(dataset)
        
        # Cập nhật từ điển nghiên cứu
        study_uid = dataset.StudyInstanceUID if hasattr(dataset, 'StudyInstanceUID') else 'unknown'
        if study_uid not in self.study_dict:
            self.study_dict[study_uid] = []
        self.study_dict[study_uid].append(dataset)
        
        # Cập nhật từ điển series
        series_uid = dataset.SeriesInstanceUID if hasattr(dataset, 'SeriesInstanceUID') else 'unknown'
        if series_uid not in self.series_dict:
            self.series_dict[series_uid] = []
        self.series_dict[series_uid].append(dataset)
    
    def add_datasets(self, datasets: List[FileDataset]) -> None:
        """
        Thêm nhiều dataset DICOM vào manager.
        
        Parameters
        ----------
        datasets : List[FileDataset]
            Danh sách các dataset DICOM cần thêm
        """
        for dataset in datasets:
            self.add_dataset(dataset)
    
    def get_patients(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các bệnh nhân.
        
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin bệnh nhân
        """
        patients = []
        
        for patient_id, datasets in self.patient_dict.items():
            if datasets:
                # Lấy thông tin bệnh nhân từ dataset đầu tiên
                patient_info = get_patient_info(datasets[0])
                patient_info['NumDatasets'] = len(datasets)
                patients.append(patient_info)
        
        return patients
    
    def get_studies(self, patient_id: str = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các nghiên cứu.
        
        Parameters
        ----------
        patient_id : str, optional
            ID bệnh nhân để lọc nghiên cứu
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin nghiên cứu
        """
        studies = []
        
        if patient_id:
            # Lọc nghiên cứu theo bệnh nhân
            if patient_id in self.patient_dict:
                patient_datasets = self.patient_dict[patient_id]
                
                # Nhóm theo StudyInstanceUID
                study_groups = {}
                for ds in patient_datasets:
                    if hasattr(ds, 'StudyInstanceUID'):
                        study_uid = ds.StudyInstanceUID
                        if study_uid not in study_groups:
                            study_groups[study_uid] = []
                        study_groups[study_uid].append(ds)
                
                # Lấy thông tin từng nghiên cứu
                for study_uid, study_datasets in study_groups.items():
                    if study_datasets:
                        study_info = get_study_info(study_datasets[0])
                        study_info['NumDatasets'] = len(study_datasets)
                        studies.append(study_info)
        else:
            # Lấy tất cả nghiên cứu
            for study_uid, study_datasets in self.study_dict.items():
                if study_datasets:
                    study_info = get_study_info(study_datasets[0])
                    study_info['NumDatasets'] = len(study_datasets)
                    studies.append(study_info)
        
        return studies
    
    def get_series(self, study_uid: str = None) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các series.
        
        Parameters
        ----------
        study_uid : str, optional
            Study Instance UID để lọc series
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin series
        """
        series_list = []
        
        if study_uid:
            # Lọc series theo nghiên cứu
            if study_uid in self.study_dict:
                study_datasets = self.study_dict[study_uid]
                
                # Nhóm theo SeriesInstanceUID
                series_groups = {}
                for ds in study_datasets:
                    if hasattr(ds, 'SeriesInstanceUID'):
                        series_uid = ds.SeriesInstanceUID
                        if series_uid not in series_groups:
                            series_groups[series_uid] = []
                        series_groups[series_uid].append(ds)
                
                # Lấy thông tin từng series
                for series_uid, series_datasets in series_groups.items():
                    if series_datasets:
                        series_info = get_series_info(series_datasets[0])
                        series_info['NumDatasets'] = len(series_datasets)
                        series_list.append(series_info)
        else:
            # Lấy tất cả series
            for series_uid, series_datasets in self.series_dict.items():
                if series_datasets:
                    series_info = get_series_info(series_datasets[0])
                    series_info['NumDatasets'] = len(series_datasets)
                    series_list.append(series_info)
        
        return series_list
    
    def get_datasets_by_patient(self, patient_id: str) -> List[FileDataset]:
        """
        Lấy tất cả các dataset của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID bệnh nhân
            
        Returns
        -------
        List[FileDataset]
            Danh sách các dataset
        """
        if patient_id in self.patient_dict:
            return self.patient_dict[patient_id]
        return []
    
    def get_datasets_by_study(self, study_uid: str) -> List[FileDataset]:
        """
        Lấy tất cả các dataset của một nghiên cứu.
        
        Parameters
        ----------
        study_uid : str
            Study Instance UID
            
        Returns
        -------
        List[FileDataset]
            Danh sách các dataset
        """
        if study_uid in self.study_dict:
            return self.study_dict[study_uid]
        return []
    
    def get_datasets_by_series(self, series_uid: str) -> List[FileDataset]:
        """
        Lấy tất cả các dataset của một series.
        
        Parameters
        ----------
        series_uid : str
            Series Instance UID
            
        Returns
        -------
        List[FileDataset]
            Danh sách các dataset
        """
        if series_uid in self.series_dict:
            return self.series_dict[series_uid]
        return []
    
    def filter_datasets(self, predicate: Callable[[FileDataset], bool]) -> List[FileDataset]:
        """
        Lọc dataset theo điều kiện.
        
        Parameters
        ----------
        predicate : Callable[[FileDataset], bool]
            Hàm điều kiện, trả về True nếu muốn giữ lại dataset
            
        Returns
        -------
        List[FileDataset]
            Danh sách các dataset đáp ứng điều kiện
        """
        return [ds for ds in self.datasets if predicate(ds)]
    
    def group_datasets_by_modality(self) -> Dict[str, List[FileDataset]]:
        """
        Nhóm các dataset theo modality.
        
        Returns
        -------
        Dict[str, List[FileDataset]]
            Từ điển ánh xạ từ modality đến danh sách các dataset
        """
        modality_groups = {}
        
        for ds in self.datasets:
            modality = ds.Modality if hasattr(ds, 'Modality') else 'unknown'
            
            if modality not in modality_groups:
                modality_groups[modality] = []
            
            modality_groups[modality].append(ds)
        
        return modality_groups
    
    def sort_datasets_by_acquisition_time(self, datasets: List[FileDataset]) -> List[FileDataset]:
        """
        Sắp xếp các dataset theo thời gian thu nhận.
        
        Parameters
        ----------
        datasets : List[FileDataset]
            Danh sách các dataset cần sắp xếp
            
        Returns
        -------
        List[FileDataset]
            Danh sách đã sắp xếp
        """
        def get_acquisition_time(ds):
            acquisition_date = None
            acquisition_time = None
            
            # Thử các tag thông dụng cho thông tin thời gian
            if hasattr(ds, 'AcquisitionDate'):
                acquisition_date = ds.AcquisitionDate
            elif hasattr(ds, 'ContentDate'):
                acquisition_date = ds.ContentDate
            elif hasattr(ds, 'SeriesDate'):
                acquisition_date = ds.SeriesDate
            
            if hasattr(ds, 'AcquisitionTime'):
                acquisition_time = ds.AcquisitionTime
            elif hasattr(ds, 'ContentTime'):
                acquisition_time = ds.ContentTime
            elif hasattr(ds, 'SeriesTime'):
                acquisition_time = ds.SeriesTime
            
            # Nếu không có thông tin thời gian, trả về giá trị mặc định
            if not acquisition_date or not acquisition_time:
                return "00000000000000"
            
            # Kết hợp ngày và thời gian
            combined = f"{acquisition_date}{acquisition_time.split('.')[0]}"
            return combined
        
        # Sắp xếp theo thời gian tăng dần
        return sorted(datasets, key=get_acquisition_time)
    
    def find_datasets_by_date_range(self, start_date: str, end_date: str) -> List[FileDataset]:
        """
        Tìm các dataset trong khoảng ngày.
        
        Parameters
        ----------
        start_date : str
            Ngày bắt đầu (định dạng YYYYMMDD)
        end_date : str
            Ngày kết thúc (định dạng YYYYMMDD)
            
        Returns
        -------
        List[FileDataset]
            Danh sách các dataset trong khoảng ngày
        """
        def is_in_date_range(ds):
            study_date = None
            
            # Thử các tag thông dụng cho thông tin ngày
            if hasattr(ds, 'StudyDate'):
                study_date = ds.StudyDate
            elif hasattr(ds, 'SeriesDate'):
                study_date = ds.SeriesDate
            elif hasattr(ds, 'ContentDate'):
                study_date = ds.ContentDate
            elif hasattr(ds, 'AcquisitionDate'):
                study_date = ds.AcquisitionDate
            
            # Nếu không có thông tin ngày, bỏ qua
            if not study_date:
                return False
            
            # Kiểm tra xem có trong khoảng không
            return start_date <= study_date <= end_date
        
        return self.filter_datasets(is_in_date_range)
    
    def save_to_directory(self, datasets: List[FileDataset], output_dir: str) -> List[str]:
        """
        Lưu các dataset vào thư mục.
        
        Parameters
        ----------
        datasets : List[FileDataset]
            Danh sách các dataset cần lưu
        output_dir : str
            Thư mục đầu ra
            
        Returns
        -------
        List[str]
            Danh sách các file đã lưu
            
        Raises
        ------
        DicomError
            Nếu không thể lưu dataset
        """
        try:
            # Tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_files = []
            
            for i, ds in enumerate(datasets):
                # Tạo tên file dựa trên các tag DICOM
                if hasattr(ds, 'SOPInstanceUID'):
                    # Sử dụng SOPInstanceUID nếu có
                    filename = f"{ds.SOPInstanceUID}.dcm"
                else:
                    # Sử dụng chỉ số nếu không có
                    filename = f"dicom_{i:04d}.dcm"
                
                # Đường dẫn đầy đủ
                output_path = os.path.join(output_dir, filename)
                
                # Lưu file
                ds.save_as(output_path)
                output_files.append(output_path)
            
            logger.info(f"Saved {len(output_files)} DICOM files to {output_dir}")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Error saving datasets to directory: {str(e)}")
            raise DicomError(f"Error saving datasets to directory: {str(e)}")
    
    def clear(self) -> None:
        """Xóa tất cả các dataset."""
        self.datasets.clear()
        self.patient_dict.clear()
        self.study_dict.clear()
        self.series_dict.clear()
        logger.info("Cleared all datasets")
