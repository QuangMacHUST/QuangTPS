#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp chức năng đọc và xử lý file DICOM cho hệ thống QuangTPS.

Module này bao gồm các lớp và hàm để tải, xử lý và tổ chức dữ liệu DICOM
từ các nguồn khác nhau, bao gồm file riêng lẻ và series DICOM.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional

# Import các thư viện PyQt cho GUI
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QTreeWidget, QTreeWidgetItem, QProgressBar,
    QSplitter, QGridLayout, QGroupBox, QComboBox, QMessageBox,
    QTabWidget, QFrame, QLineEdit, QCheckBox, QRadioButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

# Import các module cần thiết từ quangtps
from quangtps.imaging.image import Image
from quangtps.database.patient_db import Patient, Study, Series, PatientDatabase
from quangtps.database.image_db import ImageSeries, ImageDatabase
from quangtps.core.logging import get_logger

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

logger = get_logger(__name__)


class DicomSeries:
    """Lớp đại diện cho một chuỗi các file DICOM."""
    
    def __init__(self, series_id: str = "", description: str = ""):
        """
        Khởi tạo một chuỗi DICOM.
        
        Parameters
        ----------
        series_id : str
            ID của chuỗi DICOM
        description : str
            Mô tả về chuỗi DICOM
        """
        self.series_id = series_id
        self.description = description
        self.files = []  # Danh sách đường dẫn đến các file DICOM trong chuỗi
        self.metadata = {}  # Metadata của chuỗi DICOM
        self.modality = ""  # Dạng hình ảnh (CT, MR, PT, v.v.)
        self.patient_name = ""
        self.patient_id = ""
        self.study_date = ""
        self.study_description = ""
        
        # Dữ liệu hình ảnh
        self.image_data = None  # Dữ liệu 3D
        self.image = None  # Đối tượng hình ảnh để hiển thị
        self.image_position = None  # Vị trí voxel đầu tiên
        self.image_orientation = None  # Hướng của hình ảnh
        self.pixel_spacing = None  # Khoảng cách giữa các pixel
        self.slice_thickness = None  # Độ dày lát cắt
        self.base_directory = ""  # Thư mục gốc của series
        
        # Thiết lập thư mục gốc mặc định
        self.base_directory = ""
    
    def add_file(self, file_path: str) -> bool:
        """
        Thêm một file DICOM vào chuỗi.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file DICOM
        
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu thất bại
        """
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể thêm file DICOM vì thiếu thư viện pydicom")
            return False
        
        try:
            # Kiểm tra file có phải là DICOM hợp lệ không
            dicom_data = pydicom.dcmread(file_path, force=False)
            
            # Kiểm tra xem file có thuộc series này không
            if hasattr(dicom_data, 'SeriesInstanceUID') and dicom_data.SeriesInstanceUID:
                if not self.series_id:
                    # Nếu series_id chưa được thiết lập, sử dụng ID từ file đầu tiên
                    self.series_id = dicom_data.SeriesInstanceUID
                    
                    # Thiết lập các thông tin khác từ file đầu tiên
                    if hasattr(dicom_data, 'SeriesDescription') and dicom_data.SeriesDescription:
                        self.description = dicom_data.SeriesDescription
                    
                    if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
                        self.modality = dicom_data.Modality
                    
                    if hasattr(dicom_data, 'PatientName') and dicom_data.PatientName:
                        self.patient_name = str(dicom_data.PatientName)
                    
                    if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
                        self.patient_id = dicom_data.PatientID
                    
                    if hasattr(dicom_data, 'StudyDate') and dicom_data.StudyDate:
                        self.study_date = dicom_data.StudyDate
                    
                    if hasattr(dicom_data, 'StudyDescription') and dicom_data.StudyDescription:
                        self.study_description = dicom_data.StudyDescription
                
                if dicom_data.SeriesInstanceUID == self.series_id:
                    # File thuộc series này
                    self.files.append(file_path)
                    return True
                else:
                    # File không thuộc series này
                    return False
            else:
                # File không có SeriesInstanceUID
                logger.warning(f"File {file_path} không có SeriesInstanceUID")
                return False
        
        except (InvalidDicomError, Exception) as e:
            logger.error(f"Lỗi khi đọc file DICOM {file_path}: {str(e)}")
            return False
    
    def load_image_data(self) -> bool:
        """
        Tải dữ liệu hình ảnh từ các file DICOM.
        
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        if not self.files:
            logger.error("Không có file DICOM nào để tải")
            return False
        
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể tải dữ liệu hình ảnh vì thiếu thư viện pydicom")
            return False
        
        try:
            if SITK_AVAILABLE:
                # Sử dụng SimpleITK để tải dữ liệu (tốt hơn cho các chuỗi lớn)
                return self._load_with_sitk()
            else:
                # Sử dụng pydicom trực tiếp
                return self._load_with_pydicom()
        
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu hình ảnh: {str(e)}")
            return False
    
    def _load_with_sitk(self) -> bool:
        """
        Tải dữ liệu hình ảnh bằng SimpleITK.
        
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Đọc chuỗi
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(self.files)
            image = reader.Execute()
            
            # Chuyển đổi sang numpy array
            self.image_data = sitk.GetArrayFromImage(image)
            
            # Gán thuộc tính image cho hiển thị
            self.image = self.image_data
            
            # Lưu metadata
            self.pixel_spacing = image.GetSpacing()[:2]  # (x, y)
            self.slice_thickness = image.GetSpacing()[2]
            
            # Lưu thông tin vị trí và hướng
            self.image_position = image.GetOrigin()
            self.image_orientation = image.GetDirection()
            
            # Lưu base_directory (thư mục gốc của series)
            self.base_directory = os.path.dirname(self.files[0]) if self.files else ""
            
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu với SimpleITK: {str(e)}")
            return False
    
    def _load_with_pydicom(self) -> bool:
        """
        Tải dữ liệu hình ảnh bằng pydicom.
        
        Returns
        -------
        bool
            True nếu tải thành công, False nếu thất bại
        """
        try:
            # Đọc tất cả các file DICOM
            slices = [pydicom.dcmread(file) for file in self.files]
            
            # Sắp xếp các lát cắt theo vị trí
            if hasattr(slices[0], 'ImagePositionPatient'):
                # Sắp xếp theo vị trí dọc theo trục z (thường là vị trí thứ 3)
                slices = sorted(slices, key=lambda s: s.ImagePositionPatient[2])
            elif hasattr(slices[0], 'SliceLocation'):
                # Sắp xếp theo vị trí lát cắt
                slices = sorted(slices, key=lambda s: s.SliceLocation)
            else:
                # Nếu không có thông tin vị trí, sắp xếp theo InstanceNumber
                if hasattr(slices[0], 'InstanceNumber'):
                    slices = sorted(slices, key=lambda s: s.InstanceNumber)
            
            # Đảm bảo tất cả các lát cắt có cùng kích thước và loại pixel
            if len(slices) > 1:
                if slices[0].Rows != slices[1].Rows or slices[0].Columns != slices[1].Columns:
                    logger.error("Kích thước lát cắt không đồng nhất")
                    return False
            
            # Tạo mảng 3D từ các lát cắt
            img_shape = (len(slices), slices[0].Rows, slices[0].Columns)
            self.image_data = np.zeros(img_shape, dtype=np.float32)
            
            # Chuyển đổi từ các slice thành mảng 3D
            for i, slice in enumerate(slices):
                pixel_array = slice.pixel_array.astype(np.float32)
                
                # Áp dụng rescale slope và intercept nếu có
                if hasattr(slice, 'RescaleSlope') and hasattr(slice, 'RescaleIntercept'):
                    pixel_array = pixel_array * slice.RescaleSlope + slice.RescaleIntercept
                
                self.image_data[i, :, :] = pixel_array
            
            # Gán thuộc tính image cho hiển thị
            self.image = self.image_data
            
            # Lưu thông tin pixel spacing và slice thickness
            if hasattr(slices[0], 'PixelSpacing'):
                self.pixel_spacing = slices[0].PixelSpacing
            
            if hasattr(slices[0], 'SliceThickness'):
                self.slice_thickness = slices[0].SliceThickness
            
            # Lưu base_directory (thư mục gốc của series)
            self.base_directory = os.path.dirname(self.files[0]) if self.files else ""
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu với pydicom: {str(e)}")
            return False
    
    def get_metadata_summary(self) -> Dict[str, str]:
        """
        Trả về tóm tắt metadata của chuỗi DICOM.
        
        Returns
        -------
        Dict[str, str]
            Dictionary chứa các thông tin metadata chính
        """
        summary = {
            "Series ID": self.series_id,
            "Description": self.description,
            "Modality": self.modality,
            "Patient Name": self.patient_name,
            "Patient ID": self.patient_id,
            "Study Date": self.study_date,
            "Study Description": self.study_description,
            "Number of Files": str(len(self.files))
        }
        
        if self.image_data is not None:
            summary["Image Dimensions"] = f"{self.image_data.shape}"
        
        if self.pixel_spacing is not None:
            summary["Pixel Spacing"] = f"{self.pixel_spacing}"
        
        if self.slice_thickness is not None:
            summary["Slice Thickness"] = f"{self.slice_thickness}"
        
        return summary
    
    def get_slice(self, index: int, plane: str = 'axial') -> Optional[np.ndarray]:
        """
        Lấy một lát cắt từ dữ liệu 3D.
        
        Parameters
        ----------
        index : int
            Chỉ số lát cắt
        plane : str
            Mặt phẳng ('axial', 'coronal', 'sagittal')
        
        Returns
        -------
        Optional[np.ndarray]
            Dữ liệu lát cắt 2D hoặc None nếu không có dữ liệu
        """
        if self.image_data is None:
            return None
        
        if plane == 'axial':
            if 0 <= index < self.image_data.shape[0]:
                return self.image_data[index, :, :]
        elif plane == 'coronal':
            if 0 <= index < self.image_data.shape[1]:
                return self.image_data[:, index, :]
        elif plane == 'sagittal':
            if 0 <= index < self.image_data.shape[2]:
                return self.image_data[:, :, index]
        
        return None


class DicomLoader:
    """Lớp chịu trách nhiệm tải và quản lý dữ liệu DICOM."""
    
    def __init__(self):
        """Khởi tạo DicomLoader."""
        self.series_list = []  # Danh sách các chuỗi DICOM đã tải
        self.base_directory = ""  # Thư mục gốc chứa dữ liệu DICOM
        
        # Kiểm tra các thư viện cần thiết
        self._check_libraries()
    
    def _check_libraries(self):
        """Kiểm tra các thư viện cần thiết đã được cài đặt chưa."""
        if not PYDICOM_AVAILABLE:
            logger.warning("Thư viện pydicom không có sẵn. Một số chức năng có thể không hoạt động.")
        
        if not SITK_AVAILABLE:
            logger.warning("Thư viện SimpleITK không có sẵn. Hiệu suất tải DICOM có thể bị ảnh hưởng.")
    
    def load_dicom_file(self, file_path: str) -> Optional[DicomSeries]:
        """
        Tải một file DICOM và tạo chuỗi mới từ file đó.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file DICOM
        
        Returns
        -------
        Optional[DicomSeries]
            Chuỗi DICOM mới hoặc None nếu tải thất bại
        """
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể tải file DICOM vì thiếu thư viện pydicom")
            return None
        
        try:
            # Tạo chuỗi mới
            series = DicomSeries()
            
            # Thêm file vào chuỗi
            if series.add_file(file_path):
                self.series_list.append(series)
                return series
            else:
                return None
        
        except Exception as e:
            logger.error(f"Lỗi khi tải file DICOM {file_path}: {str(e)}")
            return None
    
    def load_dicom_directory(self, directory_path: str) -> List[DicomSeries]:
        """
        Tải tất cả các file DICOM từ một thư mục và tổ chức thành các chuỗi.
        
        Parameters
        ----------
        directory_path : str
            Đường dẫn đến thư mục chứa các file DICOM
        
        Returns
        -------
        List[DicomSeries]
            Danh sách các chuỗi DICOM đã tải
        """
        if not PYDICOM_AVAILABLE:
            logger.error("Không thể tải thư mục DICOM vì thiếu thư viện pydicom")
            return []
        
        # Cập nhật thư mục gốc
        self.base_directory = directory_path
        
        # Danh sách chuỗi mới
        new_series_list = []
        
        try:
            # Dictionary tạm thời lưu trữ các chuỗi theo ID
            series_dict = {}
            
            # Duyệt qua tất cả các file trong thư mục
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Kiểm tra file có phải là DICOM hợp lệ không
                        dicom_data = pydicom.dcmread(file_path, force=False)
                        
                        # Kiểm tra xem file có SeriesInstanceUID không
                        if hasattr(dicom_data, 'SeriesInstanceUID') and dicom_data.SeriesInstanceUID:
                            series_id = dicom_data.SeriesInstanceUID
                            
                            # Tạo chuỗi mới nếu cần
                            if series_id not in series_dict:
                                series_description = ""
                                if hasattr(dicom_data, 'SeriesDescription') and dicom_data.SeriesDescription:
                                    series_description = dicom_data.SeriesDescription
                                
                                series_dict[series_id] = DicomSeries(series_id, series_description)
                            
                            # Thêm file vào chuỗi
                            series_dict[series_id].add_file(file_path)
                    
                    except (InvalidDicomError, Exception) as e:
                        # Không phải file DICOM hoặc có lỗi khác
                        logger.debug(f"Bỏ qua file {file_path}: {str(e)}")
                        continue
            
            # Thêm các chuỗi mới vào danh sách
            for series in series_dict.values():
                self.series_list.append(series)
                new_series_list.append(series)
            
            return new_series_list
        
        except Exception as e:
            logger.error(f"Lỗi khi tải thư mục DICOM {directory_path}: {str(e)}")
            return []
    
    def get_series_by_id(self, series_id: str) -> Optional[DicomSeries]:
        """
        Tìm chuỗi DICOM theo ID.
        
        Parameters
        ----------
        series_id : str
            ID của chuỗi DICOM cần tìm
        
        Returns
        -------
        Optional[DicomSeries]
            Chuỗi DICOM hoặc None nếu không tìm thấy
        """
        for series in self.series_list:
            if series.series_id == series_id:
                return series
        
        return None
    
    def clear_series(self):
        """Xóa tất cả các chuỗi DICOM đã tải."""
        self.series_list.clear()


def get_dicom_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Trích xuất metadata từ file DICOM.
    
    Parameters
    ----------
    file_path : str
        Đường dẫn đến file DICOM
    
    Returns
    -------
    Dict[str, Any]
        Dictionary chứa metadata
    """
    if not PYDICOM_AVAILABLE:
        logger.error("Không thể trích xuất metadata vì thiếu thư viện pydicom")
        return {}
    
    try:
        dicom_data = pydicom.dcmread(file_path, force=True)
        
        metadata = {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path
        }
        
        # Thông tin cơ bản
        if hasattr(dicom_data, 'PatientName') and dicom_data.PatientName:
            metadata["PatientName"] = str(dicom_data.PatientName)
        
        if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
            metadata["PatientID"] = dicom_data.PatientID
        
        if hasattr(dicom_data, 'PatientBirthDate') and dicom_data.PatientBirthDate:
            metadata["PatientBirthDate"] = dicom_data.PatientBirthDate
        
        if hasattr(dicom_data, 'PatientSex') and dicom_data.PatientSex:
            metadata["PatientSex"] = dicom_data.PatientSex
        
        # Thông tin nghiên cứu
        if hasattr(dicom_data, 'StudyInstanceUID') and dicom_data.StudyInstanceUID:
            metadata["StudyInstanceUID"] = dicom_data.StudyInstanceUID
        
        if hasattr(dicom_data, 'StudyDate') and dicom_data.StudyDate:
            metadata["StudyDate"] = dicom_data.StudyDate
        
        if hasattr(dicom_data, 'StudyTime') and dicom_data.StudyTime:
            metadata["StudyTime"] = dicom_data.StudyTime
        
        if hasattr(dicom_data, 'StudyDescription') and dicom_data.StudyDescription:
            metadata["StudyDescription"] = dicom_data.StudyDescription
        
        # Thông tin chuỗi
        if hasattr(dicom_data, 'SeriesInstanceUID') and dicom_data.SeriesInstanceUID:
            metadata["SeriesInstanceUID"] = dicom_data.SeriesInstanceUID
        
        if hasattr(dicom_data, 'SeriesNumber') and dicom_data.SeriesNumber:
            metadata["SeriesNumber"] = dicom_data.SeriesNumber
        
        if hasattr(dicom_data, 'SeriesDescription') and dicom_data.SeriesDescription:
            metadata["SeriesDescription"] = dicom_data.SeriesDescription
        
        # Thông tin hình ảnh
        if hasattr(dicom_data, 'Modality') and dicom_data.Modality:
            metadata["Modality"] = dicom_data.Modality
        
        if hasattr(dicom_data, 'Manufacturer') and dicom_data.Manufacturer:
            metadata["Manufacturer"] = dicom_data.Manufacturer
        
        if hasattr(dicom_data, 'InstitutionName') and dicom_data.InstitutionName:
            metadata["InstitutionName"] = dicom_data.InstitutionName
        
        if hasattr(dicom_data, 'PixelSpacing') and dicom_data.PixelSpacing:
            metadata["PixelSpacing"] = dicom_data.PixelSpacing
        
        if hasattr(dicom_data, 'SliceThickness') and dicom_data.SliceThickness:
            metadata["SliceThickness"] = dicom_data.SliceThickness
        
        if hasattr(dicom_data, 'ImagePositionPatient') and dicom_data.ImagePositionPatient:
            metadata["ImagePositionPatient"] = dicom_data.ImagePositionPatient
        
        if hasattr(dicom_data, 'Rows') and dicom_data.Rows:
            metadata["Rows"] = dicom_data.Rows
        
        if hasattr(dicom_data, 'Columns') and dicom_data.Columns:
            metadata["Columns"] = dicom_data.Columns
        
        return metadata
    
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất metadata từ {file_path}: {str(e)}")
        return {"Error": str(e)}


class DicomLoaderWidget(QWidget):
    """
    Widget cho phép tải và quản lý dữ liệu DICOM trong giao diện người dùng.
    
    Cung cấp các chức năng:
    - Tải file DICOM từ thư mục
    - Hiển thị và tổ chức các chuỗi DICOM theo bệnh nhân và nghiên cứu
    - Xem thông tin chi tiết về chuỗi DICOM
    - Nhập dữ liệu DICOM vào hệ thống QuangTPS
    """
    
    # Tín hiệu
    series_selected = pyqtSignal(DicomSeries)
    series_imported = pyqtSignal(str, str, str)  # patient_id, study_id, series_id
    
    def __init__(self, parent=None):
        """Khởi tạo DicomLoaderWidget."""
        super().__init__(parent)
        
        # Dữ liệu
        self.dicom_loader = DicomLoader()
        self.current_directory = ""
        self.current_series = None
        
        # Khởi tạo cơ sở dữ liệu
        self.patient_db = PatientDatabase()
        self.image_db = ImageDatabase()
        
        # UI
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        
        # Thanh công cụ
        toolbar_layout = QHBoxLayout()
        
        # Nút mở thư mục
        self.open_dir_btn = QPushButton("Mở thư mục DICOM")
        self.open_dir_btn.clicked.connect(self._open_directory)
        toolbar_layout.addWidget(self.open_dir_btn)
        
        # Nút làm mới
        self.refresh_btn = QPushButton("Làm mới")
        self.refresh_btn.clicked.connect(self._refresh_current_directory)
        toolbar_layout.addWidget(self.refresh_btn)
        
        # Nút nhập dữ liệu
        self.import_btn = QPushButton("Nhập vào hệ thống")
        self.import_btn.clicked.connect(self._import_selected_series)
        self.import_btn.setEnabled(False)
        toolbar_layout.addWidget(self.import_btn)
        
        toolbar_layout.addStretch()
        
        main_layout.addLayout(toolbar_layout)
        
        # Splitter chính
        splitter = QSplitter(Qt.Horizontal)
        
        # Cây DICOM ở bên trái
        self.dicom_tree = QTreeWidget()
        self.dicom_tree.setHeaderLabels(["Thông tin DICOM"])
        self.dicom_tree.setMinimumWidth(300)
        self.dicom_tree.itemClicked.connect(self._on_tree_item_clicked)
        splitter.addWidget(self.dicom_tree)
        
        # Thông tin chi tiết ở bên phải
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        # Tiêu đề
        self.info_title = QLabel("Thông tin chi tiết")
        self.info_title.setAlignment(Qt.AlignCenter)
        self.info_title.setFont(QFont("Arial", 12, QFont.Bold))
        info_layout.addWidget(self.info_title)
        
        # Thông tin cơ bản
        basic_info_group = QGroupBox("Thông tin cơ bản")
        basic_info_layout = QGridLayout(basic_info_group)
        
        basic_info_layout.addWidget(QLabel("Bệnh nhân:"), 0, 0)
        self.patient_label = QLabel("")
        basic_info_layout.addWidget(self.patient_label, 0, 1)
        
        basic_info_layout.addWidget(QLabel("Nghiên cứu:"), 1, 0)
        self.study_label = QLabel("")
        basic_info_layout.addWidget(self.study_label, 1, 1)
        
        basic_info_layout.addWidget(QLabel("Chuỗi:"), 2, 0)
        self.series_label = QLabel("")
        basic_info_layout.addWidget(self.series_label, 2, 1)
        
        basic_info_layout.addWidget(QLabel("Dạng:"), 3, 0)
        self.modality_label = QLabel("")
        basic_info_layout.addWidget(self.modality_label, 3, 1)
        
        basic_info_layout.addWidget(QLabel("Số lượng file:"), 4, 0)
        self.files_count_label = QLabel("")
        basic_info_layout.addWidget(self.files_count_label, 4, 1)
        
        info_layout.addWidget(basic_info_group)
        
        # Metadata chi tiết
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.metadata_tree = QTreeWidget()
        self.metadata_tree.setHeaderLabels(["Thẻ", "Giá trị"])
        self.metadata_tree.setAlternatingRowColors(True)
        metadata_layout.addWidget(self.metadata_tree)
        
        info_layout.addWidget(metadata_group)
        
        # Thêm widget thông tin vào splitter
        splitter.addWidget(info_widget)
        
        # Thiết lập kích thước ban đầu cho splitter
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter, 1)
        
        # Thanh trạng thái
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Sẵn sàng")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(status_layout)
        
    def _open_directory(self):
        """Mở hộp thoại chọn thư mục và tải dữ liệu DICOM."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Chọn thư mục chứa file DICOM",
            ""
        )
        
        if directory:
            self._load_directory(directory)
    
    def _load_directory(self, directory):
        """Tải dữ liệu DICOM từ thư mục được chỉ định."""
        self.current_directory = directory
        self.status_label.setText(f"Đang tải dữ liệu từ {directory}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Xóa dữ liệu cũ
        self.dicom_loader.clear_series()
        self.dicom_tree.clear()
        self._clear_info_panel()
        
        # Tải dữ liệu mới
        try:
            series_list = self.dicom_loader.load_dicom_directory(directory)
            self._populate_dicom_tree(series_list)
            self.status_label.setText(f"Đã tải {len(series_list)} chuỗi DICOM từ {directory}")
        except Exception as e:
            self.status_label.setText(f"Lỗi khi tải: {str(e)}")
            QMessageBox.warning(self, "Lỗi tải dữ liệu", f"Không thể tải dữ liệu DICOM: {str(e)}")
        
        self.progress_bar.setVisible(False)
    
    def _refresh_current_directory(self):
        """Làm mới dữ liệu từ thư mục hiện tại."""
        if self.current_directory:
            self._load_directory(self.current_directory)
        else:
            self.status_label.setText("Không có thư mục nào được chọn")
    
    def _populate_dicom_tree(self, series_list):
        """Cập nhật cây hiển thị với danh sách chuỗi DICOM."""
        # Sắp xếp các chuỗi theo bệnh nhân và nghiên cứu
        patients = {}
        
        for series in series_list:
            metadata = series.metadata
            patient_id = metadata.get('PatientID', 'Unknown')
            patient_name = metadata.get('PatientName', 'Unknown')
            
            study_uid = metadata.get('StudyInstanceUID', 'Unknown')
            study_date = metadata.get('StudyDate', '')
            
            # Tạo entry cho bệnh nhân nếu chưa tồn tại
            if patient_id not in patients:
                patients[patient_id] = {
                    'name': patient_name,
                    'studies': {}
                }
            
            # Tạo entry cho nghiên cứu nếu chưa tồn tại
            if study_uid not in patients[patient_id]['studies']:
                patients[patient_id]['studies'][study_uid] = {
                    'date': study_date,
                    'series': []
                }
            
            # Thêm chuỗi vào nghiên cứu
            patients[patient_id]['studies'][study_uid]['series'].append(series)
        
        # Cập nhật cây hiển thị
        for patient_id, patient_info in patients.items():
            patient_item = QTreeWidgetItem(self.dicom_tree)
            patient_item.setText(0, f"{patient_info['name']} ({patient_id})")
            patient_item.setData(0, Qt.UserRole, {'type': 'patient', 'id': patient_id})
            
            for study_uid, study_info in patient_info['studies'].items():
                study_item = QTreeWidgetItem(patient_item)
                study_item.setText(0, f"{study_info['date']}")
                study_item.setData(0, Qt.UserRole, {'type': 'study', 'id': study_uid})
                
                for series in study_info['series']:
                    series_item = QTreeWidgetItem(study_item)
                    series_item.setText(0, f"{series.description} - {series.modality}")
                    series_item.setData(0, Qt.UserRole, {'type': 'series', 'id': series.series_id, 'object': series})
        
        # Mở rộng cây
        self.dicom_tree.expandAll()
    
    def _on_tree_item_clicked(self, item, column):
        """Xử lý khi một mục trong cây được chọn."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
            
        item_type = data.get('type')
        
        if item_type == 'series':
            # Hiển thị thông tin chuỗi
            series = data.get('object')
            if series:
                self.current_series = series
                self._display_series_info(series)
                self.import_btn.setEnabled(True)
                self.series_selected.emit(series)
        else:
            # Xóa thông tin chi tiết cho các loại mục khác
            self._clear_info_panel()
            self.import_btn.setEnabled(False)
            self.current_series = None
    
    def _display_series_info(self, series):
        """Hiển thị thông tin chi tiết của chuỗi DICOM."""
        # Cập nhật thông tin cơ bản
        metadata = series.metadata
        
        self.patient_label.setText(f"{metadata.get('PatientName', 'Unknown')} ({metadata.get('PatientID', 'Unknown')})")
        self.study_label.setText(metadata.get('StudyDescription', 'Unknown Study'))
        self.series_label.setText(series.description)
        self.modality_label.setText(series.modality)
        self.files_count_label.setText(str(len(series.files)))
        
        # Cập nhật cây metadata
        self.metadata_tree.clear()
        
        # Thêm các mục metadata
        for key, value in sorted(metadata.items()):
            item = QTreeWidgetItem(self.metadata_tree)
            item.setText(0, str(key))
            item.setText(1, str(value))
    
    def _clear_info_panel(self):
        """Xóa thông tin chi tiết."""
        self.patient_label.setText("")
        self.study_label.setText("")
        self.series_label.setText("")
        self.modality_label.setText("")
        self.files_count_label.setText("")
        self.metadata_tree.clear()
    
    def _import_selected_series(self):
        """Nhập chuỗi DICOM được chọn vào hệ thống QuangTPS."""
        if not self.current_series:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một chuỗi DICOM để nhập")
            return
            
        try:
            # Lấy thông tin metadata
            metadata = self.current_series.metadata
            patient_id = metadata.get('PatientID', 'Unknown')
            patient_name = metadata.get('PatientName', 'Unknown')
            study_uid = metadata.get('StudyInstanceUID', 'Unknown')
            series_id = self.current_series.series_id
            
            # Đặt trạng thái
            self.status_label.setText("Đang nhập chuỗi %s..." % self.current_series.description)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(30)
            
            # Tạo hoặc cập nhật bệnh nhân trong cơ sở dữ liệu
            birth_date = metadata.get('PatientBirthDate', None)
            gender = metadata.get('PatientSex', None)
            
            # Kiểm tra xem bệnh nhân đã tồn tại chưa
            patient = self.patient_db.get_patient(patient_id)
            if not patient:
                logger.info("Tạo bệnh nhân mới với ID: %s", patient_id)
                patient_metadata = {k: v for k, v in metadata.items() if k.startswith('Patient')}
                patient_id = self.patient_db.create_patient(
                    patient_id=patient_id,
                    name=patient_name,
                    birth_date=birth_date,
                    gender=gender,
                    metadata=patient_metadata
                )
            
            # Kiểm tra và tạo nghiên cứu nếu cần
            study = self.patient_db.get_study(study_uid)
            if not study:
                logger.info("Tạo nghiên cứu mới với ID: %s", study_uid)
                study_metadata = {k: v for k, v in metadata.items() if k.startswith('Study')}
                study_desc = metadata.get('StudyDescription', '')
                study_date = metadata.get('StudyDate', '')
                study_time = metadata.get('StudyTime', '')
                self.patient_db.create_study(
                    study_id=study_uid,
                    patient_id=patient_id,
                    description=study_desc,
                    study_date=study_date,
                    study_time=study_time,
                    metadata=study_metadata
                )
            
            self.progress_bar.setValue(50)
            
            # Tạo đối tượng hình ảnh từ chuỗi DICOM
            if SITK_AVAILABLE and self.current_series.modality in ['CT', 'MR', 'PT']:
                # Tạo đường dẫn tạm thời cho hình ảnh
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(self.current_series.files)
                sitk_image = reader.Execute()
                
                # Chuyển đổi sang đối tượng Image
                image = Image.from_sitk(sitk_image)
                
                # Thêm metadata
                for key, value in metadata.items():
                    if key not in image.metadata:
                        image.metadata[key] = value
                
                # Lưu hình ảnh vào cơ sở dữ liệu
                self.progress_bar.setValue(70)
                
                # Lưu hình ảnh sử dụng ImageDatabase
                series_desc = metadata.get('SeriesDescription', 'Unknown Series')
                saved_series_id = self.image_db.save_image(
                    image=image,
                    series_id=series_id,
                    patient_id=patient_id,
                    study_id=study_uid,
                    description=series_desc,
                    metadata=metadata
                )
                
                self.status_label.setText("Đã nhập thành công chuỗi %s" % self.current_series.description)
                self.progress_bar.setValue(100)
                
                # Phát tín hiệu đã nhập
                self.series_imported.emit(patient_id, study_uid, saved_series_id)
                
                # Hiển thị thông báo thành công
                QMessageBox.information(self, "Nhập thành công", 
                                      "Đã nhập thành công chuỗi DICOM: %s" % self.current_series.description)
            else:
                raise ValueError("Không hỗ trợ dạng hình ảnh: %s" % self.current_series.modality)
                
        except Exception as e:
            logger.error("Lỗi khi nhập chuỗi DICOM: %s", str(e), exc_info=True)
            self.status_label.setText("Lỗi khi nhập: %s" % str(e))
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Lỗi khi nhập", 
                               "Không thể nhập chuỗi DICOM:\n%s" % str(e))
