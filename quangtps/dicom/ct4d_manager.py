#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quản lý dữ liệu 4D-CT.

Module này cung cấp các chức năng để nhập, xử lý và phân tích dữ liệu 4D-CT
cho các ứng dụng xạ trị. 4D-CT là một chuỗi CT được thu thập trong nhiều 
chu kỳ hô hấp, cho phép mô phỏng chuyển động của các cơ quan và khối u.
"""

import os
import logging
import numpy as np
import pydicom
from typing import List, Dict, Any, Union, Optional, Tuple, Sequence
from collections import defaultdict

from quangtps.core.exceptions import DicomError, ValidationError
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.dicom_validator import DicomValidator

logger = logging.getLogger(__name__)

class CT4DPhase:
    """
    Lớp đại diện cho một pha của dữ liệu 4D-CT.
    
    Mỗi pha tương ứng với một vị trí trong chu kỳ hô hấp.
    """
    
    def __init__(self, phase_id: str, phase_percent: float = 0.0):
        """
        Khởi tạo một CT4DPhase.
        
        Parameters
        ----------
        phase_id : str
            ID định danh pha
        phase_percent : float, optional
            Phần trăm của chu kỳ hô hấp (0-100%)
        """
        self.phase_id = phase_id
        self.phase_percent = phase_percent
        self.ct_datasets = []
        self.slice_positions = []
        self.volume = None
        self.voxel_size = None
    
    def add_dataset(self, dataset: pydicom.dataset.FileDataset) -> bool:
        """
        Thêm một dataset CT vào pha.
        
        Parameters
        ----------
        dataset : pydicom.dataset.FileDataset
            Dataset CT của một lát cắt
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu không
        """
        try:
            # Kiểm tra dataset có phải CT không
            if not hasattr(dataset, 'Modality') or dataset.Modality != 'CT':
                logger.warning(f"Dataset không phải là CT")
                return False
            
            # Lấy vị trí của slice
            if not hasattr(dataset, 'ImagePositionPatient'):
                logger.warning(f"Dataset không có thông tin vị trí")
                return False
            
            slice_position = float(dataset.ImagePositionPatient[2])
            
            # Thêm vào danh sách
            self.ct_datasets.append(dataset)
            self.slice_positions.append(slice_position)
            
            # Reset volume (sẽ được tính lại khi cần)
            self.volume = None
            self.voxel_size = None
            
            return True
        
        except Exception as e:
            logger.error(f"Lỗi khi thêm dataset: {str(e)}")
            return False
    
    def get_sorted_datasets(self) -> List[pydicom.dataset.FileDataset]:
        """
        Lấy danh sách các dataset đã được sắp xếp theo vị trí Z.
        
        Returns
        -------
        List[pydicom.dataset.FileDataset]
            Danh sách các dataset đã sắp xếp
        """
        if not self.ct_datasets:
            return []
        
        # Sắp xếp các slice theo vị trí Z
        sorted_indices = np.argsort(self.slice_positions)
        return [self.ct_datasets[i] for i in sorted_indices]
    
    def get_volume(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Tạo và trả về volume 3D từ các slice CT.
        
        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float, float]]
            Volume 3D và kích thước voxel (dx, dy, dz)
        """
        if self.volume is not None and self.voxel_size is not None:
            return self.volume, self.voxel_size
        
        # Sắp xếp các slice
        sorted_datasets = self.get_sorted_datasets()
        
        if not sorted_datasets:
            raise DicomError("Không có slice CT nào")
        
        # Lấy kích thước voxel
        first_slice = sorted_datasets[0]
        pixel_spacing = first_slice.PixelSpacing
        dx, dy = float(pixel_spacing[0]), float(pixel_spacing[1])
        
        # Tính khoảng cách slice (dz)
        sorted_positions = sorted(self.slice_positions)
        if len(sorted_positions) > 1:
            dz = sum([abs(sorted_positions[i] - sorted_positions[i-1]) for i in range(1, len(sorted_positions))]) / (len(sorted_positions) - 1)
        else:
            dz = 1.0  # Giá trị mặc định
        
        # Tạo volume
        pixel_arrays = [s.pixel_array for s in sorted_datasets]
        volume = np.stack(pixel_arrays, axis=0)
        
        # Chuyển HU
        if hasattr(first_slice, 'RescaleIntercept') and hasattr(first_slice, 'RescaleSlope'):
            intercept = first_slice.RescaleIntercept
            slope = first_slice.RescaleSlope
            volume = volume * slope + intercept
        
        self.volume = volume
        self.voxel_size = (dx, dy, dz)
        
        return self.volume, self.voxel_size
    
    def get_slice_count(self) -> int:
        """
        Lấy số lượng slice.
        
        Returns
        -------
        int
            Số lượng slice
        """
        return len(self.ct_datasets)


class CT4DDataset:
    """
    Lớp đại diện cho một tập dữ liệu 4D-CT.
    
    Tập dữ liệu 4D-CT bao gồm nhiều pha khác nhau,
    mỗi pha tương ứng với một vị trí trong chu kỳ hô hấp.
    """
    
    def __init__(self, patient_id: str, patient_name: str, study_uid: str):
        """
        Khởi tạo CT4DDataset.
        
        Parameters
        ----------
        patient_id : str
            ID bệnh nhân
        patient_name : str
            Tên bệnh nhân
        study_uid : str
            Study Instance UID
        """
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.study_uid = study_uid
        self.phases = {}  # phase_id -> CT4DPhase
        self.phase_order = []  # danh sách các phase_id theo thứ tự
        self.metadata = {}  # metadata bổ sung
    
    def add_phase(self, phase_id: str, phase_percent: float = 0.0) -> CT4DPhase:
        """
        Thêm một pha mới vào tập dữ liệu.
        
        Parameters
        ----------
        phase_id : str
            ID định danh pha
        phase_percent : float, optional
            Phần trăm của chu kỳ hô hấp (0-100%)
            
        Returns
        -------
        CT4DPhase
            Đối tượng pha mới
        """
        if phase_id in self.phases:
            return self.phases[phase_id]
        
        # Tạo pha mới
        phase = CT4DPhase(phase_id, phase_percent)
        self.phases[phase_id] = phase
        self.phase_order.append(phase_id)
        
        # Sắp xếp lại thứ tự các pha theo phase_percent
        self.phase_order = [p_id for p_id, p in 
                           sorted([(p_id, self.phases[p_id]) for p_id in self.phases], 
                                  key=lambda x: x[1].phase_percent)]
        
        return phase
    
    def get_phase(self, phase_id: str) -> Optional[CT4DPhase]:
        """
        Lấy một pha theo ID.
        
        Parameters
        ----------
        phase_id : str
            ID định danh pha
            
        Returns
        -------
        Optional[CT4DPhase]
            Đối tượng pha nếu tồn tại, None nếu không
        """
        return self.phases.get(phase_id)
    
    def get_phases(self) -> List[CT4DPhase]:
        """
        Lấy tất cả các pha theo thứ tự.
        
        Returns
        -------
        List[CT4DPhase]
            Danh sách các pha
        """
        return [self.phases[phase_id] for phase_id in self.phase_order]
    
    def get_phase_count(self) -> int:
        """
        Lấy số lượng pha.
        
        Returns
        -------
        int
            Số lượng pha
        """
        return len(self.phases)
    
    def get_average_volume(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Tính volume trung bình từ tất cả các pha.
        
        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float, float]]
            Volume trung bình và kích thước voxel
        """
        if not self.phases:
            raise DicomError("Không có pha nào")
        
        # Lấy volume từ tất cả các pha
        volumes = []
        voxel_sizes = []
        
        for phase in self.phases.values():
            volume, voxel_size = phase.get_volume()
            volumes.append(volume)
            voxel_sizes.append(voxel_size)
        
        # Kiểm tra kích thước voxel có tương thích không
        if not all(voxel_size == voxel_sizes[0] for voxel_size in voxel_sizes):
            logger.warning("Kích thước voxel không đồng nhất giữa các pha")
        
        # Kiểm tra kích thước volume có tương thích không
        if not all(volume.shape == volumes[0].shape for volume in volumes):
            logger.warning("Kích thước volume không đồng nhất giữa các pha")
            # Nếu kích thước khác nhau, cần thực hiện nội suy
            # Đây là một tác vụ phức tạp, không được thực hiện trong phương thức này
            raise DicomError("Kích thước volume không đồng nhất giữa các pha")
        
        # Tính volume trung bình
        avg_volume = np.mean(np.array(volumes), axis=0)
        
        return avg_volume, voxel_sizes[0]
    
    def get_maximum_intensity_projection(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Tính Maximum Intensity Projection (MIP) từ tất cả các pha.
        
        Returns
        -------
        Tuple[np.ndarray, Tuple[float, float, float]]
            Volume MIP và kích thước voxel
        """
        if not self.phases:
            raise DicomError("Không có pha nào")
        
        # Lấy volume từ tất cả các pha
        volumes = []
        voxel_sizes = []
        
        for phase in self.phases.values():
            volume, voxel_size = phase.get_volume()
            volumes.append(volume)
            voxel_sizes.append(voxel_size)
        
        # Kiểm tra kích thước voxel có tương thích không
        if not all(voxel_size == voxel_sizes[0] for voxel_size in voxel_sizes):
            logger.warning("Kích thước voxel không đồng nhất giữa các pha")
        
        # Kiểm tra kích thước volume có tương thích không
        if not all(volume.shape == volumes[0].shape for volume in volumes):
            logger.warning("Kích thước volume không đồng nhất giữa các pha")
            # Nếu kích thước khác nhau, cần thực hiện nội suy
            # Đây là một tác vụ phức tạp, không được thực hiện trong phương thức này
            raise DicomError("Kích thước volume không đồng nhất giữa các pha")
        
        # Tính MIP
        mip_volume = np.max(np.array(volumes), axis=0)
        
        return mip_volume, voxel_sizes[0]
    
    def compute_motion_field(self, reference_phase_id: str) -> Dict[str, np.ndarray]:
        """
        Tính trường chuyển động từ pha tham chiếu đến các pha khác.
        
        Parameters
        ----------
        reference_phase_id : str
            ID của pha tham chiếu
            
        Returns
        -------
        Dict[str, np.ndarray]
            Trường chuyển động (phase_id -> displacement field)
        """
        # Phương thức này cần sử dụng thuật toán đăng ký hình ảnh biến dạng (DIR)
        # Đây là một tác vụ phức tạp, không được thực hiện đầy đủ trong phương thức này
        # Ở đây chỉ cung cấp một khung cơ bản
        
        if reference_phase_id not in self.phases:
            raise DicomError(f"Pha tham chiếu {reference_phase_id} không tồn tại")
        
        # Lấy volume pha tham chiếu
        ref_phase = self.phases[reference_phase_id]
        ref_volume, ref_voxel_size = ref_phase.get_volume()
        
        # Tính trường chuyển động đến các pha khác
        motion_fields = {}
        
        for phase_id, phase in self.phases.items():
            if phase_id == reference_phase_id:
                continue
            
            # Lấy volume pha hiện tại
            target_volume, target_voxel_size = phase.get_volume()
            
            # Ở đây cần gọi thuật toán DIR để tính trường chuyển động
            # Đây chỉ là ví dụ đơn giản, thực tế cần thuật toán phức tạp hơn nhiều
            # motion_field = deformable_registration(ref_volume, target_volume)
            
            # Tạm thời trả về một trường chuyển động rỗng (zeros)
            # Thực tế cần thực hiện đăng ký hình ảnh
            motion_field = np.zeros((*ref_volume.shape, 3), dtype=np.float32)
            motion_fields[phase_id] = motion_field
        
        return motion_fields
    
    @classmethod
    def from_dicom_series(cls, ct_datasets: List[pydicom.dataset.FileDataset], 
                       phase_tag: str = 'AcquisitionTime') -> 'CT4DDataset':
        """
        Tạo tập dữ liệu 4D-CT từ danh sách các dataset CT.
        
        Parameters
        ----------
        ct_datasets : List[pydicom.dataset.FileDataset]
            Danh sách các dataset CT
        phase_tag : str, optional
            Tag DICOM chứa thông tin pha
            
        Returns
        -------
        CT4DDataset
            Tập dữ liệu 4D-CT
        """
        if not ct_datasets:
            raise DicomError("Danh sách dataset CT trống")
        
        # Lấy thông tin bệnh nhân từ dataset đầu tiên
        first_ds = ct_datasets[0]
        patient_id = getattr(first_ds, 'PatientID', 'unknown')
        patient_name = str(getattr(first_ds, 'PatientName', 'Unknown'))
        study_uid = getattr(first_ds, 'StudyInstanceUID', 'unknown')
        
        # Tạo tập dữ liệu 4D-CT
        ct4d_dataset = cls(patient_id, patient_name, study_uid)
        
        # Phân loại các dataset theo pha
        phase_groups = defaultdict(list)
        
        for ds in ct_datasets:
            # Lấy thông tin pha từ tag được chỉ định
            if hasattr(ds, phase_tag):
                phase_id = str(getattr(ds, phase_tag))
            else:
                # Nếu không có tag phase, thử sử dụng SeriesNumber hoặc tag DICOM khác
                phase_id = str(getattr(ds, 'SeriesNumber', '0'))
            
            phase_groups[phase_id].append(ds)
        
        # Tạo các pha và thêm dataset vào
        for i, (phase_id, datasets) in enumerate(phase_groups.items()):
            # Tính phase_percent (ví dụ đơn giản)
            phase_percent = (i * 100) / len(phase_groups)
            
            # Tạo pha mới
            phase = ct4d_dataset.add_phase(phase_id, phase_percent)
            
            # Thêm các dataset vào pha
            for ds in datasets:
                phase.add_dataset(ds)
        
        return ct4d_dataset


def detect_4dct_series(dicom_series: Dict[str, List[pydicom.dataset.FileDataset]]) -> List[str]:
    """
    Phát hiện các series có thể là 4D-CT từ danh sách các series DICOM.
    
    Parameters
    ----------
    dicom_series : Dict[str, List[pydicom.dataset.FileDataset]]
        Từ điển mapping series_uid -> danh sách dataset
        
    Returns
    -------
    List[str]
        Danh sách các series_uid được phát hiện là 4D-CT
    """
    potential_4dct_series = []
    
    for series_uid, datasets in dicom_series.items():
        if not datasets:
            continue
        
        # Kiểm tra modality có phải CT không
        first_ds = datasets[0]
        if not hasattr(first_ds, 'Modality') or first_ds.Modality != 'CT':
            continue
        
        # Heuristic 1: Series có số lượng lớn slices
        if len(datasets) > 100:  # Ngưỡng tùy chỉnh
            potential_4dct_series.append(series_uid)
            continue
        
        # Heuristic 2: Series có tag liên quan đến 4D
        if any(hasattr(first_ds, tag) for tag in ['ReferencedPhaseNumber', 'RespiratoryTriggerTime', 'RespiratoryTriggerType']):
            potential_4dct_series.append(series_uid)
            continue
        
        # Heuristic 3: Series có thông tin về mặt thời gian
        if hasattr(first_ds, 'AcquisitionTime'):
            # Kiểm tra xem có nhiều AcquisitionTime khác nhau không
            acq_times = set(str(getattr(ds, 'AcquisitionTime', '')) for ds in datasets)
            if len(acq_times) > 1:
                potential_4dct_series.append(series_uid)
                continue
    
    return potential_4dct_series 