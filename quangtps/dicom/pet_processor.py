"""
Công cụ xử lý dữ liệu PET (Positron Emission Tomography).

Module này cung cấp các chức năng để xử lý dữ liệu PET trong xạ trị,
bao gồm việc đọc, chuyển đổi và phân tích dữ liệu PET/CT.
"""

import os
import logging
import numpy as np
import pydicom
import SimpleITK as sitk
from typing import List, Dict, Any, Tuple, Optional, Union

from quangtps.core.exceptions import DicomError
from quangtps.dicom.dicom_reader import DicomReader
from quangtps.dicom.dicom_converter import DicomConverter

logger = logging.getLogger(__name__)

class PETProcessor:
    """
    Lớp xử lý dữ liệu PET (Positron Emission Tomography).
    
    Class này cung cấp các phương thức để xử lý dữ liệu PET, bao gồm việc
    đọc dữ liệu, chuyển đổi SUV (Standardized Uptake Value), phân đoạn
    tự động, và hiển thị dữ liệu PET/CT.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng PETProcessor."""
        self.pet_datasets = None
        self.pet_volume = None
        self.pet_spacing = None
        self.pet_origin = None
        self.pet_direction = None
        self.pet_sitk_image = None
        self.ct_image = None
        self.transform = None
        self.suv_volume = None
        
        # Thông tin bệnh nhân và nghiên cứu cần thiết cho tính toán SUV
        self.patient_weight = None  # kg
        self.injected_dose = None   # Bq
        self.injection_time = None  # datetime
        self.acquisition_time = None  # datetime
        self.isotope_half_life = None  # seconds (e.g., 6588 for F-18)
    
    def load_pet_data(self, dicom_files: List[str]) -> None:
        """
        Tải dữ liệu PET từ các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM PET
            
        Raises
        ------
        DicomError
            Nếu không thể tải dữ liệu PET
        """
        try:
            reader = DicomReader()
            self.pet_datasets = reader.read_files(dicom_files)
            
            # Kiểm tra xem có phải PET không
            if not self._validate_pet_data():
                raise DicomError("Files are not PET data")
            
            # Trích xuất thông tin cần thiết cho tính toán SUV
            self._extract_suv_parameters()
            
            # Chuyển đổi thành dữ liệu khối
            converter = DicomConverter()
            self.pet_volume, self.pet_spacing, self.pet_origin, self.pet_direction = \
                converter.convert_dicom_to_volumetric_data(self.pet_datasets)
            
            # Chuyển đổi thành đối tượng SimpleITK
            self.pet_sitk_image = sitk.GetImageFromArray(self.pet_volume)
            self.pet_sitk_image.SetSpacing(self.pet_spacing)
            self.pet_sitk_image.SetOrigin(self.pet_origin)
            self.pet_sitk_image.SetDirection(self.pet_direction)
            
            logger.info(f"Loaded PET data: {len(dicom_files)} slices, "
                       f"shape: {self.pet_volume.shape}, "
                       f"spacing: {self.pet_spacing}")
            
        except Exception as e:
            logger.error(f"Error loading PET data: {str(e)}")
            raise DicomError(f"Error loading PET data: {str(e)}")
    
    def _validate_pet_data(self) -> bool:
        """
        Xác thực dữ liệu PET.
        
        Returns
        -------
        bool
            True nếu dữ liệu là PET, False nếu không phải
        """
        if not self.pet_datasets:
            return False
        
        # Kiểm tra modality
        try:
            first_dataset = self.pet_datasets[0]
            
            if hasattr(first_dataset, 'Modality') and first_dataset.Modality == 'PT':
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error validating PET data: {str(e)}")
            return False
    
    def _extract_suv_parameters(self) -> None:
        """
        Trích xuất các thông số cần thiết cho tính toán SUV từ dataset.
        
        Raises
        ------
        DicomError
            Nếu không thể trích xuất thông số SUV
        """
        try:
            if not self.pet_datasets:
                raise DicomError("PET datasets not loaded")
            
            first_dataset = self.pet_datasets[0]
            
            # Trọng lượng bệnh nhân (kg)
            if hasattr(first_dataset, 'PatientWeight'):
                self.patient_weight = float(first_dataset.PatientWeight)
            
            # Liều tiêm (Bq)
            if hasattr(first_dataset, 'RadiopharmaceuticalInformationSequence'):
                if len(first_dataset.RadiopharmaceuticalInformationSequence) > 0:
                    radiopharm_info = first_dataset.RadiopharmaceuticalInformationSequence[0]
                    
                    # Liều tiêm (Bq)
                    if hasattr(radiopharm_info, 'RadionuclideTotalDose'):
                        self.injected_dose = float(radiopharm_info.RadionuclideTotalDose)
                    
                    # Thời gian tiêm
                    if hasattr(radiopharm_info, 'RadiopharmaceuticalStartTime'):
                        start_time = radiopharm_info.RadiopharmaceuticalStartTime
                        start_date = first_dataset.SeriesDate  # Thường là SeriesDate
                        
                        # Chuyển đổi sang datetime
                        from datetime import datetime
                        self.injection_time = datetime.strptime(f"{start_date}{start_time.split('.')[0]}", "%Y%m%d%H%M%S")
                    
                    # Thời gian bán rã (giây)
                    if hasattr(radiopharm_info, 'RadionuclideHalfLife'):
                        self.isotope_half_life = float(radiopharm_info.RadionuclideHalfLife)
            
            # Thời gian thu nhận
            if hasattr(first_dataset, 'SeriesTime'):
                series_time = first_dataset.SeriesTime
                series_date = first_dataset.SeriesDate
                
                # Chuyển đổi sang datetime
                from datetime import datetime
                self.acquisition_time = datetime.strptime(f"{series_date}{series_time.split('.')[0]}", "%Y%m%d%H%M%S")
            
            logger.info(f"Extracted SUV parameters: weight={self.patient_weight}kg, "
                       f"dose={self.injected_dose}Bq, half-life={self.isotope_half_life}s")
            
        except Exception as e:
            logger.error(f"Error extracting SUV parameters: {str(e)}")
            raise DicomError(f"Error extracting SUV parameters: {str(e)}")
    
    def calculate_suv(self, suv_type: str = 'bw') -> np.ndarray:
        """
        Tính toán giá trị SUV (Standardized Uptake Value).
        
        Parameters
        ----------
        suv_type : str, optional
            Loại SUV, có thể là 'bw' (body weight), 'lbm' (lean body mass), 'bsa' (body surface area)
            
        Returns
        -------
        np.ndarray
            Mảng SUV
            
        Raises
        ------
        DicomError
            Nếu không thể tính toán SUV
        """
        try:
            if self.pet_volume is None:
                raise DicomError("PET data not loaded")
            
            if (self.patient_weight is None or self.injected_dose is None or
                self.injection_time is None or self.acquisition_time is None or
                self.isotope_half_life is None):
                raise DicomError("SUV parameters not complete")
            
            # Tính thời gian trôi qua (giây)
            import datetime
            time_diff = (self.acquisition_time - self.injection_time).total_seconds()
            
            # Tính toán suy giảm
            decay_factor = 2 ** (-time_diff / self.isotope_half_life)
            
            # Tính toán liều hiệu dụng
            effective_dose = self.injected_dose * decay_factor
            
            # Tính toán SUV dựa trên loại
            if suv_type == 'bw':  # Body weight
                suv_factor = self.patient_weight * 1000 / effective_dose  # Convert kg to g
            elif suv_type == 'lbm':  # Lean body mass
                # Ước tính LBM từ trọng lượng cơ thể
                # (Công thức đơn giản, không phân biệt giới tính)
                lbm = self.patient_weight * 0.8  # Ước lượng thô
                suv_factor = lbm * 1000 / effective_dose  # Convert kg to g
            elif suv_type == 'bsa':  # Body surface area
                # Ước tính BSA từ trọng lượng cơ thể (công thức Mosteller)
                # BSA (m²) = sqrt((height(cm) * weight(kg)) / 3600)
                # Vì không có chiều cao, nên ta ước tính dựa trên trọng lượng
                height = np.sqrt(self.patient_weight * 100)  # Ước lượng rất thô
                bsa = np.sqrt((height * self.patient_weight) / 3600)
                suv_factor = bsa * 10000 / effective_dose  # Convert m² to cm²
            else:
                raise DicomError(f"Unknown SUV type: {suv_type}")
            
            # Tính SUV
            self.suv_volume = self.pet_volume * suv_factor
            
            logger.info(f"Calculated SUV ({suv_type}): scale factor = {suv_factor}")
            
            return self.suv_volume
            
        except Exception as e:
            logger.error(f"Error calculating SUV: {str(e)}")
            raise DicomError(f"Error calculating SUV: {str(e)}")
    
    def load_ct_data(self, dicom_files: List[str]) -> None:
        """
        Tải dữ liệu CT từ các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM CT
            
        Raises
        ------
        DicomError
            Nếu không thể tải dữ liệu CT
        """
        try:
            reader = DicomReader()
            ct_datasets = reader.read_files(dicom_files)
            
            # Kiểm tra xem có phải CT không
            if not ct_datasets or not hasattr(ct_datasets[0], 'Modality') or ct_datasets[0].Modality != 'CT':
                raise DicomError("Files are not CT data")
            
            # Chuyển đổi thành dữ liệu khối
            converter = DicomConverter()
            ct_volume, ct_spacing, ct_origin, ct_direction = \
                converter.convert_dicom_to_volumetric_data(ct_datasets)
            
            # Chuyển đổi thành đối tượng SimpleITK
            self.ct_image = sitk.GetImageFromArray(ct_volume)
            self.ct_image.SetSpacing(ct_spacing)
            self.ct_image.SetOrigin(ct_origin)
            self.ct_image.SetDirection(ct_direction)
            
            logger.info(f"Loaded CT data: {len(dicom_files)} slices, "
                       f"shape: {ct_volume.shape}, "
                       f"spacing: {ct_spacing}")
            
        except Exception as e:
            logger.error(f"Error loading CT data: {str(e)}")
            raise DicomError(f"Error loading CT data: {str(e)}")
    
    def register_pet_to_ct(self, method: str = 'rigid') -> None:
        """
        Đăng ký (register) PET với CT.
        
        Parameters
        ----------
        method : str, optional
            Phương pháp đăng ký, có thể là 'rigid', 'affine', 'deformable'
            
        Raises
        ------
        DicomError
            Nếu không thể đăng ký PET với CT
        """
        if self.pet_sitk_image is None:
            raise DicomError("PET data not loaded")
            
        if self.ct_image is None:
            raise DicomError("CT data not loaded")
        
        try:
            # Lựa chọn phương pháp đăng ký
            if method == 'rigid':
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=200
                )
                registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
                    self.ct_image, self.pet_sitk_image, sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                ))
                
            elif method == 'affine':
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=300
                )
                registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
                    self.ct_image, self.pet_sitk_image, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                ))
                
            else:
                raise DicomError(f"Unsupported registration method: {method}")
            
            # Thực hiện đăng ký
            self.transform = registration_method.Execute(
                sitk.Cast(self.ct_image, sitk.sitkFloat32),
                sitk.Cast(self.pet_sitk_image, sitk.sitkFloat32)
            )
            
            logger.info(f"Completed {method} registration of PET to CT")
            
        except Exception as e:
            logger.error(f"Error registering PET to CT: {str(e)}")
            raise DicomError(f"Error registering PET to CT: {str(e)}")
    
    def get_registered_pet(self) -> sitk.Image:
        """
        Lấy ảnh PET đã đăng ký với CT.
        
        Returns
        -------
        sitk.Image
            Ảnh PET đã đăng ký
            
        Raises
        ------
        DicomError
            Nếu PET chưa được đăng ký
        """
        if self.transform is None:
            raise DicomError("PET has not been registered to CT")
        
        try:
            registered_pet = sitk.Resample(
                self.pet_sitk_image, self.ct_image, self.transform,
                sitk.sitkLinear, 0.0, self.pet_sitk_image.GetPixelID()
            )
            
            return registered_pet
            
        except Exception as e:
            logger.error(f"Error getting registered PET: {str(e)}")
            raise DicomError(f"Error getting registered PET: {str(e)}")
    
    def segment_pet_threshold(self, threshold: float = 2.5, min_volume: float = 1.0) -> Dict[int, np.ndarray]:
        """
        Phân đoạn ảnh PET dựa trên ngưỡng SUV.
        
        Parameters
        ----------
        threshold : float, optional
            Ngưỡng SUV, mặc định là 2.5 (thường dùng trong lâm sàng)
        min_volume : float, optional
            Thể tích tối thiểu (cm³) để coi là một vùng quan tâm
            
        Returns
        -------
        Dict[int, np.ndarray]
            Từ điển các vùng phân đoạn (ID, mặt nạ nhị phân)
            
        Raises
        ------
        DicomError
            Nếu không thể phân đoạn PET
        """
        try:
            if self.suv_volume is None:
                if self.pet_volume is None:
                    raise DicomError("PET data not loaded")
                # Tính SUV nếu chưa có
                self.calculate_suv()
            
            # Tạo mặt nạ nhị phân dựa trên ngưỡng
            binary_mask = (self.suv_volume > threshold).astype(np.uint8)
            
            # Phân tích các thành phần kết nối
            label_mask = sitk.GetArrayFromImage(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(binary_mask)
                )
            )
            
            # Lọc các vùng theo kích thước
            voxel_volume = np.prod(self.pet_spacing) / 1000.0  # cm³
            min_voxel_count = int(min_volume / voxel_volume)
            
            # Tạo từ điển các vùng
            regions = {}
            for label in range(1, label_mask.max() + 1):
                region_mask = (label_mask == label)
                if np.sum(region_mask) >= min_voxel_count:
                    regions[label] = region_mask
            
            logger.info(f"Segmented PET with threshold {threshold}, "
                       f"found {len(regions)} regions above {min_volume} cm³")
            
            return regions
            
        except Exception as e:
            logger.error(f"Error segmenting PET: {str(e)}")
            raise DicomError(f"Error segmenting PET: {str(e)}")
    
    def calculate_suv_statistics(self, roi_mask: np.ndarray = None) -> Dict[str, float]:
        """
        Tính toán thống kê SUV trong vùng quan tâm.
        
        Parameters
        ----------
        roi_mask : np.ndarray, optional
            Mặt nạ nhị phân của vùng quan tâm, nếu None thì tính toàn bộ ảnh
            
        Returns
        -------
        Dict[str, float]
            Từ điển các thống kê SUV (min, max, mean, std, median, volume)
            
        Raises
        ------
        DicomError
            Nếu không thể tính toán thống kê SUV
        """
        try:
            if self.suv_volume is None:
                if self.pet_volume is None:
                    raise DicomError("PET data not loaded")
                # Tính SUV nếu chưa có
                self.calculate_suv()
            
            # Nếu không có ROI, sử dụng toàn bộ ảnh
            if roi_mask is None:
                roi_mask = np.ones_like(self.suv_volume, dtype=bool)
            
            # Áp dụng mặt nạ
            suv_values = self.suv_volume[roi_mask]
            
            # Tính toán thống kê
            stats = {
                'SUVmin': np.min(suv_values),
                'SUVmax': np.max(suv_values),
                'SUVmean': np.mean(suv_values),
                'SUVstd': np.std(suv_values),
                'SUVmedian': np.median(suv_values),
                'Volume_cm3': np.sum(roi_mask) * np.prod(self.pet_spacing) / 1000.0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating SUV statistics: {str(e)}")
            raise DicomError(f"Error calculating SUV statistics: {str(e)}")
    
    def create_pet_ct_fusion(self, alpha: float = 0.4, colormap: str = 'hot') -> Tuple[np.ndarray, Tuple]:
        """
        Tạo fusion PET/CT.
        
        Parameters
        ----------
        alpha : float, optional
            Độ trong suốt của PET (0.0-1.0)
        colormap : str, optional
            Bảng màu cho PET ('hot', 'jet', 'hsv', v.v.)
            
        Returns
        -------
        Tuple[np.ndarray, Tuple]
            Mảng fusion và thông tin spacing
            
        Raises
        ------
        DicomError
            Nếu không thể tạo fusion
        """
        try:
            if self.pet_sitk_image is None or self.ct_image is None:
                raise DicomError("PET and CT data must be loaded")
            
            # Nếu đã đăng ký, sử dụng PET đã đăng ký
            if self.transform is not None:
                registered_pet = self.get_registered_pet()
                pet_array = sitk.GetArrayFromImage(registered_pet)
                
                # Chuẩn hóa PET (sử dụng SUV nếu có)
                if self.suv_volume is not None:
                    pet_array = self.suv_volume
                    
                # Chuẩn hóa về khoảng [0, 1]
                pet_norm = self._normalize_array(pet_array, clip_min=0, clip_max=10)  # Clip SUV > 10
                
                # Lấy dữ liệu CT
                ct_array = sitk.GetArrayFromImage(self.ct_image)
                
                # Chuẩn hóa CT về khoảng [0, 1] với cửa sổ phù hợp
                ct_norm = self._normalize_array(ct_array, clip_min=-250, clip_max=500)  # Window width/level
                
                # Tạo fusion
                # Đổi pet_norm thành màu sử dụng colormap
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap(colormap)
                pet_colored = cmap(pet_norm)[:, :, :, :3]  # Bỏ qua kênh alpha
                
                # Chuyển về dạng [slice, row, col, rgb]
                pet_colored = np.transpose(pet_colored, (1, 2, 3, 0))
                
                # Tạo ảnh CT grayscale 3 kênh
                ct_rgb = np.stack([ct_norm] * 3, axis=-1)
                
                # Trộn PET và CT
                fusion = ct_rgb * (1 - alpha * pet_norm[..., np.newaxis]) + pet_colored * alpha * pet_norm[..., np.newaxis]
                
                # Đảm bảo giá trị trong khoảng [0, 1]
                fusion = np.clip(fusion, 0, 1)
                
                return fusion, self.pet_spacing
                
            else:
                # Nếu chưa đăng ký, báo lỗi
                raise DicomError("PET must be registered to CT first, call register_pet_to_ct()")
            
        except Exception as e:
            logger.error(f"Error creating PET/CT fusion: {str(e)}")
            raise DicomError(f"Error creating PET/CT fusion: {str(e)}")
    
    def _normalize_array(self, array: np.ndarray, clip_min: float = None, clip_max: float = None) -> np.ndarray:
        """
        Chuẩn hóa mảng về khoảng [0, 1].
        
        Parameters
        ----------
        array : np.ndarray
            Mảng đầu vào
        clip_min : float, optional
            Giá trị tối thiểu để cắt
        clip_max : float, optional
            Giá trị tối đa để cắt
            
        Returns
        -------
        np.ndarray
            Mảng đã chuẩn hóa
        """
        # Cắt giá trị nếu cần
        if clip_min is not None or clip_max is not None:
            if clip_min is None:
                clip_min = np.min(array)
            if clip_max is None:
                clip_max = np.max(array)
            
            array = np.clip(array, clip_min, clip_max)
        
        # Chuẩn hóa
        min_val = np.min(array)
        max_val = np.max(array)
        
        if max_val == min_val:
            return np.zeros_like(array)
        
        return (array - min_val) / (max_val - min_val)
