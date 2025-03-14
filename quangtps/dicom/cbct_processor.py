"""
Công cụ xử lý dữ liệu CBCT (Cone Beam CT).

Module này cung cấp các chức năng để xử lý dữ liệu CBCT trong xạ trị, 
bao gồm việc đọc, chuyển đổi và phân tích dữ liệu CBCT.
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

class CBCTProcessor:
    """
    Lớp xử lý dữ liệu CBCT (Cone Beam CT).
    
    Class này cung cấp các phương thức để xử lý dữ liệu CBCT, bao gồm việc
    đọc dữ liệu, chuyển đổi HU (Hounsfield Unit), hiệu chỉnh nhiễu và
    đăng ký với CT quy hoạch.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng CBCTProcessor."""
        self.cbct_datasets = None
        self.cbct_volume = None
        self.cbct_spacing = None
        self.cbct_origin = None
        self.cbct_direction = None
        self.cbct_sitk_image = None
        self.planning_ct_image = None
        self.transform = None
    
    def load_cbct_data(self, dicom_files: List[str]) -> None:
        """
        Tải dữ liệu CBCT từ các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM CBCT
            
        Raises
        ------
        DicomError
            Nếu không thể tải dữ liệu CBCT
        """
        try:
            reader = DicomReader()
            self.cbct_datasets = reader.read_files(dicom_files)
            
            # Kiểm tra xem có phải CBCT không
            if not self._validate_cbct_data():
                raise DicomError("Files are not CBCT data")
            
            # Chuyển đổi thành dữ liệu khối
            converter = DicomConverter()
            self.cbct_volume, self.cbct_spacing, self.cbct_origin, self.cbct_direction = \
                converter.convert_dicom_to_volumetric_data(self.cbct_datasets)
            
            # Chuyển đổi thành đối tượng SimpleITK
            self.cbct_sitk_image = sitk.GetImageFromArray(self.cbct_volume)
            self.cbct_sitk_image.SetSpacing(self.cbct_spacing)
            self.cbct_sitk_image.SetOrigin(self.cbct_origin)
            self.cbct_sitk_image.SetDirection(self.cbct_direction)
            
            logger.info(f"Loaded CBCT data: {len(dicom_files)} slices, "
                       f"shape: {self.cbct_volume.shape}, "
                       f"spacing: {self.cbct_spacing}")
            
        except Exception as e:
            logger.error(f"Error loading CBCT data: {str(e)}")
            raise DicomError(f"Error loading CBCT data: {str(e)}")
    
    def _validate_cbct_data(self) -> bool:
        """
        Xác thực dữ liệu CBCT.
        
        Returns
        -------
        bool
            True nếu dữ liệu là CBCT, False nếu không phải
        """
        if not self.cbct_datasets:
            return False
        
        # Kiểm tra modality
        try:
            first_dataset = self.cbct_datasets[0]
            
            # Kiểm tra trực tiếp modality
            if hasattr(first_dataset, 'Modality') and first_dataset.Modality == 'CT':
                # Các hệ thống khác nhau có thể lưu CBCT với các thông tin khác nhau
                # Một số hệ thống sử dụng tag Manufacturer và Model để phân biệt
                if hasattr(first_dataset, 'Manufacturer') and 'VARIAN' in first_dataset.Manufacturer.upper():
                    return True
                
                if hasattr(first_dataset, 'ManufacturerModelName') and 'CBCT' in first_dataset.ManufacturerModelName.upper():
                    return True
                
                # Kiểm tra một số thuộc tính đặc trưng của CBCT
                if hasattr(first_dataset, 'ScanOptions') and 'CBCT' in first_dataset.ScanOptions:
                    return True
                
                # Kiểm tra khoảng cách slice thường lớn hơn CT thông thường
                if len(self.cbct_datasets) > 1:
                    if hasattr(self.cbct_datasets[0], 'SliceThickness') and \
                       hasattr(self.cbct_datasets[1], 'SliceThickness'):
                        if self.cbct_datasets[0].SliceThickness > 1.5 or \
                           abs(float(self.cbct_datasets[0].SliceLocation) - float(self.cbct_datasets[1].SliceLocation)) > 1.5:
                            # CBCT thường có độ dày lát cắt lớn hơn CT thông thường
                            return True
                
                # Nếu không có thông tin rõ ràng, giả định là CBCT nếu tệp đầu vào được chỉ định là CBCT
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error validating CBCT data: {str(e)}")
            return False
    
    def apply_noise_reduction(self, filter_type: str = 'median', params: Dict[str, Any] = None) -> None:
        """
        Áp dụng giảm nhiễu cho dữ liệu CBCT.
        
        Parameters
        ----------
        filter_type : str, optional
            Loại bộ lọc, có thể là 'median', 'gaussian', 'anisotropic_diffusion'
        params : Dict[str, Any], optional
            Tham số cho bộ lọc
            
        Raises
        ------
        DicomError
            Nếu không thể áp dụng giảm nhiễu
        """
        if self.cbct_sitk_image is None:
            raise DicomError("CBCT data not loaded")
        
        try:
            if params is None:
                params = {}
            
            if filter_type == 'median':
                radius = params.get('radius', 2)
                self.cbct_sitk_image = sitk.Median(self.cbct_sitk_image, [radius] * 3)
                logger.info(f"Applied median filter with radius {radius}")
                
            elif filter_type == 'gaussian':
                sigma = params.get('sigma', 1.0)
                self.cbct_sitk_image = sitk.DiscreteGaussian(self.cbct_sitk_image, sigma)
                logger.info(f"Applied gaussian filter with sigma {sigma}")
                
            elif filter_type == 'anisotropic_diffusion':
                time_step = params.get('time_step', 0.0625)
                conductance = params.get('conductance', 3.0)
                iterations = params.get('iterations', 5)
                self.cbct_sitk_image = sitk.CurvatureAnisotropicDiffusion(
                    self.cbct_sitk_image, timeStep=time_step, 
                    conductanceParameter=conductance, numberOfIterations=iterations
                )
                logger.info(f"Applied anisotropic diffusion with time_step={time_step}, "
                           f"conductance={conductance}, iterations={iterations}")
                
            else:
                raise DicomError(f"Unknown filter type: {filter_type}")
            
            # Cập nhật volume từ SimpleITK image
            self.cbct_volume = sitk.GetArrayFromImage(self.cbct_sitk_image)
            
        except Exception as e:
            logger.error(f"Error applying noise reduction: {str(e)}")
            raise DicomError(f"Error applying noise reduction: {str(e)}")
    
    def correct_hounsfield_units(self, method: str = 'histogram_matching', 
                               reference_ct: sitk.Image = None) -> None:
        """
        Hiệu chỉnh giá trị Hounsfield Units (HU) cho CBCT.
        
        Parameters
        ----------
        method : str, optional
            Phương pháp hiệu chỉnh HU, có thể là 'histogram_matching', 'linear_scaling', 'lookup_table'
        reference_ct : sitk.Image, optional
            Ảnh CT tham chiếu cho phương pháp histogram_matching
            
        Raises
        ------
        DicomError
            Nếu không thể hiệu chỉnh HU
        """
        if self.cbct_sitk_image is None:
            raise DicomError("CBCT data not loaded")
        
        try:
            if method == 'histogram_matching':
                if reference_ct is None and self.planning_ct_image is not None:
                    reference_ct = self.planning_ct_image
                
                if reference_ct is None:
                    raise DicomError("Reference CT is required for histogram matching")
                
                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(100)
                matcher.SetNumberOfMatchPoints(10)
                matcher.SetThresholdAtMeanIntensity(True)
                
                self.cbct_sitk_image = matcher.Execute(self.cbct_sitk_image, reference_ct)
                logger.info("Applied histogram matching for HU correction")
                
            elif method == 'linear_scaling':
                # Áp dụng phép biến đổi tuyến tính đơn giản
                # CBCT thường có thang HU bị lệch
                current_min = sitk.MinimumMaximumImageFilter().Execute(self.cbct_sitk_image).GetMinimum()
                current_max = sitk.MinimumMaximumImageFilter().Execute(self.cbct_sitk_image).GetMaximum()
                
                # Mục tiêu: Không khí ~ -1000 HU, nước ~ 0 HU, xương ~ 400-1000 HU
                target_min = -1000
                target_max = 3000
                
                # Biến đổi tuyến tính
                slope = (target_max - target_min) / (current_max - current_min)
                intercept = target_min - slope * current_min
                
                self.cbct_sitk_image = slope * self.cbct_sitk_image + intercept
                logger.info(f"Applied linear scaling for HU correction: slope={slope:.2f}, intercept={intercept:.2f}")
                
            elif method == 'lookup_table':
                # Sử dụng bảng tra cứu cho các vật liệu khác nhau
                # Đây chỉ là ví dụ đơn giản, trong thực tế cần bảng tra cứu phức tạp hơn
                
                # Chuyển đổi về mảng numpy
                cbct_array = sitk.GetArrayFromImage(self.cbct_sitk_image)
                
                # Áp dụng các quy tắc hiệu chỉnh
                # Không khí (thường < -800 HU trong CBCT)
                cbct_array[cbct_array < -800] = -1000
                
                # Mô mềm (thường từ -800 đến 0 trong CBCT)
                mask = (cbct_array >= -800) & (cbct_array < 0)
                cbct_array[mask] = -700 + 0.8 * (cbct_array[mask] + 800)
                
                # Nước và mô mềm đậm đặc (thường từ 0 đến 200 trong CBCT)
                mask = (cbct_array >= 0) & (cbct_array < 200)
                cbct_array[mask] = -100 + 0.5 * cbct_array[mask]
                
                # Xương (thường > 200 trong CBCT)
                mask = cbct_array >= 200
                cbct_array[mask] = 300 + 2.0 * (cbct_array[mask] - 200)
                
                # Chuyển trở lại SimpleITK image
                self.cbct_sitk_image = sitk.GetImageFromArray(cbct_array)
                self.cbct_sitk_image.SetSpacing(self.cbct_spacing)
                self.cbct_sitk_image.SetOrigin(self.cbct_origin)
                self.cbct_sitk_image.SetDirection(self.cbct_direction)
                
                logger.info("Applied lookup table for HU correction")
                
            else:
                raise DicomError(f"Unknown HU correction method: {method}")
            
            # Cập nhật volume từ SimpleITK image
            self.cbct_volume = sitk.GetArrayFromImage(self.cbct_sitk_image)
            
        except Exception as e:
            logger.error(f"Error correcting Hounsfield units: {str(e)}")
            raise DicomError(f"Error correcting Hounsfield units: {str(e)}")
    
    def load_planning_ct(self, dicom_files: List[str]) -> None:
        """
        Tải CT quy hoạch từ các file DICOM.
        
        Parameters
        ----------
        dicom_files : List[str]
            Danh sách các file DICOM CT
            
        Raises
        ------
        DicomError
            Nếu không thể tải CT quy hoạch
        """
        try:
            reader = DicomReader()
            planning_datasets = reader.read_files(dicom_files)
            
            # Chuyển đổi thành dữ liệu khối
            converter = DicomConverter()
            planning_volume, planning_spacing, planning_origin, planning_direction = \
                converter.convert_dicom_to_volumetric_data(planning_datasets)
            
            # Chuyển đổi thành đối tượng SimpleITK
            self.planning_ct_image = sitk.GetImageFromArray(planning_volume)
            self.planning_ct_image.SetSpacing(planning_spacing)
            self.planning_ct_image.SetOrigin(planning_origin)
            self.planning_ct_image.SetDirection(planning_direction)
            
            logger.info(f"Loaded planning CT data: {len(dicom_files)} slices, "
                       f"shape: {planning_volume.shape}, "
                       f"spacing: {planning_spacing}")
            
        except Exception as e:
            logger.error(f"Error loading planning CT data: {str(e)}")
            raise DicomError(f"Error loading planning CT data: {str(e)}")
    
    def register_to_planning_ct(self, method: str = 'rigid') -> None:
        """
        Đăng ký (register) CBCT với CT quy hoạch.
        
        Parameters
        ----------
        method : str, optional
            Phương pháp đăng ký, có thể là 'rigid', 'affine', 'deformable'
            
        Raises
        ------
        DicomError
            Nếu không thể đăng ký CBCT với CT quy hoạch
        """
        if self.cbct_sitk_image is None:
            raise DicomError("CBCT data not loaded")
            
        if self.planning_ct_image is None:
            raise DicomError("Planning CT data not loaded")
        
        try:
            # Lựa chọn phương pháp đăng ký
            if method == 'rigid':
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=200
                )
                registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
                    self.planning_ct_image, self.cbct_sitk_image, sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                ))
                
            elif method == 'affine':
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                registration_method.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=300
                )
                registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
                    self.planning_ct_image, self.cbct_sitk_image, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                ))
                
            elif method == 'deformable':
                # Đầu tiên, thực hiện đăng ký affine
                affine_transform = sitk.CenteredTransformInitializer(
                    self.planning_ct_image, self.cbct_sitk_image, sitk.AffineTransform(3),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY
                )
                
                affine_registration = sitk.ImageRegistrationMethod()
                affine_registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
                affine_registration.SetOptimizerAsRegularStepGradientDescent(
                    learningRate=1.0, minStep=0.001, numberOfIterations=100
                )
                affine_registration.SetInitialTransform(affine_transform)
                
                affine_transform = affine_registration.Execute(
                    sitk.Cast(self.planning_ct_image, sitk.sitkFloat32),
                    sitk.Cast(self.cbct_sitk_image, sitk.sitkFloat32)
                )
                
                # Sau đó, thực hiện đăng ký deformable
                displacement_field_transform = sitk.DisplacementFieldTransform(3)
                displacement_field_transform.SetSmoothingGaussianOnUpdate(
                    varianceForUpdateField=0.0, varianceForTotalField=1.5
                )
                
                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsANTSNeighborhoodCorrelation(4)
                registration_method.SetOptimizerAsGradientDescent(
                    learningRate=1.0, numberOfIterations=50, convergenceWindowSize=10
                )
                registration_method.SetInitialTransform(displacement_field_transform, inPlace=True)
                registration_method.SetInterpolator(sitk.sitkLinear)
                registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
                registration_method.SetSmoothingSigmasPerLevel([4, 2, 1])
                
                moving_resampled = sitk.Resample(
                    self.cbct_sitk_image, self.planning_ct_image, affine_transform,
                    sitk.sitkLinear, 0.0, self.cbct_sitk_image.GetPixelID()
                )
                
                self.transform = registration_method.Execute(
                    sitk.Cast(self.planning_ct_image, sitk.sitkFloat32),
                    sitk.Cast(moving_resampled, sitk.sitkFloat32)
                )
                
                # Kết hợp 2 phép biến đổi
                composite_transform = sitk.CompositeTransform(3)
                composite_transform.AddTransform(self.transform)
                composite_transform.AddTransform(affine_transform)
                
                self.transform = composite_transform
                
                logger.info("Completed deformable registration of CBCT to planning CT")
                return
                
            else:
                raise DicomError(f"Unknown registration method: {method}")
            
            # Thực hiện đăng ký
            self.transform = registration_method.Execute(
                sitk.Cast(self.planning_ct_image, sitk.sitkFloat32),
                sitk.Cast(self.cbct_sitk_image, sitk.sitkFloat32)
            )
            
            logger.info(f"Completed {method} registration of CBCT to planning CT")
            
        except Exception as e:
            logger.error(f"Error registering CBCT to planning CT: {str(e)}")
            raise DicomError(f"Error registering CBCT to planning CT: {str(e)}")
    
    def get_registered_cbct(self) -> sitk.Image:
        """
        Lấy ảnh CBCT đã đăng ký với CT quy hoạch.
        
        Returns
        -------
        sitk.Image
            Ảnh CBCT đã đăng ký
            
        Raises
        ------
        DicomError
            Nếu CBCT chưa được đăng ký
        """
        if self.transform is None:
            raise DicomError("CBCT has not been registered to planning CT")
        
        try:
            registered_cbct = sitk.Resample(
                self.cbct_sitk_image, self.planning_ct_image, self.transform,
                sitk.sitkLinear, 0.0, self.cbct_sitk_image.GetPixelID()
            )
            
            return registered_cbct
            
        except Exception as e:
            logger.error(f"Error getting registered CBCT: {str(e)}")
            raise DicomError(f"Error getting registered CBCT: {str(e)}")
    
    def compute_difference_volume(self) -> sitk.Image:
        """
        Tính toán khối khác biệt giữa CBCT đã đăng ký và CT quy hoạch.
        
        Returns
        -------
        sitk.Image
            Khối khác biệt
            
        Raises
        ------
        DicomError
            Nếu không thể tính toán khối khác biệt
        """
        if self.transform is None:
            raise DicomError("CBCT has not been registered to planning CT")
        
        try:
            # Lấy CBCT đã đăng ký
            registered_cbct = self.get_registered_cbct()
            
            # Tính toán khác biệt
            difference = sitk.Subtract(registered_cbct, self.planning_ct_image)
            
            return difference
            
        except Exception as e:
            logger.error(f"Error computing difference volume: {str(e)}")
            raise DicomError(f"Error computing difference volume: {str(e)}")
    
    def export_to_dicom(self, output_directory: str, series_description: str = "CBCT_Processed") -> List[str]:
        """
        Xuất CBCT đã xử lý thành các file DICOM.
        
        Parameters
        ----------
        output_directory : str
            Thư mục đầu ra
        series_description : str, optional
            Mô tả series
            
        Returns
        -------
        List[str]
            Danh sách các file DICOM đã tạo
            
        Raises
        ------
        DicomError
            Nếu không thể xuất DICOM
        """
        if self.cbct_sitk_image is None:
            raise DicomError("CBCT data not loaded")
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        try:
            # Lấy mẫu dataset gốc
            template_dataset = self.cbct_datasets[0]
            
            # Lấy dữ liệu từ SimpleITK image
            cbct_array = sitk.GetArrayFromImage(self.cbct_sitk_image)
            
            # Tạo một Series ID mới
            series_instance_uid = pydicom.uid.generate_uid()
            
            # Danh sách các file đã tạo
            created_files = []
            
            # Tạo file DICOM cho mỗi slice
            for i in range(cbct_array.shape[0]):
                # Tạo bản sao của template
                ds = pydicom.dataset.FileDataset(
                    filename_or_obj=None,
                    dataset=template_dataset,
                    file_meta=pydicom.dataset.FileMetaDataset(),
                    preamble=b"\0" * 128
                )
                
                # Cập nhật thông tin DICOM
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
                ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                
                ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
                ds.SeriesInstanceUID = series_instance_uid
                ds.SeriesDescription = series_description
                ds.InstanceNumber = i + 1
                
                # Cập nhật thông tin image
                ds.Rows, ds.Columns = cbct_array[i].shape
                
                # Cập nhật thông tin spatial
                ds.SliceThickness = self.cbct_spacing[2]
                ds.PixelSpacing = [self.cbct_spacing[0], self.cbct_spacing[1]]
                
                # Vị trí Z của slice hiện tại
                z_pos = self.cbct_origin[2] + i * self.cbct_spacing[2]
                ds.SliceLocation = z_pos
                
                # Image Position (Patient)
                ds.ImagePositionPatient = [self.cbct_origin[0], self.cbct_origin[1], z_pos]
                
                # Image Orientation (Patient)
                ds.ImageOrientationPatient = [
                    self.cbct_direction[0], self.cbct_direction[1], self.cbct_direction[2],
                    self.cbct_direction[3], self.cbct_direction[4], self.cbct_direction[5]
                ]
                
                # Chuyển đổi dữ liệu pixel
                # Đảm bảo định dạng phù hợp (int16 cho CT)
                pixel_array = cbct_array[i].astype(np.int16)
                ds.PixelData = pixel_array.tobytes()
                
                # Thiết lập các thông số pixel
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1  # Signed
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                
                # Thiết lập Rescale Intercept và Slope cho HU
                ds.RescaleIntercept = -1024  # Để 0 -> -1024 HU (air)
                ds.RescaleSlope = 1
                ds.RescaleType = "HU"
                
                # Lưu file
                output_file = os.path.join(output_directory, f"CBCT_{i+1:04d}.dcm")
                ds.save_as(output_file)
                created_files.append(output_file)
            
            logger.info(f"Exported {len(created_files)} CBCT DICOM files to {output_directory}")
            return created_files
            
        except Exception as e:
            logger.error(f"Error exporting CBCT to DICOM: {str(e)}")
            raise DicomError(f"Error exporting CBCT to DICOM: {str(e)}")
