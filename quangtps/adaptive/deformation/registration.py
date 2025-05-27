#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp chức năng đăng ký hình ảnh cho xạ trị thích ứng.
Hỗ trợ đăng ký hình ảnh cứng và biến dạng để phân tích các thay đổi của bệnh nhân.
"""

import os
import numpy as np
import logging
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass

from quangtps.core.types import Image, Structure
from quangtps.core.exceptions import RegistrationError

logger = logging.getLogger(__name__)


class RegistrationType(Enum):
    """Các loại đăng ký hình ảnh được hỗ trợ"""

    RIGID = auto()  # Đăng ký cứng - chỉ dịch chuyển và xoay
    AFFINE = auto()  # Đăng ký affine - thêm co dãn và cắt xén
    DEFORMABLE_BSPLINE = auto()  # Đăng ký biến dạng sử dụng BSpline
    DEFORMABLE_DEMONS = auto()  # Đăng ký biến dạng sử dụng thuật toán Demons
    DEFORMABLE_ELASTIX = auto()  # Đăng ký biến dạng sử dụng Elastix


class SimilarityMetric(Enum):
    """Các phép đo tương đồng được sử dụng trong đăng ký hình ảnh"""

    MEAN_SQUARES = auto()  # Bình phương trung bình sai số
    NORMALIZED_CORRELATION = auto()  # Tương quan chuẩn hóa
    MUTUAL_INFORMATION = auto()  # Thông tin tương hỗ
    NORMALIZED_MUTUAL_INFO = auto()  # Thông tin tương hỗ chuẩn hóa
    MATTES_MUTUAL_INFO = auto()  # Thông tin tương hỗ của Mattes


@dataclass
class RegistrationParameters:
    """Các tham số cho quá trình đăng ký hình ảnh"""

    registration_type: RegistrationType = RegistrationType.RIGID
    similarity_metric: SimilarityMetric = SimilarityMetric.MUTUAL_INFORMATION
    max_iterations: int = 200
    sampling_percentage: float = 0.10  # Phần trăm điểm ảnh được lấy mẫu
    learning_rate: float = 1.0
    convergence_threshold: float = 1e-6
    grid_size: Tuple[int, int, int] = (5, 5, 5)  # Cho đăng ký biến dạng BSpline
    smoothing_sigmas: List[float] = None
    shrink_factors: List[int] = None

    def __post_init__(self):
        """Khởi tạo các tham số mặc định cho smoothing_sigmas và shrink_factors"""
        if self.smoothing_sigmas is None:
            self.smoothing_sigmas = [3.0, 2.0, 1.0, 0.0]
        if self.shrink_factors is None:
            self.shrink_factors = [8, 4, 2, 1]


class RegistrationResult:
    """Kết quả của quá trình đăng ký hình ảnh"""

    def __init__(self):
        self.transform = None  # Phép biến đổi được tìm thấy
        self.transform_parameters = []  # Tham số của phép biến đổi
        self.metric_value = 0.0  # Giá trị của phép đo tương đồng đạt được
        self.iterations = 0  # Số vòng lặp đã thực hiện
        self.success = False  # Trạng thái thành công
        self.error_message = ""  # Thông báo lỗi nếu có
        self.fixed_image_id = ""  # ID của hình ảnh cố định
        self.moving_image_id = ""  # ID của hình ảnh di chuyển

    def get_transform_as_sitk(self) -> sitk.Transform:
        """Lấy đối tượng phép biến đổi SimpleITK"""
        return self.transform

    def apply_transform_to_image(self, image: Image) -> Image:
        """
        Áp dụng phép biến đổi cho hình ảnh

        Parameters
        ----------
        image : Image
            Hình ảnh cần biến đổi

        Returns
        -------
        Image
            Hình ảnh đã được biến đổi
        """
        # Chuyển đổi hình ảnh sang đối tượng SimpleITK
        sitk_image = sitk.GetImageFromArray(image.pixel_array)
        sitk_image.SetSpacing(image.spacing)
        sitk_image.SetOrigin(image.origin)
        sitk_image.SetDirection(image.direction.flatten())

        # Áp dụng phép biến đổi
        transformed_sitk_image = sitk.Resample(
            sitk_image,
            sitk_image,
            self.transform,
            sitk.sitkLinear,
            0.0,
            sitk_image.GetPixelID(),
        )

        # Chuyển đổi trở lại đối tượng Image
        transformed_array = sitk.GetArrayFromImage(transformed_sitk_image)

        # Tạo đối tượng Image mới từ dữ liệu đã biến đổi
        transformed_image = Image(image_data=transformed_array)

        return transformed_image

    def apply_transform_to_structure(self, structure: Structure) -> Structure:
        """
        Áp dụng phép biến đổi cho cấu trúc

        Parameters
        ----------
        structure : Structure
            Cấu trúc cần biến đổi

        Returns
        -------
        Structure
            Cấu trúc đã được biến đổi
        """
        # Trong thực tế, cần chuyển đổi các điểm của cấu trúc
        # Đây chỉ là phác thảo của hàm

        # Tạo một cấu trúc mới
        transformed_structure = Structure(name=structure.name)

        # Áp dụng phép biến đổi cho contour (simplify implementation)
        # Trong thực tế, cần thực hiện chi tiết hơn

        return transformed_structure

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi kết quả thành từ điển để lưu trữ hoặc hiển thị"""
        return {
            "transform_parameters": self.transform_parameters,
            "metric_value": self.metric_value,
            "iterations": self.iterations,
            "success": self.success,
            "error_message": self.error_message,
            "fixed_image_id": self.fixed_image_id,
            "moving_image_id": self.moving_image_id,
        }


class ImageRegistration:
    """
    Lớp cơ sở cho các thuật toán đăng ký hình ảnh
    """

    def __init__(self, parameters: RegistrationParameters = None):
        """
        Khởi tạo đối tượng đăng ký hình ảnh

        Parameters
        ----------
        parameters : RegistrationParameters, optional
            Các tham số đăng ký hình ảnh
        """
        self.parameters = parameters or RegistrationParameters()

    def _get_sitk_metric(self, metric: SimilarityMetric):
        """
        Lấy đối tượng phép đo tương đồng SimpleITK tương ứng

        Parameters
        ----------
        metric : SimilarityMetric
            Loại phép đo tương đồng

        Returns
        -------
        sitk.ImageMetric
            Đối tượng phép đo tương đồng SimpleITK
        """
        # Trả về string để sử dụng với registration method
        if metric == SimilarityMetric.MEAN_SQUARES:
            return "MeanSquares"
        elif metric == SimilarityMetric.NORMALIZED_CORRELATION:
            return "Correlation"
        elif metric == SimilarityMetric.MUTUAL_INFORMATION:
            return "JointHistogramMutualInformation"
        elif metric == SimilarityMetric.NORMALIZED_MUTUAL_INFO:
            return "NormalizedMutualInformation"
        elif metric == SimilarityMetric.MATTES_MUTUAL_INFO:
            return "MattesMutualInformation"
        else:
            raise RegistrationError(f"Phép đo tương đồng không được hỗ trợ: {metric}")

    def register(self, fixed_image: Image, moving_image: Image) -> RegistrationResult:
        """
        Thực hiện đăng ký hình ảnh di chuyển với hình ảnh cố định

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (cần được biến đổi)

        Returns
        -------
        RegistrationResult
            Kết quả của quá trình đăng ký
        """
        # Phương thức này sẽ được ghi đè trong các lớp con
        raise NotImplementedError(
            "Phương thức register cần được triển khai trong lớp con"
        )

    def _convert_to_sitk_image(self, image: Image) -> sitk.Image:
        """
        Chuyển đổi đối tượng Image thành đối tượng SimpleITK Image

        Parameters
        ----------
        image : Image
            Đối tượng Image cần chuyển đổi

        Returns
        -------
        sitk.Image
            Đối tượng SimpleITK Image tương ứng
        """
        sitk_image = sitk.GetImageFromArray(image.pixel_array)
        sitk_image.SetSpacing(image.spacing)
        sitk_image.SetOrigin(image.origin)
        sitk_image.SetDirection(image.direction.flatten())
        return sitk_image

    def _convert_from_sitk_image(
        self, sitk_image: sitk.Image, original_image: Image
    ) -> Image:
        """
        Chuyển đổi đối tượng SimpleITK Image thành đối tượng Image

        Parameters
        ----------
        sitk_image : sitk.Image
            Đối tượng SimpleITK Image cần chuyển đổi
        original_image : Image
            Đối tượng Image gốc để lấy thông tin metadata

        Returns
        -------
        Image
            Đối tượng Image tương ứng
        """
        array = sitk.GetArrayFromImage(sitk_image)

        image = Image(image_data=array)

        return image


class RigidRegistration(ImageRegistration):
    """
    Đăng ký hình ảnh cứng (chỉ dịch chuyển và xoay)
    """

    def __init__(self, parameters: RegistrationParameters = None):
        """
        Khởi tạo đối tượng đăng ký hình ảnh cứng

        Parameters
        ----------
        parameters : RegistrationParameters, optional
            Các tham số đăng ký hình ảnh
        """
        if parameters is None:
            parameters = RegistrationParameters(
                registration_type=RegistrationType.RIGID
            )
        else:
            parameters.registration_type = RegistrationType.RIGID

        super().__init__(parameters)

    def register(self, fixed_image: Image, moving_image: Image) -> RegistrationResult:
        """
        Thực hiện đăng ký hình ảnh cứng

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (cần được biến đổi)

        Returns
        -------
        RegistrationResult
            Kết quả của quá trình đăng ký
        """
        result = RegistrationResult()
        result.fixed_image_id = fixed_image.id
        result.moving_image_id = moving_image.id

        try:
            # Chuyển đổi sang đối tượng SimpleITK
            fixed_sitk = self._convert_to_sitk_image(fixed_image)
            moving_sitk = self._convert_to_sitk_image(moving_image)

            # Chuẩn bị phép đăng ký
            registration_method = sitk.ImageRegistrationMethod()

            # Thiết lập phép đo tương đồng
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(
                self.parameters.sampling_percentage
            )

            # Thiết lập bộ tối ưu hóa
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.parameters.learning_rate,
                numberOfIterations=self.parameters.max_iterations,
                convergenceMinimumValue=self.parameters.convergence_threshold,
                convergenceWindowSize=10,
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()

            # Thiết lập phép nội suy
            registration_method.SetInterpolator(sitk.sitkLinear)

            # Thiết lập phép biến đổi ban đầu
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

            registration_method.SetInitialTransform(initial_transform, inPlace=False)

            # Thực hiện đa phân giải
            registration_method.SetShrinkFactorsPerLevel(self.parameters.shrink_factors)
            registration_method.SetSmoothingSigmasPerLevel(
                self.parameters.smoothing_sigmas
            )
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Thực hiện đăng ký
            transform = registration_method.Execute(fixed_sitk, moving_sitk)

            # Lưu kết quả
            result.transform = transform
            result.transform_parameters = transform.GetParameters()
            result.metric_value = registration_method.GetMetricValue()
            result.iterations = registration_method.GetOptimizerIteration()
            result.success = True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình đăng ký cứng: {str(e)}")
            result.success = False
            result.error_message = str(e)

        return result


class AffineRegistration(ImageRegistration):
    """
    Đăng ký hình ảnh affine (dịch chuyển, xoay, co dãn, cắt xén)
    """

    def __init__(self, parameters: RegistrationParameters = None):
        """
        Khởi tạo đối tượng đăng ký hình ảnh affine

        Parameters
        ----------
        parameters : RegistrationParameters, optional
            Các tham số đăng ký hình ảnh
        """
        if parameters is None:
            parameters = RegistrationParameters(
                registration_type=RegistrationType.AFFINE
            )
        else:
            parameters.registration_type = RegistrationType.AFFINE

        super().__init__(parameters)

    def register(self, fixed_image: Image, moving_image: Image) -> RegistrationResult:
        """
        Thực hiện đăng ký hình ảnh affine

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (cần được biến đổi)

        Returns
        -------
        RegistrationResult
            Kết quả của quá trình đăng ký
        """
        result = RegistrationResult()
        result.fixed_image_id = fixed_image.id
        result.moving_image_id = moving_image.id

        try:
            # Chuyển đổi sang đối tượng SimpleITK
            fixed_sitk = self._convert_to_sitk_image(fixed_image)
            moving_sitk = self._convert_to_sitk_image(moving_image)

            # Chuẩn bị phép đăng ký
            registration_method = sitk.ImageRegistrationMethod()

            # Thiết lập phép đo tương đồng
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(
                self.parameters.sampling_percentage
            )

            # Thiết lập bộ tối ưu hóa
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.parameters.learning_rate,
                numberOfIterations=self.parameters.max_iterations,
                convergenceMinimumValue=self.parameters.convergence_threshold,
                convergenceWindowSize=10,
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()

            # Thiết lập phép nội suy
            registration_method.SetInterpolator(sitk.sitkLinear)

            # Thiết lập phép biến đổi ban đầu
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.AffineTransform(fixed_sitk.GetDimension()),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

            registration_method.SetInitialTransform(initial_transform, inPlace=False)

            # Thực hiện đa phân giải
            registration_method.SetShrinkFactorsPerLevel(self.parameters.shrink_factors)
            registration_method.SetSmoothingSigmasPerLevel(
                self.parameters.smoothing_sigmas
            )
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Thực hiện đăng ký
            transform = registration_method.Execute(fixed_sitk, moving_sitk)

            # Lưu kết quả
            result.transform = transform
            result.transform_parameters = transform.GetParameters()
            result.metric_value = registration_method.GetMetricValue()
            result.iterations = registration_method.GetOptimizerIteration()
            result.success = True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình đăng ký affine: {str(e)}")
            result.success = False
            result.error_message = str(e)

        return result


class BSplineDeformableRegistration(ImageRegistration):
    """
    Đăng ký hình ảnh biến dạng sử dụng BSpline
    """

    def __init__(self, parameters: RegistrationParameters = None):
        """
        Khởi tạo đối tượng đăng ký hình ảnh biến dạng BSpline

        Parameters
        ----------
        parameters : RegistrationParameters, optional
            Các tham số đăng ký hình ảnh
        """
        if parameters is None:
            parameters = RegistrationParameters(
                registration_type=RegistrationType.DEFORMABLE_BSPLINE
            )
        else:
            parameters.registration_type = RegistrationType.DEFORMABLE_BSPLINE

        super().__init__(parameters)

    def register(self, fixed_image: Image, moving_image: Image) -> RegistrationResult:
        """
        Thực hiện đăng ký hình ảnh biến dạng BSpline

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (cần được biến đổi)

        Returns
        -------
        RegistrationResult
            Kết quả của quá trình đăng ký
        """
        result = RegistrationResult()
        result.fixed_image_id = fixed_image.id
        result.moving_image_id = moving_image.id

        try:
            # Chuyển đổi sang đối tượng SimpleITK
            fixed_sitk = self._convert_to_sitk_image(fixed_image)
            moving_sitk = self._convert_to_sitk_image(moving_image)

            # Trước tiên, thực hiện đăng ký cứng để lấy phép biến đổi ban đầu tốt hơn
            rigid_registration = RigidRegistration()
            rigid_result = rigid_registration.register(fixed_image, moving_image)

            if not rigid_result.success:
                raise RegistrationError("Đăng ký cứng ban đầu thất bại")

            # Áp dụng phép biến đổi cứng cho hình ảnh di chuyển
            moving_sitk = sitk.Resample(
                moving_sitk,
                fixed_sitk,
                rigid_result.transform,
                sitk.sitkLinear,
                0.0,
                moving_sitk.GetPixelID(),
            )

            # Chuẩn bị phép đăng ký biến dạng
            registration_method = sitk.ImageRegistrationMethod()

            # Thiết lập phép đo tương đồng
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50
            )
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(
                self.parameters.sampling_percentage
            )

            # Thiết lập bộ tối ưu hóa
            registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=self.parameters.convergence_threshold,
                numberOfIterations=self.parameters.max_iterations,
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=1000,
                costFunctionConvergenceFactor=1e7,
            )

            # Thiết lập phép nội suy
            registration_method.SetInterpolator(sitk.sitkLinear)

            # Tạo lưới biến dạng BSpline
            transform_domain_mesh_size = self.parameters.grid_size
            transform_domain_physical_dimensions = [
                fixed_sitk.GetSize()[i] * fixed_sitk.GetSpacing()[i]
                for i in range(fixed_sitk.GetDimension())
            ]
            transform_domain_origin = fixed_sitk.GetOrigin()
            transform_domain_direction = fixed_sitk.GetDirection()

            # Tạo phép biến đổi BSpline
            transform = sitk.BSplineTransformInitializer(
                fixed_sitk,
                transform_domain_mesh_size,
                3,  # Order of BSpline (3 = cubic)
            )

            registration_method.SetInitialTransform(transform, inPlace=True)

            # Thực hiện đa phân giải
            registration_method.SetShrinkFactorsPerLevel(self.parameters.shrink_factors)
            registration_method.SetSmoothingSigmasPerLevel(
                self.parameters.smoothing_sigmas
            )
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            # Thực hiện đăng ký
            transform = registration_method.Execute(fixed_sitk, moving_sitk)

            # Kết hợp phép biến đổi cứng và biến dạng
            composite_transform = sitk.CompositeTransform(fixed_sitk.GetDimension())
            composite_transform.AddTransform(rigid_result.transform)
            composite_transform.AddTransform(transform)

            # Lưu kết quả
            result.transform = composite_transform
            result.transform_parameters = [
                rigid_result.transform.GetParameters(),
                transform.GetParameters(),
            ]
            result.metric_value = registration_method.GetMetricValue()
            result.iterations = registration_method.GetOptimizerIteration()
            result.success = True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình đăng ký biến dạng BSpline: {str(e)}")
            result.success = False
            result.error_message = str(e)

        return result


class DemonsDeformableRegistration(ImageRegistration):
    """
    Đăng ký hình ảnh biến dạng sử dụng thuật toán Demons
    """

    def __init__(self, parameters: RegistrationParameters = None):
        """
        Khởi tạo đối tượng đăng ký hình ảnh biến dạng Demons

        Parameters
        ----------
        parameters : RegistrationParameters, optional
            Các tham số đăng ký hình ảnh
        """
        if parameters is None:
            parameters = RegistrationParameters(
                registration_type=RegistrationType.DEFORMABLE_DEMONS
            )
        else:
            parameters.registration_type = RegistrationType.DEFORMABLE_DEMONS

        super().__init__(parameters)

    def register(self, fixed_image: Image, moving_image: Image) -> RegistrationResult:
        """
        Thực hiện đăng ký hình ảnh biến dạng Demons

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (cần được biến đổi)

        Returns
        -------
        RegistrationResult
            Kết quả của quá trình đăng ký
        """
        result = RegistrationResult()
        result.fixed_image_id = fixed_image.id
        result.moving_image_id = moving_image.id

        try:
            # Chuyển đổi sang đối tượng SimpleITK
            fixed_sitk = self._convert_to_sitk_image(fixed_image)
            moving_sitk = self._convert_to_sitk_image(moving_image)

            # Chuẩn hóa giá trị cường độ pixel
            fixed_sitk = sitk.Normalize(fixed_sitk)
            moving_sitk = sitk.Normalize(moving_sitk)

            # Trước tiên, thực hiện đăng ký cứng để lấy phép biến đổi ban đầu tốt hơn
            rigid_registration = RigidRegistration()
            rigid_result = rigid_registration.register(fixed_image, moving_image)

            if not rigid_result.success:
                raise RegistrationError("Đăng ký cứng ban đầu thất bại")

            # Áp dụng phép biến đổi cứng cho hình ảnh di chuyển
            moving_sitk = sitk.Resample(
                moving_sitk,
                fixed_sitk,
                rigid_result.transform,
                sitk.sitkLinear,
                0.0,
                moving_sitk.GetPixelID(),
            )

            # Thiết lập phép biến đổi Demons
            demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
            demons_filter.SetNumberOfIterations(self.parameters.max_iterations)
            demons_filter.SetStandardDeviations(1.0)

            # Thực hiện đăng ký Demons
            displacement_field = demons_filter.Execute(fixed_sitk, moving_sitk)

            # Tạo phép biến đổi từ trường dịch chuyển
            transform = sitk.DisplacementFieldTransform(displacement_field)

            # Kết hợp phép biến đổi cứng và biến dạng
            composite_transform = sitk.CompositeTransform(fixed_sitk.GetDimension())
            composite_transform.AddTransform(rigid_result.transform)
            composite_transform.AddTransform(transform)

            # Lưu kết quả
            result.transform = composite_transform
            result.transform_parameters = [
                rigid_result.transform.GetParameters(),
                "displacement_field",  # Không thể lưu trữ dưới dạng tham số
            ]
            result.metric_value = demons_filter.GetMetric()
            result.iterations = demons_filter.GetElapsedIterations()
            result.success = True

        except Exception as e:
            logger.error(f"Lỗi trong quá trình đăng ký biến dạng Demons: {str(e)}")
            result.success = False
            result.error_message = str(e)

        return result


# Factory function to create the appropriate registration object
def create_registration(
    registration_type: RegistrationType, parameters: RegistrationParameters = None
) -> ImageRegistration:
    """
    Tạo đối tượng đăng ký hình ảnh phù hợp dựa trên loại được chỉ định

    Parameters
    ----------
    registration_type : RegistrationType
        Loại đăng ký hình ảnh cần tạo
    parameters : RegistrationParameters, optional
        Các tham số đăng ký hình ảnh

    Returns
    -------
    ImageRegistration
        Đối tượng đăng ký hình ảnh phù hợp
    """
    if registration_type == RegistrationType.RIGID:
        return RigidRegistration(parameters)
    elif registration_type == RegistrationType.AFFINE:
        return AffineRegistration(parameters)
    elif registration_type == RegistrationType.DEFORMABLE_BSPLINE:
        return BSplineDeformableRegistration(parameters)
    elif registration_type == RegistrationType.DEFORMABLE_DEMONS:
        return DemonsDeformableRegistration(parameters)
    else:
        raise RegistrationError(f"Loại đăng ký không được hỗ trợ: {registration_type}")
