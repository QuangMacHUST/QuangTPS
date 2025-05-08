#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module đăng ký hình ảnh cho QuangTPS.

Module này cung cấp các chức năng đăng ký hình ảnh, bao gồm đăng ký cứng (rigid) và
đăng ký biến dạng (deformable), hỗ trợ cho việc so sánh hình ảnh và các cấu trúc
giữa các thời điểm khác nhau trong quá trình điều trị.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from quangtps.core.types import Image, Structure
from quangtps.core.exceptions import RegistrationError

logger = logging.getLogger(__name__)

# Kiểm tra nếu có SimpleITK để dùng cho đăng ký hình ảnh
try:
    import SimpleITK as sitk

    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    logger.warning(
        "SimpleITK không được cài đặt. Một số tính năng đăng ký hình ảnh sẽ không khả dụng."
    )


def register_images(
    fixed_image: Image,
    moving_image: Image,
    method: str = "rigid",
    params: Optional[Dict[str, Any]] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Image, Dict[str, Any]]:
    """
    Đăng ký hình ảnh di chuyển vào hình ảnh cố định.

    Parameters
    ----------
    fixed_image : Image
        Hình ảnh cố định (hình ảnh tham chiếu)
    moving_image : Image
        Hình ảnh di chuyển (hình ảnh cần đăng ký)
    method : str, optional
        Phương pháp đăng ký ('rigid', 'affine', 'deformable'), mặc định là 'rigid'
    params : Optional[Dict[str, Any]], optional
        Tham số cho phương pháp đăng ký, mặc định là None
    mask : Optional[np.ndarray], optional
        Mặt nạ xác định vùng quan tâm cho đăng ký, mặc định là None

    Returns
    -------
    Tuple[Image, Dict[str, Any]]
        Hình ảnh đã đăng ký và thông tin về biến đổi

    Raises
    ------
    RegistrationError
        Nếu SimpleITK không được cài đặt hoặc có lỗi trong quá trình đăng ký
    """
    if not HAS_SITK:
        raise RegistrationError(
            "SimpleITK không được cài đặt. Không thể thực hiện đăng ký hình ảnh."
        )

    # Tham số mặc định cho đăng ký
    default_params = {
        "verbose": False,  # In thông tin chi tiết trong quá trình đăng ký
        "iterations": 100,  # Số lần lặp tối đa
        "sampling_percentage": 0.3,  # Phần trăm voxel được sử dụng
    }

    # Cập nhật với tham số đầu vào
    if params:
        default_params.update(params)

    try:
        # Chuyển đổi hình ảnh sang định dạng SimpleITK
        fixed_sitk = _convert_to_sitk(fixed_image)
        moving_sitk = _convert_to_sitk(moving_image)

        # Chuyển đổi mask sang SimpleITK nếu có
        mask_sitk = None
        if mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_sitk.CopyInformation(fixed_sitk)

        # Thực hiện đăng ký dựa trên phương pháp
        if method.lower() == "rigid":
            registered_sitk, transform_params = _rigid_registration(
                fixed_sitk, moving_sitk, mask_sitk, default_params
            )
        elif method.lower() == "affine":
            registered_sitk, transform_params = _affine_registration(
                fixed_sitk, moving_sitk, mask_sitk, default_params
            )
        elif method.lower() == "deformable":
            registered_sitk, transform_params = _deformable_registration(
                fixed_sitk, moving_sitk, mask_sitk, default_params
            )
        else:
            raise ValueError(f"Phương pháp đăng ký không hợp lệ: {method}")

        # Chuyển đổi kết quả trở lại định dạng Image
        registered_image = _convert_from_sitk(registered_sitk, fixed_image)

        # Trả về hình ảnh đã đăng ký và thông tin biến đổi
        return registered_image, transform_params

    except Exception as e:
        raise RegistrationError(f"Lỗi trong quá trình đăng ký hình ảnh: {str(e)}")


def register_structures(
    fixed_image: Image,
    moving_image: Image,
    structures: List[Structure],
    method: str = "rigid",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Structure], Dict[str, Any]]:
    """
    Đăng ký các cấu trúc từ hình ảnh di chuyển sang hình ảnh cố định.

    Parameters
    ----------
    fixed_image : Image
        Hình ảnh cố định (hình ảnh tham chiếu)
    moving_image : Image
        Hình ảnh di chuyển (hình ảnh có chứa cấu trúc)
    structures : List[Structure]
        Danh sách cấu trúc cần đăng ký
    method : str, optional
        Phương pháp đăng ký ('rigid', 'affine', 'deformable'), mặc định là 'rigid'
    params : Optional[Dict[str, Any]], optional
        Tham số cho phương pháp đăng ký, mặc định là None

    Returns
    -------
    Tuple[List[Structure], Dict[str, Any]]
        Danh sách cấu trúc đã đăng ký và thông tin về biến đổi

    Raises
    ------
    RegistrationError
        Nếu SimpleITK không được cài đặt hoặc có lỗi trong quá trình đăng ký
    """
    if not HAS_SITK:
        raise RegistrationError(
            "SimpleITK không được cài đặt. Không thể thực hiện đăng ký cấu trúc."
        )

    try:
        # Đăng ký hình ảnh trước
        _, transform_params = register_images(fixed_image, moving_image, method, params)

        # Biến đổi từng cấu trúc
        transformed_structures = []
        for structure in structures:
            transformed_structure = _transform_structure(
                structure, transform_params, fixed_image
            )
            transformed_structures.append(transformed_structure)

        return transformed_structures, transform_params

    except Exception as e:
        raise RegistrationError(f"Lỗi trong quá trình đăng ký cấu trúc: {str(e)}")


def _convert_to_sitk(image: Image) -> "sitk.Image":
    """
    Chuyển đổi đối tượng Image sang định dạng SimpleITK.

    Parameters
    ----------
    image : Image
        Đối tượng Image cần chuyển đổi

    Returns
    -------
    sitk.Image
        Đối tượng hình ảnh SimpleITK
    """
    if not HAS_SITK:
        raise RegistrationError(
            "SimpleITK không được cài đặt. Không thể chuyển đổi sang SimpleITK."
        )

    # Lấy dữ liệu numpy từ Image
    if image.data is None:
        raise ValueError("Hình ảnh không có dữ liệu.")

    # Chuyển đổi thành SimpleITK Image
    sitk_image = sitk.GetImageFromArray(image.data)

    # Thiết lập thông tin không gian
    spacing = [image.pixel_spacing[0], image.pixel_spacing[1], image.slice_thickness]
    sitk_image.SetSpacing(spacing)

    # Thông tin khác nếu có (origin, direction)
    if hasattr(image, "origin") and image.origin is not None:
        sitk_image.SetOrigin(image.origin)

    if hasattr(image, "direction") and image.direction is not None:
        sitk_image.SetDirection(image.direction)

    return sitk_image


def _convert_from_sitk(sitk_image: "sitk.Image", reference_image: Image) -> Image:
    """
    Chuyển đổi đối tượng SimpleITK.Image sang định dạng Image.

    Parameters
    ----------
    sitk_image : sitk.Image
        Đối tượng SimpleITK Image cần chuyển đổi
    reference_image : Image
        Đối tượng Image tham chiếu để sao chép metadata

    Returns
    -------
    Image
        Đối tượng Image
    """
    if not HAS_SITK:
        raise RegistrationError(
            "SimpleITK không được cài đặt. Không thể chuyển đổi từ SimpleITK."
        )

    # Tạo bản sao của đối tượng tham chiếu
    new_image = Image(reference_image.id + "_registered", reference_image.modality)

    # Lấy dữ liệu numpy từ SimpleITK
    new_image.data = sitk.GetArrayFromImage(sitk_image)

    # Sao chép thông tin không gian
    spacing = sitk_image.GetSpacing()
    new_image.pixel_spacing = (spacing[0], spacing[1])
    new_image.slice_thickness = spacing[2]

    # Sao chép thông tin khác nếu có (origin, direction)
    if hasattr(reference_image, "origin"):
        new_image.origin = sitk_image.GetOrigin()

    if hasattr(reference_image, "direction"):
        new_image.direction = sitk_image.GetDirection()

    return new_image


def _rigid_registration(
    fixed_image: "sitk.Image",
    moving_image: "sitk.Image",
    mask: Optional["sitk.Image"] = None,
    params: Dict[str, Any] = {},
) -> Tuple["sitk.Image", Dict[str, Any]]:
    """
    Thực hiện đăng ký cứng (rigid) giữa hai hình ảnh.

    Parameters
    ----------
    fixed_image : sitk.Image
        Hình ảnh cố định (hình ảnh tham chiếu)
    moving_image : sitk.Image
        Hình ảnh di chuyển (hình ảnh cần đăng ký)
    mask : Optional[sitk.Image], optional
        Mặt nạ xác định vùng quan tâm cho đăng ký, mặc định là None
    params : Dict[str, Any], optional
        Tham số cho đăng ký, mặc định là {}

    Returns
    -------
    Tuple[sitk.Image, Dict[str, Any]]
        Hình ảnh đã đăng ký và thông tin về biến đổi
    """
    # Thiết lập các tham số mặc định nếu không được cung cấp
    iterations = params.get("iterations", 100)
    sampling_percentage = params.get("sampling_percentage", 0.3)
    verbose = params.get("verbose", False)

    # Tạo đối tượng bộ lọc đăng ký
    registration_method = sitk.ImageRegistrationMethod()

    # Thiết lập metric để đo lường sự giống nhau
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)

    # Thiết lập bộ nội suy
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Thiết lập tối ưu hóa
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Thiết lập biến đổi
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Áp dụng mask nếu có
    if mask is not None:
        registration_method.SetMetricFixedMask(mask)

    # Thiết lập callback nếu cần
    if verbose:

        def _command_iteration(method):
            print(f"Iteration: {method.GetOptimizerIteration()}")
            print(f"Metric value: {method.GetMetricValue()}")

        registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: _command_iteration(registration_method)
        )

    # Thực hiện đăng ký
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Tính toán hình ảnh đã đăng ký
    registered_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    # Trích xuất thông tin biến đổi
    transform_params = {
        "type": "rigid",
        "transform": final_transform,
        "transform_parameters": final_transform.GetParameters(),
        "fixed_parameters": final_transform.GetFixedParameters(),
        "metric_value": registration_method.GetMetricValue(),
        "iterations": registration_method.GetOptimizerIteration(),
        "stop_condition": registration_method.GetOptimizerStopConditionDescription(),
    }

    return registered_image, transform_params


def _affine_registration(
    fixed_image: "sitk.Image",
    moving_image: "sitk.Image",
    mask: Optional["sitk.Image"] = None,
    params: Dict[str, Any] = {},
) -> Tuple["sitk.Image", Dict[str, Any]]:
    """
    Thực hiện đăng ký affine giữa hai hình ảnh.

    Parameters
    ----------
    fixed_image : sitk.Image
        Hình ảnh cố định (hình ảnh tham chiếu)
    moving_image : sitk.Image
        Hình ảnh di chuyển (hình ảnh cần đăng ký)
    mask : Optional[sitk.Image], optional
        Mặt nạ xác định vùng quan tâm cho đăng ký, mặc định là None
    params : Dict[str, Any], optional
        Tham số cho đăng ký, mặc định là {}

    Returns
    -------
    Tuple[sitk.Image, Dict[str, Any]]
        Hình ảnh đã đăng ký và thông tin về biến đổi
    """
    # Thiết lập các tham số mặc định nếu không được cung cấp
    iterations = params.get("iterations", 200)
    sampling_percentage = params.get("sampling_percentage", 0.3)
    verbose = params.get("verbose", False)

    # Thực hiện đăng ký cứng trước để có điểm khởi đầu tốt
    rigid_registered, rigid_params = _rigid_registration(
        fixed_image, moving_image, mask, params
    )
    rigid_transform = rigid_params["transform"]

    # Tạo đối tượng bộ lọc đăng ký
    registration_method = sitk.ImageRegistrationMethod()

    # Thiết lập metric để đo lường sự giống nhau
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)

    # Thiết lập bộ nội suy
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Thiết lập tối ưu hóa
    registration_method.SetOptimizerAsConjugateGradientLineSearch(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Chuyển đổi từ biến đổi cứng sang biến đổi affine
    affine_transform = sitk.AffineTransform(3)
    affine_transform.SetTranslation(rigid_transform.GetTranslation())
    affine_transform.SetMatrix(rigid_transform.GetMatrix())
    affine_transform.SetCenter(rigid_transform.GetCenter())

    registration_method.SetInitialTransform(affine_transform, inPlace=False)

    # Áp dụng mask nếu có
    if mask is not None:
        registration_method.SetMetricFixedMask(mask)

    # Thiết lập callback nếu cần
    if verbose:

        def _command_iteration(method):
            print(f"Iteration: {method.GetOptimizerIteration()}")
            print(f"Metric value: {method.GetMetricValue()}")

        registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: _command_iteration(registration_method)
        )

    # Thực hiện đăng ký
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Tính toán hình ảnh đã đăng ký
    registered_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    # Trích xuất thông tin biến đổi
    transform_params = {
        "type": "affine",
        "transform": final_transform,
        "transform_parameters": final_transform.GetParameters(),
        "fixed_parameters": final_transform.GetFixedParameters(),
        "metric_value": registration_method.GetMetricValue(),
        "iterations": registration_method.GetOptimizerIteration(),
        "stop_condition": registration_method.GetOptimizerStopConditionDescription(),
    }

    return registered_image, transform_params


def _deformable_registration(
    fixed_image: "sitk.Image",
    moving_image: "sitk.Image",
    mask: Optional["sitk.Image"] = None,
    params: Dict[str, Any] = {},
) -> Tuple["sitk.Image", Dict[str, Any]]:
    """
    Thực hiện đăng ký biến dạng (deformable) giữa hai hình ảnh.

    Parameters
    ----------
    fixed_image : sitk.Image
        Hình ảnh cố định (hình ảnh tham chiếu)
    moving_image : sitk.Image
        Hình ảnh di chuyển (hình ảnh cần đăng ký)
    mask : Optional[sitk.Image], optional
        Mặt nạ xác định vùng quan tâm cho đăng ký, mặc định là None
    params : Dict[str, Any], optional
        Tham số cho đăng ký, mặc định là {}

    Returns
    -------
    Tuple[sitk.Image, Dict[str, Any]]
        Hình ảnh đã đăng ký và thông tin về biến đổi
    """
    # Thiết lập các tham số mặc định nếu không được cung cấp
    iterations = params.get("iterations", 100)
    grid_spacing = params.get("grid_spacing", [50, 50, 50])
    smoothing_sigmas = params.get("smoothing_sigmas", [2, 1, 0])
    shrink_factors = params.get("shrink_factors", [4, 2, 1])
    verbose = params.get("verbose", False)

    # Thực hiện đăng ký affine trước để có điểm khởi đầu tốt
    affine_registered, affine_params = _affine_registration(
        fixed_image, moving_image, mask, params
    )
    affine_transform = affine_params["transform"]

    # Tạo đối tượng bộ lọc đăng ký
    registration_method = sitk.ImageRegistrationMethod()

    # Thiết lập metric để đo lường sự giống nhau
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    # Thiết lập bộ nội suy
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Đặt các mức độ (level) của biến đổi B-spline
    registration_method.SetShrinkFactorsPerLevel(shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Thiết lập tối ưu hóa
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    # Thiết lập biến đổi B-spline
    transform_domain_mesh_size = [8] * fixed_image.GetDimension()
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_image, transform_domain_mesh_size
    )

    # Kết hợp biến đổi affine với biến đổi B-spline
    composite_transform = sitk.CompositeTransform([affine_transform, bspline_transform])
    registration_method.SetInitialTransform(composite_transform, inPlace=False)

    # Áp dụng mask nếu có
    if mask is not None:
        registration_method.SetMetricFixedMask(mask)

    # Thiết lập callback nếu cần
    if verbose:

        def _command_iteration(method):
            print(f"Iteration: {method.GetOptimizerIteration()}")
            print(f"Metric value: {method.GetMetricValue()}")

        registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: _command_iteration(registration_method)
        )

    # Thực hiện đăng ký
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Tính toán hình ảnh đã đăng ký
    registered_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    # Trích xuất thông tin biến đổi
    transform_params = {
        "type": "deformable",
        "transform": final_transform,
        "metric_value": registration_method.GetMetricValue(),
        "iterations": registration_method.GetOptimizerIteration(),
        "stop_condition": registration_method.GetOptimizerStopConditionDescription(),
    }

    # Tính toán và lưu trường biến dạng
    displacement_field = sitk.TransformToDisplacementField(
        final_transform,
        sitk.sitkVectorFloat64,
        fixed_image.GetSize(),
        fixed_image.GetOrigin(),
        fixed_image.GetSpacing(),
        fixed_image.GetDirection(),
    )

    transform_params["displacement_field"] = displacement_field

    return registered_image, transform_params


def _transform_structure(
    structure: Structure, transform_params: Dict[str, Any], reference_image: Image
) -> Structure:
    """
    Áp dụng biến đổi cho một cấu trúc.

    Parameters
    ----------
    structure : Structure
        Cấu trúc cần biến đổi
    transform_params : Dict[str, Any]
        Thông tin biến đổi từ đăng ký hình ảnh
    reference_image : Image
        Hình ảnh tham chiếu để xác định không gian đích

    Returns
    -------
    Structure
        Cấu trúc đã biến đổi
    """
    if not HAS_SITK:
        raise RegistrationError(
            "SimpleITK không được cài đặt. Không thể biến đổi cấu trúc."
        )

    # Tạo bản sao của cấu trúc
    transformed_structure = type(structure)(
        structure.id + "_transformed", structure.name
    )
    transformed_structure.type = structure.type
    transformed_structure.color = structure.color

    # Lấy mặt nạ từ cấu trúc
    structure_mask = structure.get_binary_mask()

    # Chuyển đổi mặt nạ sang SimpleITK
    structure_sitk = sitk.GetImageFromArray(structure_mask.astype(np.uint8))

    # Thiết lập thông tin không gian từ cấu trúc gốc
    voxel_spacing = structure.get_voxel_spacing()
    structure_sitk.SetSpacing(voxel_spacing)

    # Chuyển đổi hình ảnh tham chiếu sang SimpleITK
    reference_sitk = _convert_to_sitk(reference_image)

    # Áp dụng biến đổi cho mặt nạ cấu trúc
    transform = transform_params["transform"]
    transformed_sitk = sitk.Resample(
        structure_sitk,
        reference_sitk,
        transform,
        sitk.sitkNearestNeighbor,  # Sử dụng nearest neighbor cho mặt nạ nhị phân
        0,
        structure_sitk.GetPixelID(),
    )

    # Chuyển đổi trở lại numpy
    transformed_mask = sitk.GetArrayFromImage(transformed_sitk).astype(bool)

    # Cập nhật cấu trúc mới
    transformed_structure.set_binary_mask(transformed_mask)
    transformed_structure.set_voxel_spacing(
        reference_image.pixel_spacing + (reference_image.slice_thickness,)
    )

    # Tính lại thể tích
    transformed_structure.calculate_volume()

    return transformed_structure


class ImageRegistration:
    """
    Lớp quản lý đăng ký hình ảnh với nhiều phương pháp.

    Lớp này cung cấp giao diện thống nhất cho các phương pháp đăng ký hình ảnh
    khác nhau, cho phép lưu trữ và sử dụng lại các biến đổi đã tính toán.
    """

    def __init__(self):
        """Khởi tạo đối tượng đăng ký hình ảnh."""
        self.transforms = {}  # Lưu trữ các biến đổi đã tính toán
        self._check_dependencies()

    def _check_dependencies(self):
        """Kiểm tra các thư viện phụ thuộc có khả dụng không."""
        if not HAS_SITK:
            logger.warning(
                "SimpleITK không được cài đặt. Một số tính năng đăng ký hình ảnh sẽ không khả dụng."
            )

    def register(
        self,
        fixed_image: Image,
        moving_image: Image,
        method: str = "rigid",
        params: Optional[Dict[str, Any]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[Image, Dict[str, Any]]:
        """
        Đăng ký hình ảnh di chuyển vào hình ảnh cố định.

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (hình ảnh tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (hình ảnh cần đăng ký)
        method : str, optional
            Phương pháp đăng ký ('rigid', 'affine', 'deformable'), mặc định là 'rigid'
        params : Optional[Dict[str, Any]], optional
            Tham số cho phương pháp đăng ký, mặc định là None
        mask : Optional[np.ndarray], optional
            Mặt nạ xác định vùng quan tâm cho đăng ký, mặc định là None

        Returns
        -------
        Tuple[Image, Dict[str, Any]]
            Hình ảnh đã đăng ký và thông tin về biến đổi
        """
        # Gọi hàm đăng ký
        registered_image, transform_params = register_images(
            fixed_image, moving_image, method, params, mask
        )

        # Lưu biến đổi để sử dụng lại
        transform_key = f"{fixed_image.id}_{moving_image.id}_{method}"
        self.transforms[transform_key] = transform_params

        return registered_image, transform_params

    def register_structures(
        self,
        fixed_image: Image,
        moving_image: Image,
        structures: List[Structure],
        method: str = "rigid",
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Structure], Dict[str, Any]]:
        """
        Đăng ký các cấu trúc từ hình ảnh di chuyển sang hình ảnh cố định.

        Parameters
        ----------
        fixed_image : Image
            Hình ảnh cố định (hình ảnh tham chiếu)
        moving_image : Image
            Hình ảnh di chuyển (hình ảnh có chứa cấu trúc)
        structures : List[Structure]
            Danh sách cấu trúc cần đăng ký
        method : str, optional
            Phương pháp đăng ký ('rigid', 'affine', 'deformable'), mặc định là 'rigid'
        params : Optional[Dict[str, Any]], optional
            Tham số cho phương pháp đăng ký, mặc định là None

        Returns
        -------
        Tuple[List[Structure], Dict[str, Any]]
            Danh sách cấu trúc đã đăng ký và thông tin về biến đổi
        """
        # Kiểm tra xem biến đổi đã được tính toán trước đó chưa
        transform_key = f"{fixed_image.id}_{moving_image.id}_{method}"
        if transform_key in self.transforms:
            transform_params = self.transforms[transform_key]
        else:
            # Đăng ký hình ảnh để lấy biến đổi
            _, transform_params = self.register(
                fixed_image, moving_image, method, params
            )

        # Biến đổi từng cấu trúc
        transformed_structures = []
        for structure in structures:
            transformed_structure = _transform_structure(
                structure, transform_params, fixed_image
            )
            transformed_structures.append(transformed_structure)

        return transformed_structures, transform_params

    def get_transform(
        self, fixed_image_id: str, moving_image_id: str, method: str = "rigid"
    ) -> Optional[Dict[str, Any]]:
        """
        Lấy biến đổi đã tính toán trước đó.

        Parameters
        ----------
        fixed_image_id : str
            ID của hình ảnh cố định
        moving_image_id : str
            ID của hình ảnh di chuyển
        method : str, optional
            Phương pháp đăng ký, mặc định là 'rigid'

        Returns
        -------
        Optional[Dict[str, Any]]
            Thông tin biến đổi hoặc None nếu không tìm thấy
        """
        transform_key = f"{fixed_image_id}_{moving_image_id}_{method}"
        return self.transforms.get(transform_key)


# Hàm tiện ích để nội suy hình ảnh giữa các thời điểm
def interpolate_images(
    images: List[Image],
    weights: List[float],
) -> Image:
    """
    Nội suy hình ảnh từ một tập hợp các hình ảnh với trọng số.

    Parameters
    ----------
    images : List[Image]
        Danh sách hình ảnh đầu vào
    weights : List[float]
        Danh sách trọng số tương ứng, tổng bằng 1

    Returns
    -------
    Image
        Hình ảnh đã nội suy
    """
    if len(images) != len(weights):
        raise ValueError("Số lượng hình ảnh và trọng số phải bằng nhau.")

    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Tổng trọng số phải bằng 1.")

    # Tạo một hình ảnh mới với cùng thuộc tính như hình ảnh đầu tiên
    reference_image = images[0]
    interpolated_image = Image(
        reference_image.id + "_interpolated", reference_image.modality
    )
    interpolated_image.pixel_spacing = reference_image.pixel_spacing
    interpolated_image.slice_thickness = reference_image.slice_thickness

    # Khởi tạo dữ liệu hình ảnh với 0
    interpolated_image.data = np.zeros_like(reference_image.data)

    # Nội suy dữ liệu
    for image, weight in zip(images, weights):
        interpolated_image.data += image.data * weight

    return interpolated_image
