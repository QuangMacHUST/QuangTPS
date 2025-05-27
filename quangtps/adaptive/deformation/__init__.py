#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module xử lý biến dạng trong kế hoạch thích ứng.
Cung cấp các chức năng để đăng ký và theo dõi sự biến dạng giải phẫu.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RegistrationResult:
    """Kết quả của quá trình đăng ký ảnh"""

    def __init__(self, transformation_matrix=None, displacement_field=None):
        self.transformation_matrix = transformation_matrix
        self.displacement_field = displacement_field
        self.registration_accuracy = 0.0
        self.computation_time = 0.0
        self.success = True


class BaseRegistration(ABC):
    """Base class cho tất cả các phương pháp đăng ký ảnh"""

    def __init__(self):
        self.registered = False
        self.result = None

    @abstractmethod
    def register(self, fixed_image, moving_image):
        """Thực hiện đăng ký ảnh"""
        pass


class RigidRegistration(BaseRegistration):
    """
    Đăng ký ảnh cứng (rigid registration).
    Chỉ cho phép translation và rotation.
    """

    def __init__(self, translation_tolerance=1.0, rotation_tolerance=1.0):
        """
        Khởi tạo rigid registration

        Parameters
        ----------
        translation_tolerance : float
            Độ dung sai cho translation (mm)
        rotation_tolerance : float
            Độ dung sai cho rotation (degrees)
        """
        super().__init__()
        self.translation_tolerance = translation_tolerance
        self.rotation_tolerance = rotation_tolerance

    def register(self, fixed_image, moving_image):
        """
        Thực hiện rigid registration

        Parameters
        ----------
        fixed_image : ndarray
            Ảnh tham chiếu cố định
        moving_image : ndarray
            Ảnh cần đăng ký

        Returns
        -------
        RegistrationResult
            Kết quả đăng ký với transformation matrix
        """
        try:
            logger.info("Bắt đầu rigid registration")

            # Mock rigid registration với random transformation
            # Trong thực tế cần dùng ITK hoặc SimpleElastix

            # Tạo transformation matrix ngẫu nhiên trong phạm vi tolerance
            translation = np.random.uniform(
                -self.translation_tolerance, self.translation_tolerance, 3
            )
            rotation = np.random.uniform(
                -self.rotation_tolerance, self.rotation_tolerance, 3
            )

            # Tạo transformation matrix 4x4
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, 3] = translation

            self.result = RegistrationResult(
                transformation_matrix=transformation_matrix
            )
            self.result.registration_accuracy = np.random.uniform(0.8, 0.95)

            self.registered = True
            logger.info(
                f"Rigid registration hoàn thành với độ chính xác: {self.result.registration_accuracy:.3f}"
            )

            return self.result

        except Exception as e:
            logger.error(f"Lỗi trong rigid registration: {e}")
            self.result = RegistrationResult()
            self.result.success = False
            return self.result


class DeformableRegistration(BaseRegistration):
    """
    Đăng ký ảnh biến dạng (deformable registration).
    Cho phép biến dạng local của ảnh.
    """

    def __init__(self, grid_spacing=10.0, regularization_weight=0.1):
        """
        Khởi tạo deformable registration

        Parameters
        ----------
        grid_spacing : float
            Khoảng cách lưới cho B-spline transformation (mm)
        regularization_weight : float
            Trọng số regularization để smooth transformation
        """
        super().__init__()
        self.grid_spacing = grid_spacing
        self.regularization_weight = regularization_weight

    def register(self, fixed_image, moving_image):
        """
        Thực hiện deformable registration

        Parameters
        ----------
        fixed_image : ndarray
            Ảnh tham chiếu cố định
        moving_image : ndarray
            Ảnh cần đăng ký

        Returns
        -------
        RegistrationResult
            Kết quả đăng ký với displacement field
        """
        try:
            logger.info("Bắt đầu deformable registration")

            # Mock deformable registration
            # Trong thực tế cần dùng ITK hoặc SimpleElastix

            # Tạo displacement field ngẫu nhiên
            if hasattr(fixed_image, "shape"):
                displacement_shape = fixed_image.shape + (3,)
            else:
                # Mock shape nếu không có ảnh thực
                displacement_shape = (64, 64, 30, 3)

            displacement_field = np.random.normal(0, 2.0, displacement_shape)

            self.result = RegistrationResult(displacement_field=displacement_field)
            self.result.registration_accuracy = np.random.uniform(0.85, 0.98)

            self.registered = True
            logger.info(
                f"Deformable registration hoàn thành với độ chính xác: {self.result.registration_accuracy:.3f}"
            )

            return self.result

        except Exception as e:
            logger.error(f"Lỗi trong deformable registration: {e}")
            self.result = RegistrationResult()
            self.result.success = False
            return self.result

    def get_deformed_structure(self, structure_mask):
        """
        Áp dụng displacement field lên structure mask

        Parameters
        ----------
        structure_mask : ndarray
            Mask của structure cần biến dạng

        Returns
        -------
        ndarray
            Structure mask sau khi biến dạng
        """
        if not self.registered or self.result.displacement_field is None:
            logger.warning(
                "Chưa thực hiện registration hoặc không có displacement field"
            )
            return structure_mask

        # Mock deformation
        # Trong thực tế cần interpolation với displacement field
        deformed_mask = structure_mask.copy()

        # Thêm random noise nhỏ để mô phỏng biến dạng
        noise = np.random.normal(0, 0.1, structure_mask.shape)
        deformed_mask = np.clip(deformed_mask + noise, 0, 1)

        return deformed_mask


__all__ = [
    "RegistrationResult",
    "BaseRegistration",
    "RigidRegistration",
    "DeformableRegistration",
]
