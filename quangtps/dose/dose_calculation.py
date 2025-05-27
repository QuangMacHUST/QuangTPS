"""
Module tính toán liều lượng cho hệ thống QuangTPS.

Module này cung cấp các lớp và hàm thực hiện tính toán liều lượng chi tiết,
tích hợp với các thuật toán tính toán và cung cấp kết quả để hiển thị và phân tích.
"""

import os
import time
import datetime
import logging
import numpy as np
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Union, Type, Callable

from quangtps.core.exceptions import ValidationError, CalculationError
from quangtps.core.config import Config
from quangtps.dose.dose_engine import DoseEngine, DoseCalculationAlgorithm
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)


class DoseCalculationStatus:
    """Lớp biểu diễn trạng thái tính toán liều."""

    def __init__(self):
        """Khởi tạo trạng thái tính toán liều."""
        self.start_time = None
        self.end_time = None
        self.progress = 0.0
        self.status = "Not Started"
        self.completed = False
        self.error = None
        self.calculation_id = None
        self.details = {}

    def start(self):
        """Bắt đầu tính toán."""
        self.start_time = time.time()
        self.status = "Running"
        self.progress = 0.0
        self.completed = False
        self.error = None

    def update_progress(self, progress: float, status: str = None):
        """
        Cập nhật tiến độ tính toán.

        Parameters:
            progress (float): Tiến độ từ 0.0 đến 1.0
            status (str, optional): Thông tin trạng thái
        """
        self.progress = max(0.0, min(1.0, progress))
        if status:
            self.status = status

    def complete(self):
        """Hoàn thành tính toán."""
        self.end_time = time.time()
        self.progress = 1.0
        self.status = "Completed"
        self.completed = True

    def fail(self, error_message: str):
        """
        Đánh dấu tính toán thất bại.

        Parameters:
            error_message (str): Thông tin lỗi
        """
        self.end_time = time.time()
        self.status = "Failed"
        self.error = error_message
        self.completed = False

    def get_elapsed_time(self) -> float:
        """
        Lấy thời gian đã trôi qua.

        Returns:
            float: Thời gian tính toán (giây)
        """
        if not self.start_time:
            return 0.0

        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """
        Lấy tóm tắt trạng thái tính toán.

        Returns:
            dict: Tóm tắt trạng thái
        """
        elapsed = self.get_elapsed_time()

        return {
            "calculation_id": self.calculation_id,
            "status": self.status,
            "progress": self.progress,
            "elapsed_time": elapsed,
            "elapsed_formatted": str(datetime.timedelta(seconds=int(elapsed))),
            "completed": self.completed,
            "error": self.error,
            "details": self.details,
        }


class DoseRegionOfInterest:
    """
    Lớp xác định vùng tính toán liều, có thể là toàn bộ hình ảnh hoặc
    một vùng cụ thể liên quan đến một cấu trúc.
    """

    def __init__(
        self,
        name: str = "ROI",
        structure_name: Optional[str] = None,
        margin_mm: float = 10.0,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Khởi tạo vùng tính toán liều.

        Parameters:
            name (str): Tên vùng tính toán
            structure_name (str, optional): Tên cấu trúc liên quan
            margin_mm (float): Khoảng mở rộng ra ngoài cấu trúc (mm)
            bounds (list, optional): Ranh giới vùng tính toán [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        self.name = name
        self.structure_name = structure_name
        self.margin_mm = margin_mm
        self.bounds = bounds

    def calculate_bounds_from_structure(
        self, structures: Dict[str, np.ndarray], image_info: Dict[str, Any]
    ) -> List[Tuple[float, float]]:
        """
        Tính toán ranh giới vùng tính toán từ cấu trúc.

        Parameters:
            structures (dict): Dict các cấu trúc
            image_info (dict): Thông tin hình ảnh

        Returns:
            list: Danh sách ranh giới [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

        Raises:
            ValueError: Nếu không tìm thấy cấu trúc
        """
        if not self.structure_name or self.structure_name not in structures:
            raise ValueError(f"Structure '{self.structure_name}' not found")

        # Lấy thông tin vị trí và kích thước voxel
        spacing = image_info.get("spacing", [1.0, 1.0, 1.0])
        origin = image_info.get("origin", [0.0, 0.0, 0.0])

        # Lấy mặt nạ cấu trúc
        structure_mask = structures[self.structure_name]

        # Tìm các vị trí voxel khác 0 (nơi có cấu trúc)
        indices = np.where(structure_mask > 0)

        if len(indices[0]) == 0:
            raise ValueError(f"Structure '{self.structure_name}' is empty")

        # Tính toán min/max theo từng chiều
        min_indices = [np.min(idx) for idx in indices]
        max_indices = [np.max(idx) for idx in indices]

        # Chuyển đổi từ chỉ số voxel sang tọa độ thực (mm)
        bounds = []
        for i in range(3):
            min_pos = origin[i] + min_indices[i] * spacing[i] - self.margin_mm
            max_pos = origin[i] + max_indices[i] * spacing[i] + self.margin_mm
            bounds.append((min_pos, max_pos))

        return bounds

    def get_bounds(
        self,
        structures: Dict[str, np.ndarray] = None,
        image_info: Dict[str, Any] = None,
    ) -> List[Tuple[float, float]]:
        """
        Lấy ranh giới vùng tính toán.

        Parameters:
            structures (dict, optional): Dict các cấu trúc
            image_info (dict, optional): Thông tin hình ảnh

        Returns:
            list: Danh sách ranh giới [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        if self.bounds:
            return self.bounds

        if self.structure_name and structures and image_info:
            return self.calculate_bounds_from_structure(structures, image_info)

        # Default bounds (full image)
        if image_info:
            size = image_info.get("size", [100, 100, 100])
            spacing = image_info.get("spacing", [1.0, 1.0, 1.0])
            origin = image_info.get("origin", [0.0, 0.0, 0.0])

            return [
                (origin[0], origin[0] + size[0] * spacing[0]),
                (origin[1], origin[1] + size[1] * spacing[1]),
                (origin[2], origin[2] + size[2] * spacing[2]),
            ]

        # Fallback to a default cubic region
        return [(-100, 100), (-100, 100), (-100, 100)]


class DoseCalculator:
    """
    Lớp quản lý và thực hiện tính toán liều.

    DoseCalculator điều phối quá trình tính toán liều, bao gồm thiết lập
    tham số, chuẩn bị dữ liệu, và thực hiện tính toán liều từ một hoặc nhiều chùm tia.
    """

    def __init__(
        self,
        algorithm: Union[str, DoseCalculationAlgorithm] = DoseCalculationAlgorithm.CCC,
    ):
        """
        Khởi tạo DoseCalculator.

        Parameters:
            algorithm (str or DoseCalculationAlgorithm, optional): Thuật toán tính toán liều
        """
        self.config = Config()
        self.dose_engine = DoseEngine(algorithm)
        self.parameters = {}
        self.roi = None
        self.calculation_status = DoseCalculationStatus()
        self.result_grid = None
        self.callback = None
        self.max_workers = self.config.get("dose.max_workers", 4)

    def set_algorithm(self, algorithm: Union[str, DoseCalculationAlgorithm]) -> bool:
        """
        Đặt thuật toán tính toán liều.

        Parameters:
            algorithm (str or DoseCalculationAlgorithm): Thuật toán tính toán liều

        Returns:
            bool: True nếu thành công
        """
        return self.dose_engine.set_algorithm(algorithm)

    def set_parameter(self, name: str, value: Any):
        """
        Đặt tham số tính toán.

        Parameters:
            name (str): Tên tham số
            value: Giá trị tham số
        """
        self.parameters[name] = value

        # Chuyển tiếp tham số nếu nó là tham số của dose_engine
        if name.startswith("engine."):
            engine_param = name[7:]  # Remove 'engine.' prefix
            self.dose_engine.set_parameter(engine_param, value)

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Đặt nhiều tham số tính toán.

        Parameters:
            parameters (dict): Dict của các tham số
        """
        for name, value in parameters.items():
            self.set_parameter(name, value)

    def set_region_of_interest(self, roi: DoseRegionOfInterest):
        """
        Đặt vùng tính toán liều.

        Parameters:
            roi (DoseRegionOfInterest): Vùng tính toán liều
        """
        self.roi = roi

    def set_callback(self, callback: Callable[[DoseCalculationStatus], None]):
        """
        Đặt callback để nhận thông báo về tiến độ tính toán.

        Parameters:
            callback (Callable): Hàm callback
        """
        self.callback = callback

    def _update_status(self, progress: float = None, status: str = None):
        """
        Cập nhật trạng thái tính toán và gọi callback.

        Parameters:
            progress (float, optional): Tiến độ từ 0.0 đến 1.0
            status (str, optional): Thông tin trạng thái
        """
        if progress is not None:
            self.calculation_status.update_progress(progress, status)
        elif status:
            self.calculation_status.status = status

        if self.callback:
            self.callback(self.calculation_status)

    def _prepare_dose_grid(
        self,
        patient_ct: sitk.Image,
        roi: DoseRegionOfInterest,
        structures: Dict[str, np.ndarray],
    ) -> DoseGrid:
        """
        Chuẩn bị lưới liều dựa trên vùng tính toán.

        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT của bệnh nhân
            roi (DoseRegionOfInterest): Vùng tính toán liều
            structures (dict): Dict các cấu trúc

        Returns:
            DoseGrid: Lưới liều đã chuẩn bị
        """
        # Lấy thông tin hình ảnh
        size = patient_ct.GetSize()
        spacing = patient_ct.GetSpacing()
        origin = patient_ct.GetOrigin()

        image_info = {
            "size": size,
            "spacing": spacing,
            "origin": origin,
            "direction": patient_ct.GetDirection(),
        }

        # Tính toán ranh giới vùng tính toán
        if not roi:
            # Sử dụng toàn bộ hình ảnh nếu không có ROI
            roi = DoseRegionOfInterest(name="FullImage")

        bounds = roi.get_bounds(structures, image_info)

        # Tính toán kích thước lưới
        grid_resolution = self.parameters.get("grid_resolution", spacing)

        grid_size = []
        grid_origin = []

        for i in range(3):
            bound_min, bound_max = bounds[i]

            # Tính số điểm lưới
            num_points = int((bound_max - bound_min) / grid_resolution[i]) + 1

            grid_size.append(num_points)
            grid_origin.append(bound_min)

        # Tạo lưới liều
        dose_grid = DoseGrid(
            size=grid_size,
            spacing=grid_resolution,
            origin=grid_origin,
            direction=patient_ct.GetDirection(),
        )

        return dose_grid

    def calculate_dose(
        self,
        patient_ct: sitk.Image,
        structures: Dict[str, np.ndarray],
        beams: List[Dict[str, Any]],
        roi: Optional[DoseRegionOfInterest] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> DoseGrid:
        """
        Tính toán phân bố liều.

        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT của bệnh nhân
            structures (dict): Dict các cấu trúc
            beams (list): Danh sách các chùm tia
            roi (DoseRegionOfInterest, optional): Vùng tính toán liều
            parameters (dict, optional): Các tham số tính toán

        Returns:
            DoseGrid: Kết quả tính toán liều

        Raises:
            CalculationError: Nếu có lỗi trong quá trình tính toán
        """
        try:
            # Thiết lập tham số nếu có
            if parameters:
                self.set_parameters(parameters)

            # Sử dụng ROI đã cung cấp hoặc ROI mặc định
            if roi:
                self.roi = roi

            # Bắt đầu tính toán
            self.calculation_status = DoseCalculationStatus()
            self.calculation_status.start()

            # Tạo ID cho lần tính toán này
            import uuid

            self.calculation_status.calculation_id = str(uuid.uuid4())

            self._update_status(0.01, "Preparing calculation grid")

            # Chuẩn bị lưới liều
            reference_grid = self._prepare_dose_grid(patient_ct, self.roi, structures)

            self._update_status(0.05, "Validating input data")

            # Kiểm tra dữ liệu đầu vào
            if not beams:
                raise ValidationError("No beams provided for dose calculation")

            # Thực hiện tính toán liều
            self._update_status(0.1, "Starting dose calculation")

            # Tính toán liều cho từng chùm tia và tổng hợp
            if (
                self.parameters.get("parallel_beam_calculation", True)
                and len(beams) > 1
            ):
                # Tính toán song song
                beam_results = self._calculate_beams_parallel(
                    patient_ct, structures, beams, reference_grid
                )
            else:
                # Tính toán tuần tự
                beam_results = self._calculate_beams_sequential(
                    patient_ct, structures, beams, reference_grid
                )

            # Tổng hợp kết quả
            self._update_status(0.9, "Combining beam doses")

            # Nếu chỉ có một chùm tia, sử dụng kết quả trực tiếp
            if len(beam_results) == 1:
                self.result_grid = beam_results[0]
            else:
                # Tổng hợp liều từ nhiều chùm tia
                self.result_grid = reference_grid.copy()

                # Cộng liều từ tất cả các chùm tia
                for beam_grid in beam_results:
                    self.result_grid.add_dose(beam_grid)

            # Chuẩn hóa liều
            if self.parameters.get("normalize_to_prescription", False):
                prescription_dose = self.parameters.get("prescription_dose", 0.0)
                if prescription_dose > 0:
                    self._update_status(0.95, "Normalizing dose")
                    self.result_grid.normalize(prescription_dose)

            # Hoàn tất tính toán
            self.calculation_status.complete()
            self._update_status(1.0, "Calculation completed")

            return self.result_grid

        except Exception as e:
            logger.error(f"Error in dose calculation: {str(e)}")
            self.calculation_status.fail(str(e))
            self._update_status(status=f"Error: {str(e)}")
            raise CalculationError(f"Dose calculation failed: {str(e)}")

    def _calculate_beams_sequential(
        self,
        patient_ct: sitk.Image,
        structures: Dict[str, np.ndarray],
        beams: List[Dict[str, Any]],
        reference_grid: DoseGrid,
    ) -> List[DoseGrid]:
        """
        Tính toán liều tuần tự cho từng chùm tia.

        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT của bệnh nhân
            structures (dict): Dict các cấu trúc
            beams (list): Danh sách các chùm tia
            reference_grid (DoseGrid): Lưới liều tham chiếu

        Returns:
            list: Danh sách kết quả tính toán liều cho từng chùm tia
        """
        results = []

        # Tính toán liều cho từng chùm tia
        for i, beam in enumerate(beams):
            beam_progress_start = 0.1 + (0.8 * i / len(beams))
            beam_progress_end = 0.1 + (0.8 * (i + 1) / len(beams))

            progress = beam_progress_start
            self._update_status(
                progress,
                f"Calculating beam {i + 1}/{len(beams)}: {beam.get('name', f'Beam {i + 1}')}",
            )

            # Tính toán liều cho chùm tia
            beam_grid = self.dose_engine.calculate_dose(
                patient_ct=patient_ct,
                structures=structures,
                beams=[beam],  # Single beam
                reference_grid=reference_grid,
                parameters=self.parameters,
            )

            # Áp dụng trọng số chùm tia nếu có
            beam_weight = beam.get("weight", 1.0)
            if beam_weight != 1.0:
                beam_grid.scale_dose(beam_weight)

            results.append(beam_grid)
            progress = beam_progress_end
            self._update_status(progress)

        return results

    def _calculate_beams_parallel(
        self,
        patient_ct: sitk.Image,
        structures: Dict[str, np.ndarray],
        beams: List[Dict[str, Any]],
        reference_grid: DoseGrid,
    ) -> List[DoseGrid]:
        """
        Tính toán liều song song cho các chùm tia.

        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT của bệnh nhân
            structures (dict): Dict các cấu trúc
            beams (list): Danh sách các chùm tia
            reference_grid (DoseGrid): Lưới liều tham chiếu

        Returns:
            list: Danh sách kết quả tính toán liều cho từng chùm tia
        """
        max_workers = min(self.max_workers, len(beams))
        results = [None] * len(beams)
        completed = 0

        def calculate_beam(index, beam):
            # Clone dose engine cho mỗi beam để tránh xung đột
            beam_engine = DoseEngine(self.dose_engine.algorithm)
            for param_name, param_value in self.dose_engine.get_parameters().items():
                beam_engine.set_parameter(param_name, param_value)

            # Tính toán liều cho chùm tia
            beam_grid = beam_engine.calculate_dose(
                patient_ct=patient_ct,
                structures=structures,
                beams=[beam],  # Single beam
                reference_grid=reference_grid,
                parameters=self.parameters,
            )

            # Áp dụng trọng số chùm tia nếu có
            beam_weight = beam.get("weight", 1.0)
            if beam_weight != 1.0:
                beam_grid.scale_dose(beam_weight)

            return index, beam_grid

        def update_progress(index, result):
            nonlocal completed
            results[index] = result
            completed += 1
            progress = 0.1 + (0.8 * completed / len(beams))
            self._update_status(progress, f"Completed {completed}/{len(beams)} beams")

        # Sử dụng ThreadPoolExecutor để tính toán song song
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for i, beam in enumerate(beams):
                future = executor.submit(calculate_beam, i, beam)
                futures[future] = i

            # Xử lý kết quả khi hoàn thành
            for future in futures:
                try:
                    idx, result = future.result()
                    results[idx] = result
                    update_progress(idx, result)
                except Exception as e:
                    logger.error(f"Error calculating beam {futures[future]}: {str(e)}")
                    raise

        return results

    def get_calculation_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái tính toán hiện tại.

        Returns:
            dict: Trạng thái tính toán
        """
        return self.calculation_status.get_summary()

    def get_result(self) -> Optional[DoseGrid]:
        """
        Lấy kết quả tính toán liều.

        Returns:
            DoseGrid: Kết quả tính toán liều hoặc None nếu chưa có kết quả
        """
        return self.result_grid

    def export_dose(self, output_file: str, format: str = "dicom"):
        """
        Xuất phân bố liều ra file.

        Parameters:
            output_file (str): Đường dẫn đến file đầu ra
            format (str, optional): Định dạng đầu ra (dicom, nrrd, ...)

        Returns:
            bool: True nếu thành công

        Raises:
            ValueError: Nếu chưa có kết quả hoặc định dạng không được hỗ trợ
        """
        if not self.result_grid:
            raise ValueError("No calculation result available")

        # Kiểm tra định dạng
        format = format.lower()
        if format not in ["dicom", "nrrd", "mha", "mhd", "nii", "nii.gz"]:
            raise ValueError(f"Unsupported format: {format}")

        # Đảm bảo thư mục đích tồn tại
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Xuất theo định dạng
        if format == "dicom":
            # TODO: Triển khai xuất DICOM RT Dose
            return self.result_grid.to_dicom(output_file)
        else:
            # Xuất định dạng khác
            return self.result_grid.to_file(output_file, format)


# Alias for backward compatibility
DoseCalculation = DoseCalculator
