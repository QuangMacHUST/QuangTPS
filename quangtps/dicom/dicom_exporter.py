"""
Xuất dữ liệu sang định dạng DICOM.

Module này cung cấp các công cụ để xuất dữ liệu từ hệ thống xạ trị sang định dạng DICOM,
bao gồm xuất CT, cấu trúc, kế hoạch xạ trị và phân bố liều.
"""

import os
import logging
import pydicom
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Tuple

from quangtps.core.exceptions import DicomError
from quangtps.dicom.dicom_factory import DicomFactory
from quangtps.dicom.dicom_utils import generate_uid

logger = logging.getLogger(__name__)


class DicomExporter:
    """
    Lớp xuất dữ liệu sang định dạng DICOM.

    Class này cung cấp các phương thức để xuất dữ liệu từ hệ thống xạ trị
    sang định dạng DICOM, chẳng hạn như CT, cấu trúc, kế hoạch và phân bố liều.
    """

    def __init__(self):
        """Khởi tạo đối tượng DicomExporter."""
        self.factory = DicomFactory()
        self.study_uid = None
        self.frame_of_reference_uid = None

    def set_uids(
        self, study_uid: str = None, frame_of_reference_uid: str = None
    ) -> None:
        """
        Thiết lập các UID sử dụng cho xuất dữ liệu.

        Parameters
        ----------
        study_uid : str, optional
            Study Instance UID, nếu None sẽ tạo mới
        frame_of_reference_uid : str, optional
            Frame of Reference UID, nếu None sẽ tạo mới
        """
        if study_uid is None:
            study_uid = generate_uid()

        if frame_of_reference_uid is None:
            frame_of_reference_uid = generate_uid()

        self.study_uid = study_uid
        self.frame_of_reference_uid = frame_of_reference_uid

    def export_ct_volume(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        output_directory: str,
        patient_info: Dict[str, Any] = None,
    ) -> List[str]:
        """
        Xuất dữ liệu khối sang series CT DICOM.

        Parameters
        ----------
        volume : np.ndarray
            Mảng 3D chứa dữ liệu CT
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các voxel (mm)
        origin : Tuple[float, float, float]
            Vị trí gốc của khối (mm)
        output_directory : str
            Thư mục xuất file
        patient_info : Dict[str, Any], optional
            Thông tin bệnh nhân

        Returns
        -------
        List[str]
            Danh sách đường dẫn đến các file DICOM đã tạo

        Raises
        ------
        DicomError
            Nếu không thể xuất dữ liệu
        """
        try:
            # Tạo thư mục đầu ra nếu chưa tồn tại
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Kiểm tra và tạo UID nếu cần
            if self.study_uid is None or self.frame_of_reference_uid is None:
                self.set_uids()

            # Tạo Series Instance UID
            series_uid = generate_uid()

            # Tạo thông tin bệnh nhân mặc định nếu không được cung cấp
            if patient_info is None:
                patient_info = {
                    "PatientName": "ANONYMOUS^PATIENT",
                    "PatientID": f"ANON{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "PatientBirthDate": "",
                    "PatientSex": "",
                }

            # Thông tin chung cho toàn bộ series
            series_description = "Exported CT Series"
            series_number = 1

            # Tạo ngày và giờ hiện tại
            current_date = datetime.now().strftime("%Y%m%d")
            current_time = datetime.now().strftime("%H%M%S")

            # Tạo hướng chuẩn
            direction = (1, 0, 0, 0, 1, 0)  # LPS orientation

            # Danh sách các file đã tạo
            output_files = []

            # Tạo từng file CT slice
            for z in range(volume.shape[0]):
                # Tạo tên file
                filename = f"CT.{z:04d}.dcm"
                filepath = os.path.join(output_directory, filename)

                # Thiết lập thông tin slice
                position = [origin[0], origin[1], origin[2] + z * spacing[2]]
                slice_data = volume[z].astype(np.int16)

                # Tạo dataset mới với các tham số cần thiết
                ds = self.factory.create_ct_dataset(
                    pixel_data=slice_data,
                    pixel_spacing=[spacing[0], spacing[1]],
                    image_position=position,
                    image_orientation=list(direction),
                    slice_thickness=spacing[2],
                    patient_id=patient_info["PatientID"],
                    patient_name=patient_info["PatientName"],
                    study_uid=self.study_uid,
                    series_uid=series_uid,
                )

                # Thiết lập thông tin bệnh nhân bổ sung
                ds.PatientBirthDate = patient_info["PatientBirthDate"]
                ds.PatientSex = patient_info["PatientSex"]

                # Thiết lập thông tin nghiên cứu
                ds.StudyDate = current_date
                ds.StudyTime = current_time
                ds.StudyDescription = "Exported CT Study"

                # Thiết lập thông tin series
                ds.SeriesDate = current_date
                ds.SeriesTime = current_time
                ds.SeriesDescription = series_description
                ds.SeriesNumber = series_number

                # Thiết lập vị trí slice
                ds.SliceLocation = position[2]

                # Thiết lập thông tin hình ảnh
                ds.Rows = volume.shape[1]
                ds.Columns = volume.shape[2]

                # Thiết lập tham chiếu khung
                ds.FrameOfReferenceUID = self.frame_of_reference_uid

                # Thiết lập thông tin HU
                ds.RescaleIntercept = -1024.0
                ds.RescaleSlope = 1.0
                ds.RescaleType = "HU"

                # Thiết lập SOP Instance UID
                ds.SOPInstanceUID = generate_uid()

                # Lưu file
                ds.save_as(filepath)
                output_files.append(filepath)

            logger.info(f"Exported {len(output_files)} CT slices to {output_directory}")

            return output_files

        except Exception as e:
            logger.error(f"Error exporting CT volume: {str(e)}")
            raise DicomError(f"Error exporting CT volume: {str(e)}")

    def export_rt_structure(
        self,
        structures: Dict[str, np.ndarray],
        structure_names: Dict[str, str],
        reference_ct_series_uid: str,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        output_path: str,
        patient_info: Dict[str, Any] = None,
    ) -> str:
        """
        Xuất cấu trúc sang DICOM RT Structure.

        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Từ điển các cấu trúc, khóa là ID cấu trúc, giá trị là mặt nạ nhị phân
        structure_names : Dict[str, str]
            Từ điển tên cấu trúc, khóa là ID cấu trúc, giá trị là tên
        reference_ct_series_uid : str
            Series Instance UID của CT tham chiếu
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các voxel (mm)
        origin : Tuple[float, float, float]
            Vị trí gốc của khối (mm)
        output_path : str
            Đường dẫn đến file đầu ra
        patient_info : Dict[str, Any], optional
            Thông tin bệnh nhân

        Returns
        -------
        str
            Đường dẫn đến file DICOM đã tạo

        Raises
        ------
        DicomError
            Nếu không thể xuất cấu trúc
        """
        try:
            # Tạo thư mục cha nếu chưa tồn tại
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Kiểm tra và tạo UID nếu cần
            if self.study_uid is None or self.frame_of_reference_uid is None:
                self.set_uids()

            # Tạo thông tin bệnh nhân mặc định nếu không được cung cấp
            if patient_info is None:
                patient_info = {
                    "PatientName": "ANONYMOUS^PATIENT",
                    "PatientID": f"ANON{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "PatientBirthDate": "",
                    "PatientSex": "",
                }

            # Tạo dataset mới
            ds = self.factory.create_rt_structure_dataset()

            # Thiết lập thông tin bệnh nhân
            ds.PatientName = patient_info["PatientName"]
            ds.PatientID = patient_info["PatientID"]
            ds.PatientBirthDate = patient_info["PatientBirthDate"]
            ds.PatientSex = patient_info["PatientSex"]

            # Tạo ngày và giờ hiện tại
            current_date = datetime.now().strftime("%Y%m%d")
            current_time = datetime.now().strftime("%H%M%S")

            # Thiết lập thông tin nghiên cứu
            ds.StudyDate = current_date
            ds.StudyTime = current_time
            ds.StudyDescription = "Exported RT Structure Study"
            ds.StudyInstanceUID = self.study_uid

            # Thiết lập thông tin series
            ds.SeriesDate = current_date
            ds.SeriesTime = current_time
            ds.SeriesDescription = "RT Structure Set"
            ds.SeriesNumber = 1
            ds.SeriesInstanceUID = generate_uid()

            # Thiết lập tham chiếu khung
            ds.FrameOfReferenceUID = self.frame_of_reference_uid

            # Thiết lập thông tin cấu trúc cơ bản
            ds.StructureSetLabel = "RTSTRUCT"
            ds.StructureSetName = "RT Structure Set"
            ds.StructureSetDate = current_date
            ds.StructureSetTime = current_time

            # Tham chiếu đến series CT
            ref_series_sequence = []
            ref_series = pydicom.dataset.Dataset()
            ref_series.SeriesInstanceUID = reference_ct_series_uid
            ref_series_sequence.append(ref_series)

            ds.ReferencedFrameOfReferenceSequence = []
            frame_of_ref = pydicom.dataset.Dataset()
            frame_of_ref.FrameOfReferenceUID = self.frame_of_reference_uid
            frame_of_ref.RTReferencedStudySequence = []

            rt_ref_study = pydicom.dataset.Dataset()
            rt_ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
            rt_ref_study.ReferencedSOPInstanceUID = self.study_uid
            rt_ref_study.RTReferencedSeriesSequence = []

            rt_ref_series = pydicom.dataset.Dataset()
            rt_ref_series.SeriesInstanceUID = reference_ct_series_uid
            rt_ref_study.RTReferencedSeriesSequence.append(rt_ref_series)

            frame_of_ref.RTReferencedStudySequence.append(rt_ref_study)
            ds.ReferencedFrameOfReferenceSequence.append(frame_of_ref)

            # Tạo các ROI
            ds.StructureSetROISequence = []
            ds.ROIContourSequence = []
            ds.RTROIObservationsSequence = []

            # Xử lý từng cấu trúc
            for i, (roi_id, roi_mask) in enumerate(structures.items(), 1):
                # Lấy tên cấu trúc
                roi_name = structure_names.get(roi_id, f"Structure{i}")

                # Tạo StructureSetROI
                structure_set_roi = pydicom.dataset.Dataset()
                structure_set_roi.ROINumber = i
                structure_set_roi.ROIName = roi_name
                structure_set_roi.ROIGenerationAlgorithm = "MANUAL"
                structure_set_roi.ReferencedFrameOfReferenceUID = (
                    self.frame_of_reference_uid
                )
                ds.StructureSetROISequence.append(structure_set_roi)

                # Tạo ROIContour
                roi_contour = pydicom.dataset.Dataset()
                roi_contour.ROIDisplayColor = [255, 0, 0]  # Default to red
                roi_contour.ReferencedROINumber = i
                roi_contour.ContourSequence = []

                # Chuyển đổi mặt nạ nhị phân thành đường viền
                contours = self._mask_to_contours(roi_mask, spacing, origin)

                # Thêm các đường viền vào ContourSequence
                for j, contour_data in enumerate(contours):
                    contour = pydicom.dataset.Dataset()
                    contour.ContourGeometricType = "CLOSED_PLANAR"
                    contour.NumberOfContourPoints = len(contour_data) // 3
                    contour.ContourData = contour_data
                    roi_contour.ContourSequence.append(contour)

                ds.ROIContourSequence.append(roi_contour)

                # Tạo RTROIObservation
                roi_observation = pydicom.dataset.Dataset()
                roi_observation.ObservationNumber = i
                roi_observation.ReferencedROINumber = i
                roi_observation.ROIObservationLabel = roi_name
                roi_observation.RTROIInterpretedType = "ORGAN"
                ds.RTROIObservationsSequence.append(roi_observation)

            # Thiết lập SOP Instance UID
            ds.SOPInstanceUID = generate_uid()

            # Lưu file
            ds.save_as(output_path)

            logger.info(
                f"Exported RT Structure Set with {len(structures)} structures to {output_path}"
            )

            return output_path

        except Exception as e:
            logger.error(f"Error exporting RT Structure: {str(e)}")
            raise DicomError(f"Error exporting RT Structure: {str(e)}")

    def _mask_to_contours(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ) -> List[List[float]]:
        """
        Chuyển đổi mặt nạ nhị phân thành danh sách contour.

        Parameters
        ----------
        mask : np.ndarray
            Mặt nạ nhị phân 3D của cấu trúc
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các voxel (mm)
        origin : Tuple[float, float, float]
            Vị trí gốc của khối (mm)

        Returns
        -------
        List[List[float]]
            Danh sách các contour, mỗi contour là một danh sách điểm 3D
        """
        try:
            # Kiểm tra mask
            if mask is None or mask.size == 0:
                logger.error("Mặt nạ cấu trúc rỗng hoặc None")
                return []

            if not isinstance(mask, np.ndarray):
                logger.error(f"Mặt nạ phải là mảng numpy, nhận được {type(mask)}")
                return []

            if len(mask.shape) != 3:
                logger.error(f"Mặt nạ phải là mảng 3D, nhận được {len(mask.shape)}D")
                return []

            # Import các thư viện cần thiết
            import cv2

            # Danh sách kết quả
            all_contours = []

            # Xử lý từng lớp
            for z in range(mask.shape[0]):
                # Lấy lớp hiện tại và chuyển đổi sang uint8
                layer = mask[z, :, :].astype(np.uint8) * 255

                # Bỏ qua lớp trống
                if np.sum(layer) == 0:
                    continue

                # Tìm contour trong lớp, bao gồm cả các lỗ (holes)
                contours, hierarchy = cv2.findContours(
                    layer,
                    cv2.RETR_CCOMP,  # Tìm cả contour ngoài và lỗ
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                # Nếu không có contour, bỏ qua lớp
                if len(contours) == 0:
                    continue

                # Xử lý từng contour
                for i, contour in enumerate(contours):
                    # Bỏ qua contour quá nhỏ (ít hơn 3 điểm không thể tạo thành đa giác)
                    if len(contour) < 3:
                        continue

                    # Đơn giản hóa contour để giảm số điểm nhưng vẫn giữ hình dạng
                    # epsilon = 0.01 * cv2.arcLength(contour, True)
                    # contour = cv2.approxPolyDP(contour, epsilon, True)

                    # Nếu contour vẫn quá nhỏ sau khi đơn giản hóa, bỏ qua
                    if len(contour) < 3:
                        continue

                    # Chuyển đổi contour sang định dạng DICOM
                    dicom_contour = []
                    z_coord = origin[2] + z * spacing[2]

                    for point in contour:
                        # Chuyển đổi tọa độ từ pixel sang tọa độ thế giới (mm)
                        # Lưu ý: OpenCV trả về tọa độ theo định dạng [y, x]
                        x = origin[0] + point[0][1] * spacing[1]
                        y = origin[1] + point[0][0] * spacing[0]

                        dicom_contour.extend([x, y, z_coord])

                    # Đảm bảo contour kết thúc tại điểm bắt đầu (khép kín)
                    if len(dicom_contour) >= 6:  # Ít nhất 2 điểm
                        if (
                            dicom_contour[0] != dicom_contour[-3]
                            or dicom_contour[1] != dicom_contour[-2]
                        ):
                            # Thêm điểm đầu tiên vào cuối để khép kín contour
                            dicom_contour.extend(
                                [dicom_contour[0], dicom_contour[1], dicom_contour[2]]
                            )

                    # Thêm contour vào danh sách kết quả
                    all_contours.append(dicom_contour)

            # Kiểm tra số lượng contour tìm được
            if len(all_contours) == 0:
                logger.warning("Không tìm thấy contour nào từ mặt nạ")
            else:
                logger.debug(f"Đã tìm thấy {len(all_contours)} contour từ mặt nạ")

            return all_contours

        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi mask thành contour: {str(e)}")
            import traceback

            logger.debug(f"Chi tiết lỗi: {traceback.format_exc()}")
            return []

    def export_rt_dose(
        self,
        dose_volume: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        dose_grid_scaling: float,
        output_path: str,
        reference_rt_plan_uid: str = None,
        dose_type: str = "PHYSICAL",
        dose_comment: str = "",
        patient_info: Dict[str, Any] = None,
    ) -> str:
        """
        Xuất phân bố liều sang DICOM RT Dose.

        Parameters
        ----------
        dose_volume : np.ndarray
            Mảng 3D chứa dữ liệu liều (Gy)
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các voxel (mm)
        origin : Tuple[float, float, float]
            Vị trí gốc của khối (mm)
        dose_grid_scaling : float
            Hệ số scaling của lưới liều
        output_path : str
            Đường dẫn đến file đầu ra
        reference_rt_plan_uid : str, optional
            SOP Instance UID của RT Plan tham chiếu
        dose_type : str, optional
            Loại liều, mặc định là 'PHYSICAL'
        dose_comment : str, optional
            Bình luận về liều
        patient_info : Dict[str, Any], optional
            Thông tin bệnh nhân

        Returns
        -------
        str
            Đường dẫn đến file DICOM đã tạo

        Raises
        ------
        DicomError
            Nếu không thể xuất liều
        """
        try:
            # Tạo thư mục cha nếu chưa tồn tại
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Kiểm tra và tạo UID nếu cần
            if self.study_uid is None or self.frame_of_reference_uid is None:
                self.set_uids()

            # Tạo thông tin bệnh nhân mặc định nếu không được cung cấp
            if patient_info is None:
                patient_info = {
                    "PatientName": "ANONYMOUS^PATIENT",
                    "PatientID": f"ANON{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "PatientBirthDate": "",
                    "PatientSex": "",
                }

            # Tạo dataset mới
            ds = self.factory.create_rt_dose_dataset()

            # Thiết lập thông tin bệnh nhân
            ds.PatientName = patient_info["PatientName"]
            ds.PatientID = patient_info["PatientID"]
            ds.PatientBirthDate = patient_info["PatientBirthDate"]
            ds.PatientSex = patient_info["PatientSex"]

            # Tạo ngày và giờ hiện tại
            current_date = datetime.now().strftime("%Y%m%d")
            current_time = datetime.now().strftime("%H%M%S")

            # Thiết lập thông tin nghiên cứu
            ds.StudyDate = current_date
            ds.StudyTime = current_time
            ds.StudyDescription = "Exported RT Dose Study"
            ds.StudyInstanceUID = self.study_uid

            # Thiết lập thông tin series
            ds.SeriesDate = current_date
            ds.SeriesTime = current_time
            ds.SeriesDescription = "RT Dose"
            ds.SeriesNumber = 1
            ds.SeriesInstanceUID = generate_uid()

            # Thiết lập tham chiếu khung
            ds.FrameOfReferenceUID = self.frame_of_reference_uid

            # Thiết lập thông tin liều
            ds.DoseUnits = "GY"
            ds.DoseType = dose_type
            ds.DoseComment = dose_comment
            ds.DoseSummationType = "PLAN"
            ds.DoseGridScaling = dose_grid_scaling

            # Tham chiếu đến RT Plan nếu có
            if reference_rt_plan_uid:
                ds.ReferencedRTPlanSequence = []
                ref_plan = pydicom.dataset.Dataset()
                ref_plan.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.5"
                ref_plan.ReferencedSOPInstanceUID = reference_rt_plan_uid
                ds.ReferencedRTPlanSequence.append(ref_plan)

            # Thiết lập thông tin hình ảnh
            ds.ImagePositionPatient = origin
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # LPS orientation
            ds.PixelSpacing = [spacing[0], spacing[1]]
            ds.SliceThickness = spacing[2]

            # Thiết lập thông tin grid
            ds.Rows = dose_volume.shape[1]
            ds.Columns = dose_volume.shape[2]
            ds.NumberOfFrames = dose_volume.shape[0]

            # Thiết lập thông tin vị trí grid
            grid_frame_offset_vector = []
            for z in range(dose_volume.shape[0]):
                grid_frame_offset_vector.append(z * spacing[2])
            ds.GridFrameOffsetVector = grid_frame_offset_vector

            # Chuyển đổi dữ liệu liều sang dạng phù hợp
            # Chia cho dose_grid_scaling để lưu dưới dạng số nguyên
            pixel_data = (dose_volume / dose_grid_scaling).astype(np.uint16)
            ds.PixelData = pixel_data.tobytes()

            # Thiết lập SOP Instance UID
            ds.SOPInstanceUID = generate_uid()

            # Lưu file
            ds.save_as(output_path)

            logger.info(f"Exported RT Dose to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error exporting RT Dose: {str(e)}")
            raise DicomError(f"Error exporting RT Dose: {str(e)}")

    def export_rt_plan(
        self,
        beam_data: List[Dict[str, Any]],
        output_path: str,
        reference_structure_uid: str = None,
        patient_info: Dict[str, Any] = None,
        plan_info: Dict[str, Any] = None,
    ) -> str:
        """
        Xuất kế hoạch xạ trị sang DICOM RT Plan.

        Parameters
        ----------
        beam_data : List[Dict[str, Any]]
            Danh sách thông tin các beam
        output_path : str
            Đường dẫn đến file đầu ra
        reference_structure_uid : str, optional
            SOP Instance UID của RT Structure tham chiếu
        patient_info : Dict[str, Any], optional
            Thông tin bệnh nhân
        plan_info : Dict[str, Any], optional
            Thông tin kế hoạch

        Returns
        -------
        str
            Đường dẫn đến file DICOM đã tạo

        Raises
        ------
        DicomError
            Nếu không thể xuất kế hoạch
        """
        # Phương thức này quá phức tạp để triển khai đầy đủ ở đây
        # Thực tế cần thêm nhiều thông tin chi tiết về beam, collimator, MLC, v.v.
        logger.warning("export_rt_plan is not fully implemented yet")
        return ""
