"""
Factory cho đối tượng DICOM.

Module này cung cấp các phương thức tạo và khởi tạo các đối tượng DICOM
một cách thuận tiện, bao gồm các dataset DICOM mới cho RT Structure,
RT Dose, RT Plan và các đối tượng khác.
"""

import os
import logging
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

from quangtps.core.exceptions import DicomError

logger = logging.getLogger(__name__)

class DicomFactory:
    """Lớp factory cho đối tượng DICOM"""
    
    @staticmethod
    def create_basic_dataset(file_meta=None, **kwargs):
        """
        Tạo dataset DICOM cơ bản.
        
        Parameters
        ----------
        file_meta : pydicom.dataset.Dataset, optional
            File meta dataset
        **kwargs : dict
            Các thuộc tính cho dataset
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset DICOM cơ bản
        """
        # Tạo file meta nếu chưa có
        if file_meta is None:
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'  # CR Image Storage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
            file_meta.ImplementationVersionName = 'QuangTPS'
        
        # Tạo dataset
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Thiết lập các thuộc tính cơ bản
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Thiết lập thuộc tính từ kwargs
        for key, value in kwargs.items():
            setattr(ds, key, value)
        
        return ds
    
    @staticmethod
    def create_rt_structure_dataset(referenced_study_uid: str, referenced_series_uid: str, 
                                    patient_id: str = None, patient_name: str = None, 
                                    structure_label: str = None):
        """
        Tạo dataset RT Structure mới.
        
        Parameters
        ----------
        referenced_study_uid : str
            Study Instance UID tham chiếu
        referenced_series_uid : str
            Series Instance UID tham chiếu
        patient_id : str, optional
            ID bệnh nhân
        patient_name : str, optional
            Tên bệnh nhân
        structure_label : str, optional
            Nhãn RT Structure
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset RT Structure
        """
        # Tạo file meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Thiết lập thông tin cơ bản
        if structure_label is None:
            structure_label = 'RT Structure Set'
        
        if patient_id is None:
            patient_id = 'ANONYMOUS'
        
        if patient_name is None:
            patient_name = 'ANONYMOUS'
        
        # Tạo dataset
        ds = DicomFactory.create_basic_dataset(file_meta,
                                              Modality='RTSTRUCT',
                                              PatientID=patient_id,
                                              PatientName=patient_name,
                                              StructureSetLabel=structure_label,
                                              StructureSetName=structure_label,
                                              StudyInstanceUID=referenced_study_uid,
                                              SeriesInstanceUID=pydicom.uid.generate_uid())
        
        # Thêm tham chiếu đến series ảnh
        referenced_frame_of_reference = Dataset()
        referenced_frame_of_reference.FrameOfReferenceUID = pydicom.uid.generate_uid()
        
        referenced_study = Dataset()
        referenced_study.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        referenced_study.ReferencedSOPInstanceUID = referenced_study_uid
        
        rt_referenced_series = Dataset()
        rt_referenced_series.SeriesInstanceUID = referenced_series_uid
        rt_referenced_series.ContourImageSequence = Sequence([referenced_study])
        
        rt_referenced_study = Dataset()
        rt_referenced_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'  # Study Component Management SOPClass
        rt_referenced_study.ReferencedSOPInstanceUID = referenced_study_uid
        rt_referenced_study.RTReferencedSeriesSequence = Sequence([rt_referenced_series])
        
        referenced_frame_of_reference.RTReferencedStudySequence = Sequence([rt_referenced_study])
        
        ds.ReferencedFrameOfReferenceSequence = Sequence([referenced_frame_of_reference])
        
        # Khởi tạo sequence cho cấu trúc
        ds.StructureSetROISequence = Sequence([])
        ds.ROIContourSequence = Sequence([])
        ds.RTROIObservationsSequence = Sequence([])
        
        return ds
    
    @staticmethod
    def create_rt_dose_dataset(referenced_study_uid: str, referenced_series_uid: str,
                              ref_rtplan_uid: str = None, patient_id: str = None, 
                              patient_name: str = None, dose_type: str = 'PHYSICAL'):
        """
        Tạo dataset RT Dose mới.
        
        Parameters
        ----------
        referenced_study_uid : str
            Study Instance UID tham chiếu
        referenced_series_uid : str
            Series Instance UID tham chiếu
        ref_rtplan_uid : str, optional
            RT Plan Instance UID tham chiếu
        patient_id : str, optional
            ID bệnh nhân
        patient_name : str, optional
            Tên bệnh nhân
        dose_type : str, optional
            Loại liều, mặc định là 'PHYSICAL'
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset RT Dose
        """
        # Tạo file meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'  # RT Dose Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Thiết lập thông tin cơ bản
        if patient_id is None:
            patient_id = 'ANONYMOUS'
        
        if patient_name is None:
            patient_name = 'ANONYMOUS'
        
        # Tạo dataset
        ds = DicomFactory.create_basic_dataset(file_meta,
                                              Modality='RTDOSE',
                                              PatientID=patient_id,
                                              PatientName=patient_name,
                                              StudyInstanceUID=referenced_study_uid,
                                              SeriesInstanceUID=pydicom.uid.generate_uid())
        
        # Thiết lập các thuộc tính dose
        ds.DoseUnits = 'GY'
        ds.DoseType = dose_type
        ds.DoseSummationType = 'PLAN'
        ds.DoseGridScaling = 1.0
        
        # Thêm tham chiếu đến RT Plan nếu có
        if ref_rtplan_uid is not None:
            ref_rtplan = Dataset()
            ref_rtplan.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'  # RT Plan Storage
            ref_rtplan.ReferencedSOPInstanceUID = ref_rtplan_uid
            ds.ReferencedRTPlanSequence = Sequence([ref_rtplan])
        
        return ds
    
    @staticmethod
    def create_rt_plan_dataset(referenced_study_uid: str, patient_id: str = None, 
                             patient_name: str = None, plan_label: str = None):
        """
        Tạo dataset RT Plan mới.
        
        Parameters
        ----------
        referenced_study_uid : str
            Study Instance UID tham chiếu
        patient_id : str, optional
            ID bệnh nhân
        patient_name : str, optional
            Tên bệnh nhân
        plan_label : str, optional
            Nhãn RT Plan
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset RT Plan
        """
        # Tạo file meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'  # RT Plan Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Thiết lập thông tin cơ bản
        if plan_label is None:
            plan_label = 'RT Plan'
        
        if patient_id is None:
            patient_id = 'ANONYMOUS'
        
        if patient_name is None:
            patient_name = 'ANONYMOUS'
        
        # Tạo dataset
        ds = DicomFactory.create_basic_dataset(file_meta,
                                              Modality='RTPLAN',
                                              PatientID=patient_id,
                                              PatientName=patient_name,
                                              RTPlanLabel=plan_label,
                                              RTPlanName=plan_label,
                                              StudyInstanceUID=referenced_study_uid,
                                              SeriesInstanceUID=pydicom.uid.generate_uid())
        
        # Khởi tạo sequence cho plan
        ds.BeamSequence = Sequence([])
        ds.DoseReferenceSequence = Sequence([])
        ds.FractionGroupSequence = Sequence([])
        
        # Thiết lập giá trị cơ bản
        ds.RTPlanDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.RTPlanTime = datetime.datetime.now().strftime('%H%M%S')
        ds.PlanIntent = 'PLAN'
        
        return ds
    
    @staticmethod
    def add_structure_to_rt_structure(rt_struct_dataset, roi_number: int, roi_name: str,
                                     contour_data: List[np.ndarray], color: List[int] = None):
        """
        Thêm cấu trúc mới vào dataset RT Structure.
        
        Parameters
        ----------
        rt_struct_dataset : pydicom.dataset.FileDataset
            Dataset RT Structure
        roi_number : int
            Số ROI
        roi_name : str
            Tên ROI
        contour_data : List[np.ndarray]
            Dữ liệu contour, mỗi phần tử là một mảng numpy (Nx3)
        color : List[int], optional
            Màu RGB của ROI, mặc định là [255, 0, 0]
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset RT Structure đã cập nhật
        """
        if color is None:
            color = [255, 0, 0]
        
        # Tạo ROI trong StructureSetROISequence
        roi = Dataset()
        roi.ROINumber = roi_number
        roi.ROIName = roi_name
        roi.ROIGenerationAlgorithm = 'MANUAL'
        
        # Thêm vào StructureSetROISequence
        rt_struct_dataset.StructureSetROISequence.append(roi)
        
        # Tạo ROI contour
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = color
        roi_contour.ReferencedROINumber = roi_number
        roi_contour.ContourSequence = Sequence([])
        
        # Thêm các contour
        for i, points in enumerate(contour_data):
            contour = Dataset()
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = points.shape[0]
            contour.ContourData = points.flatten().tolist()
            roi_contour.ContourSequence.append(contour)
        
        # Thêm vào ROIContourSequence
        rt_struct_dataset.ROIContourSequence.append(roi_contour)
        
        # Tạo ROI observation
        roi_observation = Dataset()
        roi_observation.ObservationNumber = roi_number
        roi_observation.ReferencedROINumber = roi_number
        roi_observation.ROIObservationLabel = roi_name
        roi_observation.RTROIInterpretedType = 'ORGAN'
        
        # Thêm vào RTROIObservationsSequence
        rt_struct_dataset.RTROIObservationsSequence.append(roi_observation)
        
        return rt_struct_dataset
    
    @staticmethod
    def add_beam_to_rt_plan(rt_plan_dataset, beam_number: int, beam_name: str, beam_type: str,
                           gantry_angle: float, collimator_angle: float, couch_angle: float,
                           isocenter: List[float], jaw_positions: Dict[str, List[float]] = None):
        """
        Thêm beam mới vào dataset RT Plan.
        
        Parameters
        ----------
        rt_plan_dataset : pydicom.dataset.FileDataset
            Dataset RT Plan
        beam_number : int
            Số beam
        beam_name : str
            Tên beam
        beam_type : str
            Loại beam ('STATIC', 'DYNAMIC', etc.)
        gantry_angle : float
            Góc gantry (độ)
        collimator_angle : float
            Góc collimator (độ)
        couch_angle : float
            Góc couch (độ)
        isocenter : List[float]
            Tọa độ isocenter [x, y, z] (mm)
        jaw_positions : Dict[str, List[float]], optional
            Vị trí jaw, ví dụ {'x': [-100, 100], 'y': [-100, 100]}
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset RT Plan đã cập nhật
        """
        # Tạo beam mới
        beam = Dataset()
        beam.BeamNumber = beam_number
        beam.BeamName = beam_name
        beam.BeamType = beam_type
        beam.RadiationType = 'PHOTON'
        beam.PrimaryDosimeterUnit = 'MU'
        beam.NumberOfControlPoints = 1
        beam.ControlPointSequence = Sequence([])
        
        # Tạo control point
        cp = Dataset()
        cp.ControlPointIndex = 0
        cp.CumulativeMetersetWeight = 1.0
        cp.NominalBeamEnergy = 6.0  # 6 MV
        cp.DoseRateSet = 300.0  # 300 MU/min
        cp.GantryAngle = gantry_angle
        cp.BeamLimitingDeviceAngle = collimator_angle
        cp.PatientSupportAngle = couch_angle
        cp.IsocenterPosition = isocenter
        cp.BeamLimitingDevicePositionSequence = Sequence([])
        
        # Thêm vị trí jaw nếu có
        if jaw_positions is not None:
            if 'x' in jaw_positions:
                x_positions = Dataset()
                x_positions.RTBeamLimitingDeviceType = 'ASYMX'
                x_positions.LeafJawPositions = jaw_positions['x']
                cp.BeamLimitingDevicePositionSequence.append(x_positions)
            
            if 'y' in jaw_positions:
                y_positions = Dataset()
                y_positions.RTBeamLimitingDeviceType = 'ASYMY'
                y_positions.LeafJawPositions = jaw_positions['y']
                cp.BeamLimitingDevicePositionSequence.append(y_positions)
        
        # Thêm control point vào beam
        beam.ControlPointSequence.append(cp)
        
        # Thêm beam vào BeamSequence
        rt_plan_dataset.BeamSequence.append(beam)
        
        # Kiểm tra và tạo FractionGroupSequence nếu cần
        if not hasattr(rt_plan_dataset, 'FractionGroupSequence') or not rt_plan_dataset.FractionGroupSequence:
            fg = Dataset()
            fg.FractionGroupNumber = 1
            fg.NumberOfFractions = 1
            fg.ReferencedBeamSequence = Sequence([])
            rt_plan_dataset.FractionGroupSequence = Sequence([fg])
        
        # Thêm tham chiếu beam vào FractionGroupSequence
        ref_beam = Dataset()
        ref_beam.ReferencedBeamNumber = beam_number
        ref_beam.BeamMeterset = 100.0  # 100 MU
        rt_plan_dataset.FractionGroupSequence[0].ReferencedBeamSequence.append(ref_beam)
        
        return rt_plan_dataset
    
    @staticmethod
    def create_ct_dataset(pixel_data: np.ndarray, pixel_spacing: List[float], 
                        image_position: List[float], image_orientation: List[float],
                        slice_thickness: float, patient_id: str = None, patient_name: str = None,
                        study_uid: str = None, series_uid: str = None, instance_uid: str = None):
        """
        Tạo dataset CT mới.
        
        Parameters
        ----------
        pixel_data : np.ndarray
            Dữ liệu hình ảnh 2D
        pixel_spacing : List[float]
            Khoảng cách giữa các pixel [row_spacing, col_spacing] (mm)
        image_position : List[float]
            Vị trí của góc trên bên trái của hình ảnh (mm)
        image_orientation : List[float]
            Hướng của hình ảnh (6 giá trị)
        slice_thickness : float
            Độ dày lát cắt (mm)
        patient_id : str, optional
            ID bệnh nhân
        patient_name : str, optional
            Tên bệnh nhân
        study_uid : str, optional
            Study Instance UID
        series_uid : str, optional
            Series Instance UID
        instance_uid : str, optional
            SOP Instance UID
            
        Returns
        -------
        pydicom.dataset.FileDataset
            Dataset CT
        """
        # Tạo UID mới nếu chưa có
        if study_uid is None:
            study_uid = pydicom.uid.generate_uid()
        
        if series_uid is None:
            series_uid = pydicom.uid.generate_uid()
        
        if instance_uid is None:
            instance_uid = pydicom.uid.generate_uid()
        
        # Thiết lập thông tin cơ bản
        if patient_id is None:
            patient_id = 'ANONYMOUS'
        
        if patient_name is None:
            patient_name = 'ANONYMOUS'
        
        # Tạo file meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = instance_uid
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Tạo dataset
        ds = DicomFactory.create_basic_dataset(file_meta,
                                              Modality='CT',
                                              PatientID=patient_id,
                                              PatientName=patient_name,
                                              StudyInstanceUID=study_uid,
                                              SeriesInstanceUID=series_uid,
                                              SOPInstanceUID=instance_uid)
        
        # Thiết lập thông tin hình ảnh
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        
        # Thiết lập kích thước
        ds.Rows = pixel_data.shape[0]
        ds.Columns = pixel_data.shape[1]
        
        # Thiết lập các thuộc tính không gian
        ds.PixelSpacing = pixel_spacing
        ds.ImagePositionPatient = image_position
        ds.ImageOrientationPatient = image_orientation
        ds.SliceThickness = slice_thickness
        ds.SliceLocation = image_position[2]
        
        # Thang điểm Hounsfield
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        
        # Gán dữ liệu pixel
        if pixel_data.dtype != np.uint16:
            pixel_data = pixel_data.astype(np.uint16)
        ds.PixelData = pixel_data.tobytes()
        
        return ds
