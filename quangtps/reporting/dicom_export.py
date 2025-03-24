#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module xuất dữ liệu kế hoạch điều trị ra định dạng DICOM.

Module này cung cấp các lớp và hàm để xuất kế hoạch điều trị, liều lượng,
và cấu trúc từ QuangTPS ra định dạng DICOM để sử dụng trong các hệ thống khác.
"""

import os
import logging
import datetime
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.dicom.dicom_writer import DicomWriter
from quangtps.core.config import Config
from quangtps.dicom.dicom_factory import DicomFactory

logger = logging.getLogger(__name__)


class DicomExporter:
    """
    Lớp xuất dữ liệu kế hoạch điều trị ra định dạng DICOM.
    
    Lớp này cung cấp các phương thức để xuất kế hoạch điều trị, liều lượng,
    và cấu trúc từ QuangTPS ra định dạng DICOM để sử dụng trong các hệ thống khác.
    """
    
    def __init__(self, output_dir: Optional[str] = None, config: Optional[Config] = None):
        """
        Khởi tạo lớp DicomExporter.
        
        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra để lưu file DICOM
        config : Config, optional
            Đối tượng cấu hình
        """
        self.output_dir = output_dir or os.path.expanduser("~/Documents/QuangTPS/DICOM")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.config = config or Config()
        self.dicom_writer = DicomWriter()
        self.dicom_factory = DicomFactory()
    
    def export_rt_plan(self, patient: Patient, plan: Plan, output_dir: Optional[str] = None) -> str:
        """
        Xuất kế hoạch xạ trị ra file DICOM RT Plan.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        output_dir : str, optional
            Thư mục đầu ra, mặc định là output_dir khởi tạo
            
        Returns
        -------
        str
            Đường dẫn đến file DICOM đã tạo
        """
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = self.output_dir
        
        # Tạo thư mục cho bệnh nhân nếu chưa tồn tại
        patient_dir = os.path.join(output_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Tạo tên file RT Plan
        rt_plan_file = os.path.join(patient_dir, f"{plan.name}_RTPlan.dcm")
        
        try:
            # Tạo dataset DICOM RT Plan
            ds = self._create_rt_plan_dataset(patient, plan)
            
            # Lưu file
            ds.save_as(rt_plan_file, write_like_original=False)
            
            logger.info(f"Đã xuất kế hoạch xạ trị ra file DICOM: {rt_plan_file}")
            return rt_plan_file
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất kế hoạch xạ trị ra DICOM: {str(e)}", exc_info=True)
            raise
    
    def export_rt_dose(self, patient: Patient, plan: Plan, output_dir: Optional[str] = None) -> str:
        """
        Xuất phân bố liều ra file DICOM RT Dose.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        output_dir : str, optional
            Thư mục đầu ra, mặc định là output_dir khởi tạo
            
        Returns
        -------
        str
            Đường dẫn đến file DICOM đã tạo
        """
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = self.output_dir
        
        # Tạo thư mục cho bệnh nhân nếu chưa tồn tại
        patient_dir = os.path.join(output_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Tạo tên file RT Dose
        rt_dose_file = os.path.join(patient_dir, f"{plan.name}_RTDose.dcm")
        
        try:
            # Kiểm tra xem kế hoạch có phân bố liều không
            if not hasattr(plan, 'dose') or plan.dose is None:
                logger.warning(f"Kế hoạch {plan.name} không có phân bố liều.")
                return ""
            
            # Tạo dataset DICOM RT Dose
            ds = self._create_rt_dose_dataset(patient, plan)
            
            # Lưu file
            ds.save_as(rt_dose_file, write_like_original=False)
            
            logger.info(f"Đã xuất phân bố liều ra file DICOM: {rt_dose_file}")
            return rt_dose_file
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất phân bố liều ra DICOM: {str(e)}", exc_info=True)
            raise
    
    def export_rt_structure_set(self, patient: Patient, plan: Plan, output_dir: Optional[str] = None) -> str:
        """
        Xuất tập cấu trúc ra file DICOM RT Structure Set.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        output_dir : str, optional
            Thư mục đầu ra, mặc định là output_dir khởi tạo
            
        Returns
        -------
        str
            Đường dẫn đến file DICOM đã tạo
        """
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = self.output_dir
        
        # Tạo thư mục cho bệnh nhân nếu chưa tồn tại
        patient_dir = os.path.join(output_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Tạo tên file RT Structure Set
        rt_struct_file = os.path.join(patient_dir, f"{plan.name}_RTStruct.dcm")
        
        try:
            # Kiểm tra xem kế hoạch có cấu trúc không
            if not hasattr(plan, 'structures') or not plan.structures:
                logger.warning(f"Kế hoạch {plan.name} không có cấu trúc.")
                return ""
            
            # Tạo dataset DICOM RT Structure Set
            ds = self._create_rt_structure_dataset(patient, plan)
            
            # Lưu file
            ds.save_as(rt_struct_file, write_like_original=False)
            
            logger.info(f"Đã xuất tập cấu trúc ra file DICOM: {rt_struct_file}")
            return rt_struct_file
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất tập cấu trúc ra DICOM: {str(e)}", exc_info=True)
            raise
    
    def export_full_plan(self, patient: Patient, plan: Plan, output_dir: Optional[str] = None) -> List[str]:
        """
        Xuất đầy đủ kế hoạch xạ trị ra các file DICOM.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
        output_dir : str, optional
            Thư mục đầu ra, mặc định là output_dir khởi tạo
            
        Returns
        -------
        List[str]
            Danh sách đường dẫn đến các file DICOM đã tạo
        """
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = self.output_dir
        
        # Tạo thư mục cho bệnh nhân nếu chưa tồn tại
        patient_dir = os.path.join(output_dir, patient.patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Tạo thư mục cho kế hoạch nếu chưa tồn tại
        plan_dir = os.path.join(patient_dir, plan.name)
        os.makedirs(plan_dir, exist_ok=True)
        
        # Danh sách các file đã xuất
        exported_files = []
        
        try:
            # Xuất RT Plan
            rt_plan_file = self.export_rt_plan(patient, plan, plan_dir)
            if rt_plan_file:
                exported_files.append(rt_plan_file)
            
            # Xuất RT Dose
            rt_dose_file = self.export_rt_dose(patient, plan, plan_dir)
            if rt_dose_file:
                exported_files.append(rt_dose_file)
            
            # Xuất RT Structure Set
            rt_struct_file = self.export_rt_structure_set(patient, plan, plan_dir)
            if rt_struct_file:
                exported_files.append(rt_struct_file)
            
            # Nếu cần, có thể xuất cả CT images liên quan
            # self._export_ct_images(patient, plan, plan_dir)
            
            logger.info(f"Đã xuất đầy đủ kế hoạch xạ trị ra thư mục: {plan_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Lỗi khi xuất đầy đủ kế hoạch xạ trị: {str(e)}", exc_info=True)
            raise
    
    def _create_rt_plan_dataset(self, patient: Patient, plan: Plan) -> Dataset:
        """
        Tạo dataset DICOM RT Plan.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
            
        Returns
        -------
        Dataset
            Dataset DICOM RT Plan
        """
        # Tạo file DICOM mới
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'  # RT Plan Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(tempfile.NamedTemporaryFile(suffix='.dcm').name, dataset={}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Thiết lập các thuộc tính bắt buộc
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'  # RT Plan Storage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = 'RTPLAN'
        
        # Thông tin bệnh nhân
        ds.PatientName = patient.name
        ds.PatientID = patient.patient_id
        ds.PatientBirthDate = patient.date_of_birth.replace('-', '') if hasattr(patient, 'date_of_birth') else ''
        ds.PatientSex = patient.gender[0] if hasattr(patient, 'gender') else ''
        
        # Thông tin nghiên cứu
        ds.StudyInstanceUID = generate_uid()
        ds.StudyID = "1"
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        
        # Thông tin series
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = "1"
        
        # Thông tin kế hoạch
        ds.RTPlanLabel = plan.name
        ds.RTPlanName = plan.name
        ds.RTPlanDate = plan.creation_date.replace('-', '') if hasattr(plan, 'creation_date') else datetime.datetime.now().strftime('%Y%m%d')
        ds.RTPlanTime = datetime.datetime.now().strftime('%H%M%S')
        ds.RTPlanGeometry = 'PATIENT'
        
        # Kê đơn
        if hasattr(plan, 'prescriptions') and plan.prescriptions:
            ds.DoseReferenceSequence = Sequence()
            
            for i, prescription in enumerate(plan.prescriptions):
                dose_ref = Dataset()
                dose_ref.DoseReferenceNumber = str(i + 1)
                dose_ref.DoseReferenceStructureType = 'SITE'
                dose_ref.DoseReferenceDescription = prescription.structure_name
                dose_ref.TargetPrescriptionDose = float(prescription.dose)
                
                ds.DoseReferenceSequence.append(dose_ref)
        
        # Thông tin chùm tia
        if hasattr(plan, 'beams') and plan.beams:
            ds.BeamSequence = Sequence()
            ds.FractionGroupSequence = Sequence()
            
            # Tạo fraction group
            fg = Dataset()
            fg.FractionGroupNumber = "1"
            fg.NumberOfFractionsPlanned = plan.number_of_fractions
            fg.ReferencedBeamSequence = Sequence()
            
            for i, beam in enumerate(plan.beams):
                # Tạo beam
                b = Dataset()
                b.BeamNumber = str(i + 1)
                b.BeamName = beam.name
                b.BeamType = beam.beam_type
                b.RadiationType = 'PHOTON' if 'PHOTON' in beam.energy else 'ELECTRON'
                b.TreatmentMachineName = 'TRUEBEAM'  # Hoặc từ config
                b.SourceAxisDistance = beam.sad
                
                # Thêm thông tin năng lượng
                b.BeamEnergyMin = float(beam.energy.replace('MV', '').replace('FFF', '').strip())
                b.BeamEnergyMax = b.BeamEnergyMin
                
                # Thông tin MLC nếu có
                if hasattr(beam, 'mlc') and beam.mlc:
                    # Thêm thông tin MLC vào beam
                    pass  # Chi tiết triển khai tùy thuộc vào cấu trúc dữ liệu MLC
                
                # Thêm vào BeamSequence
                ds.BeamSequence.append(b)
                
                # Tạo referenced beam trong fraction group
                ref_beam = Dataset()
                ref_beam.ReferencedBeamNumber = b.BeamNumber
                ref_beam.BeamDose = beam.monitor_units
                ref_beam.BeamDoseSpecificationPoint = [0.0, 0.0, 0.0]  # Điểm gốc hoặc isocenter
                
                fg.ReferencedBeamSequence.append(ref_beam)
            
            # Thêm fraction group vào sequence
            ds.FractionGroupSequence.append(fg)
        
        # Các thuộc tính DICOM chung
        ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
        ds.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8
        
        return ds
    
    def _create_rt_dose_dataset(self, patient: Patient, plan: Plan) -> Dataset:
        """
        Tạo dataset DICOM RT Dose.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
            
        Returns
        -------
        Dataset
            Dataset DICOM RT Dose
        """
        # Tạo file DICOM mới
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'  # RT Dose Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(tempfile.NamedTemporaryFile(suffix='.dcm').name, dataset={}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Thiết lập các thuộc tính bắt buộc
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'  # RT Dose Storage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = 'RTDOSE'
        
        # Thông tin bệnh nhân
        ds.PatientName = patient.name
        ds.PatientID = patient.patient_id
        ds.PatientBirthDate = patient.date_of_birth.replace('-', '') if hasattr(patient, 'date_of_birth') else ''
        ds.PatientSex = patient.gender[0] if hasattr(patient, 'gender') else ''
        
        # Thông tin nghiên cứu (giống với RT Plan)
        ds.StudyInstanceUID = generate_uid()  # Nên sử dụng cùng StudyInstanceUID với RT Plan
        ds.StudyID = "1"
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        
        # Thông tin series
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = "2"  # Khác với RT Plan
        
        # Thông tin dose grid
        if hasattr(plan, 'dose') and plan.dose is not None:
            dose_grid = plan.dose
            
            # Thiết lập thông tin hình ảnh
            ds.ImagePositionPatient = list(dose_grid.origin)
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # LPS orientation
            ds.PixelSpacing = [dose_grid.spacing[0], dose_grid.spacing[1]]
            ds.SliceThickness = dose_grid.spacing[2]
            
            # Thiết lập thông tin pixel
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0  # unsigned
            
            # Chuyển đổi dữ liệu liều thành pixel data
            # (Chi tiết triển khai tùy thuộc vào cấu trúc dữ liệu liều)
            import numpy as np
            
            # Giả sử dose_grid.data là numpy array 3D
            if hasattr(dose_grid, 'data') and isinstance(dose_grid.data, np.ndarray):
                # Chuyển đổi đơn vị liều (Gy) sang scale phù hợp cho DICOM
                dose_scaling_factor = 100.0  # Scale to cGy or any other factor
                
                # Nhân với scale factor và chuyển sang uint16
                scaled_data = (dose_grid.data * dose_scaling_factor).astype(np.uint16)
                
                # Thiết lập thông tin frame
                ds.NumberOfFrames = dose_grid.data.shape[2]
                ds.Rows = dose_grid.data.shape[0]
                ds.Columns = dose_grid.data.shape[1]
                
                # Thiết lập PixelData
                ds.PixelData = scaled_data.tobytes()
                
                # Thiết lập thông tin liều
                ds.DoseUnits = 'GY'
                ds.DoseType = 'PHYSICAL'
                ds.DoseGridScaling = float(1.0 / dose_scaling_factor)
                ds.DoseSummationType = 'PLAN'
        
        # Các thuộc tính DICOM chung
        ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
        ds.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8
        
        return ds
    
    def _create_rt_structure_dataset(self, patient: Patient, plan: Plan) -> Dataset:
        """
        Tạo dataset DICOM RT Structure Set.
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        plan : Plan
            Kế hoạch điều trị
            
        Returns
        -------
        Dataset
            Dataset DICOM RT Structure Set
        """
        # Tạo file DICOM mới
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(tempfile.NamedTemporaryFile(suffix='.dcm').name, dataset={}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Thiết lập các thuộc tính bắt buộc
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Storage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = 'RTSTRUCT'
        
        # Thông tin bệnh nhân
        ds.PatientName = patient.name
        ds.PatientID = patient.patient_id
        ds.PatientBirthDate = patient.date_of_birth.replace('-', '') if hasattr(patient, 'date_of_birth') else ''
        ds.PatientSex = patient.gender[0] if hasattr(patient, 'gender') else ''
        
        # Thông tin nghiên cứu (giống với RT Plan)
        ds.StudyInstanceUID = generate_uid()  # Nên sử dụng cùng StudyInstanceUID với RT Plan
        ds.StudyID = "1"
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        
        # Thông tin series
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = "3"  # Khác với RT Plan và RT Dose
        
        # Thông tin structure set
        ds.StructureSetLabel = f"{plan.name}_Structures"
        ds.StructureSetName = f"{plan.name}_Structures"
        ds.StructureSetDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StructureSetTime = datetime.datetime.now().strftime('%H%M%S')
        
        # Nếu có thông tin tham chiếu đến CT series
        if hasattr(plan, 'reference_series_uid'):
            ds.ReferencedFrameOfReferenceSequence = Sequence()
            ref_for = Dataset()
            ref_for.FrameOfReferenceUID = generate_uid()  # Hoặc sử dụng UID của CT
            
            ref_for.RTReferencedStudySequence = Sequence()
            ref_study = Dataset()
            ref_study.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
            ref_study.ReferencedSOPInstanceUID = plan.reference_series_uid
            
            ref_for.RTReferencedStudySequence.append(ref_study)
            ds.ReferencedFrameOfReferenceSequence.append(ref_for)
        
        # Thông tin cấu trúc
        if hasattr(plan, 'structures') and plan.structures:
            ds.StructureSetROISequence = Sequence()
            ds.ROIContourSequence = Sequence()
            ds.RTROIObservationsSequence = Sequence()
            
            for i, structure in enumerate(plan.structures):
                # Thêm vào StructureSetROISequence
                ssroi = Dataset()
                ssroi.ROINumber = i + 1
                ssroi.ROIName = structure.name
                ssroi.ROIGenerationAlgorithm = 'MANUAL'
                
                ds.StructureSetROISequence.append(ssroi)
                
                # Thêm vào ROIContourSequence
                roi_contour = Dataset()
                roi_contour.ROIDisplayColor = list(structure.color) if hasattr(structure, 'color') else [255, 0, 0]
                roi_contour.ReferencedROINumber = ssroi.ROINumber
                
                # Contour Sequence - chi tiết triển khai tùy thuộc vào cấu trúc dữ liệu contour
                roi_contour.ContourSequence = Sequence()
                
                # Giả sử structure.contours là danh sách các contour trên các lớp cắt
                if hasattr(structure, 'contours'):
                    for j, contour in enumerate(structure.contours):
                        contour_ds = Dataset()
                        contour_ds.ContourGeometricType = 'CLOSED_PLANAR'
                        contour_ds.NumberOfContourPoints = len(contour.points) if hasattr(contour, 'points') else 0
                        
                        # Chuyển đổi các điểm thành danh sách phẳng
                        if hasattr(contour, 'points'):
                            flat_points = []
                            for point in contour.points:
                                flat_points.extend(point)
                            
                            contour_ds.ContourData = flat_points
                        
                        # Thêm vào ContourSequence
                        roi_contour.ContourSequence.append(contour_ds)
                
                ds.ROIContourSequence.append(roi_contour)
                
                # Thêm vào RTROIObservationsSequence
                rt_roi_obs = Dataset()
                rt_roi_obs.ObservationNumber = ssroi.ROINumber
                rt_roi_obs.ReferencedROINumber = ssroi.ROINumber
                rt_roi_obs.ROIObservationLabel = structure.name
                
                # Xác định loại ROI
                if 'PTV' in structure.name:
                    rt_roi_obs.RTROIInterpretedType = 'PTV'
                elif 'CTV' in structure.name:
                    rt_roi_obs.RTROIInterpretedType = 'CTV'
                elif 'GTV' in structure.name:
                    rt_roi_obs.RTROIInterpretedType = 'GTV'
                elif 'OAR' in structure.name:
                    rt_roi_obs.RTROIInterpretedType = 'ORGAN'
                else:
                    rt_roi_obs.RTROIInterpretedType = 'ORGAN'
                
                ds.RTROIObservationsSequence.append(rt_roi_obs)
        
        # Các thuộc tính DICOM chung
        ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
        ds.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8
        
        return ds
