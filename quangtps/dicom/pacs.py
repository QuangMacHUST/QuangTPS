
\"""Tích hợp với hệ thống PACS.

Module này cung cấp các công cụ để kết nối và tương tác với hệ thống PACS (Picture Archiving and Communication System),
cho phép truy vấn, tìm kiếm và tải dữ liệu DICOM từ máy chủ PACS.
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Union, Optional, Tuple

import pydicom
from pynetdicom import AE, evt, StoragePresentationContexts, QueryRetrievePresentationContexts
# SOP Classes chính xác cho Query/Retrieve
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelGet,
    PatientRootQueryRetrieveInformationModelMove,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelGet,
    StudyRootQueryRetrieveInformationModelMove,
    VerificationSOPClass
)

from quangtps.core.config import Config
from quangtps.core.exceptions import NetworkError, AuthenticationError, DicomError

logger = logging.getLogger(__name__)

class PACSClient:
    """Lớp xử lý việc kết nối và tương tác với hệ thống PACS"""
    
    def __init__(self, ae_title: str = None, output_dir: str = None):
        """
        Khởi tạo PACSClient.
        
        Parameters
        ----------
        ae_title : str, optional
            Application Entity Title cho local AE
        output_dir : str, optional
            Thư mục để lưu các file DICOM tải về
        """
        config = Config()
        self.ae_title = ae_title or config.get('pacs', 'ae_title', fallback='QUANGTPS')
        self.output_dir = output_dir or config.get('pacs', 'output_dir', fallback=tempfile.gettempdir())
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Khởi tạo Application Entity
        self.ae = AE(ae_title=self.ae_title)
        
        # Thêm context cho các dịch vụ
        self.ae.requested_contexts = QueryRetrievePresentationContexts
        
        # Thêm context cho lưu trữ (storage)
        for context in StoragePresentationContexts:
            self.ae.add_requested_context(context.abstract_syntax)
    
    def echo(self, host: str, port: int, ae_title: str) -> bool:
        """
        Kiểm tra kết nối đến máy chủ PACS bằng C-ECHO.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
            
        Returns
        -------
        bool
            True nếu kết nối thành công
            
        Raises
        ------
        NetworkError
            Nếu không thể kết nối đến máy chủ PACS
        """
        # Thêm context cho dịch vụ verification
        self.ae.add_requested_context(VerificationSOPClass)
        
        try:
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if assoc.is_established:
                # Gửi C-ECHO
                status = assoc.send_c_echo()
                
                # Giải phóng kết nối
                assoc.release()
                
                # Kiểm tra kết quả
                return status.Status == 0
            else:
                raise NetworkError(f"Association rejected, aborted or never connected with {host}:{port}")
        except Exception as e:
            logger.error(f"C-ECHO failed: {str(e)}")
            raise NetworkError(f"C-ECHO failed: {str(e)}", url=f"{host}:{port}")
    
    def find_patients(self, host: str, port: int, ae_title: str, 
                     patient_id: str = None, patient_name: str = None) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bệnh nhân trên máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        patient_id : str, optional
            ID bệnh nhân cần tìm
        patient_name : str, optional
            Tên bệnh nhân cần tìm
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các bệnh nhân tìm thấy
            
        Raises
        ------
        NetworkError
            Nếu không thể kết nối đến máy chủ PACS
        """
        # Tạo dataset cho truy vấn
        ds = pydicom.dataset.Dataset()
        ds.QueryRetrieveLevel = 'PATIENT'
        
        # Các trường cần trả về
        ds.PatientID = ''
        ds.PatientName = ''
        ds.PatientBirthDate = ''
        ds.PatientSex = ''
        
        # Thêm điều kiện tìm kiếm
        if patient_id:
            ds.PatientID = patient_id
        if patient_name:
            ds.PatientName = patient_name
        
        try:
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                raise NetworkError(f"Association rejected, aborted or never connected with {host}:{port}")
            
            # Gửi C-FIND
            responses = []
            for (status, dataset) in assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind):
                if status.Status == 0xFF00:  # Pending
                    responses.append({
                        'PatientID': getattr(dataset, 'PatientID', ''),
                        'PatientName': str(getattr(dataset, 'PatientName', '')),
                        'PatientBirthDate': getattr(dataset, 'PatientBirthDate', ''),
                        'PatientSex': getattr(dataset, 'PatientSex', '')
                    })
            
            # Giải phóng kết nối
            assoc.release()
            
            return responses
        except Exception as e:
            logger.error(f"C-FIND patients failed: {str(e)}")
            raise NetworkError(f"C-FIND patients failed: {str(e)}", url=f"{host}:{port}")
    
    def find_studies(self, host: str, port: int, ae_title: str, 
                    patient_id: str = None, study_date: str = None) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các nghiên cứu (study) trên máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        patient_id : str, optional
            ID bệnh nhân cần tìm
        study_date : str, optional
            Ngày nghiên cứu (định dạng YYYYMMDD hoặc YYYYMMDD-YYYYMMDD)
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các nghiên cứu tìm thấy
            
        Raises
        ------
        NetworkError
            Nếu không thể kết nối đến máy chủ PACS
        """
        # Tạo dataset cho truy vấn
        ds = pydicom.dataset.Dataset()
        ds.QueryRetrieveLevel = 'STUDY'
        
        # Các trường cần trả về
        ds.PatientID = ''
        ds.PatientName = ''
        ds.StudyInstanceUID = ''
        ds.StudyDescription = ''
        ds.StudyDate = ''
        ds.StudyTime = ''
        ds.ModalitiesInStudy = ''
        ds.NumberOfStudyRelatedSeries = ''
        
        # Thêm điều kiện tìm kiếm
        if patient_id:
            ds.PatientID = patient_id
        if study_date:
            ds.StudyDate = study_date
        
        try:
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                raise NetworkError(f"Association rejected, aborted or never connected with {host}:{port}")
            
            # Gửi C-FIND
            responses = []
            for (status, dataset) in assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind):
                if status.Status == 0xFF00:  # Pending
                    responses.append({
                        'PatientID': getattr(dataset, 'PatientID', ''),
                        'PatientName': str(getattr(dataset, 'PatientName', '')),
                        'StudyInstanceUID': getattr(dataset, 'StudyInstanceUID', ''),
                        'StudyDescription': getattr(dataset, 'StudyDescription', ''),
                        'StudyDate': getattr(dataset, 'StudyDate', ''),
                        'StudyTime': getattr(dataset, 'StudyTime', ''),
                        'ModalitiesInStudy': getattr(dataset, 'ModalitiesInStudy', ''),
                        'NumberOfStudyRelatedSeries': getattr(dataset, 'NumberOfStudyRelatedSeries', '')
                    })
            
            # Giải phóng kết nối
            assoc.release()
            
            return responses
        except Exception as e:
            logger.error(f"C-FIND studies failed: {str(e)}")
            raise NetworkError(f"C-FIND studies failed: {str(e)}", url=f"{host}:{port}")
    
    def find_series(self, host: str, port: int, ae_title: str, 
                   study_instance_uid: str) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các series trong một nghiên cứu trên máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        study_instance_uid : str
            Study Instance UID của nghiên cứu
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các series tìm thấy
            
        Raises
        ------
        NetworkError
            Nếu không thể kết nối đến máy chủ PACS
        """
        # Tạo dataset cho truy vấn
        ds = pydicom.dataset.Dataset()
        ds.QueryRetrieveLevel = 'SERIES'
        
        # Các trường cần trả về
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = ''
        ds.SeriesDescription = ''
        ds.SeriesNumber = ''
        ds.Modality = ''
        ds.NumberOfSeriesRelatedInstances = ''
        
        try:
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                raise NetworkError(f"Association rejected, aborted or never connected with {host}:{port}")
            
            # Gửi C-FIND
            responses = []
            for (status, dataset) in assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind):
                if status.Status == 0xFF00:  # Pending
                    responses.append({
                        'StudyInstanceUID': getattr(dataset, 'StudyInstanceUID', ''),
                        'SeriesInstanceUID': getattr(dataset, 'SeriesInstanceUID', ''),
                        'SeriesDescription': getattr(dataset, 'SeriesDescription', ''),
                        'SeriesNumber': getattr(dataset, 'SeriesNumber', ''),
                        'Modality': getattr(dataset, 'Modality', ''),
                        'NumberOfSeriesRelatedInstances': getattr(dataset, 'NumberOfSeriesRelatedInstances', '')
                    })
            
            # Giải phóng kết nối
            assoc.release()
            
            return responses
        except Exception as e:
            logger.error(f"C-FIND series failed: {str(e)}")
            raise NetworkError(f"C-FIND series failed: {str(e)}", url=f"{host}:{port}")
    
    def find_instances(self, host: str, port: int, ae_
