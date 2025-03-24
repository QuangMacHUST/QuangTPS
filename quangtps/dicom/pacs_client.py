"""
Giao tiếp với hệ thống PACS (Picture Archiving and Communication System).

Module này cung cấp các chức năng để giao tiếp với hệ thống PACS,
cho phép tìm kiếm, truy xuất và lưu trữ dữ liệu DICOM.
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Tuple, Optional, Union
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
from pynetdicom import AE, evt, StoragePresentationContexts, QueryRetrievePresentationContexts
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    PatientRootQueryRetrieveInformationModelGet,
    PatientRootQueryRetrieveInformationModelMove,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelGet,
    StudyRootQueryRetrieveInformationModelMove,
    VerificationServiceClass
)
import pydicom

from quangtps.core.exceptions import DicomError, NetworkError

logger = logging.getLogger(__name__)

class PACSClient:
    """
    Lớp giao tiếp với hệ thống PACS.
    
    Class này cung cấp các phương thức để truy xuất dữ liệu từ hệ thống PACS,
    bao gồm tìm kiếm bệnh nhân, nghiên cứu, series và tải dữ liệu DICOM.
    """
    
    def __init__(self, local_ae_title: str = 'QUANGTPS'):
        """
        Khởi tạo đối tượng PACSClient.
        
        Parameters
        ----------
        local_ae_title : str, optional
            Application Entity Title của hệ thống QuangTPS
        """
        self.local_ae_title = local_ae_title
        self.ae = None
        self._init_application_entity()
    
    def _init_application_entity(self) -> None:
        """Khởi tạo Application Entity."""
        self.ae = AE(ae_title=self.local_ae_title)
        
        # Thêm context cho verification (ưu tiên quan trọng nhất)
        self.ae.add_requested_context('1.2.840.10008.1.1')  # Verification Service Class UID
        
        # Thêm context cho các hoạt động query/retrieve (ưu tiên cao)
        # Thường sẽ có ít hơn 10 context
        successful_contexts = 0
        failed_contexts = 0
        for context in QueryRetrievePresentationContexts:
            try:
                self.ae.add_requested_context(context)
                successful_contexts += 1
            except (ValueError, TypeError):
                failed_contexts += 1
                continue
        
        # Thêm context cho các hoạt động lưu trữ phổ biến nhất
        # Giới hạn ở 110 context để đảm bảo tổng số không vượt quá 128
        most_common_storage_contexts = StoragePresentationContexts[:110]
        storage_successful = 0
        storage_failed = 0
        for context in most_common_storage_contexts:
            try:
                self.ae.add_requested_context(context)
                storage_successful += 1
            except ValueError:
                # Nếu đã đạt giới hạn, dừng thêm context
                logger.warning("Đã đạt giới hạn số lượng context")
                break
            except TypeError:
                storage_failed += 1
                continue
        
        # Thử thêm các SOP class cụ thể cần thiết cho xạ trị
        rt_contexts = [
            '1.2.840.10008.5.1.4.1.1.481.1',  # RT Image Storage
            '1.2.840.10008.5.1.4.1.1.481.2',  # RT Dose Storage
            '1.2.840.10008.5.1.4.1.1.481.3',  # RT Structure Set Storage
            '1.2.840.10008.5.1.4.1.1.481.4',  # RT Beams Treatment Record Storage
            '1.2.840.10008.5.1.4.1.1.481.5',  # RT Plan Storage
            '1.2.840.10008.5.1.4.1.1.481.6',  # RT Brachy Treatment Record Storage
            '1.2.840.10008.5.1.4.1.1.481.7',  # RT Treatment Summary Record Storage
            '1.2.840.10008.5.1.4.1.1.481.8',  # RT Ion Plan Storage
            '1.2.840.10008.5.1.4.1.1.481.9',  # RT Ion Beams Treatment Record Storage
        ]
        
        rt_successful = 0
        rt_failed = 0
        for context in rt_contexts:
            try:
                self.ae.add_requested_context(context)
                rt_successful += 1
                logger.debug(f"Added RT context: {context}")
            except Exception as e:
                rt_failed += 1
        
        # Ghi log tổng kết
        if failed_contexts > 0:
            logger.debug(f"Không thể thêm {failed_contexts} context QR")
        if storage_failed > 0:
            logger.debug(f"Không thể thêm {storage_failed} context lưu trữ DICOM")
        if rt_failed > 0:
            logger.debug(f"Không thể thêm {rt_failed} context RT DICOM")

        logger.info(f"Initialized AE with title {self.local_ae_title} and {len(self.ae.requested_contexts)} contexts")
    
    def verify_connection(self, host: str, port: int, ae_title: str, timeout: int = 5) -> bool:
        """
        Kiểm tra kết nối đến máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        timeout : int, optional
            Thời gian chờ kết nối (giây)
            
        Returns
        -------
        bool
            True nếu kết nối thành công
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        """
        try:
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title, timeout=timeout)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                return False
            
            # Gửi C-ECHO request
            status = assoc.send_c_echo()
            
            # Giải phóng kết nối
            assoc.release()
            
            # Kiểm tra kết quả
            if status:
                logger.info(f"Connection to {host}:{port} ({ae_title}) verified")
                return True
            else:
                logger.error(f"C-ECHO to {host}:{port} ({ae_title}) failed with status {status}")
                return False
            
        except Exception as e:
            logger.error(f"Error verifying connection to {host}:{port}: {str(e)}")
            raise NetworkError(f"Error verifying connection to {host}:{port}: {str(e)}")
    
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
            Danh sách thông tin bệnh nhân
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        """
        try:
            # Tạo dataset cho C-FIND
            ds = Dataset()
            ds.QueryRetrieveLevel = 'PATIENT'
            
            # Các trường cần truy vấn
            ds.PatientID = ''
            ds.PatientName = ''
            ds.PatientBirthDate = ''
            ds.PatientSex = ''
            
            # Thêm điều kiện tìm kiếm
            if patient_id:
                ds.PatientID = patient_id
            if patient_name:
                ds.PatientName = patient_name
            
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Gửi C-FIND request
            responses = []
            for response in assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind):
                if response.is_pending:
                    # Chuyển đổi response thành dictionary
                    patient_info = {
                        'PatientID': getattr(response.identifier, 'PatientID', ''),
                        'PatientName': str(getattr(response.identifier, 'PatientName', '')),
                        'PatientBirthDate': getattr(response.identifier, 'PatientBirthDate', ''),
                        'PatientSex': getattr(response.identifier, 'PatientSex', '')
                    }
                    responses.append(patient_info)
            
            # Giải phóng kết nối
            assoc.release()
            
            logger.info(f"Found {len(responses)} patients on {host}:{port}")
            
            return responses
            
        except Exception as e:
            logger.error(f"Error finding patients on {host}:{port}: {str(e)}")
            raise NetworkError(f"Error finding patients on {host}:{port}: {str(e)}")
    
    def find_studies(self, host: str, port: int, ae_title: str, 
                     patient_id: str = None, study_date: str = None) -> List[Dict[str, Any]]:
        """
        Tìm kiếm nghiên cứu trên máy chủ PACS.
        
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
            Ngày nghiên cứu cần tìm (định dạng YYYYMMDD)
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin nghiên cứu
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        """
        try:
            # Tạo dataset cho C-FIND
            ds = Dataset()
            ds.QueryRetrieveLevel = 'STUDY'
            
            # Các trường cần truy vấn
            ds.PatientID = ''
            ds.PatientName = ''
            ds.StudyInstanceUID = ''
            ds.StudyDate = ''
            ds.StudyTime = ''
            ds.StudyDescription = ''
            ds.NumberOfStudyRelatedSeries = ''
            
            # Thêm điều kiện tìm kiếm
            if patient_id:
                ds.PatientID = patient_id
            if study_date:
                ds.StudyDate = study_date
            
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Gửi C-FIND request
            responses = []
            for response in assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind):
                if response.is_pending:
                    # Chuyển đổi response thành dictionary
                    study_info = {
                        'PatientID': getattr(response.identifier, 'PatientID', ''),
                        'PatientName': str(getattr(response.identifier, 'PatientName', '')),
                        'StudyInstanceUID': getattr(response.identifier, 'StudyInstanceUID', ''),
                        'StudyDate': getattr(response.identifier, 'StudyDate', ''),
                        'StudyTime': getattr(response.identifier, 'StudyTime', ''),
                        'StudyDescription': getattr(response.identifier, 'StudyDescription', ''),
                        'NumberOfStudyRelatedSeries': getattr(response.identifier, 'NumberOfStudyRelatedSeries', '')
                    }
                    responses.append(study_info)
            
            # Giải phóng kết nối
            assoc.release()
            
            logger.info(f"Found {len(responses)} studies on {host}:{port}")
            
            return responses
            
        except Exception as e:
            logger.error(f"Error finding studies on {host}:{port}: {str(e)}")
            raise NetworkError(f"Error finding studies on {host}:{port}: {str(e)}")
    
    def find_series(self, host: str, port: int, ae_title: str, 
                   study_instance_uid: str) -> List[Dict[str, Any]]:
        """
        Tìm kiếm series trên máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        study_instance_uid : str
            Study Instance UID cần tìm
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin series
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        """
        try:
            # Tạo dataset cho C-FIND
            ds = Dataset()
            ds.QueryRetrieveLevel = 'SERIES'
            
            # Các trường cần truy vấn
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = ''
            ds.SeriesNumber = ''
            ds.Modality = ''
            ds.SeriesDescription = ''
            ds.NumberOfSeriesRelatedInstances = ''
            
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Gửi C-FIND request
            responses = []
            for response in assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind):
                if response.is_pending:
                    # Chuyển đổi response thành dictionary
                    series_info = {
                        'StudyInstanceUID': getattr(response.identifier, 'StudyInstanceUID', ''),
                        'SeriesInstanceUID': getattr(response.identifier, 'SeriesInstanceUID', ''),
                        'SeriesNumber': getattr(response.identifier, 'SeriesNumber', ''),
                        'Modality': getattr(response.identifier, 'Modality', ''),
                        'SeriesDescription': getattr(response.identifier, 'SeriesDescription', ''),
                        'NumberOfSeriesRelatedInstances': getattr(response.identifier, 'NumberOfSeriesRelatedInstances', '')
                    }
                    responses.append(series_info)
            
            # Giải phóng kết nối
            assoc.release()
            
            logger.info(f"Found {len(responses)} series for study {study_instance_uid}")
            
            return responses
            
        except Exception as e:
            logger.error(f"Error finding series on {host}:{port}: {str(e)}")
            raise NetworkError(f"Error finding series on {host}:{port}: {str(e)}")
    
    def find_instances(self, host: str, port: int, ae_title: str, 
                      study_instance_uid: str, series_instance_uid: str) -> List[Dict[str, Any]]:
        """
        Tìm kiếm instances (SOP Instances) trên máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        study_instance_uid : str
            Study Instance UID cần tìm
        series_instance_uid : str
            Series Instance UID cần tìm
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin instances
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        """
        try:
            # Tạo dataset cho C-FIND
            ds = Dataset()
            ds.QueryRetrieveLevel = 'IMAGE'
            
            # Các trường cần truy vấn
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.SOPInstanceUID = ''
            ds.InstanceNumber = ''
            ds.SOPClassUID = ''
            
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Gửi C-FIND request
            responses = []
            for response in assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind):
                if response.is_pending:
                    # Chuyển đổi response thành dictionary
                    instance_info = {
                        'StudyInstanceUID': getattr(response.identifier, 'StudyInstanceUID', ''),
                        'SeriesInstanceUID': getattr(response.identifier, 'SeriesInstanceUID', ''),
                        'SOPInstanceUID': getattr(response.identifier, 'SOPInstanceUID', ''),
                        'InstanceNumber': getattr(response.identifier, 'InstanceNumber', ''),
                        'SOPClassUID': getattr(response.identifier, 'SOPClassUID', '')
                    }
                    responses.append(instance_info)
            
            # Giải phóng kết nối
            assoc.release()
            
            logger.info(f"Found {len(responses)} instances for series {series_instance_uid}")
            
            return responses
            
        except Exception as e:
            logger.error(f"Error finding instances on {host}:{port}: {str(e)}")
            raise NetworkError(f"Error finding instances on {host}:{port}: {str(e)}")
    
    def get_dicom_files(self, host: str, port: int, ae_title: str, output_dir: str,
                       study_instance_uid: str, series_instance_uid: str = None,
                       sop_instance_uid: str = None) -> List[str]:
        """
        Lấy file DICOM từ máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        output_dir : str
            Thư mục lưu file DICOM
        study_instance_uid : str
            Study Instance UID cần lấy
        series_instance_uid : str, optional
            Series Instance UID cần lấy
        sop_instance_uid : str, optional
            SOP Instance UID cần lấy
            
        Returns
        -------
        List[str]
            Danh sách đường dẫn đến các file DICOM đã lấy
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        DicomError
            Nếu có lỗi khi lưu file
        """
        try:
            # Tạo thư mục lưu trữ nếu chưa tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Tạo dataset cho C-GET
            ds = Dataset()
            ds.QueryRetrieveLevel = 'STUDY'
            ds.StudyInstanceUID = study_instance_uid
            
            # Nếu có Series Instance UID
            if series_instance_uid:
                ds.QueryRetrieveLevel = 'SERIES'
                ds.SeriesInstanceUID = series_instance_uid
            
            # Nếu có SOP Instance UID
            if sop_instance_uid:
                ds.QueryRetrieveLevel = 'IMAGE'
                ds.SOPInstanceUID = sop_instance_uid
            
            # Thiết lập handler cho sự kiện C-STORE
            def handle_store(event):
                """Xử lý sự kiện C-STORE."""
                ds = event.dataset
                
                # Tạo tên file
                if hasattr(ds, 'SOPInstanceUID'):
                    filename = f"{ds.SOPInstanceUID}.dcm"
                else:
                    filename = f"dicom_{hash(str(ds))}.dcm"
                
                # Đường dẫn đầy đủ
                filepath = os.path.join(output_dir, filename)
                
                # Lưu file
                try:
                    # Thiết lập transfer syntax
                    if event.context.transfer_syntax in [ImplicitVRLittleEndian, ExplicitVRLittleEndian]:
                        ds.file_meta = pydicom.dataset.FileMetaDataset()
                        ds.file_meta.TransferSyntaxUID = event.context.transfer_syntax
                    
                    # Lưu dataset
                    ds.save_as(filepath, write_like_original=False)
                    
                    # Thêm vào danh sách file đã lưu
                    if filepath not in retrieved_files:
                        retrieved_files.append(filepath)
                    
                    return 0x0000  # Success
                except Exception as e:
                    logger.error(f"Error saving file {filepath}: {str(e)}")
                    return 0xC001  # Failure
            
            # Thiết lập handler
            handlers = [(evt.EVT_C_STORE, handle_store)]
            
            # Danh sách file đã lấy
            retrieved_files = []
            
            # Thiết lập kết nối và thực hiện C-GET
            self.ae.on_c_store = handle_store
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Gửi C-GET request
            responses = assoc.send_c_get(ds, StudyRootQueryRetrieveInformationModelGet)
            for response in responses:
                pass  # Chỉ cần lặp qua responses để đảm bảo hoàn thành
            
            # Giải phóng kết nối
            assoc.release()
            
            logger.info(f"Retrieved {len(retrieved_files)} DICOM files to {output_dir}")
            
            return retrieved_files
            
        except Exception as e:
            logger.error(f"Error retrieving DICOM files from {host}:{port}: {str(e)}")
            raise NetworkError(f"Error retrieving DICOM files from {host}:{port}: {str(e)}")
    
    def store_dicom_files(self, host: str, port: int, ae_title: str, 
                         file_paths: List[str]) -> int:
        """
        Lưu trữ file DICOM lên máy chủ PACS.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        file_paths : List[str]
            Danh sách đường dẫn đến các file DICOM cần lưu trữ
            
        Returns
        -------
        int
            Số file đã lưu trữ thành công
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        DicomError
            Nếu có lỗi khi đọc file
        """
        try:
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Đếm số file đã lưu trữ thành công
            successful_count = 0
            
            # Gửi từng file
            for file_path in file_paths:
                try:
                    # Đọc file DICOM
                    ds = pydicom.dcmread(file_path)
                    
                    # Gửi C-STORE request
                    status = assoc.send_c_store(ds)
                    
                    if status:
                        successful_count += 1
                        logger.debug(f"Successfully stored {file_path}")
                    else:
                        logger.warning(f"Failed to store {file_path}")
                
                except Exception as e:
                    logger.error(f"Error reading or storing file {file_path}: {str(e)}")
            
            # Giải phóng kết nối
            assoc.release()
            
            logger.info(f"Stored {successful_count}/{len(file_paths)} files to {host}:{port}")
            
            return successful_count
            
        except Exception as e:
            logger.error(f"Error storing DICOM files to {host}:{port}: {str(e)}")
            raise NetworkError(f"Error storing DICOM files to {host}:{port}: {str(e)}")
    
    def move_dicom_files(self, host: str, port: int, ae_title: str, destination_ae: str,
                        study_instance_uid: str, series_instance_uid: str = None,
                        sop_instance_uid: str = None) -> bool:
        """
        Di chuyển file DICOM từ máy chủ PACS đến AE đích.
        
        Parameters
        ----------
        host : str
            Địa chỉ máy chủ PACS
        port : int
            Cổng máy chủ PACS
        ae_title : str
            Application Entity Title của máy chủ PACS
        destination_ae : str
            Application Entity Title đích
        study_instance_uid : str
            Study Instance UID cần di chuyển
        series_instance_uid : str, optional
            Series Instance UID cần di chuyển
        sop_instance_uid : str, optional
            SOP Instance UID cần di chuyển
            
        Returns
        -------
        bool
            True nếu thành công
            
        Raises
        ------
        NetworkError
            Nếu có lỗi kết nối
        """
        try:
            # Tạo dataset cho C-MOVE
            ds = Dataset()
            ds.QueryRetrieveLevel = 'STUDY'
            ds.StudyInstanceUID = study_instance_uid
            
            # Nếu có Series Instance UID
            if series_instance_uid:
                ds.QueryRetrieveLevel = 'SERIES'
                ds.SeriesInstanceUID = series_instance_uid
            
            # Nếu có SOP Instance UID
            if sop_instance_uid:
                ds.QueryRetrieveLevel = 'IMAGE'
                ds.SOPInstanceUID = sop_instance_uid
            
            # Thiết lập kết nối
            assoc = self.ae.associate(host, port, ae_title=ae_title)
            
            if not assoc.is_established:
                logger.error(f"Association with {host}:{port} ({ae_title}) failed")
                raise NetworkError(f"Association with {host}:{port} ({ae_title}) failed")
            
            # Gửi C-MOVE request
            responses = assoc.send_c_move(ds, destination_ae, StudyRootQueryRetrieveInformationModelMove)
            
            success = True
            for response in responses:
                if response.Status != 0x0000:  # Status khác Success
                    success = False
                    logger.warning(f"C-MOVE failed with status {response.Status}")
            
            # Giải phóng kết nối
            assoc.release()
            
            if success:
                logger.info(f"Successfully moved DICOM data to {destination_ae}")
            else:
                logger.warning(f"Some issues occurred when moving DICOM data to {destination_ae}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error moving DICOM files to {destination_ae}: {str(e)}")
            raise NetworkError(f"Error moving DICOM files to {destination_ae}: {str(e)}")
