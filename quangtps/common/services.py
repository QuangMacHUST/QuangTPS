#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các dịch vụ chung của QuangTPS.

Module này cung cấp các dịch vụ và giao diện chung cho toàn bộ ứng dụng,
và hoạt động cùng với hệ thống dịch vụ trong core/services.py.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union, Callable, TypeVar, cast

from quangtps.core.services import ServiceBase as CoreServiceBase
from quangtps.core.services import ServiceRegistry as CoreServiceRegistry
from quangtps.core.patient import Patient

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceBase(CoreServiceBase):
    """
    Lớp cơ sở cho các dịch vụ chung.
    Kế thừa từ ServiceBase trong core để đảm bảo tính tương thích.
    """
    pass

class ServiceRegistry(CoreServiceRegistry):
    """
    Registry cho các dịch vụ chung, kế thừa từ ServiceRegistry trong core.
    Cung cấp điểm truy cập toàn cục cho các dịch vụ từ bất kỳ phần nào của ứng dụng.
    """
    pass

# Sử dụng registry cốt lõi để đảm bảo tính nhất quán
service_registry = ServiceRegistry.get_instance()

class PatientService(ServiceBase):
    """
    Dịch vụ quản lý bệnh nhân.
    Cung cấp các phương thức để truy xuất và quản lý dữ liệu bệnh nhân.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo dịch vụ quản lý bệnh nhân.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Cấu hình cho dịch vụ
        """
        super().__init__(config)
        self.patient_db = None
    
    def initialize(self) -> bool:
        """
        Khởi tạo dịch vụ quản lý bệnh nhân.
        Kết nối đến cơ sở dữ liệu bệnh nhân.
        
        Returns
        -------
        bool
            True nếu khởi tạo thành công, False nếu không
        """
        try:
            from quangtps.database.patient_db import PatientDB
            self.patient_db = PatientDB()
            
            logger.info("Dịch vụ quản lý bệnh nhân đã khởi tạo thành công")
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Không thể khởi tạo dịch vụ quản lý bệnh nhân: {str(e)}")
            return False
    
    def shutdown(self) -> bool:
        """
        Tắt dịch vụ quản lý bệnh nhân.
        Đóng kết nối đến cơ sở dữ liệu bệnh nhân.
        
        Returns
        -------
        bool
            True nếu tắt thành công, False nếu không
        """
        try:
            if hasattr(self, 'patient_db') and self.patient_db:
                self.patient_db.close()
            self.initialized = False
            logger.info("Dịch vụ quản lý bệnh nhân đã tắt thành công")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tắt dịch vụ quản lý bệnh nhân: {str(e)}")
            return False
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả bệnh nhân.
        
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin của tất cả bệnh nhân
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return []
        
        try:
            return self.patient_db.get_all_patients()
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách bệnh nhân: {str(e)}")
            return []
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin của một bệnh nhân theo ID.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân cần lấy thông tin
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Thông tin của bệnh nhân nếu tìm thấy, None nếu không
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return None
        
        try:
            return self.patient_db.get_patient(patient_id)
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin bệnh nhân: {str(e)}")
            return None
    
    def add_patient(self, patient_data: Dict[str, Any]) -> str:
        """
        Thêm một bệnh nhân mới.
        
        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu của bệnh nhân cần thêm
            
        Returns
        -------
        str
            ID của bệnh nhân mới nếu thêm thành công, chuỗi rỗng nếu không
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return ""
        
        try:
            result = self.patient_db.add_patient(patient_data)
            return result if result else ""
        except Exception as e:
            logger.error(f"Lỗi khi thêm bệnh nhân: {str(e)}")
            return ""
    
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """
        Cập nhật thông tin của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân cần cập nhật
        patient_data : Dict[str, Any]
            Dữ liệu mới của bệnh nhân
            
        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu không
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return False
        
        try:
            return self.patient_db.update_patient(patient_id, patient_data)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật thông tin bệnh nhân: {str(e)}")
            return False
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Xóa một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return False
        
        try:
            return self.patient_db.delete_patient(patient_id)
        except Exception as e:
            logger.error(f"Lỗi khi xóa bệnh nhân: {str(e)}")
            return False
    
    def search_patients(self, query: Optional[str] = None, **filters: Any) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bệnh nhân theo từ khóa hoặc các bộ lọc.
        
        Parameters
        ----------
        query : Optional[str], optional
            Từ khóa tìm kiếm
        **filters : Any
            Các bộ lọc tìm kiếm khác
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin của các bệnh nhân phù hợp
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return []
        
        try:
            # Convert query to empty string if None before passing to the database function
            search_query = "" if query is None else query
            
            # Type safety: explicitly cast the filters to Dict[str, Any]
            search_filters = dict(filters)
            
            # Use explicit try-except to catch type errors
            try:
                return self.patient_db.search_patients(search_query, **search_filters)
            except TypeError:
                # Fall back to direct implementation if type error occurs
                logger.warning("Falling back to simplified search due to type mismatch")
                patients = self.get_all_patients()
                if not search_query:
                    return patients
                    
                result = []
                search_query = search_query.lower()
                for patient in patients:
                    for key, value in patient.items():
                        if isinstance(value, str) and search_query in value.lower():
                            result.append(patient)
                            break
                return result
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm bệnh nhân: {str(e)}")
            return []
    
    def get_patient_studies(self, patient_id: str, include_series: bool = False) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các nghiên cứu của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        include_series : bool, optional
            Có bao gồm thông tin về các chuỗi không, mặc định là False
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin của các nghiên cứu
        """
        if not self.initialized or not self.patient_db:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return []
        
        try:
            return self.patient_db.get_patient_studies(patient_id, include_series)
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách nghiên cứu của bệnh nhân: {str(e)}")
            return []
    
    def create_patient_object(self, patient_data: Dict[str, Any]) -> Optional[Any]:
        """
        Tạo đối tượng Patient từ dữ liệu.
        
        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân
            
        Returns
        -------
        Optional[Any]
            Đối tượng Patient nếu tạo thành công, None nếu không
        """
        if not self.initialized:
            logger.error("Dịch vụ quản lý bệnh nhân chưa được khởi tạo")
            return None
        
        try:
            # Import here to avoid circular imports
            from quangtps.core.patient import Patient
            from datetime import date
            
            # Extract required fields or use defaults
            patient_id = str(patient_data.get('id', ''))
            name = str(patient_data.get('name', ''))
            
            # Parse date of birth
            dob_str = patient_data.get('dob') or patient_data.get('birth_date')
            try:
                dob = date.fromisoformat(dob_str) if dob_str else date.today()
            except (ValueError, TypeError):
                dob = date.today()
                
            gender = str(patient_data.get('gender', ''))
            
            # Create a dictionary of kwargs for the Patient constructor
            patient_kwargs = {
                'id': patient_id,
                'name': name,
                'dob': dob,
                'gender': gender,
            }
            
            # Add optional fields that are explicitly defined in the Patient class
            optional_fields = [
                'address', 'phone', 'email', 'diagnosis', 'notes',
                'mrn', 'primary_physician', 'referring_physician', 'hospital_id',
                'insurance_id', 'allergies', 'height_cm', 'weight_kg',
                'diagnosis_code', 'site', 'technique', 'treatment_intent'
            ]
            
            for field in optional_fields:
                if field in patient_data:
                    patient_kwargs[field] = patient_data[field]
            
            # Handle metadata separately if present
            if 'metadata' in patient_data:
                patient_kwargs['metadata'] = patient_data['metadata']
            
            # Create the Patient object with unpacked kwargs
            return Patient(**patient_kwargs)
        except Exception as e:
            logger.error(f"Lỗi khi tạo đối tượng Patient: {str(e)}")
            return None

# Đăng ký dịch vụ quản lý bệnh nhân
patient_service = PatientService()

# Sử dụng ServiceRegistry để đăng ký dịch vụ
# Đặt typehint cho NOOP để tránh lỗi linter
_ = service_registry.register("PatientService", patient_service) 