"""
Quản lý cơ sở dữ liệu bệnh nhân.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class Patient:
    """
    Lớp biểu diễn thông tin bệnh nhân.
    """
    
    def __init__(self, patient_id=None, name=None, birth_date=None, gender=None, metadata=None):
        """
        Khởi tạo đối tượng Patient.
        
        Args:
            patient_id (str): ID của bệnh nhân
            name (str): Tên của bệnh nhân
            birth_date (str): Ngày sinh của bệnh nhân
            gender (str): Giới tính của bệnh nhân
            metadata (dict): Metadata bổ sung của bệnh nhân
        """
        self.id = patient_id or str(uuid.uuid4())
        self.name = name or ""
        self.birth_date = birth_date
        self.gender = gender
        self.metadata = metadata or {}
        self.studies = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_study(self, study):
        """
        Thêm một nghiên cứu vào bệnh nhân.
        
        Args:
            study (Study): Đối tượng nghiên cứu
        """
        self.studies.append(study)
        study.patient_id = self.id
    
    def get_study_by_id(self, study_id):
        """
        Lấy nghiên cứu theo ID.
        
        Args:
            study_id (str): ID của nghiên cứu cần tìm
            
        Returns:
            Study: Đối tượng nghiên cứu hoặc None nếu không tìm thấy
        """
        for study in self.studies:
            if study.id == study_id:
                return study
        return None

    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary.
        
        Returns:
            dict: Thông tin bệnh nhân dưới dạng dictionary
        """
        return {
            "id": self.id,
            "name": self.name,
            "birth_date": self.birth_date,
            "gender": self.gender,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "studies": [study.to_dict() for study in self.studies]
        }


class Study:
    """
    Lớp biểu diễn thông tin nghiên cứu y tế.
    """
    
    def __init__(self, study_id=None, description=None, date=None, patient_id=None, metadata=None):
        """
        Khởi tạo đối tượng Study.
        
        Args:
            study_id (str): ID của nghiên cứu
            description (str): Mô tả về nghiên cứu
            date (str): Ngày thực hiện nghiên cứu
            patient_id (str): ID của bệnh nhân liên quan
            metadata (dict): Metadata bổ sung của nghiên cứu
        """
        self.id = study_id or str(uuid.uuid4())
        self.description = description or ""
        self.date = date or datetime.now().isoformat()
        self.patient_id = patient_id
        self.metadata = metadata or {}
        self.series_list = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_series(self, series):
        """
        Thêm một chuỗi vào nghiên cứu.
        
        Args:
            series (Series): Đối tượng chuỗi
        """
        self.series_list.append(series)
        series.study_id = self.id
    
    def get_series_by_id(self, series_id):
        """
        Lấy chuỗi theo ID.
        
        Args:
            series_id (str): ID của chuỗi cần tìm
            
        Returns:
            Series: Đối tượng chuỗi hoặc None nếu không tìm thấy
        """
        for series in self.series_list:
            if series.id == series_id:
                return series
        return None
    
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary.
        
        Returns:
            dict: Thông tin nghiên cứu dưới dạng dictionary
        """
        return {
            "id": self.id,
            "description": self.description,
            "date": self.date,
            "patient_id": self.patient_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "series": [series.to_dict() for series in self.series_list]
        }


class Series:
    """
    Lớp biểu diễn thông tin chuỗi dữ liệu y tế.
    """
    
    def __init__(self, series_id=None, description=None, modality=None, study_id=None, metadata=None):
        """
        Khởi tạo đối tượng Series.
        
        Args:
            series_id (str): ID của chuỗi
            description (str): Mô tả về chuỗi
            modality (str): Dạng hình ảnh (CT, MR, PT, v.v.)
            study_id (str): ID của nghiên cứu liên quan
            metadata (dict): Metadata bổ sung của chuỗi
        """
        self.id = series_id or str(uuid.uuid4())
        self.description = description or ""
        self.modality = modality or ""
        self.study_id = study_id
        self.metadata = metadata or {}
        self.file_paths = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.data_path = None  # Đường dẫn đến dữ liệu xử lý
    
    def add_file(self, file_path):
        """
        Thêm một file vào chuỗi.
        
        Args:
            file_path (str): Đường dẫn đến file
        """
        if file_path not in self.file_paths:
            self.file_paths.append(file_path)
    
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary.
        
        Returns:
            dict: Thông tin chuỗi dưới dạng dictionary
        """
        return {
            "id": self.id,
            "description": self.description,
            "modality": self.modality,
            "study_id": self.study_id,
            "metadata": self.metadata,
            "file_paths": self.file_paths,
            "data_path": self.data_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class PatientDatabase:
    """
    Class quản lý thông tin bệnh nhân trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng PatientDatabase.
        """
        self.db = DBConnector()

    def create_patient(self, name, birth_date=None, gender=None, metadata=None):
        """
        Tạo một bệnh nhân mới trong cơ sở dữ liệu.
        
        Args:
            name (str): Tên của bệnh nhân
            birth_date (str, optional): Ngày sinh của bệnh nhân
            gender (str, optional): Giới tính của bệnh nhân
            metadata (dict, optional): Siêu dữ liệu bổ sung
            
        Returns:
            str: ID của bệnh nhân mới tạo
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi tạo bệnh nhân
        """
        try:
            patient_id = str(uuid.uuid4())
            created_at = datetime.now().isoformat()
            updated_at = created_at
            
            # Chuyển đổi metadata thành chuỗi JSON (nếu có)
            metadata_str = json.dumps(metadata) if metadata else None
            
            # Chuẩn bị câu truy vấn
            query = """
                INSERT INTO patients (id, name, birth_date, gender, created_date, updated_date, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            # Chuẩn bị tham số
            params = (patient_id, name, birth_date, gender, created_at, updated_at, metadata_str)
            
            # Thực hiện câu truy vấn
            self.db.execute_insert(query, params)
            
            logger.info("Đã tạo bệnh nhân mới: %s, ID: %s", name, patient_id)
            return patient_id
        except Exception as e:
            logger.error("Lỗi khi tạo bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể tạo bệnh nhân: %s" % str(e)) from e
            
    def get_patient(self, patient_id):
        """
        Lấy thông tin của một bệnh nhân theo ID.
        
        Args:
            patient_id (str): ID của bệnh nhân
            
        Returns:
            dict: Thông tin của bệnh nhân hoặc None nếu không tìm thấy
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi truy vấn
        """
        try:
            # Chuẩn bị câu truy vấn
            query = "SELECT * FROM patients WHERE id = ?"
            
            # Thực hiện câu truy vấn
            result = self.db.execute_query(query, (patient_id,))
            
            if not result:
                logger.warning("Không tìm thấy bệnh nhân với ID: %s", patient_id)
                return None
                
            # Xử lý kết quả truy vấn
            patient = {
                'id': result[0],
                'name': result[1],
                'birth_date': result[2],
                'gender': result[3],
                'created_at': result[4],
                'updated_at': result[5],
                'metadata': json.loads(result[6]) if result[6] else None
            }
            
            logger.debug("Đã lấy thông tin bệnh nhân ID: %s", patient_id)
            return patient
        except Exception as e:
            logger.error("Lỗi khi lấy thông tin bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể lấy thông tin bệnh nhân: %s" % str(e)) from e
            
    def update_patient(self, patient_id, name=None, birth_date=None, gender=None, metadata=None):
        """
        Cập nhật thông tin của một bệnh nhân.
        
        Args:
            patient_id (str): ID của bệnh nhân
            name (str, optional): Tên mới của bệnh nhân
            birth_date (str, optional): Ngày sinh mới
            gender (str, optional): Giới tính mới
            metadata (dict, optional): Siêu dữ liệu mới
            
        Returns:
            bool: True nếu cập nhật thành công, False nếu không tìm thấy bệnh nhân
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi cập nhật
        """
        try:
            # Kiểm tra xem bệnh nhân có tồn tại không
            patient = self.get_patient(patient_id)
            if not patient:
                logger.warning("Không thể cập nhật: Không tìm thấy bệnh nhân với ID: %s", patient_id)
                return False
                
            # Chuẩn bị các trường cần cập nhật
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
                
            if birth_date is not None:
                updates.append("birth_date = ?")
                params.append(birth_date)
                
            if gender is not None:
                updates.append("gender = ?")
                params.append(gender)
                
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
                
            # Luôn cập nhật thởi gian cập nhật
            updated_at = datetime.now().isoformat()
            updates.append("updated_date = ?")
            params.append(updated_at)
            
            if not updates:
                logger.debug("Không có thông tin nào cần cập nhật cho bệnh nhân ID: %s", patient_id)
                return True
                
            # Chuẩn bị câu truy vấn
            query = "UPDATE patients SET " + ", ".join(updates) + " WHERE id = ?"
            params.append(patient_id)
            
            # Thực hiện cập nhật
            self.db.execute_update(query, params)
            
            logger.info("Đã cập nhật bệnh nhân ID: %s", patient_id)
            return True
        except Exception as e:
            logger.error("Lỗi khi cập nhật bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể cập nhật bệnh nhân: %s" % str(e)) from e
            
    def delete_patient(self, patient_id):
        """
        Xóa một bệnh nhân và tất cả các dữ liệu liên quan.
        
        Args:
            patient_id (str): ID của bệnh nhân cần xóa
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy bệnh nhân
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi xóa
        """
        try:
            # Kiểm tra xem bệnh nhân có tồn tại không
            patient = self.get_patient(patient_id)
            if not patient:
                logger.warning("Không thể xóa: Không tìm thấy bệnh nhân với ID: %s", patient_id)
                return False
                
            # Xóa tất cả kế hoạch điều trị liên quan trước
            self._delete_all_plans(patient_id)
                
            # Thực hiện xóa bệnh nhân
            query = "DELETE FROM patients WHERE id = ?"
            params = (patient_id,)
            
            self.db.execute_update(query, params)
            logger.info("Đã xóa bệnh nhân ID: %s", patient_id)
            
            return True
        except Exception as e:
            logger.error("Lỗi khi xóa bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể xóa bệnh nhân: %s" % str(e)) from e

    def get_all_patients(self):
        """
        Lấy danh sách tất cả các bệnh nhân.

        Returns:
            list: Danh sách các bệnh nhân

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình lấy danh sách
        """
        try:
            query = "SELECT * FROM patients ORDER BY name"
            patients = self.db.execute_query(query, fetchall=True)
            
            # Parse metadata từ JSON nếu có
            for patient in patients:
                if patient.get('metadata'):
                    try:
                        patient['metadata'] = json.loads(patient['metadata'])
                    except json.JSONDecodeError:
                        logger.warning("Không thể parse metadata cho bệnh nhân ID: %s", patient['id'])
            
            logger.info("Đã lấy danh sách %d bệnh nhân", len(patients))
            return patients
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách bệnh nhân: %s", str(e))
            raise DatabaseError("Không thể lấy danh sách bệnh nhân") from e

    def search_patients(self, name=None, birth_date=None, gender=None, limit=100, offset=0):
        """
        Tìm kiếm bệnh nhân theo các tiêu chí.

        Args:
            name (str, optional): Tên hoặc một phần tên của bệnh nhân.
            birth_date (str, optional): Ngày sinh của bệnh nhân.
            gender (str, optional): Giới tính của bệnh nhân.
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách các bệnh nhân thỏa mãn tiêu chí tìm kiếm.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tìm kiếm.
        """
        try:
            conditions = []
            params = []
            
            if name:
                conditions.append("name LIKE ?")
                params.append("%" + name + "%")
            
            if birth_date:
                conditions.append("birth_date = ?")
                params.append(birth_date)
            
            if gender:
                conditions.append("gender = ?")
                params.append(gender)
            
            # Xây dựng câu truy vấn
            query = "SELECT * FROM patients"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY name LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Thực hiện truy vấn
            results = self.db.execute_query(query, params, fetchall=True)
            
            # Xử lý kết quả
            patients = []
            for row in results:
                patient = {
                    'id': row[0],
                    'name': row[1],
                    'birth_date': row[2],
                    'gender': row[3],
                    'created_at': row[4],
                    'updated_at': row[5],
                    'metadata': json.loads(row[6]) if row[6] else None
                }
                patients.append(patient)
            
            logger.debug("Tìm thấy %d bệnh nhân thỏa mãn tiêu chí", len(patients))
            return patients
        except Exception as e:
            logger.error("Lỗi khi tìm kiếm bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể tìm kiếm bệnh nhân: %s" % str(e)) from e

    def count_patients(self, name=None, birth_date=None, gender=None):
        """
        Đếm số lượng bệnh nhân thỏa mãn tiêu chí tìm kiếm.

        Args:
            name (str, optional): Tên hoặc một phần tên của bệnh nhân.
            birth_date (str, optional): Ngày sinh của bệnh nhân.
            gender (str, optional): Giới tính của bệnh nhân.

        Returns:
            int: Số lượng bệnh nhân thỏa mãn tiêu chí.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình đếm.
        """
        try:
            conditions = []
            params = []
            
            if name:
                conditions.append("name LIKE ?")
                params.append("%" + name + "%")
            
            if birth_date:
                conditions.append("birth_date = ?")
                params.append(birth_date)
            
            if gender:
                conditions.append("gender = ?")
                params.append(gender)
            
            # Xây dựng câu truy vấn
            query = "SELECT COUNT(*) FROM patients"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Thực hiện truy vấn
            result = self.db.execute_query(query, params)
            
            # Xử lý kết quả
            count = result[0] if result else 0
            logger.debug("Số lượng bệnh nhân thỏa mãn tiêu chí: %d", count)
            return count
        except Exception as e:
            logger.error("Lỗi khi đếm bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể đếm bệnh nhân: %s" % str(e)) from e

    def get_all_patients_paged(self, limit=100, offset=0):
        """
        Lấy danh sách tất cả các bệnh nhân với phân trang.

        Args:
            limit (int, optional): Số lượng bệnh nhân tối đa cần lấy. Mặc định là 100.
            offset (int, optional): Vị trí bắt đầu lấy dữ liệu. Mặc định là 0.

        Returns:
            list: Danh sách bệnh nhân

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình lấy danh sách bệnh nhân
        """
        try:
            query = "SELECT * FROM patients ORDER BY name LIMIT ? OFFSET ?"
            params = (limit, offset)
            
            patients = self.db.execute_query(query, params, fetchall=True)
            
            # Parse metadata từ JSON nếu có
            for patient in patients:
                if patient.get('metadata'):
                    try:
                        patient['metadata'] = json.loads(patient['metadata'])
                    except json.JSONDecodeError:
                        logger.warning("Không thể parse metadata cho bệnh nhân ID: %s", patient['id'])
            
            logger.info("Đã lấy danh sách %d bệnh nhân (limit=%d, offset=%d)", len(patients), limit, offset)
            return patients
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách bệnh nhân: %s", str(e))
            raise DatabaseError("Không thể lấy danh sách bệnh nhân") from e

    def get_patient_studies(self, patient_id):
        """
        Lấy danh sách các study của bệnh nhân.

        Args:
            patient_id (str): ID của bệnh nhân.

        Returns:
            list: Danh sách các study của bệnh nhân.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM studies WHERE patient_id = ? ORDER BY created_at DESC"
            results = self.db.execute_query(query, (patient_id,), fetchall=True)
            
            studies = []
            for row in results:
                study = {
                    'id': row[0],
                    'patient_id': row[1],
                    'study_uid': row[2],
                    'study_date': row[3],
                    'study_description': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                studies.append(study)
            
            return studies
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách study của bệnh nhân: %s", str(e))
            raise DatabaseError("Không thể lấy danh sách study của bệnh nhân") from e

    def get_patient_plans(self, patient_id):
        """
        Lấy danh sách các kế hoạch điều trị của bệnh nhân.

        Args:
            patient_id (str): ID của bệnh nhân.

        Returns:
            list: Danh sách các kế hoạch điều trị của bệnh nhân.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM plans WHERE patient_id = ? ORDER BY created_at DESC"
            results = self.db.execute_query(query, (patient_id,), fetchall=True)
            
            plans = []
            for row in results:
                plan = {
                    'id': row[0],
                    'patient_id': row[1],
                    'study_uid': row[2],
                    'name': row[3],
                    'description': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                plans.append(plan)
            
            return plans
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách kế hoạch điều trị của bệnh nhân: %s", str(e))
            raise DatabaseError("Không thể lấy danh sách kế hoạch điều trị của bệnh nhân") from e

    def _delete_all_plans(self, patient_id):
        """
        Xóa tất cả các kế hoạch điều trị liên quan đến bệnh nhân.

        Args:
            patient_id (str): ID của bệnh nhân.

        Returns:
            bool: True nếu xóa thành công, False nếu có lỗi.

        Note:
            Phương thức này được sử dụng nội bộ trước khi xóa bệnh nhân.
        """
        try:
            # Xóa kế hoạch điều trị (plans)
            query = "DELETE FROM plans WHERE patient_id = ?"
            self.db.execute_update(query, (patient_id,))
            
            # Xóa cấu trúc (structures) nếu có
            query = "DELETE FROM structures WHERE patient_id = ?"
            self.db.execute_update(query, (patient_id,))
            
            # Xóa liều (doses) nếu có
            query = "DELETE FROM doses WHERE patient_id = ?"
            self.db.execute_update(query, (patient_id,))
            
            # Xóa các nghiên cứu (studies) nếu có
            query = "DELETE FROM studies WHERE patient_id = ?"
            self.db.execute_update(query, (patient_id,))
            
            # Xóa các series nếu có
            query = "DELETE FROM series WHERE patient_id = ?"
            self.db.execute_update(query, (patient_id,))
            
            # Xóa các instances nếu có
            query = "DELETE FROM instances WHERE patient_id = ?"
            self.db.execute_update(query, (patient_id,))
            
            logger.info("Đã xóa tất cả kế hoạch điều trị liên quan đến bệnh nhân ID: %s", patient_id)
            return True
        except Exception as e:
            logger.error("Lỗi khi xóa kế hoạch điều trị: %s", str(e))
            # Không raise lỗi ở đây vì đây là phương thức nội bộ, lỗi sẽ được xử lý ở phương thức gọi
            return False