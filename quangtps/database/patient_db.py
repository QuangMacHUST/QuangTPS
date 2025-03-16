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
        Tạo bản ghi bệnh nhân mới trong cơ sở dữ liệu.

        Args:
            name (str): Tên của bệnh nhân.
            birth_date (str, optional): Ngày sinh của bệnh nhân (định dạng ISO).
            gender (str, optional): Giới tính của bệnh nhân.
            metadata (dict, optional): Metadata bổ sung của bệnh nhân.

        Returns:
            str: ID của bệnh nhân vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo bệnh nhân.
        """
        try:
            patient_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO patients (id, name, birth_date, gender, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (patient_id, name, birth_date, gender, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo bệnh nhân mới với ID: {patient_id}")
            
            return patient_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể tạo bệnh nhân: {str(e)}")

    def get_patient(self, patient_id):
        """
        Lấy thông tin bệnh nhân theo ID.

        Args:
            patient_id (str): ID của bệnh nhân.

        Returns:
            dict: Thông tin bệnh nhân hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM patients WHERE id = ?"
            result = self.db.execute_query(query, (patient_id,), fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy bệnh nhân với ID: {patient_id}")
                return None
            
            patient = {
                'id': result[0],
                'name': result[1],
                'birth_date': result[2],
                'gender': result[3],
                'created_at': result[4],
                'updated_at': result[5],
                'metadata': json.loads(result[6]) if result[6] else None
            }
            
            return patient
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin bệnh nhân: {str(e)}")

    def update_patient(self, patient_id, name=None, birth_date=None, gender=None, metadata=None):
        """
        Cập nhật thông tin bệnh nhân.

        Args:
            patient_id (str): ID của bệnh nhân.
            name (str, optional): Tên mới của bệnh nhân.
            birth_date (str, optional): Ngày sinh mới của bệnh nhân.
            gender (str, optional): Giới tính mới của bệnh nhân.
            metadata (dict, optional): Metadata mới của bệnh nhân.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của bệnh nhân
            current_patient = self.get_patient(patient_id)
            if not current_patient:
                logger.warning(f"Không thể cập nhật bệnh nhân không tồn tại: {patient_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if name is not None:
                update_data['name'] = name
            if birth_date is not None:
                update_data['birth_date'] = birth_date
            if gender is not None:
                update_data['gender'] = gender
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_patient.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho bệnh nhân: {patient_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE patients SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(patient_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật bệnh nhân: {patient_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật bệnh nhân: {str(e)}")

    def delete_patient(self, patient_id):
        """
        Xóa bệnh nhân khỏi cơ sở dữ liệu.

        Args:
            patient_id (str): ID của bệnh nhân.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra bệnh nhân có tồn tại không
            patient = self.get_patient(patient_id)
            if not patient:
                logger.warning(f"Không thể xóa bệnh nhân không tồn tại: {patient_id}")
                return False
            
            # Thực hiện xóa bệnh nhân và tất cả dữ liệu liên quan
            self.db.execute_transaction([
                ("DELETE FROM plans WHERE patient_id = ?", (patient_id,)),
                ("DELETE FROM patients WHERE id = ?", (patient_id,))
            ])
            
            logger.info(f"Đã xóa bệnh nhân: {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể xóa bệnh nhân: {str(e)}")

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
                params.append(f"%{name}%")
            
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
            
            return patients
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể tìm kiếm bệnh nhân: {str(e)}")

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
                params.append(f"%{name}%")
            
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
            result = self.db.execute_query(query, params, fetchone=True)
            
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Lỗi khi đếm bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể đếm bệnh nhân: {str(e)}")

    def get_all_patients(self, limit=100, offset=0):
        """
        Lấy danh sách tất cả bệnh nhân.

        Args:
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách tất cả bệnh nhân.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        return self.search_patients(limit=limit, offset=offset)

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
            logger.error(f"Lỗi khi lấy danh sách study của bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách study của bệnh nhân: {str(e)}")

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
            logger.error(f"Lỗi khi lấy danh sách kế hoạch điều trị của bệnh nhân: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách kế hoạch điều trị của bệnh nhân: {str(e)}")