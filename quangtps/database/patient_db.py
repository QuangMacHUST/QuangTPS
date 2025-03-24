"""
Quản lý cơ sở dữ liệu bệnh nhân.
"""

import json
import uuid
import logging
from datetime import datetime, date
from typing import List, Optional

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
        self._create_tables()

    def _create_tables(self):
        """Tạo các bảng cần thiết"""
        # Lấy kết nối từ DBConnector
        conn = self.db.connection()
        cursor = conn.cursor()
        
        # Bảng bệnh nhân
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dob TEXT NOT NULL,
                gender TEXT NOT NULL,
                address TEXT,
                phone TEXT,
                email TEXT,
                diagnosis TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Bảng lịch sử khám
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                visit_date TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
                    ON DELETE CASCADE
            )
        """)
        
        conn.commit()

    def create_patient(self, name, birth_date=None, gender=None, metadata=None):
        """
        Tạo một bệnh nhân mới trong cơ sở dữ liệu.
        
        Args:
            name (str): Tên của bệnh nhân
            birth_date (str or datetime, optional): Ngày sinh của bệnh nhân
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
            
            # Xử lý ngày sinh nếu là đối tượng datetime
            if birth_date and isinstance(birth_date, datetime):
                birth_date = birth_date.strftime('%Y-%m-%d')
                
            # Chuyển đổi metadata thành JSON
            metadata_json = json.dumps(metadata) if metadata else "{}"
            
            # Thử chèn với created_date trước
            try:
                query = '''
                    INSERT INTO patients (id, name, dob, gender, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
                
                self.db.execute_query(
                    query, 
                    (patient_id, name, birth_date, gender, created_at, created_at, metadata_json)
                )
            except Exception as insert_error:
                # Nếu gặp lỗi, ghi log và raise lỗi
                logger.error("Lỗi khi chèn bệnh nhân: %s", str(insert_error), exc_info=True)
                raise
            
            logger.info("Đã tạo bệnh nhân mới với ID: %s", patient_id)
            return patient_id
            
        except Exception as e:
            logger.error("Lỗi khi tạo bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi tạo bệnh nhân: %s" % str(e)) from e
            
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
            # Thực hiện truy vấn
            query = "SELECT * FROM patients WHERE id = ?"
            patient_data = self.db.execute_query(query, (patient_id,), fetchall=False)
            
            if not patient_data:
                logger.warning("Không tìm thấy bệnh nhân với ID: %s", patient_id)
                return None
            
            # Chuyển đổi dữ liệu về dict
            patient_dict = dict(patient_data)
            
            # Giải mã metadata từ JSON
            if 'metadata' in patient_dict and patient_dict['metadata']:
                try:
                    patient_dict['metadata'] = json.loads(patient_dict['metadata'])
                except json.JSONDecodeError:
                    patient_dict['metadata'] = {}
            else:
                patient_dict['metadata'] = {}
            
            logger.info("Đã lấy thông tin bệnh nhân: %s", patient_id)
            return patient_dict
            
        except Exception as e:
            logger.error("Lỗi khi lấy thông tin bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lấy thông tin bệnh nhân: %s" % str(e)) from e
            
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
            # Lấy thông tin hiện tại của bệnh nhân
            current_patient = self.get_patient(patient_id)
            if not current_patient:
                logger.warning("Không thể cập nhật: Bệnh nhân không tồn tại với ID: %s", patient_id)
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            updated_at = datetime.now().isoformat()
            
            # Chuẩn bị metadata
            if metadata is None:
                # Sử dụng metadata hiện có
                metadata_json = current_patient.get('metadata', {})
                if isinstance(metadata_json, dict):
                    metadata_json = json.dumps(metadata_json)
            else:
                # Sử dụng metadata mới
                metadata_json = json.dumps(metadata)
            
            # Chuẩn bị các giá trị cập nhật
            set_clauses = []
            params = []
            
            if name is not None:
                set_clauses.append("name = ?")
                params.append(name)
            
            if birth_date is not None:
                set_clauses.append("dob = ?")
                params.append(birth_date)
            
            if gender is not None:
                set_clauses.append("gender = ?")
                params.append(gender)
            
            if metadata is not None:
                set_clauses.append("metadata = ?")
                params.append(metadata_json)
            
            # Thêm ngày cập nhật mới nhất
            set_clauses.append("updated_at = ?")
            params.append(updated_at)
            
            # Nếu không có gì cần cập nhật, trả về True (coi như đã cập nhật)
            if not set_clauses:
                logger.info("Không có thông tin nào cần cập nhật cho bệnh nhân %s", patient_id)
                return True
            
            # Tạo câu truy vấn UPDATE
            query = f"UPDATE patients SET {', '.join(set_clauses)} WHERE id = ?"
            params.append(patient_id)
            
            # Thực hiện câu truy vấn
            self.db.execute_query(query, tuple(params))
            
            logger.info("Đã cập nhật thông tin bệnh nhân %s", patient_id)
            return True
            
        except Exception as e:
            logger.error("Lỗi khi cập nhật bệnh nhân: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi cập nhật bệnh nhân: %s" % str(e)) from e
            
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
            results = self.db.execute_query(query, fetchall=True)
            
            # Chuyển đổi sqlite3.Row thành dictionaries
            patients = []
            for row in results:
                patient = dict(row)
                
                # Parse metadata từ JSON nếu có
                if patient.get('metadata'):
                    try:
                        patient['metadata'] = json.loads(patient['metadata'])
                    except json.JSONDecodeError:
                        logger.warning("Không thể parse metadata cho bệnh nhân ID: %s", patient['id'])
                
                patients.append(patient)
            
            logger.info("Đã lấy danh sách %d bệnh nhân", len(patients))
            return patients
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách bệnh nhân: %s", str(e))
            raise DatabaseError("Không thể lấy danh sách bệnh nhân") from e

    def search_patients(self, query=None, limit=100, offset=0):
        """
        Tìm kiếm bệnh nhân theo các tiêu chí.
        
        Args:
            query (dict, optional): Từ điển các tiêu chí tìm kiếm
                Các khóa có thể là: 'name', 'gender', 'birth_date', 'metadata'
            limit (int, optional): Số lượng kết quả tối đa
            offset (int, optional): Vị trí bắt đầu
            
        Returns:
            list: Danh sách các bệnh nhân phù hợp
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi truy vấn
        """
        try:
            # Chuẩn bị câu truy vấn cơ sở
            base_query = "SELECT * FROM patients"
            where_clauses = []
            params = []
            
            # Thêm các điều kiện tìm kiếm nếu có
            if query:
                if 'name' in query and query['name']:
                    where_clauses.append("name LIKE ?")
                    params.append(f"%{query['name']}%")
                
                if 'gender' in query and query['gender']:
                    where_clauses.append("gender = ?")
                    params.append(query['gender'])
                
                if 'birth_date' in query and query['birth_date']:
                    where_clauses.append("dob = ?")
                    params.append(query['birth_date'])
                
                # Tìm kiếm trong metadata (cần xử lý đặc biệt)
                if 'metadata' in query and query['metadata']:
                    for key, value in query['metadata'].items():
                        where_clauses.append("metadata LIKE ?")
                        params.append(f"%\"{key}\":\"{value}\"%")
                        
                # Tìm theo DICOM ID (lưu trong metadata)
                if 'dicom_id' in query and query['dicom_id']:
                    where_clauses.append("metadata LIKE ?")
                    params.append(f"%\"dicom_id\":\"{query['dicom_id']}\"%")
            
            # Hoàn thiện câu truy vấn
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
            
            # Thêm giới hạn kết quả
            base_query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.append(limit)
            params.append(offset)
            
            # Thực hiện truy vấn
            results = self.db.execute_query(base_query, tuple(params))
            
            # Chuyển đổi kết quả thành danh sách dict
            patients = []
            for row in results:
                patient_dict = dict(row)
                
                # Giải mã metadata từ JSON
                if 'metadata' in patient_dict and patient_dict['metadata']:
                    try:
                        patient_dict['metadata'] = json.loads(patient_dict['metadata'])
                    except json.JSONDecodeError:
                        patient_dict['metadata'] = {}
                else:
                    patient_dict['metadata'] = {}
                
                patients.append(patient_dict)
                
            logger.info(f"Tìm thấy {len(patients)} bệnh nhân phù hợp")
            return patients
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm bệnh nhân: {str(e)}", exc_info=True)
            raise DatabaseError(f"Lỗi khi tìm kiếm bệnh nhân: {str(e)}") from e

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
                conditions.append("dob = ?")
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
            
            results = self.db.execute_query(query, params, fetchall=True)
            
            # Chuyển đổi sqlite3.Row thành dictionaries
            patients = []
            for row in results:
                patient = dict(row)
                
                # Parse metadata từ JSON nếu có
                if patient.get('metadata'):
                    try:
                        patient['metadata'] = json.loads(patient['metadata'])
                    except json.JSONDecodeError:
                        logger.warning("Không thể parse metadata cho bệnh nhân ID: %s", patient['id'])
                
                patients.append(patient)
            
            logger.info("Đã lấy danh sách %d bệnh nhân (limit=%d, offset=%d)", len(patients), limit, offset)
            return patients
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách bệnh nhân: %s", str(e))
            raise DatabaseError("Không thể lấy danh sách bệnh nhân") from e

    def get_patient_studies(self, patient_id, include_series=False):
        """
        Lấy danh sách nghiên cứu của một bệnh nhân.
        
        Args:
            patient_id (str): ID của bệnh nhân
            include_series (bool): Có bao gồm thông tin series không
            
        Returns:
            list: Danh sách các nghiên cứu
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi truy vấn
        """
        try:
            # Kiểm tra xem bệnh nhân có tồn tại không
            patient = self.get_patient(patient_id)
            if not patient:
                logger.warning(f"Bệnh nhân không tồn tại với ID: {patient_id}")
                return []
            
            # Truy vấn danh sách nghiên cứu
            query = "SELECT * FROM studies WHERE patient_id = ? ORDER BY date DESC"
            studies_data = self.db.execute_query(query, (patient_id,))
            
            # Chuyển đổi kết quả thành danh sách dict
            studies = []
            for row in studies_data:
                study_dict = dict(row)
                
                # Giải mã metadata từ JSON
                if 'metadata' in study_dict and study_dict['metadata']:
                    try:
                        study_dict['metadata'] = json.loads(study_dict['metadata'])
                    except json.JSONDecodeError:
                        study_dict['metadata'] = {}
                else:
                    study_dict['metadata'] = {}
                
                # Thêm thông tin series nếu yêu cầu
                if include_series:
                    study_dict['series'] = self.get_study_series(study_dict['id'])
                
                studies.append(study_dict)
                
            logger.info(f"Đã lấy {len(studies)} nghiên cứu của bệnh nhân {patient_id}")
            return studies
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách nghiên cứu: {str(e)}", exc_info=True)
            raise DatabaseError(f"Lỗi khi lấy danh sách nghiên cứu: {str(e)}") from e
            
    def get_study_series(self, study_id):
        """
        Lấy danh sách series của một nghiên cứu.
        
        Args:
            study_id (str): ID của nghiên cứu
            
        Returns:
            list: Danh sách các series
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi truy vấn
        """
        try:
            # Truy vấn danh sách series
            query = "SELECT * FROM series WHERE study_id = ?"
            series_data = self.db.execute_query(query, (study_id,))
            
            # Chuyển đổi kết quả thành danh sách dict
            series_list = []
            for row in series_data:
                series_dict = dict(row)
                
                # Giải mã metadata từ JSON
                if 'metadata' in series_dict and series_dict['metadata']:
                    try:
                        series_dict['metadata'] = json.loads(series_dict['metadata'])
                    except json.JSONDecodeError:
                        series_dict['metadata'] = {}
                else:
                    series_dict['metadata'] = {}
                
                # Lấy danh sách file_paths
                query = "SELECT file_path FROM files WHERE series_id = ?"
                files_data = self.db.execute_query(query, (series_dict['id'],))
                series_dict['file_paths'] = [row['file_path'] for row in files_data]
                
                series_list.append(series_dict)
                
            return series_list
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách series: {str(e)}", exc_info=True)
            raise DatabaseError(f"Lỗi khi lấy danh sách series: {str(e)}") from e

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

    def add_study_to_patient(self, patient_id, study):
        """
        Thêm một nghiên cứu mới cho bệnh nhân.
        
        Args:
            patient_id (str): ID của bệnh nhân
            study (Study): Đối tượng nghiên cứu
            
        Returns:
            str: ID của nghiên cứu
            
        Raises:
            DatabaseError: Nếu có lỗi xảy ra khi thêm nghiên cứu
        """
        try:
            # Kiểm tra xem bệnh nhân có tồn tại không
            patient = self.get_patient(patient_id)
            if not patient:
                raise ValueError(f"Bệnh nhân không tồn tại với ID: {patient_id}")
            
            # Chuyển đổi Study thành dict
            study_data = study.to_dict()
            
            # Lưu nghiên cứu vào bảng studies
            query = '''
                INSERT INTO studies (id, description, date, patient_id, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            
            self.db.execute_query(
                query, 
                (
                    study_data['id'],
                    study_data['description'],
                    study_data['date'],
                    patient_id,
                    study_data['created_at'],
                    study_data['updated_at'],
                    json.dumps(study_data['metadata'])
                )
            )
            
            # Lưu từng series
            for series_data in study_data['series']:
                query = '''
                    INSERT INTO series (id, description, modality, study_id, created_at, updated_at, metadata, data_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
                
                self.db.execute_query(
                    query, 
                    (
                        series_data['id'],
                        series_data['description'],
                        series_data['modality'],
                        study_data['id'],
                        series_data['created_at'],
                        series_data['updated_at'],
                        json.dumps(series_data['metadata']),
                        series_data['data_path']
                    )
                )
                
                # Lưu danh sách files
                for file_path in series_data['file_paths']:
                    query = '''
                        INSERT INTO files (series_id, file_path, created_at)
                        VALUES (?, ?, ?)
                    '''
                    
                    self.db.execute_query(
                        query, 
                        (
                            series_data['id'],
                            file_path,
                            datetime.now().isoformat()
                        )
                    )
            
            logger.info(f"Đã thêm nghiên cứu {study_data['id']} cho bệnh nhân {patient_id}")
            return study_data['id']
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm nghiên cứu: {str(e)}", exc_info=True)
            raise DatabaseError(f"Lỗi khi thêm nghiên cứu: {str(e)}") from e

    def add_history(self, patient_id: str, description: str) -> None:
        """Thêm lịch sử khám cho bệnh nhân"""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patient_history (
                    patient_id, visit_date, description
                ) VALUES (?, CURRENT_DATE, ?)
            """, (patient_id, description))
            conn.commit()
    
    def get_history(self, patient_id: str) -> List[dict]:
        """Lấy lịch sử khám của bệnh nhân"""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT visit_date, description
                FROM patient_history
                WHERE patient_id = ?
                ORDER BY visit_date DESC
            """, (patient_id,))
            
            return [
                {
                    "visit_date": date.fromisoformat(row[0]),
                    "description": row[1]
                }
                for row in cursor.fetchall()
            ]