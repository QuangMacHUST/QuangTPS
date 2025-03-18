"""
Kết nối cơ sở dữ liệu cho QuangTPS.
Cung cấp các lớp và phương thức để tương tác với cơ sở dữ liệu.
"""

import os
import sqlite3
import logging
from typing import List, Optional, Tuple

from quangtps.core.config import Config
from quangtps.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class DBConnector:
    """
    Lớp kết nối và quản lý cơ sở dữ liệu SQLite cho hệ thống QuangTPS.
    
    Lớp này triển khai mẫu thiết kế Singleton để đảm bảo chỉ có một kết nối
    đến cơ sở dữ liệu trong suốt thời gian chạy ứng dụng.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Tạo một thể hiện duy nhất của lớp."""
        if cls._instance is None:
            cls._instance = super(DBConnector, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Trả về instance duy nhất của DBConnector"""
        if cls._instance is None:
            return cls()
        return cls._instance
    
    def __init__(self):
        """Khởi tạo kết nối cơ sở dữ liệu."""
        if not self._initialized:
            # Khởi tạo tất cả các thuộc tính của lớp
            self._initialized = True
            self.config = Config.get_instance()
            self.db_dir = os.path.join(self.config.data_dir, 'database')
            self.db_path = os.path.join(self.db_dir, 'quangtps.db')
            self.connection = None
            
            # Thực hiện thiết lập thực tế
            self._setup_database()
            self._connect()
            
            # Tạo bảng nếu chưa tồn tại
            self._create_tables()
    
    def _setup_database(self):
        """Đảm bảo thư mục cơ sở dữ liệu tồn tại."""
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            logger.info("Đã tạo thư mục cơ sở dữ liệu: %s", self.db_dir)
    
    def _connect(self):
        """Kết nối đến cơ sở dữ liệu SQLite"""
        try:
            # Gán giá trị cho thuộc tính đã được khởi tạo trong __init__
            if not self.connection:
                self.connection = sqlite3.connect(self.db_path)
                self.connection.row_factory = sqlite3.Row
                logger.info("Đã kết nối đến cơ sở dữ liệu: %s", self.db_path)
        except Exception as e:
            logger.error("Lỗi kết nối cơ sở dữ liệu: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi kết nối cơ sở dữ liệu: %s" % str(e)) from e
    
    def _disconnect(self):
        """Đóng kết nối cơ sở dữ liệu"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Đã ngắt kết nối cơ sở dữ liệu")
    
    def _create_tables(self):
        """Tạo các bảng cơ sở dữ liệu nếu chưa tồn tại"""
        try:
            # Tạo bảng patients
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    birth_date TEXT,
                    gender TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT
                )
            ''')
            
            # Tạo bảng studies
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    description TEXT,
                    study_date TEXT,
                    study_time TEXT,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')
            
            # Tạo bảng series
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS series (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    modality TEXT,
                    description TEXT,
                    series_date TEXT,
                    series_time TEXT,
                    metadata TEXT,
                    FOREIGN KEY (study_id) REFERENCES studies (id) ON DELETE CASCADE,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')
            
            # Tạo bảng instances (images)
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS instances (
                    id TEXT PRIMARY KEY,
                    series_id TEXT NOT NULL,
                    study_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    instance_number INTEGER,
                    file_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY (series_id) REFERENCES series (id) ON DELETE CASCADE,
                    FOREIGN KEY (study_id) REFERENCES studies (id) ON DELETE CASCADE,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')
            
            # Tạo bảng cấu trúc (structures)
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS structures (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    structure_type TEXT,
                    color TEXT,
                    opacity REAL,
                    data TEXT,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')
            
            # Tạo bảng kế hoạch (plans)
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    study_id TEXT,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE,
                    FOREIGN KEY (study_id) REFERENCES studies (id) ON DELETE CASCADE
                )
            ''')
            
            # Tạo bảng liều (doses)
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS doses (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    dose_type TEXT,
                    dose_unit TEXT,
                    dose_data TEXT,
                    metadata TEXT,
                    FOREIGN KEY (plan_id) REFERENCES plans (id) ON DELETE CASCADE,
                    FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                )
            ''')
            
            # Lưu các thay đổi
            self.connection.commit()
            logger.info("Đã tạo các bảng cơ sở dữ liệu nếu chưa tồn tại")
            
        except Exception as e:
            logger.error("Lỗi khi tạo bảng cơ sở dữ liệu: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi tạo bảng cơ sở dữ liệu: %s" % str(e)) from e
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, fetchall: bool = False):
        """
        Thực thi một câu truy vấn SQL và trả về kết quả.
        
        Parameters:
            query (str): Câu lệnh SQL
            params (tuple, optional): Tham số cho câu lệnh
            fetchall (bool): True để lấy tất cả các kết quả, False để lấy dòng đầu tiên
        
        Returns:
            list/dict: Kết quả của câu truy vấn
        """
        if not self.connection:
            self._connect()
            
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            if fetchall:
                result = cursor.fetchall()
                return result
            else:
                result = cursor.fetchone()
                return result
        except Exception as e:
            self.connection.rollback()
            logger.error("Lỗi thực thi truy vấn: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi thực thi truy vấn: %s" % str(e)) from e
            
    def execute_insert(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Thực thi một câu lệnh chèn dữ liệu.
        
        Parameters:
            query (str): Câu lệnh SQL INSERT
            params (tuple, optional): Tham số cho câu lệnh
            
        Returns:
            int: ID của dòng vừa chèn
        """
        if not self.connection:
            self._connect()
            
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            self.connection.commit()
            return cursor.lastrowid
        except Exception as e:
            self.connection.rollback()
            logger.error("Lỗi chèn dữ liệu: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi chèn dữ liệu: %s" % str(e)) from e
            
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Thực thi một câu lệnh cập nhật dữ liệu.
        
        Parameters:
            query (str): Câu lệnh SQL UPDATE hoặc DELETE
            params (tuple, optional): Tham số cho câu lệnh
            
        Returns:
            int: Số dòng bị ảnh hưởng
        """
        if not self.connection:
            self._connect()
            
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            self.connection.rollback()
            logger.error("Lỗi cập nhật dữ liệu: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi cập nhật dữ liệu: %s" % str(e)) from e
            
    def execute_transaction(self, queries: List[Tuple[str, Optional[Tuple]]]) -> bool:
        """
        Thực thi một giao dịch với nhiều câu lệnh SQL.
        
        Parameters:
            queries (list): Danh sách các tuple (query, params)
            
        Returns:
            bool: True nếu giao dịch thành công, False nếu thất bại
        """
        if not self.connection:
            self._connect()
            
        # Đánh dấu điểm bắt đầu giao dịch
        try:
            # Thực thi từng câu lệnh trong giao dịch
            cursor = self.connection.cursor()
            
            for query, params in queries:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                    
            # Commit giao dịch
            self.connection.commit()
            logger.info("Đã thực thi giao dịch thành công với %d câu lệnh", len(queries))
            return True
        except Exception as e:
            # Rollback nếu có lỗi
            self.connection.rollback()
            logger.error("Lỗi thực thi giao dịch: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi thực thi giao dịch: %s" % str(e)) from e
            
    def close(self):
        """Đóng kết nối cơ sở dữ liệu"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Đã đóng kết nối cơ sở dữ liệu")