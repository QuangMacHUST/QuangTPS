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
            self.conn = None
            
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
            if not self.conn:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row
                logger.info("Đã kết nối đến cơ sở dữ liệu: %s", self.db_path)
        except Exception as e:
            logger.error("Lỗi kết nối cơ sở dữ liệu: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi kết nối cơ sở dữ liệu: %s" % str(e)) from e
    
    def _disconnect(self):
        """Ngắt kết nối đến cơ sở dữ liệu."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
            logger.debug("Đã ngắt kết nối cơ sở dữ liệu")
    
    def connection(self):
        """Trả về kết nối đến cơ sở dữ liệu."""
        if not self.conn:
            self._connect()
        return self.conn
    
    def _create_tables(self):
        """
        Tạo các bảng cần thiết cho cơ sở dữ liệu nếu chưa tồn tại.
        """
        # Tạo bảng patients
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                birth_date TEXT,
                gender TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_date TEXT,
                metadata TEXT
            )
        """)
        
        # Tạo bảng studies
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS studies (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                description TEXT,
                date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        # Tạo bảng series
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS series (
                id TEXT PRIMARY KEY,
                study_id TEXT NOT NULL,
                description TEXT,
                modality TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                data_path TEXT,
                FOREIGN KEY (study_id) REFERENCES studies (id)
            )
        """)
        
        # Tạo bảng files để lưu trữ danh sách file thuộc về một series
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (series_id) REFERENCES series (id)
            )
        """)
        
        # Tạo bảng structures
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS structures (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT,
                color TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                file_path TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        # Tạo bảng structure_sets
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS structure_sets (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                study_id TEXT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id),
                FOREIGN KEY (study_id) REFERENCES studies (id)
            )
        """)
        
        # Tạo bảng structure_set_items để lưu trữ cấu trúc nào thuộc tập hợp nào
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS structure_set_items (
                structure_set_id TEXT NOT NULL,
                structure_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (structure_set_id, structure_id),
                FOREIGN KEY (structure_set_id) REFERENCES structure_sets (id),
                FOREIGN KEY (structure_id) REFERENCES structures (id)
            )
        """)
        
        # Tạo bảng plans
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                technique TEXT,
                prescribed_dose REAL,
                fraction_count INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        """)
        
        # Tạo bảng beams
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS beams (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                name TEXT NOT NULL,
                gantry_angle REAL,
                collimator_angle REAL,
                couch_angle REAL,
                isocenter_x REAL,
                isocenter_y REAL,
                isocenter_z REAL,
                monitor_units REAL,
                weight REAL,
                energy TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (plan_id) REFERENCES plans (id)
            )
        """)
        
        # Tạo bảng dose_distributions
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS dose_distributions (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                beam_id TEXT,
                type TEXT NOT NULL,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (plan_id) REFERENCES plans (id),
                FOREIGN KEY (beam_id) REFERENCES beams (id)
            )
        """)
        
        # Tạo bảng dvh
        self.execute_query("""
            CREATE TABLE IF NOT EXISTS dvh (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                structure_id TEXT NOT NULL,
                file_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (plan_id) REFERENCES plans (id),
                FOREIGN KEY (structure_id) REFERENCES structures (id)
            )
        """)
    
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
        if not self.conn:
            self._connect()
            
        cursor = self.conn.cursor()
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
            self.conn.rollback()
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