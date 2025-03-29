#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý cấu trúc cơ sở dữ liệu (schema) cho hệ thống QuangTPS.

Module này cung cấp định nghĩa cho tất cả các bảng trong cơ sở dữ liệu,
bao gồm các trường, kiểu dữ liệu, khóa ngoại và các ràng buộc.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Định nghĩa các kiểu dữ liệu
class DataType(str, Enum):
    """Enum định nghĩa các kiểu dữ liệu cho cơ sở dữ liệu."""
    INTEGER = "INTEGER"
    REAL = "REAL"
    TEXT = "TEXT"
    BLOB = "BLOB"
    BOOLEAN = "INTEGER"  # SQLite không có kiểu Boolean, sử dụng INTEGER (0/1)
    TIMESTAMP = "TEXT"  # ISO8601 timestamp: YYYY-MM-DD HH:MM:SS.SSS
    JSON = "TEXT"  # JSON được lưu dưới dạng TEXT

# Định nghĩa các ràng buộc
class Constraint(str, Enum):
    """Enum định nghĩa các ràng buộc cho cơ sở dữ liệu."""
    NOT_NULL = "NOT NULL"
    UNIQUE = "UNIQUE"
    PRIMARY_KEY = "PRIMARY KEY"
    FOREIGN_KEY = "REFERENCES"
    DEFAULT = "DEFAULT"
    CHECK = "CHECK"

# Định nghĩa các bảng
SCHEMA_TABLES = {
    # Bảng bệnh nhân
    "patient": {
        "columns": {
            "patient_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "name": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "gender": {"type": DataType.TEXT},
            "birth_date": {"type": DataType.TIMESTAMP},
            "address": {"type": DataType.TEXT},
            "phone": {"type": DataType.TEXT},
            "email": {"type": DataType.TEXT},
            "medical_record_num": {"type": DataType.TEXT, "constraints": [Constraint.UNIQUE]},
            "creation_date": {"type": DataType.TIMESTAMP, "constraints": [Constraint.NOT_NULL]},
            "modification_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["medical_record_num", "name"]
    },

    # Bảng nghiên cứu (study)
    "study": {
        "columns": {
            "study_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "patient_id": {"type": DataType.TEXT, "constraints": ["REFERENCES patient(patient_id)"]},
            "description": {"type": DataType.TEXT},
            "date": {"type": DataType.TIMESTAMP, "constraints": [Constraint.NOT_NULL]},
            "referring_physician": {"type": DataType.TEXT},
            "institution": {"type": DataType.TEXT},
            "study_uid": {"type": DataType.TEXT, "constraints": [Constraint.UNIQUE]},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["patient_id", "date"]
    },
    
    # Bảng chuỗi (series)
    "series": {
        "columns": {
            "series_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "study_id": {"type": DataType.TEXT, "constraints": ["REFERENCES study(study_id)"]},
            "modality": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "description": {"type": DataType.TEXT},
            "date": {"type": DataType.TIMESTAMP},
            "series_uid": {"type": DataType.TEXT, "constraints": [Constraint.UNIQUE]},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["study_id", "modality"]
    },
    
    # Bảng hình ảnh (image)
    "image": {
        "columns": {
            "image_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "series_id": {"type": DataType.TEXT, "constraints": ["REFERENCES series(series_id)"]},
            "instance_number": {"type": DataType.INTEGER},
            "position": {"type": DataType.JSON},  # [x, y, z]
            "orientation": {"type": DataType.JSON},  # [x1, y1, z1, x2, y2, z2]
            "pixel_spacing": {"type": DataType.JSON},  # [x, y]
            "slice_thickness": {"type": DataType.REAL},
            "rows": {"type": DataType.INTEGER},
            "columns": {"type": DataType.INTEGER},
            "pixel_data": {"type": DataType.BLOB},
            "sop_instance_uid": {"type": DataType.TEXT, "constraints": [Constraint.UNIQUE]},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["series_id", "instance_number"]
    },
    
    # Bảng cấu trúc (structure)
    "structure": {
        "columns": {
            "structure_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "series_id": {"type": DataType.TEXT, "constraints": ["REFERENCES series(series_id)"]},
            "name": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "color": {"type": DataType.JSON},  # [r, g, b]
            "type": {"type": DataType.TEXT},  # PTV, CTV, OAR, etc.
            "description": {"type": DataType.TEXT},
            "creation_date": {"type": DataType.TIMESTAMP},
            "modification_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["series_id", "name"]
    },
    
    # Bảng đường viền (contour)
    "contour": {
        "columns": {
            "contour_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "structure_id": {"type": DataType.TEXT, "constraints": ["REFERENCES structure(structure_id)"]},
            "slice_index": {"type": DataType.INTEGER},
            "points": {"type": DataType.JSON},  # [[x1, y1, z1], [x2, y2, z2], ...]
            "closed": {"type": DataType.BOOLEAN},
            "creation_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["structure_id", "slice_index"]
    },
    
    # Bảng kế hoạch điều trị (plan)
    "plan": {
        "columns": {
            "plan_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "series_id": {"type": DataType.TEXT, "constraints": ["REFERENCES series(series_id)"]},
            "name": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "description": {"type": DataType.TEXT},
            "technique": {"type": DataType.TEXT},  # 3DCRT, IMRT, VMAT, etc.
            "energy": {"type": DataType.TEXT},  # 6MV, 10MV, etc.
            "creation_date": {"type": DataType.TIMESTAMP},
            "approval_status": {"type": DataType.TEXT},  # Draft, Under Review, Approved, etc.
            "approved_by": {"type": DataType.TEXT},
            "approval_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["series_id", "name"]
    },
    
    # Bảng chùm tia (beam)
    "beam": {
        "columns": {
            "beam_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "plan_id": {"type": DataType.TEXT, "constraints": ["REFERENCES plan(plan_id)"]},
            "name": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "description": {"type": DataType.TEXT},
            "gantry_angle": {"type": DataType.REAL},
            "collimator_angle": {"type": DataType.REAL},
            "couch_angle": {"type": DataType.REAL},
            "sad": {"type": DataType.REAL},  # Source-to-Axis Distance
            "energy": {"type": DataType.TEXT},  # 6MV, 10MV, etc.
            "dose_rate": {"type": DataType.REAL},
            "monitor_units": {"type": DataType.REAL},
            "isocenter": {"type": DataType.JSON},  # [x, y, z]
            "field_size": {"type": DataType.JSON},  # [width, height]
            "beam_type": {"type": DataType.TEXT},  # Static, Arc, etc.
            "beam_modifiers": {"type": DataType.JSON},  # List of modifiers
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["plan_id", "name"]
    },
    
    # Bảng fluence map (used for IMRT)
    "fluence_map": {
        "columns": {
            "fluence_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "beam_id": {"type": DataType.TEXT, "constraints": ["REFERENCES beam(beam_id)"]},
            "resolution": {"type": DataType.JSON},  # [x, y]
            "dimensions": {"type": DataType.JSON},  # [width, height]
            "data": {"type": DataType.BLOB},  # The fluence map data
            "creation_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["beam_id"]
    },
    
    # Bảng MLC (Multi-Leaf Collimator)
    "mlc": {
        "columns": {
            "mlc_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "beam_id": {"type": DataType.TEXT, "constraints": ["REFERENCES beam(beam_id)"]},
            "control_point": {"type": DataType.INTEGER},
            "leaf_positions": {"type": DataType.JSON},  # [[p1A, p1B], [p2A, p2B], ...]
            "jaw_positions": {"type": DataType.JSON},  # [x1, y1, x2, y2]
            "gantry_angle": {"type": DataType.REAL},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["beam_id", "control_point"]
    },
    
    # Bảng liều lượng (dose)
    "dose": {
        "columns": {
            "dose_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "plan_id": {"type": DataType.TEXT, "constraints": ["REFERENCES plan(plan_id)"]},
            "description": {"type": DataType.TEXT},
            "algorithm": {"type": DataType.TEXT},  # PB, CC, MC, etc.
            "dimensions": {"type": DataType.JSON},  # [x, y, z]
            "resolution": {"type": DataType.JSON},  # [dx, dy, dz]
            "data": {"type": DataType.BLOB},  # The dose grid data
            "scaling_factor": {"type": DataType.REAL},
            "creation_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["plan_id"]
    },
    
    # Bảng DVH (Dose-Volume Histogram)
    "dvh": {
        "columns": {
            "dvh_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "dose_id": {"type": DataType.TEXT, "constraints": ["REFERENCES dose(dose_id)"]},
            "structure_id": {"type": DataType.TEXT, "constraints": ["REFERENCES structure(structure_id)"]},
            "type": {"type": DataType.TEXT},  # Cumulative, Differential
            "bin_count": {"type": DataType.INTEGER},
            "max_dose": {"type": DataType.REAL},
            "min_dose": {"type": DataType.REAL},
            "mean_dose": {"type": DataType.REAL},
            "median_dose": {"type": DataType.REAL},
            "mode_dose": {"type": DataType.REAL},
            "std_dev": {"type": DataType.REAL},
            "volume": {"type": DataType.REAL},
            "data": {"type": DataType.JSON},  # [[dose1, volume1], [dose2, volume2], ...]
            "creation_date": {"type": DataType.TIMESTAMP},
        },
        "indices": ["dose_id", "structure_id"]
    },
    
    # Bảng đơn thuốc (prescription)
    "prescription": {
        "columns": {
            "prescription_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "plan_id": {"type": DataType.TEXT, "constraints": ["REFERENCES plan(plan_id)"]},
            "target_structure_id": {"type": DataType.TEXT, "constraints": ["REFERENCES structure(structure_id)"]},
            "prescribed_dose": {"type": DataType.REAL},
            "fractions": {"type": DataType.INTEGER},
            "dose_per_fraction": {"type": DataType.REAL},
            "method": {"type": DataType.TEXT},  # Isocentric, SSD, etc.
            "dose_type": {"type": DataType.TEXT},  # Max, Min, Mean, etc.
            "creation_date": {"type": DataType.TIMESTAMP},
            "approval_status": {"type": DataType.TEXT},
            "approved_by": {"type": DataType.TEXT},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["plan_id", "target_structure_id"]
    },
    
    # Bảng mục tiêu liều (dose_objective)
    "dose_objective": {
        "columns": {
            "objective_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "plan_id": {"type": DataType.TEXT, "constraints": ["REFERENCES plan(plan_id)"]},
            "structure_id": {"type": DataType.TEXT, "constraints": ["REFERENCES structure(structure_id)"]},
            "type": {"type": DataType.TEXT},  # Max Dose, Min Dose, Mean Dose, D95, V20, etc.
            "target_value": {"type": DataType.REAL},
            "priority": {"type": DataType.INTEGER},
            "weight": {"type": DataType.REAL},
            "is_constraint": {"type": DataType.BOOLEAN},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["plan_id", "structure_id"]
    },
    
    # Bảng chỉ số kế hoạch (plan_metric)
    "plan_metric": {
        "columns": {
            "metric_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "plan_id": {"type": DataType.TEXT, "constraints": ["REFERENCES plan(plan_id)"]},
            "name": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "value": {"type": DataType.REAL},
            "reference_value": {"type": DataType.REAL},
            "unit": {"type": DataType.TEXT},
            "description": {"type": DataType.TEXT},
            "calculation_date": {"type": DataType.TIMESTAMP},
            "metadata": {"type": DataType.JSON},
        },
        "indices": ["plan_id", "name"]
    },
    
    # Bảng ghi làm việc (activity_log)
    "activity_log": {
        "columns": {
            "log_id": {"type": DataType.TEXT, "constraints": [Constraint.PRIMARY_KEY]},
            "user_id": {"type": DataType.TEXT},
            "action": {"type": DataType.TEXT, "constraints": [Constraint.NOT_NULL]},
            "entity_type": {"type": DataType.TEXT},  # patient, plan, beam, etc.
            "entity_id": {"type": DataType.TEXT},
            "details": {"type": DataType.JSON},
            "timestamp": {"type": DataType.TIMESTAMP, "constraints": [Constraint.NOT_NULL]},
        },
        "indices": ["user_id", "entity_type", "entity_id", "timestamp"]
    },
}

def get_create_table_statement(table_name: str) -> str:
    """
    Tạo câu lệnh CREATE TABLE cho một bảng.
    
    Parameters
    ----------
    table_name : str
        Tên bảng cần tạo
        
    Returns
    -------
    str
        Câu lệnh SQL để tạo bảng
    """
    if table_name not in SCHEMA_TABLES:
        raise ValueError(f"Table '{table_name}' not defined in schema")
    
    table_def = SCHEMA_TABLES[table_name]
    columns = table_def["columns"]
    
    # Tạo định nghĩa các cột
    col_defs = []
    for col_name, col_props in columns.items():
        constraints = col_props.get("constraints", [])
        col_def = f"{col_name} {col_props['type'].value}"
        
        if constraints:
            # Process each constraint separately to handle foreign keys correctly
            constraint_parts = []
            for constraint in constraints:
                if isinstance(constraint, Constraint):
                    # Simple constraints like NOT NULL, UNIQUE, etc.
                    constraint_parts.append(constraint.value)
                elif isinstance(constraint, str) and constraint.startswith("REFERENCES"):
                    # Foreign key references
                    constraint_parts.append(constraint)
                elif isinstance(constraint, tuple) and constraint[0] == Constraint.DEFAULT:
                    # Default values
                    if isinstance(constraint[1], str):
                        # Wrap string defaults in quotes
                        constraint_parts.append(f"DEFAULT '{constraint[1]}'")
                    else:
                        constraint_parts.append(f"DEFAULT {constraint[1]}")
                else:
                    # Other constraints
                    constraint_parts.append(str(constraint))
            
            if constraint_parts:
                col_def += " " + " ".join(constraint_parts)
        
        col_defs.append(col_def)
    
    # Tạo câu lệnh CREATE TABLE
    create_stmt = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    create_stmt += ",\n".join(f"    {col_def}" for col_def in col_defs)
    create_stmt += "\n);"
    
    return create_stmt

def get_create_index_statements(table_name: str) -> list:
    """
    Tạo câu lệnh CREATE INDEX cho một bảng.
    
    Parameters
    ----------
    table_name : str
        Tên bảng cần tạo index
        
    Returns
    -------
    list
        Danh sách các câu lệnh SQL để tạo index
    """
    if table_name not in SCHEMA_TABLES:
        raise ValueError(f"Table '{table_name}' not defined in schema")
    
    table_def = SCHEMA_TABLES[table_name]
    indices = table_def.get("indices", [])
    
    index_stmts = []
    for col_name in indices:
        index_name = f"idx_{table_name}_{col_name}"
        stmt = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({col_name});"
        index_stmts.append(stmt)
    
    return index_stmts

def validate_data_for_table(table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kiểm tra và chuẩn hóa dữ liệu cho một bảng.
    
    Parameters
    ----------
    table_name : str
        Tên bảng cần kiểm tra dữ liệu
    data : Dict[str, Any]
        Dữ liệu cần kiểm tra và chuẩn hóa
        
    Returns
    -------
    Dict[str, Any]
        Dữ liệu đã được chuẩn hóa
        
    Raises
    ------
    ValueError
        Nếu dữ liệu không hợp lệ cho bảng
    """
    if table_name not in SCHEMA_TABLES:
        raise ValueError(f"Table '{table_name}' not defined in schema")
    
    table_def = SCHEMA_TABLES[table_name]
    columns = table_def["columns"]
    
    # Kiểm tra các trường bắt buộc
    for col_name, col_props in columns.items():
        constraints = col_props.get("constraints", [])
        
        # Kiểm tra trường NOT NULL
        if Constraint.NOT_NULL in constraints and col_name not in data:
            raise ValueError(f"Column '{col_name}' is required for table '{table_name}'")
    
    # Lọc và chuẩn hóa dữ liệu
    valid_data = {}
    for key, value in data.items():
        if key in columns:
            # Có thể thêm logic chuẩn hóa dữ liệu theo kiểu ở đây
            valid_data[key] = value
        else:
            logger.warning(f"Column '{key}' not defined for table '{table_name}', ignoring")
    
    return valid_data

def get_table_schema(table_name: str) -> Dict[str, Any]:
    """
    Lấy schema của một bảng.
    
    Parameters
    ----------
    table_name : str
        Tên bảng cần lấy schema
        
    Returns
    -------
    Dict[str, Any]
        Schema của bảng
        
    Raises
    ------
    ValueError
        Nếu bảng không tồn tại trong schema
    """
    if table_name not in SCHEMA_TABLES:
        raise ValueError(f"Table '{table_name}' not defined in schema")
    
    return SCHEMA_TABLES[table_name]

def get_all_tables() -> List[str]:
    """
    Lấy danh sách tất cả các bảng trong schema.
    
    Returns
    -------
    List[str]
        Danh sách tên các bảng
    """
    return list(SCHEMA_TABLES.keys())

def get_schema_version() -> str:
    """
    Lấy phiên bản của schema.
    
    Returns
    -------
    str
        Phiên bản của schema
    """
    return "1.0.0" 