"""
Module cơ sở dữ liệu của QuangTPS.
Cung cấp các lớp và phương thức để tương tác với cơ sở dữ liệu.
"""

from quangtps.database.db_connector import DBConnector
from quangtps.database.patient_db import PatientDB, PatientDatabase, Patient, Study, Series
from quangtps.database.study_db import StudyDB
from quangtps.database.series_db import SeriesDB
from quangtps.database.plan_db import PlanDB
from quangtps.database.structure_db import StructureDatabase
from quangtps.database.dose_db import DoseDB
from quangtps.database.beam_db import BeamDB
from quangtps.database.prescription_db import PrescriptionDB
from quangtps.database.query import QueryBuilder

# Create an alias for PrescriptionDB for compatibility
PrescriptionDatabase = PrescriptionDB

__all__ = [
    'DBConnector',
    'PatientDB',
    'PatientDatabase',
    'Patient',
    'Study',
    'Series',
    'StudyDB',
    'SeriesDB',
    'PlanDB',
    'StructureDatabase',
    'DoseDB',
    'BeamDB',
    'PrescriptionDB',
    'PrescriptionDatabase',
    'QueryBuilder'
]