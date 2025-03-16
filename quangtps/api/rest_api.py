"""
REST API cho hệ thống lập kế hoạch xạ trị QuangTPS.

API này cung cấp các endpoint để tương tác với hệ thống QuangTPS từ xa, 
hỗ trợ giao diện web và các ứng dụng khách khác.
"""

import logging
import os
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from quangtps.core.logging import get_logger
from quangtps.core.config import Config
from quangtps.database.patient_db import PatientDatabase
from quangtps.treatment.plan import TreatmentPlan
from quangtps.evaluation.qa.treatment_qa import (
    TreatmentQAManager, TreatmentQATest, TreatmentQAResult, 
    QATestType, QAProtocol, QAStatus
)

# Thiết lập logger
logger = get_logger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="QuangTPS API",
    description="REST API cho hệ thống lập kế hoạch xạ trị QuangTPS",
    version="1.0.0",
)

# Cấu hình CORS để cho phép truy cập từ frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn gốc trong quá trình phát triển
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo các thành phần cần thiết
config = Config.get_instance()
patient_db = PatientDatabase()
qa_manager = TreatmentQAManager()

# Định nghĩa các model dữ liệu cho API
class PatientBase(BaseModel):
    """Model cơ bản cho thông tin bệnh nhân."""
    patient_id: Optional[str] = None
    name: str
    birth_date: str
    gender: str
    medical_record_number: Optional[str] = None
    
class PatientCreate(PatientBase):
    """Model để tạo bệnh nhân mới."""
    pass

class PatientResponse(PatientBase):
    """Model phản hồi thông tin bệnh nhân."""
    patient_id: str
    created_date: str
    
class PlanBase(BaseModel):
    """Model cơ bản cho kế hoạch điều trị."""
    plan_name: str
    description: Optional[str] = None
    prescribed_dose: float
    fractionation: int
    target_structures: List[str]
    oar_structures: List[str]
    
class PlanCreate(PlanBase):
    """Model để tạo kế hoạch điều trị mới."""
    patient_id: str
    
class PlanResponse(PlanBase):
    """Model phản hồi thông tin kế hoạch điều trị."""
    plan_id: str
    patient_id: str
    created_date: str
    status: str
    
class QATestBase(BaseModel):
    """Model cơ bản cho bài kiểm tra QA."""
    test_name: str
    test_type: str
    protocol: str
    description: Optional[str] = None
    
class QATestCreate(QATestBase):
    """Model để tạo bài kiểm tra QA mới."""
    plan_id: Optional[str] = None
    patient_id: Optional[str] = None
    machine_id: Optional[str] = None
    
class QATestResponse(QATestBase):
    """Model phản hồi thông tin bài kiểm tra QA."""
    test_id: str
    created_date: str
    status: str
    overall_result: Optional[bool] = None
    
class MetricResultModel(BaseModel):
    """Model cho kết quả chỉ số đánh giá."""
    name: str
    value: float
    reference: float
    tolerance: float
    unit: Optional[str] = None
    description: Optional[str] = None
    is_acceptable: bool
    
class ErrorResponse(BaseModel):
    """Model cho phản hồi lỗi."""
    detail: str

# Các route cho bệnh nhân
@app.get("/patients/", response_model=List[PatientResponse], tags=["Patients"])
async def get_patients():
    """Lấy danh sách tất cả bệnh nhân."""
    try:
        patients = patient_db.get_all_patients()
        return [
            PatientResponse(
                patient_id=p.patient_id,
                name=p.name,
                birth_date=p.birth_date.isoformat(),
                gender=p.gender,
                medical_record_number=p.medical_record_number,
                created_date=p.created_date.isoformat()
            )
            for p in patients
        ]
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách bệnh nhân: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patients/", response_model=PatientResponse, tags=["Patients"])
async def create_patient(patient: PatientCreate):
    """Tạo bệnh nhân mới."""
    try:
        new_patient = patient_db.create_patient(
            name=patient.name,
            birth_date=datetime.fromisoformat(patient.birth_date),
            gender=patient.gender,
            medical_record_number=patient.medical_record_number
        )
        return PatientResponse(
            patient_id=new_patient.patient_id,
            name=new_patient.name,
            birth_date=new_patient.birth_date.isoformat(),
            gender=new_patient.gender,
            medical_record_number=new_patient.medical_record_number,
            created_date=new_patient.created_date.isoformat()
        )
    except Exception as e:
        logger.error(f"Lỗi khi tạo bệnh nhân mới: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}", response_model=PatientResponse, tags=["Patients"])
async def get_patient(patient_id: str = PathParam(..., description="ID của bệnh nhân")):
    """Lấy thông tin chi tiết của một bệnh nhân."""
    try:
        patient = patient_db.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Bệnh nhân không tồn tại")
        return PatientResponse(
            patient_id=patient.patient_id,
            name=patient.name,
            birth_date=patient.birth_date.isoformat(),
            gender=patient.gender,
            medical_record_number=patient.medical_record_number,
            created_date=patient.created_date.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin bệnh nhân: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Các route cho kế hoạch điều trị
@app.get("/plans/", response_model=List[PlanResponse], tags=["Plans"])
async def get_plans(patient_id: Optional[str] = Query(None, description="Lọc theo ID bệnh nhân")):
    """Lấy danh sách kế hoạch điều trị."""
    try:
        plans = []
        if patient_id:
            patient = patient_db.get_patient(patient_id)
            if not patient:
                raise HTTPException(status_code=404, detail="Bệnh nhân không tồn tại")
            plans = patient_db.get_patient_plans(patient_id)
        else:
            # Lấy tất cả kế hoạch từ mọi bệnh nhân
            patients = patient_db.get_all_patients()
            for patient in patients:
                plans.extend(patient_db.get_patient_plans(patient.patient_id))
        
        return [
            PlanResponse(
                plan_id=p.plan_id,
                patient_id=p.patient_id,
                plan_name=p.plan_name,
                description=p.description,
                prescribed_dose=p.prescribed_dose,
                fractionation=p.fractionation.num_fractions if hasattr(p, 'fractionation') and p.fractionation else 0,
                target_structures=p.target_structures,
                oar_structures=p.oar_structures,
                created_date=p.created_date.isoformat(),
                status=p.status
            )
            for p in plans
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách kế hoạch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plans/", response_model=PlanResponse, tags=["Plans"])
async def create_plan(plan: PlanCreate):
    """Tạo kế hoạch điều trị mới."""
    try:
        patient = patient_db.get_patient(plan.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Bệnh nhân không tồn tại")
        
        # Thực hiện logic tạo kế hoạch
        new_plan = TreatmentPlan(
            plan_name=plan.plan_name,
            patient_id=plan.patient_id,
            prescribed_dose=plan.prescribed_dose
        )
        new_plan.description = plan.description
        new_plan.target_structures = plan.target_structures
        new_plan.oar_structures = plan.oar_structures
        
        # Lưu kế hoạch vào cơ sở dữ liệu
        patient_db.add_plan_to_patient(plan.patient_id, new_plan)
        
        return PlanResponse(
            plan_id=new_plan.plan_id,
            patient_id=new_plan.patient_id,
            plan_name=new_plan.plan_name,
            description=new_plan.description,
            prescribed_dose=new_plan.prescribed_dose,
            fractionation=plan.fractionation,
            target_structures=new_plan.target_structures,
            oar_structures=new_plan.oar_structures,
            created_date=new_plan.created_date.isoformat(),
            status=new_plan.status
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi tạo kế hoạch mới: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plans/{plan_id}", response_model=PlanResponse, tags=["Plans"])
async def get_plan(plan_id: str = PathParam(..., description="ID của kế hoạch")):
    """Lấy thông tin chi tiết của một kế hoạch điều trị."""
    try:
        # Cần cài đặt logic để tìm kiếm kế hoạch theo ID
        plan = None
        patients = patient_db.get_all_patients()
        for patient in patients:
            plans = patient_db.get_patient_plans(patient.patient_id)
            for p in plans:
                if p.plan_id == plan_id:
                    plan = p
                    break
            if plan:
                break
                
        if not plan:
            raise HTTPException(status_code=404, detail="Kế hoạch không tồn tại")
            
        return PlanResponse(
            plan_id=plan.plan_id,
            patient_id=plan.patient_id,
            plan_name=plan.plan_name,
            description=plan.description,
            prescribed_dose=plan.prescribed_dose,
            fractionation=plan.fractionation.num_fractions if hasattr(plan, 'fractionation') and plan.fractionation else 0,
            target_structures=plan.target_structures,
            oar_structures=plan.oar_structures,
            created_date=plan.created_date.isoformat(),
            status=plan.status
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin kế hoạch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Các route cho module QA
@app.get("/qa/tests/", response_model=List[QATestResponse], tags=["Quality Assurance"])
async def get_qa_tests(
    plan_id: Optional[str] = Query(None, description="Lọc theo ID kế hoạch"),
    patient_id: Optional[str] = Query(None, description="Lọc theo ID bệnh nhân"),
    machine_id: Optional[str] = Query(None, description="Lọc theo ID máy xạ trị")
):
    """Lấy danh sách các bài kiểm tra QA."""
    try:
        tests = []
        if plan_id:
            tests = qa_manager.get_plan_qa_tests(plan_id)
        elif patient_id:
            tests = qa_manager.get_patient_qa_tests(patient_id)
        elif machine_id:
            tests = qa_manager.get_machine_qa_tests(machine_id)
        else:
            # Lấy tất cả bài kiểm tra
            tests = list(qa_manager.qa_tests.values())
            
        return [
            QATestResponse(
                test_id=test.test_id,
                test_name=test.test_name,
                test_type=test.test_type.value,
                protocol=test.protocol.value,
                description=test.description,
                created_date=test.created_date.isoformat(),
                status=test.status.value,
                overall_result=test.overall_result
            )
            for test in tests
        ]
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách bài kiểm tra QA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/qa/tests/", response_model=QATestResponse, tags=["Quality Assurance"])
async def create_qa_test(test: QATestCreate):
    """Tạo bài kiểm tra QA mới."""
    try:
        test_type = getattr(QATestType, test.test_type) if hasattr(QATestType, test.test_type) else QATestType.PRE_TREATMENT
        protocol = getattr(QAProtocol, test.protocol) if hasattr(QAProtocol, test.protocol) else QAProtocol.CUSTOM
        
        test_id = qa_manager.create_test(
            test_name=test.test_name,
            test_type=test_type,
            protocol=protocol,
            plan_id=test.plan_id,
            patient_id=test.patient_id,
            machine_id=test.machine_id,
            description=test.description
        )
        
        new_test = qa_manager.get_test(test_id)
        if not new_test:
            raise HTTPException(status_code=500, detail="Không thể tạo bài kiểm tra QA")
            
        return QATestResponse(
            test_id=new_test.test_id,
            test_name=new_test.test_name,
            test_type=new_test.test_type.value,
            protocol=new_test.protocol.value,
            description=new_test.description,
            created_date=new_test.created_date.isoformat(),
            status=new_test.status.value,
            overall_result=new_test.overall_result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi tạo bài kiểm tra QA mới: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa/tests/{test_id}", response_model=QATestResponse, tags=["Quality Assurance"])
async def get_qa_test(test_id: str = PathParam(..., description="ID của bài kiểm tra QA")):
    """Lấy thông tin chi tiết của một bài kiểm tra QA."""
    try:
        test = qa_manager.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Bài kiểm tra QA không tồn tại")
            
        return QATestResponse(
            test_id=test.test_id,
            test_name=test.test_name,
            test_type=test.test_type.value,
            protocol=test.protocol.value,
            description=test.description,
            created_date=test.created_date.isoformat(),
            status=test.status.value,
            overall_result=test.overall_result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin bài kiểm tra QA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa/tests/{test_id}/metrics", response_model=List[MetricResultModel], tags=["Quality Assurance"])
async def get_qa_test_metrics(test_id: str = PathParam(..., description="ID của bài kiểm tra QA")):
    """Lấy danh sách các chỉ số đánh giá của một bài kiểm tra QA."""
    try:
        test = qa_manager.get_test(test_id)
        if not test:
            raise HTTPException(status_code=404, detail="Bài kiểm tra QA không tồn tại")
            
        return [
            MetricResultModel(
                name=metric.name,
                value=metric.value,
                reference=metric.reference,
                tolerance=metric.tolerance,
                unit=metric.unit,
                description=metric.description,
                is_acceptable=metric.is_acceptable
            )
            for metric in test.metrics
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách chỉ số đánh giá: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Hàm chính để chạy API
def start_api(host="0.0.0.0", port=8000):
    """Khởi động REST API server."""
    import uvicorn
    logger.info(f"Khởi động QuangTPS REST API trên {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api()