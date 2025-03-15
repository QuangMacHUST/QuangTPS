#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý toàn bộ quy trình điều trị xạ trị.

Module này tích hợp các module khác nhau trong hệ thống QuangTPS để quản lý toàn bộ quy trình điều trị,
từ kế hoạch điều trị đến thực hiện điều trị và theo dõi tiến trình.
"""

import logging
import datetime
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import os
import json
import pickle

from quangtps.core.patient import Patient
from quangtps.treatment.plan import TreatmentPlan
from quangtps.treatment.fractionation import Fractionation
from quangtps.treatment.treatment_delivery import TreatmentCourse, TreatmentStatus, FractionStatus, TreatmentFraction
from quangtps.treatment.scheduler import TreatmentScheduler, TimeSlot
from quangtps.treatment.machine.treatment_machine import TreatmentMachine, MachineType, MachineStatus
from quangtps.treatment.machine.machine_specs import MachineSpecification

logger = logging.getLogger(__name__)


class TreatmentManager:
    """
    Lớp quản lý toàn bộ quy trình điều trị xạ trị.
    
    Lớp này tích hợp các thành phần khác nhau của hệ thống QuangTPS để quản lý toàn bộ quy trình điều trị,
    từ kế hoạch điều trị đến thực hiện điều trị và theo dõi tiến trình.
    """
    
    def __init__(self, data_path: str = "data/treatment"):
        """
        Khởi tạo quản lý điều trị.
        
        Parameters
        ----------
        data_path : str, optional
            Đường dẫn thư mục dữ liệu
        """
        self.data_path = data_path
        
        # Đảm bảo thư mục dữ liệu tồn tại
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(os.path.join(data_path, "patients"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "plans"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "courses"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "machines"), exist_ok=True)
        os.makedirs(os.path.join(data_path, "schedulers"), exist_ok=True)
        
        # Các đối tượng quản lý
        self.patients: Dict[str, Patient] = {}
        self.plans: Dict[str, TreatmentPlan] = {}
        self.courses: Dict[str, TreatmentCourse] = {}
        self.machines: Dict[str, TreatmentMachine] = {}
        self.schedulers: Dict[str, TreatmentScheduler] = {}
        
        # Load dữ liệu
        self._load_patients()
        self._load_plans()
        self._load_courses()
        self._load_machines()
        self._load_schedulers()
    
    def _load_patients(self):
        """Load dữ liệu bệnh nhân từ file."""
        patients_dir = os.path.join(self.data_path, "patients")
        for filename in os.listdir(patients_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(patients_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        patient = Patient.from_dict(data)
                        self.patients[patient.patient_id] = patient
                except Exception as e:
                    logger.error(f"Lỗi khi load bệnh nhân từ file {file_path}: {e}")
    
    def _load_plans(self):
        """Load dữ liệu kế hoạch điều trị từ file."""
        plans_dir = os.path.join(self.data_path, "plans")
        for filename in os.listdir(plans_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(plans_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        plan = TreatmentPlan.from_dict(data)
                        self.plans[plan.plan_id] = plan
                except Exception as e:
                    logger.error(f"Lỗi khi load kế hoạch điều trị từ file {file_path}: {e}")
    
    def _load_courses(self):
        """Load dữ liệu đợt điều trị từ file."""
        courses_dir = os.path.join(self.data_path, "courses")
        for filename in os.listdir(courses_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(courses_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        course = TreatmentCourse.from_dict(data)
                        self.courses[course.course_id] = course
                except Exception as e:
                    logger.error(f"Lỗi khi load đợt điều trị từ file {file_path}: {e}")
    
    def _load_machines(self):
        """Load dữ liệu máy xạ trị từ file."""
        machines_dir = os.path.join(self.data_path, "machines")
        for filename in os.listdir(machines_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(machines_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        machine = TreatmentMachine.from_dict(data)
                        self.machines[machine.machine_id] = machine
                except Exception as e:
                    logger.error(f"Lỗi khi load máy xạ trị từ file {file_path}: {e}")
    
    def _load_schedulers(self):
        """Load dữ liệu lịch điều trị từ file."""
        schedulers_dir = os.path.join(self.data_path, "schedulers")
        for filename in os.listdir(schedulers_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(schedulers_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        scheduler = TreatmentScheduler.from_dict(data)
                        self.schedulers[scheduler.machine_id] = scheduler
                except Exception as e:
                    logger.error(f"Lỗi khi load lịch điều trị từ file {file_path}: {e}")
    
    def _save_patient(self, patient: Patient):
        """
        Lưu dữ liệu bệnh nhân vào file.
        
        Parameters
        ----------
        patient : Patient
            Bệnh nhân cần lưu
        """
        file_path = os.path.join(self.data_path, "patients", f"{patient.patient_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(patient.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu bệnh nhân vào file {file_path}: {e}")
    
    def _save_plan(self, plan: TreatmentPlan):
        """
        Lưu dữ liệu kế hoạch điều trị vào file.
        
        Parameters
        ----------
        plan : TreatmentPlan
            Kế hoạch điều trị cần lưu
        """
        file_path = os.path.join(self.data_path, "plans", f"{plan.plan_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu kế hoạch điều trị vào file {file_path}: {e}")
    
    def _save_course(self, course: TreatmentCourse):
        """
        Lưu dữ liệu đợt điều trị vào file.
        
        Parameters
        ----------
        course : TreatmentCourse
            Đợt điều trị cần lưu
        """
        file_path = os.path.join(self.data_path, "courses", f"{course.course_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(course.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu đợt điều trị vào file {file_path}: {e}")
    
    def _save_machine(self, machine: TreatmentMachine):
        """
        Lưu dữ liệu máy xạ trị vào file.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy xạ trị cần lưu
        """
        file_path = os.path.join(self.data_path, "machines", f"{machine.machine_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(machine.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu máy xạ trị vào file {file_path}: {e}")
    
    def _save_scheduler(self, scheduler: TreatmentScheduler):
        """
        Lưu dữ liệu lịch điều trị vào file.
        
        Parameters
        ----------
        scheduler : TreatmentScheduler
            Lịch điều trị cần lưu
        """
        file_path = os.path.join(self.data_path, "schedulers", f"{scheduler.machine_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(scheduler.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu lịch điều trị vào file {file_path}: {e}")
    
    # Các phương thức CRUD cho Patient
    
    def add_patient(self, patient: Patient) -> bool:
        """
        Thêm bệnh nhân mới.
        
        Parameters
        ----------
        patient : Patient
            Bệnh nhân cần thêm
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if patient.patient_id in self.patients:
            logger.warning(f"Bệnh nhân có ID {patient.patient_id} đã tồn tại.")
            return False
            
        self.patients[patient.patient_id] = patient
        self._save_patient(patient)
        return True
    
    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """
        Lấy thông tin bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        Optional[Patient]
            Đối tượng Patient nếu tồn tại, None nếu không
        """
        return self.patients.get(patient_id)
    
    def update_patient(self, patient: Patient) -> bool:
        """
        Cập nhật thông tin bệnh nhân.
        
        Parameters
        ----------
        patient : Patient
            Bệnh nhân cần cập nhật
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if patient.patient_id not in self.patients:
            logger.warning(f"Bệnh nhân có ID {patient.patient_id} không tồn tại.")
            return False
            
        self.patients[patient.patient_id] = patient
        self._save_patient(patient)
        return True
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Xóa bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if patient_id not in self.patients:
            logger.warning(f"Bệnh nhân có ID {patient_id} không tồn tại.")
            return False
            
        # Xóa từ bộ nhớ
        del self.patients[patient_id]
        
        # Xóa file
        file_path = os.path.join(self.data_path, "patients", f"{patient_id}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Lỗi khi xóa file bệnh nhân {file_path}: {e}")
            return False
            
        return True
    
    def get_all_patients(self) -> List[Patient]:
        """
        Lấy danh sách tất cả bệnh nhân.
        
        Returns
        -------
        List[Patient]
            Danh sách bệnh nhân
        """
        return list(self.patients.values())
    
    # Các phương thức CRUD cho TreatmentPlan
    
    def add_plan(self, plan: TreatmentPlan) -> bool:
        """
        Thêm kế hoạch điều trị mới.
        
        Parameters
        ----------
        plan : TreatmentPlan
            Kế hoạch điều trị cần thêm
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if plan.plan_id in self.plans:
            logger.warning(f"Kế hoạch điều trị có ID {plan.plan_id} đã tồn tại.")
            return False
            
        self.plans[plan.plan_id] = plan
        self._save_plan(plan)
        return True
    
    def get_plan(self, plan_id: str) -> Optional[TreatmentPlan]:
        """
        Lấy thông tin kế hoạch điều trị.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
            
        Returns
        -------
        Optional[TreatmentPlan]
            Đối tượng TreatmentPlan nếu tồn tại, None nếu không
        """
        return self.plans.get(plan_id)
    
    def update_plan(self, plan: TreatmentPlan) -> bool:
        """
        Cập nhật thông tin kế hoạch điều trị.
        
        Parameters
        ----------
        plan : TreatmentPlan
            Kế hoạch điều trị cần cập nhật
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if plan.plan_id not in self.plans:
            logger.warning(f"Kế hoạch điều trị có ID {plan.plan_id} không tồn tại.")
            return False
            
        self.plans[plan.plan_id] = plan
        self._save_plan(plan)
        return True
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Xóa kế hoạch điều trị.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if plan_id not in self.plans:
            logger.warning(f"Kế hoạch điều trị có ID {plan_id} không tồn tại.")
            return False
            
        # Xóa từ bộ nhớ
        del self.plans[plan_id]
        
        # Xóa file
        file_path = os.path.join(self.data_path, "plans", f"{plan_id}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Lỗi khi xóa file kế hoạch điều trị {file_path}: {e}")
            return False
            
        return True
    
    def get_patient_plans(self, patient_id: str) -> List[TreatmentPlan]:
        """
        Lấy danh sách kế hoạch điều trị của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        List[TreatmentPlan]
            Danh sách kế hoạch điều trị
        """
        return [plan for plan in self.plans.values() if plan.patient_id == patient_id]
    
    # Các phương thức CRUD cho TreatmentCourse
    
    def add_course(self, course: TreatmentCourse) -> bool:
        """
        Thêm đợt điều trị mới.
        
        Parameters
        ----------
        course : TreatmentCourse
            Đợt điều trị cần thêm
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if course.course_id in self.courses:
            logger.warning(f"Đợt điều trị có ID {course.course_id} đã tồn tại.")
            return False
            
        self.courses[course.course_id] = course
        self._save_course(course)
        return True
    
    def get_course(self, course_id: str) -> Optional[TreatmentCourse]:
        """
        Lấy thông tin đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
            
        Returns
        -------
        Optional[TreatmentCourse]
            Đối tượng TreatmentCourse nếu tồn tại, None nếu không
        """
        return self.courses.get(course_id)
    
    def update_course(self, course: TreatmentCourse) -> bool:
        """
        Cập nhật thông tin đợt điều trị.
        
        Parameters
        ----------
        course : TreatmentCourse
            Đợt điều trị cần cập nhật
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if course.course_id not in self.courses:
            logger.warning(f"Đợt điều trị có ID {course.course_id} không tồn tại.")
            return False
            
        self.courses[course.course_id] = course
        self._save_course(course)
        return True
    
    def delete_course(self, course_id: str) -> bool:
        """
        Xóa đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if course_id not in self.courses:
            logger.warning(f"Đợt điều trị có ID {course_id} không tồn tại.")
            return False
            
        # Xóa từ bộ nhớ
        del self.courses[course_id]
        
        # Xóa file
        file_path = os.path.join(self.data_path, "courses", f"{course_id}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Lỗi khi xóa file đợt điều trị {file_path}: {e}")
            return False
            
        return True
    
    def get_patient_courses(self, patient_id: str) -> List[TreatmentCourse]:
        """
        Lấy danh sách đợt điều trị của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        List[TreatmentCourse]
            Danh sách đợt điều trị
        """
        return [course for course in self.courses.values() if course.patient_id == patient_id]
    
    def update_fraction_status(
        self,
        course_id: str,
        fraction_number: int,
        status: FractionStatus,
        **kwargs
    ) -> bool:
        """
        Cập nhật trạng thái của một phân đoạn.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
        fraction_number : int
            Số thứ tự của phân đoạn
        status : FractionStatus
            Trạng thái mới
        **kwargs : dict
            Các thông số bổ sung
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        course = self.get_course(course_id)
        if not course:
            logger.warning(f"Đợt điều trị có ID {course_id} không tồn tại.")
            return False
            
        success = course.update_fraction_status(fraction_number, status, **kwargs)
        if success:
            self._save_course(course)
            
        return success
    
    def create_course_from_plan(
        self,
        plan_id: str,
        course_name: str,
        fractionation: Optional[Fractionation] = None,
        start_date: Optional[datetime.date] = None,
        days_of_week: Optional[List[int]] = None
    ) -> Optional[TreatmentCourse]:
        """
        Tạo đợt điều trị từ kế hoạch điều trị.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        course_name : str
            Tên của đợt điều trị
        fractionation : Fractionation, optional
            Thông tin phân đoạn. Nếu không cung cấp, sẽ sử dụng fractionation từ kế hoạch.
        start_date : datetime.date, optional
            Ngày bắt đầu điều trị. Mặc định là ngày hiện tại.
        days_of_week : List[int], optional
            Các ngày trong tuần để điều trị (0=Monday, 6=Sunday).
            Mặc định là [0, 1, 2, 3, 4] (Thứ 2 đến Thứ 6).
            
        Returns
        -------
        Optional[TreatmentCourse]
            Đối tượng TreatmentCourse nếu thành công, None nếu thất bại
        """
        plan = self.get_plan(plan_id)
        if not plan:
            logger.warning(f"Kế hoạch điều trị có ID {plan_id} không tồn tại.")
            return None
            
        patient_id = plan.patient_id
        
        # Sử dụng fractionation từ kế hoạch nếu không cung cấp
        if not fractionation:
            fractionation = plan.fractionation
            
        if not fractionation:
            logger.warning(f"Không tìm thấy thông tin phân đoạn cho kế hoạch {plan_id}.")
            return None
            
        # Tạo đợt điều trị mới
        course = TreatmentCourse(
            patient_id=patient_id,
            plan_id=plan_id,
            course_name=course_name,
            fractionation=fractionation
        )
        
        # Tạo các phân đoạn
        if not start_date:
            start_date = datetime.date.today()
            
        course.generate_fractions(start_date, days_of_week)
        
        # Lưu đợt điều trị
        self.add_course(course)
        
        return course
    
    # Các phương thức CRUD cho TreatmentMachine
    
    def add_machine(self, machine: TreatmentMachine) -> bool:
        """
        Thêm máy xạ trị mới.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy xạ trị cần thêm
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if machine.machine_id in self.machines:
            logger.warning(f"Máy xạ trị có ID {machine.machine_id} đã tồn tại.")
            return False
            
        self.machines[machine.machine_id] = machine
        self._save_machine(machine)
        
        # Tạo lịch điều trị cho máy nếu chưa có
        if machine.machine_id not in self.schedulers:
            scheduler = TreatmentScheduler(machine_id=machine.machine_id)
            self.schedulers[machine.machine_id] = scheduler
            self._save_scheduler(scheduler)
            
        return True
    
    def get_machine(self, machine_id: str) -> Optional[TreatmentMachine]:
        """
        Lấy thông tin máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
            
        Returns
        -------
        Optional[TreatmentMachine]
            Đối tượng TreatmentMachine nếu tồn tại, None nếu không
        """
        return self.machines.get(machine_id)
    
    def update_machine(self, machine: TreatmentMachine) -> bool:
        """
        Cập nhật thông tin máy xạ trị.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy xạ trị cần cập nhật
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if machine.machine_id not in self.machines:
            logger.warning(f"Máy xạ trị có ID {machine.machine_id} không tồn tại.")
            return False
            
        self.machines[machine.machine_id] = machine
        self._save_machine(machine)
        return True
    
    def delete_machine(self, machine_id: str) -> bool:
        """
        Xóa máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if machine_id not in self.machines:
            logger.warning(f"Máy xạ trị có ID {machine_id} không tồn tại.")
            return False
            
        # Xóa từ bộ nhớ
        del self.machines[machine_id]
        
        # Xóa file
        file_path = os.path.join(self.data_path, "machines", f"{machine_id}.json")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Lỗi khi xóa file máy xạ trị {file_path}: {e}")
            return False
        
        # Xóa lịch điều trị tương ứng
        if machine_id in self.schedulers:
            del self.schedulers[machine_id]
            scheduler_path = os.path.join(self.data_path, "schedulers", f"{machine_id}.json")
            try:
                if os.path.exists(scheduler_path):
                    os.remove(scheduler_path)
            except Exception as e:
                logger.error(f"Lỗi khi xóa file lịch điều trị {scheduler_path}: {e}")
            
        return True
    
    def get_all_machines(self) -> List[TreatmentMachine]:
        """
        Lấy danh sách tất cả máy xạ trị.
        
        Returns
        -------
        List[TreatmentMachine]
            Danh sách máy xạ trị
        """
        return list(self.machines.values())
    
    def get_available_machines(self) -> List[TreatmentMachine]:
        """
        Lấy danh sách máy xạ trị đang hoạt động.
        
        Returns
        -------
        List[TreatmentMachine]
            Danh sách máy xạ trị đang hoạt động
        """
        return [machine for machine in self.machines.values() if machine.status == MachineStatus.OPERATIONAL]
    
    # Các phương thức cho TreatmentScheduler
    
    def get_scheduler(self, machine_id: str) -> Optional[TreatmentScheduler]:
        """
        Lấy lịch điều trị của một máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
            
        Returns
        -------
        Optional[TreatmentScheduler]
            Đối tượng TreatmentScheduler nếu tồn tại, None nếu không
        """
        return self.schedulers.get(machine_id)
    
    def generate_machine_schedule(
        self,
        machine_id: str,
        start_date: datetime.date,
        end_date: datetime.date,
        override_existing: bool = False
    ) -> bool:
        """
        Tạo lịch điều trị cho một máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
        start_date : datetime.date
            Ngày bắt đầu
        end_date : datetime.date
            Ngày kết thúc
        override_existing : bool, optional
            Ghi đè lên lịch điều trị đã tồn tại
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        scheduler = self.get_scheduler(machine_id)
        if not scheduler:
            logger.warning(f"Không tìm thấy lịch điều trị cho máy {machine_id}.")
            return False
            
        scheduler.generate_slots(start_date, end_date, override_existing)
        self._save_scheduler(scheduler)
        return True
    
    def schedule_course(self, course_id: str, machine_id: str) -> bool:
        """
        Lên lịch cho một đợt điều trị trên một máy xạ trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
        machine_id : str
            ID của máy xạ trị
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        course = self.get_course(course_id)
        if not course:
            logger.warning(f"Đợt điều trị có ID {course_id} không tồn tại.")
            return False
            
        scheduler = self.get_scheduler(machine_id)
        if not scheduler:
            logger.warning(f"Không tìm thấy lịch điều trị cho máy {machine_id}.")
            return False
            
        # Đảm bảo có đủ slot cho tất cả các phân đoạn
        if course.fractions:
            first_fraction = course.fractions[0]
            last_fraction = course.fractions[-1]
            
            if first_fraction.scheduled_date and last_fraction.scheduled_date:
                # Tạo slot từ ngày điều trị đầu tiên đến ngày điều trị cuối cùng
                scheduler.generate_slots(
                    first_fraction.scheduled_date,
                    last_fraction.scheduled_date,
                    override_existing=False
                )
        
        # Lên lịch cho đợt điều trị
        success = scheduler.schedule_course(course)
        if success:
            self._save_scheduler(scheduler)
            
        return success
    
    def cancel_course_schedule(self, course_id: str, machine_id: str) -> bool:
        """
        Hủy lịch của một đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
        machine_id : str
            ID của máy xạ trị
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        scheduler = self.get_scheduler(machine_id)
        if not scheduler:
            logger.warning(f"Không tìm thấy lịch điều trị cho máy {machine_id}.")
            return False
            
        success = scheduler.cancel_course(course_id)
        if success:
            self._save_scheduler(scheduler)
            
        return success
    
    def get_machine_schedule(
        self,
        machine_id: str,
        date: datetime.date
    ) -> List[TimeSlot]:
        """
        Lấy lịch điều trị của một máy xạ trị trong một ngày.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
        date : datetime.date
            Ngày cần lấy lịch
            
        Returns
        -------
        List[TimeSlot]
            Danh sách các khoảng thời gian
        """
        scheduler = self.get_scheduler(machine_id)
        if not scheduler:
            logger.warning(f"Không tìm thấy lịch điều trị cho máy {machine_id}.")
            return []
            
        return scheduler.get_schedule(date)
    
    def get_patient_schedule(
        self,
        patient_id: str,
        machine_id: Optional[str] = None
    ) -> Dict[datetime.date, List[TimeSlot]]:
        """
        Lấy lịch điều trị của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        machine_id : str, optional
            ID của máy xạ trị. Nếu không cung cấp, sẽ tìm kiếm trên tất cả các máy.
            
        Returns
        -------
        Dict[datetime.date, List[TimeSlot]]
            Lịch điều trị theo ngày
        """
        result = {}
        
        if machine_id:
            # Tìm kiếm trên một máy cụ thể
            scheduler = self.get_scheduler(machine_id)
            if scheduler:
                result = scheduler.get_patient_schedule(patient_id)
        else:
            # Tìm kiếm trên tất cả các máy
            for scheduler_id, scheduler in self.schedulers.items():
                patient_schedule = scheduler.get_patient_schedule(patient_id)
                
                # Gộp lịch
                for date, slots in patient_schedule.items():
                    if date in result:
                        result[date].extend(slots)
                    else:
                        result[date] = slots
        
        return result

    # Các phương thức báo cáo và thống kê
    
    def get_course_progress(self, course_id: str) -> Dict[str, Any]:
        """
        Lấy tiến trình của một đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
            
        Returns
        -------
        Dict[str, Any]
            Thông tin tiến trình
        """
        course = self.get_course(course_id)
        if not course:
            logger.warning(f"Đợt điều trị có ID {course_id} không tồn tại.")
            return {}
            
        total_fractions = len(course.fractions)
        completed_fractions = sum(1 for f in course.fractions if f.status == FractionStatus.COMPLETED)
        scheduled_fractions = sum(1 for f in course.fractions if f.status == FractionStatus.SCHEDULED)
        remaining_fractions = total_fractions - completed_fractions
        
        progress = {
            "course_id": course_id,
            "patient_id": course.patient_id,
            "plan_id": course.plan_id,
            "course_name": course.course_name,
            "start_date": course.start_date.isoformat() if course.start_date else None,
            "expected_end_date": course.expected_end_date.isoformat() if course.expected_end_date else None,
            "actual_end_date": course.actual_end_date.isoformat() if course.actual_end_date else None,
            "status": course.status.name,
            "total_fractions": total_fractions,
            "completed_fractions": completed_fractions,
            "scheduled_fractions": scheduled_fractions,
            "remaining_fractions": remaining_fractions,
            "completion_percentage": (completed_fractions / total_fractions * 100) if total_fractions > 0 else 0,
            "last_treatment_date": course.last_treatment_date.isoformat() if course.last_treatment_date else None,
            "next_treatment_date": None
        }
        
        # Tìm ngày điều trị tiếp theo
        next_fraction = None
        today = datetime.date.today()
        for fraction in course.fractions:
            if fraction.status == FractionStatus.SCHEDULED and fraction.scheduled_date:
                if fraction.scheduled_date >= today and (next_fraction is None or fraction.scheduled_date < next_fraction.scheduled_date):
                    next_fraction = fraction
                    
        if next_fraction and next_fraction.scheduled_date:
            progress["next_treatment_date"] = next_fraction.scheduled_date.isoformat()
            
        return progress
    
    def get_patient_treatment_summary(self, patient_id: str) -> Dict[str, Any]:
        """
        Lấy tổng hợp điều trị của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        Dict[str, Any]
            Thông tin tổng hợp
        """
        patient = self.get_patient(patient_id)
        if not patient:
            logger.warning(f"Bệnh nhân có ID {patient_id} không tồn tại.")
            return {}
            
        courses = self.get_patient_courses(patient_id)
        plans = self.get_patient_plans(patient_id)
        
        total_fractions = 0
        completed_fractions = 0
        scheduled_fractions = 0
        remaining_fractions = 0
        active_courses = 0
        completed_courses = 0
        
        for course in courses:
            course_fractions = len(course.fractions)
            total_fractions += course_fractions
            completed_fractions += sum(1 for f in course.fractions if f.status == FractionStatus.COMPLETED)
            scheduled_fractions += sum(1 for f in course.fractions if f.status == FractionStatus.SCHEDULED)
            
            if course.status == TreatmentStatus.COMPLETED:
                completed_courses += 1
            elif course.status == TreatmentStatus.IN_PROGRESS:
                active_courses += 1
                
        remaining_fractions = total_fractions - completed_fractions
        
        summary = {
            "patient_id": patient_id,
            "patient_name": f"{patient.last_name}, {patient.first_name}",
            "patient_mrn": patient.medical_record_number,
            "total_plans": len(plans),
            "total_courses": len(courses),
            "active_courses": active_courses,
            "completed_courses": completed_courses,
            "total_fractions": total_fractions,
            "completed_fractions": completed_fractions,
            "scheduled_fractions": scheduled_fractions,
            "remaining_fractions": remaining_fractions,
            "completion_percentage": (completed_fractions / total_fractions * 100) if total_fractions > 0 else 0,
            "courses": [self.get_course_progress(course.course_id) for course in courses]
        }
        
        return summary
    
    def get_machine_utilization(
        self, 
        machine_id: str,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None
    ) -> Dict[str, Any]:
        """
        Lấy thông tin sử dụng của một máy xạ trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
        start_date : datetime.date, optional
            Ngày bắt đầu. Mặc định là ngày hiện tại.
        end_date : datetime.date, optional
            Ngày kết thúc. Mặc định là 7 ngày sau ngày bắt đầu.
            
        Returns
        -------
        Dict[str, Any]
            Thông tin sử dụng
        """
        machine = self.get_machine(machine_id)
        if not machine:
            logger.warning(f"Máy xạ trị có ID {machine_id} không tồn tại.")
            return {}
            
        scheduler = self.get_scheduler(machine_id)
        if not scheduler:
            logger.warning(f"Không tìm thấy lịch điều trị cho máy {machine_id}.")
            return {}
            
        if not start_date:
            start_date = datetime.date.today()
            
        if not end_date:
            end_date = start_date + datetime.timedelta(days=7)
            
        # Tổng số slot có sẵn và đã đặt lịch
        total_slots = 0
        booked_slots = 0
        free_slots = 0
        
        daily_stats = {}
        current_date = start_date
        
        while current_date <= end_date:
            # Lấy lịch của ngày hiện tại
            slots = scheduler.get_schedule(current_date)
            
            day_total = len(slots)
            day_booked = sum(1 for s in slots if s.is_booked)
            day_free = day_total - day_booked
            
            daily_stats[current_date.isoformat()] = {
                "total_slots": day_total,
                "booked_slots": day_booked,
                "free_slots": day_free,
                "utilization_percentage": (day_booked / day_total * 100) if day_total > 0 else 0
            }
            
            total_slots += day_total
            booked_slots += day_booked
            free_slots += day_free
            
            current_date += datetime.timedelta(days=1)
            
        utilization = {
            "machine_id": machine_id,
            "machine_name": machine.name,
            "machine_type": machine.machine_type.name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_slots": total_slots,
            "booked_slots": booked_slots,
            "free_slots": free_slots,
            "utilization_percentage": (booked_slots / total_slots * 100) if total_slots > 0 else 0,
            "daily_stats": daily_stats
        }
        
        return utilization
    
    def generate_treatment_report(self, course_id: str) -> Dict[str, Any]:
        """
        Tạo báo cáo điều trị cho một đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
            
        Returns
        -------
        Dict[str, Any]
            Báo cáo điều trị
        """
        course = self.get_course(course_id)
        if not course:
            logger.warning(f"Đợt điều trị có ID {course_id} không tồn tại.")
            return {}
            
        plan = self.get_plan(course.plan_id)
        if not plan:
            logger.warning(f"Không tìm thấy kế hoạch điều trị có ID {course.plan_id}.")
            return {}
            
        patient = self.get_patient(course.patient_id)
        if not patient:
            logger.warning(f"Không tìm thấy bệnh nhân có ID {course.patient_id}.")
            return {}
            
        # Phân tích trạng thái các phân đoạn
        fraction_stats = {status.name: 0 for status in FractionStatus}
        
        for fraction in course.fractions:
            fraction_stats[fraction.status.name] += 1
            
        # Tính toán lịch sử liều
        total_delivered_dose = 0
        for fraction in course.fractions:
            if fraction.status == FractionStatus.COMPLETED and fraction.delivered_dose:
                total_delivered_dose += fraction.delivered_dose
                
        expected_total_dose = course.fractionation.total_dose if course.fractionation else 0
        remaining_dose = max(0, expected_total_dose - total_delivered_dose)
        
        # Gộp thông tin lịch sử điều trị cho báo cáo
        treatment_history = []
        for fraction in sorted(course.fractions, key=lambda f: f.fraction_number):
            history_item = {
                "fraction_number": fraction.fraction_number,
                "status": fraction.status.name,
                "scheduled_date": fraction.scheduled_date.isoformat() if fraction.scheduled_date else None,
                "treatment_date": fraction.treatment_date.isoformat() if fraction.treatment_date else None,
                "delivered_dose": fraction.delivered_dose,
                "machine_id": fraction.machine_id,
                "operator": fraction.operator,
                "notes": fraction.notes,
                "errors": fraction.errors
            }
            treatment_history.append(history_item)
            
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_date": datetime.datetime.now().isoformat(),
            "course_id": course_id,
            "course_name": course.course_name,
            "patient_id": patient.patient_id,
            "patient_name": f"{patient.last_name}, {patient.first_name}",
            "patient_mrn": patient.medical_record_number,
            "plan_id": plan.plan_id,
            "plan_name": plan.name,
            "diagnosis": patient.diagnosis,
            "fractionation_scheme": {
                "total_dose": course.fractionation.total_dose if course.fractionation else 0,
                "number_of_fractions": course.fractionation.number_of_fractions if course.fractionation else 0,
                "dose_per_fraction": course.fractionation.dose_per_fraction if course.fractionation else 0
            },
            "treatment_status": course.status.name,
            "start_date": course.start_date.isoformat() if course.start_date else None,
            "expected_end_date": course.expected_end_date.isoformat() if course.expected_end_date else None,
            "actual_end_date": course.actual_end_date.isoformat() if course.actual_end_date else None,
            "total_fractions": len(course.fractions),
            "fraction_stats": fraction_stats,
            "total_delivered_dose": total_delivered_dose,
            "expected_total_dose": expected_total_dose,
            "remaining_dose": remaining_dose,
            "dose_completion_percentage": (total_delivered_dose / expected_total_dose * 100) if expected_total_dose > 0 else 0,
            "treatment_history": treatment_history
        }
        
        return report
    
    # Các phương thức kiểm tra chất lượng
    
    def verify_treatment_consistency(self, course_id: str) -> Dict[str, Any]:
        """
        Kiểm tra tính nhất quán của một đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
            
        Returns
        -------
        Dict[str, Any]
            Kết quả kiểm tra
        """
        course = self.get_course(course_id)
        if not course:
            logger.warning(f"Đợt điều trị có ID {course_id} không tồn tại.")
            return {
                "verified": False,
                "errors": ["Đợt điều trị không tồn tại"]
            }
            
        plan = self.get_plan(course.plan_id)
        if not plan:
            logger.warning(f"Không tìm thấy kế hoạch điều trị có ID {course.plan_id}.")
            return {
                "verified": False,
                "errors": ["Kế hoạch điều trị không tồn tại"]
            }
            
        # Kiểm tra tính nhất quán
        errors = []
        warnings = []
        
        # Kiểm tra số lượng phân đoạn
        expected_fractions = course.fractionation.number_of_fractions if course.fractionation else 0
        actual_fractions = len(course.fractions)
        
        if expected_fractions != actual_fractions:
            error = f"Số lượng phân đoạn không khớp: mong đợi {expected_fractions}, thực tế {actual_fractions}"
            errors.append(error)
            
        # Kiểm tra tính liên tục của số thứ tự phân đoạn
        fraction_numbers = [f.fraction_number for f in course.fractions]
        expected_numbers = list(range(1, actual_fractions + 1))
        
        if sorted(fraction_numbers) != expected_numbers:
            error = f"Số thứ tự phân đoạn không liên tục: {fraction_numbers}"
            errors.append(error)
            
        # Kiểm tra tính nhất quán của liều lượng
        completed_fractions = [f for f in course.fractions if f.status == FractionStatus.COMPLETED and f.delivered_dose is not None]
        
        if completed_fractions:
            expected_dose_per_fraction = course.fractionation.dose_per_fraction if course.fractionation else 0
            dose_variations = []
            
            for fraction in completed_fractions:
                variation = abs(fraction.delivered_dose - expected_dose_per_fraction) / expected_dose_per_fraction * 100
                dose_variations.append(variation)
                
                if variation > 5:  # Sai lệch > 5%
                    warning = f"Phân đoạn {fraction.fraction_number}: Sai lệch liều {variation:.2f}% (mong đợi {expected_dose_per_fraction}, thực tế {fraction.delivered_dose})"
                    warnings.append(warning)
            
            avg_variation = sum(dose_variations) / len(dose_variations)
            if avg_variation > 3:  # Sai lệch trung bình > 3%
                warning = f"Sai lệch liều trung bình: {avg_variation:.2f}%"
                warnings.append(warning)
                
        # Kiểm tra máy xạ trị nhất quán
        machine_ids = set(f.machine_id for f in course.fractions if f.machine_id is not None)
        if len(machine_ids) > 1:
            warning = f"Nhiều máy xạ trị được sử dụng: {', '.join(machine_ids)}"
            warnings.append(warning)
            
        # Kiểm tra tính liên tục của ngày điều trị
        treatment_dates = [f.treatment_date for f in completed_fractions if f.treatment_date is not None]
        if treatment_dates:
            sorted_dates = sorted(treatment_dates)
            date_diffs = []
            
            for i in range(1, len(sorted_dates)):
                diff = (sorted_dates[i] - sorted_dates[i-1]).days
                date_diffs.append(diff)
                
                if diff > 5:  # Khoảng cách > 5 ngày
                    warning = f"Khoảng cách lớn giữa các lần điều trị: {sorted_dates[i-1].isoformat()} -> {sorted_dates[i].isoformat()} ({diff} ngày)"
                    warnings.append(warning)
            
        result = {
            "verified": len(errors) == 0,
            "course_id": course_id,
            "plan_id": course.plan_id,
            "patient_id": course.patient_id,
            "errors": errors,
            "warnings": warnings
        }
        
        return result
    
    # Các phương thức tiện ích
    
    def export_treatment_data(self, course_id: str, export_path: str) -> bool:
        """
        Xuất dữ liệu điều trị ra file JSON.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
        export_path : str
            Đường dẫn xuất dữ liệu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        report = self.generate_treatment_report(course_id)
        if not report:
            logger.warning(f"Không thể tạo báo cáo điều trị cho đợt điều trị {course_id}.")
            return False
            
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xuất dữ liệu điều trị ra file {export_path}: {e}")
            return False
    
    def find_available_machine(
        self,
        date: datetime.date,
        machine_type: Optional[MachineType] = None
    ) -> List[Tuple[str, List[TimeSlot]]]:
        """
        Tìm kiếm máy xạ trị có sẵn vào một ngày cụ thể.
        
        Parameters
        ----------
        date : datetime.date
            Ngày cần tìm
        machine_type : MachineType, optional
            Loại máy xạ trị
            
        Returns
        -------
        List[Tuple[str, List[TimeSlot]]]
            Danh sách các máy có sẵn và các khung giờ còn trống
        """
        available_machines = []
        
        for machine_id, machine in self.machines.items():
            # Kiểm tra loại máy
            if machine_type and machine.machine_type != machine_type:
                continue
                
            # Kiểm tra trạng thái máy
            if machine.status != MachineStatus.OPERATIONAL:
                continue
                
            # Lấy lịch điều trị
            scheduler = self.get_scheduler(machine_id)
            if not scheduler:
                continue
                
            # Lấy các slot còn trống
            slots = scheduler.get_schedule(date)
            free_slots = [slot for slot in slots if not slot.is_booked]
            
            if free_slots:
                available_machines.append((machine_id, free_slots))
                
        return available_machines
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng thể của hệ thống.
        
        Returns
        -------
        Dict[str, Any]
            Thống kê hệ thống
        """
        total_patients = len(self.patients)
        total_plans = len(self.plans)
        total_courses = len(self.courses)
        total_machines = len(self.machines)
        
        active_courses = 0
        completed_courses = 0
        scheduled_fractions = 0
        completed_fractions = 0
        
        for course in self.courses.values():
            if course.status == TreatmentStatus.IN_PROGRESS:
                active_courses += 1
            elif course.status == TreatmentStatus.COMPLETED:
                completed_courses += 1
                
            for fraction in course.fractions:
                if fraction.status == FractionStatus.SCHEDULED:
                    scheduled_fractions += 1
                elif fraction.status == FractionStatus.COMPLETED:
                    completed_fractions += 1
                    
        # Phân loại máy theo loại và trạng thái
        machine_types = {}
        machine_status = {status.name: 0 for status in MachineStatus}
        
        for machine in self.machines.values():
            type_name = machine.machine_type.name
            if type_name in machine_types:
                machine_types[type_name] += 1
            else:
                machine_types[type_name] = 1
                
            machine_status[machine.status.name] += 1
            
        # Số lượng kế hoạch điều trị theo loại
        plan_types = {}
        for plan in self.plans.values():
            type_name = plan.plan_type.name if hasattr(plan, "plan_type") and plan.plan_type else "Unknown"
            if type_name in plan_types:
                plan_types[type_name] += 1
            else:
                plan_types[type_name] = 1
                
        statistics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_patients": total_patients,
            "total_plans": total_plans,
            "total_courses": total_courses,
            "active_courses": active_courses,
            "completed_courses": completed_courses,
            "total_machines": total_machines,
            "machine_by_type": machine_types,
            "machine_by_status": machine_status,
            "plan_by_type": plan_types,
            "scheduled_fractions": scheduled_fractions,
            "completed_fractions": completed_fractions,
            "system_load": {
                "active_courses_per_machine": active_courses / total_machines if total_machines > 0 else 0,
                "scheduled_fractions_per_active_course": scheduled_fractions / active_courses if active_courses > 0 else 0
            }
        }
        
        return statistics
