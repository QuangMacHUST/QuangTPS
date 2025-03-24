#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thực hiện điều trị trong QuangTPS.

Module này cung cấp các lớp và phương thức để mô phỏng quá trình thực hiện
điều trị trên máy gia tốc tuyến tính, bao gồm thiết lập và ghi nhận các thông số
điều trị như góc gantry, góc collimator, vị trí MLC, và đơn vị monitor.
"""

import os
import logging
import numpy as np
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

from quangtps.core.exceptions import TreatmentDeliveryError
from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.plan import TreatmentPlan
from quangtps.database.plan_db import PlanDB
from quangtps.database.beam_db import BeamDB
from quangtps.treatment.fractionation import FractionationScheme
from quangtps.planning.mlc import MLCSequence
from quangtps.treatment.machine.treatment_machine import TreatmentMachine

logger = logging.getLogger(__name__)


class DeliveryStatus:
    """Enum cho trạng thái của quá trình thực hiện điều trị."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"
    PAUSED = "PAUSED"


class BeamDelivery:
    """
    Lớp đại diện cho quá trình thực hiện điều trị của một chùm tia cụ thể.
    """

    def __init__(self, beam: Beam, fraction_number: int = 1):
        """
        Khởi tạo đối tượng BeamDelivery.

        Parameters
        ----------
        beam : Beam
            Chùm tia cần thực hiện
        fraction_number : int, optional
            Số thứ tự phân liều, mặc định là 1
        """
        self.beam = beam
        self.fraction_number = fraction_number
        self.status = DeliveryStatus.PENDING
        self.delivered_mu = 0.0
        self.start_time = None
        self.end_time = None
        self.current_control_point = 0
        self.interruptions = []
        self.errors = []
        self.delivery_record = {
            'beam_id': beam.id,
            'beam_name': beam.name,
            'fraction_number': fraction_number,
            'planned_mu': beam.monitor_units,
            'delivered_mu': 0.0,
            'control_points': [],
            'status': DeliveryStatus.PENDING,
            'start_time': None,
            'end_time': None
        }

    def start_delivery(self):
        """
        Bắt đầu quy trình thực hiện điều trị chùm tia.

        Returns
        -------
        bool
            True nếu bắt đầu thành công, False nếu không
        """
        if self.status not in [DeliveryStatus.PENDING, DeliveryStatus.PAUSED]:
            logger.warning(
                f"Cannot start delivery for beam {self.beam.name}: current status is {self.status}")
            return False

        self.status = DeliveryStatus.IN_PROGRESS
        self.start_time = datetime.datetime.now()
        self.delivery_record['start_time'] = self.start_time.isoformat()
        self.delivery_record['status'] = DeliveryStatus.IN_PROGRESS

        logger.info(
            f"Started delivery for beam {self.beam.name} at {self.start_time}")
        return True

    def deliver_control_point(self, control_point_index: int):
        """
        Thực hiện điều trị tại một control point cụ thể.
        
        Parameters
        ----------
        control_point_index : int
            Chỉ số của control point cần thực hiện

        Returns
        -------
        bool
            True nếu thực hiện thành công, False nếu không
        """
        if self.status != DeliveryStatus.IN_PROGRESS:
            logger.warning(
                f"Cannot deliver control point: beam {self.beam.name} is not in progress")
            return False

        try:
            # Lấy thông tin control point
            control_points = self.beam.control_points
            if control_point_index >= len(control_points):
                logger.error(
                    f"Control point index {control_point_index} out of range for beam {self.beam.name}")
                return False

            control_point = control_points[control_point_index]
            cp_mu = control_point.get(
                'cumulative_meterset_weight', 0) * self.beam.monitor_units
            current_mu = cp_mu - self.delivered_mu

            # Ghi nhận control point đã thực hiện
            cp_record = {
                'index': control_point_index,
                'gantry_angle': control_point.get('gantry_angle'),
                'collimator_angle': control_point.get('collimator_angle'),
                'couch_angle': control_point.get('couch_angle'),
                'cumulative_mu': cp_mu,
                'delivered_mu': current_mu,
                'mlc_positions': control_point.get('mlc_positions', []),
                'jaw_positions': control_point.get('jaw_positions', []),
                'time': datetime.datetime.now().isoformat()
            }

            self.delivery_record['control_points'].append(cp_record)
            self.delivered_mu = cp_mu
            self.delivery_record['delivered_mu'] = self.delivered_mu
            self.current_control_point = control_point_index

            logger.info(
                f"Delivered control point {control_point_index} for beam {self.beam.name}, cumulative MU: {cp_mu}")
            return True

        except Exception as e:
            error_msg = f"Error delivering control point {control_point_index} for beam {self.beam.name}: {str(e)}"
            logger.error(error_msg)
            self.errors.append({
                'time': datetime.datetime.now().isoformat(),
                'message': error_msg
            })
            return False

    def pause_delivery(self, reason: str = ""):
        """
        Tạm dừng quá trình thực hiện điều trị.
        
        Parameters
        ----------
        reason : str, optional
            Lý do tạm dừng

        Returns
        -------
        bool
            True nếu tạm dừng thành công, False nếu không
        """
        if self.status != DeliveryStatus.IN_PROGRESS:
            return False

        self.status = DeliveryStatus.PAUSED
        self.delivery_record['status'] = DeliveryStatus.PAUSED
        self.interruptions.append({
            'time': datetime.datetime.now().isoformat(),
            'delivered_mu': self.delivered_mu,
            'control_point': self.current_control_point,
            'reason': reason
        })

        logger.info(
            f"Paused delivery for beam {self.beam.name} at MU: {self.delivered_mu}")
        return True

    def resume_delivery(self):
        """
        Tiếp tục quá trình thực hiện điều trị sau khi tạm dừng.

        Returns
        -------
        bool
            True nếu tiếp tục thành công, False nếu không
        """
        if self.status != DeliveryStatus.PAUSED:
            return False

        self.status = DeliveryStatus.IN_PROGRESS
        self.delivery_record['status'] = DeliveryStatus.IN_PROGRESS

        logger.info(f"Resumed delivery for beam {self.beam.name}")
        return True

    def complete_delivery(self):
        """
        Hoàn thành quá trình thực hiện điều trị.

        Returns
        -------
        bool
            True nếu hoàn thành thành công, False nếu không
        """
        if self.status != DeliveryStatus.IN_PROGRESS:
            return False

        self.status = DeliveryStatus.COMPLETED
        self.end_time = datetime.datetime.now()
        self.delivery_record['status'] = DeliveryStatus.COMPLETED
        self.delivery_record['end_time'] = self.end_time.isoformat()

        # Kiểm tra xem tất cả MU đã được thực hiện chưa
        if abs(self.delivered_mu - self.beam.monitor_units) > 0.1:
            logger.warning(
                f"Beam {self.beam.name} completed with delivered MU {self.delivered_mu} vs planned MU {self.beam.monitor_units}")

        logger.info(
            f"Completed delivery for beam {self.beam.name} at {self.end_time}")
        return True

    def abort_delivery(self, reason: str):
        """
        Dừng hẳn quá trình thực hiện điều trị.
        
        Parameters
        ----------
        reason : str
            Lý do dừng điều trị

        Returns
        -------
        bool
            True nếu dừng thành công, False nếu không
        """
        if self.status not in [DeliveryStatus.IN_PROGRESS, DeliveryStatus.PAUSED]:
            return False

        self.status = DeliveryStatus.INTERRUPTED
        self.end_time = datetime.datetime.now()
        self.delivery_record['status'] = DeliveryStatus.INTERRUPTED
        self.delivery_record['end_time'] = self.end_time.isoformat()
        self.interruptions.append({
            'time': self.end_time.isoformat(),
            'delivered_mu': self.delivered_mu,
            'control_point': self.current_control_point,
            'reason': reason,
            'type': 'abort'
        })

        logger.warning(
            f"Aborted delivery for beam {self.beam.name} at MU: {self.delivered_mu}. Reason: {reason}")
        return True

    def get_delivery_record(self) -> Dict[str, Any]:
        """
        Lấy bản ghi về quá trình thực hiện điều trị.

        Returns
        -------
        Dict[str, Any]
            Bản ghi chi tiết về quá trình thực hiện điều trị
        """
        # Cập nhật trạng thái mới nhất
        self.delivery_record['status'] = self.status
        self.delivery_record['delivered_mu'] = self.delivered_mu
        self.delivery_record['current_control_point'] = self.current_control_point
        self.delivery_record['interruptions'] = self.interruptions
        self.delivery_record['errors'] = self.errors

        return self.delivery_record


class TreatmentSession:
    """
    Lớp đại diện cho một buổi điều trị hoàn chỉnh.
    """

    def __init__(self, patient_id: str, plan_id: str, fraction_number: int = 1):
        """
        Khởi tạo buổi điều trị.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        plan_id : str
            ID của kế hoạch điều trị
        fraction_number : int, optional
            Số thứ tự phân liều, mặc định là 1
        """
        self.patient_id = patient_id
        self.plan_id = plan_id
        self.fraction_number = fraction_number
        self.status = DeliveryStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.beam_deliveries = []
        self.current_beam_index = -1
        self.setup_errors = {}
        self.session_notes = []

        # Lấy thông tin kế hoạch
        self.plan_db = PlanDB()
        self.beam_db = BeamDB()
        self.plan = self.plan_db.get_plan(plan_id)

        if not self.plan:
            raise TreatmentDeliveryError(f"Plan with ID {plan_id} not found")

        # Lấy danh sách chùm tia
        self.beams = self.beam_db.get_beams_by_plan(plan_id)
        if not self.beams:
            raise TreatmentDeliveryError(
                f"No beams found for plan with ID {plan_id}")

        # Tạo đối tượng BeamDelivery cho từng chùm tia
        for beam in self.beams:
            self.beam_deliveries.append(BeamDelivery(beam, fraction_number))

    def start_session(self):
        """
        Bắt đầu buổi điều trị.

        Returns
        -------
        bool
            True nếu bắt đầu thành công, False nếu không
        """
        if self.status != DeliveryStatus.PENDING:
            logger.warning(
                f"Cannot start session: current status is {self.status}")
            return False

        self.status = DeliveryStatus.IN_PROGRESS
        self.start_time = datetime.datetime.now()

        logger.info(
            f"Started treatment session for plan {self.plan_id}, fraction {self.fraction_number}")
        return True

    def setup_patient(self, setup_details: Dict[str, Any]):
        """
        Ghi nhận thông tin thiết lập bệnh nhân.
        
        Parameters
        ----------
        setup_details : Dict[str, Any]
            Chi tiết về thiết lập bệnh nhân

        Returns
        -------
        bool
            True nếu thiết lập thành công, False nếu không
        """
        if self.status != DeliveryStatus.IN_PROGRESS:
            logger.warning("Cannot setup patient: session is not in progress")
            return False

        # Lưu thông tin thiết lập
        self.setup_details = {
            'time': datetime.datetime.now().isoformat(),
            'details': setup_details
        }

        # Ghi nhận sai số thiết lập nếu có
        if 'setup_errors' in setup_details:
            self.setup_errors = setup_details['setup_errors']

        logger.info(
            f"Patient setup completed for session {self.plan_id}, fraction {self.fraction_number}")
        return True

    def start_next_beam(self):
        """
        Bắt đầu thực hiện chùm tia tiếp theo.

        Returns
        -------
        BeamDelivery
            Đối tượng BeamDelivery cho chùm tia tiếp theo, hoặc None nếu không còn chùm tia nào
        """
        if self.status != DeliveryStatus.IN_PROGRESS:
            logger.warning(
                "Cannot start next beam: session is not in progress")
            return None

        if self.current_beam_index >= len(self.beam_deliveries) - 1:
            logger.warning("No more beams to deliver in this session")
            return None

        # Tăng chỉ số chùm tia hiện tại
        self.current_beam_index += 1
        current_beam = self.beam_deliveries[self.current_beam_index]

        # Bắt đầu thực hiện chùm tia
        success = current_beam.start_delivery()
        if not success:
            logger.error(
                f"Failed to start delivery for beam {current_beam.beam.name}")
            return None

        logger.info(
            f"Started delivery for beam {current_beam.beam.name}, {self.current_beam_index + 1}/{len(self.beam_deliveries)}")
        return current_beam

    def complete_current_beam(self):
        """
        Hoàn thành chùm tia hiện tại.

        Returns
        -------
        bool
            True nếu hoàn thành thành công, False nếu không
        """
        if self.status != DeliveryStatus.IN_PROGRESS or self.current_beam_index < 0:
            logger.warning("No current beam to complete")
            return False

        current_beam = self.beam_deliveries[self.current_beam_index]
        success = current_beam.complete_delivery()

        if success:
            logger.info(
                f"Completed delivery for beam {current_beam.beam.name}")

            # Kiểm tra xem đã thực hiện tất cả chùm tia chưa
            if self.current_beam_index == len(self.beam_deliveries) - 1:
                self.complete_session()

            return True
        else:
            logger.error(
                f"Failed to complete delivery for beam {current_beam.beam.name}")
            return False

    def pause_session(self, reason: str = ""):
        """
        Tạm dừng buổi điều trị.
        
        Parameters
        ----------
        reason : str, optional
            Lý do tạm dừng

        Returns
        -------
        bool
            True nếu tạm dừng thành công, False nếu không
        """
        if self.status != DeliveryStatus.IN_PROGRESS:
            return False

        self.status = DeliveryStatus.PAUSED

        # Nếu đang thực hiện một chùm tia, tạm dừng chùm tia đó
        if 0 <= self.current_beam_index < len(self.beam_deliveries):
            current_beam = self.beam_deliveries[self.current_beam_index]
            current_beam.pause_delivery(reason)

        logger.info(
            f"Paused treatment session for plan {self.plan_id}. Reason: {reason}")
        return True

    def resume_session(self):
        """
        Tiếp tục buổi điều trị sau khi tạm dừng.

        Returns
        -------
        bool
            True nếu tiếp tục thành công, False nếu không
        """
        if self.status != DeliveryStatus.PAUSED:
            return True

        self.status = DeliveryStatus.IN_PROGRESS

        # Nếu đang ở giữa một chùm tia, tiếp tục chùm tia đó
        if 0 <= self.current_beam_index < len(self.beam_deliveries):
            current_beam = self.beam_deliveries[self.current_beam_index]
            if current_beam.status == DeliveryStatus.PAUSED:
                current_beam.resume_delivery()

        logger.info(f"Resumed treatment session for plan {self.plan_id}")
        return True

    def complete_session(self):
        """
        Hoàn thành buổi điều trị.

        Returns
        -------
        bool
            True nếu hoàn thành thành công, False nếu không
        """
        if self.status not in [DeliveryStatus.IN_PROGRESS, DeliveryStatus.PAUSED]:
            logger.warning(
                f"Cannot complete session: current status is {self.status}")
            return False

        # Hoàn thành chùm tia hiện tại nếu đang thực hiện
        if 0 <= self.current_beam_index < len(self.beam_deliveries):
            current_beam = self.beam_deliveries[self.current_beam_index]
            if current_beam.status == DeliveryStatus.IN_PROGRESS:
                current_beam.complete_delivery()

        self.status = DeliveryStatus.COMPLETED
        self.end_time = datetime.datetime.now()

        logger.info(
            f"Completed treatment session for plan {self.plan_id}, fraction {self.fraction_number}")
        return True

    def abort_session(self, reason: str):
        """
        Hủy bỏ buổi điều trị.

        Parameters
        ----------
        reason : str
            Lý do hủy bỏ

        Returns
        -------
        bool
            True nếu hủy bỏ thành công, False nếu không
        """
        if self.status not in [DeliveryStatus.IN_PROGRESS, DeliveryStatus.PAUSED]:
            return False

        # Hủy chùm tia hiện tại nếu đang thực hiện
        if 0 <= self.current_beam_index < len(self.beam_deliveries):
            current_beam = self.beam_deliveries[self.current_beam_index]
            if current_beam.status in [DeliveryStatus.IN_PROGRESS, DeliveryStatus.PAUSED]:
                current_beam.abort_delivery(reason)

        self.status = DeliveryStatus.INTERRUPTED
        self.end_time = datetime.datetime.now()

        logger.warning(
            f"Aborted treatment session for plan {self.plan_id}. Reason: {reason}")
        return True

    def add_note(self, note: str, author: str = "System"):
        """
        Thêm ghi chú cho buổi điều trị.

        Parameters
        ----------
        note : str
            Nội dung ghi chú
        author : str, optional
            Người tạo ghi chú, mặc định là "System"
        """
        self.session_notes.append({
            'time': datetime.datetime.now().isoformat(),
            'note': note,
            'author': author
        })

        logger.info(f"Added note to session {self.plan_id}: {note}")

    def get_session_record(self) -> Dict[str, Any]:
        """
        Lấy bản ghi về buổi điều trị.
        
        Returns
        -------
        Dict[str, Any]
            Bản ghi chi tiết về buổi điều trị
        """
        beam_records = [bd.get_delivery_record()
                        for bd in self.beam_deliveries]

        return {
            'patient_id': self.patient_id,
            'plan_id': self.plan_id,
            'plan_name': self.plan.get('name', ''),
            'fraction_number': self.fraction_number,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'setup_details': getattr(self, 'setup_details', {}),
            'setup_errors': self.setup_errors,
            'notes': self.session_notes,
            'beams': beam_records,
            'current_beam_index': self.current_beam_index
        }


class TreatmentStatus(str, Enum):
    """Enum đại diện cho trạng thái của chuỗi điều trị."""
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    CANCELED = "CANCELED"
    ON_HOLD = "ON_HOLD"


class FractionStatus(str, Enum):
    """Enum đại diện cho trạng thái của phân liều điều trị."""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    MISSED = "MISSED"
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"


class TreatmentFraction:
    """
    Lớp đại diện cho một phân liều điều trị.
    """

    def __init__(self, fraction_number: int, plan_id: str, scheduled_date=None):
        """
        Khởi tạo đối tượng TreatmentFraction.
        
        Parameters
        ----------
        fraction_number : int
            Số thứ tự phân liều
        plan_id : str
            ID của kế hoạch điều trị
        scheduled_date : datetime.date, optional
            Ngày dự kiến điều trị, mặc định là None
        """
        self.fraction_number = fraction_number
        self.plan_id = plan_id
        self.scheduled_date = scheduled_date
        self.status = FractionStatus.PENDING
        self.actual_date = None
        self.beam_records = []
        self.delivery_notes = []
        self.completion_percentage = 0.0
        self.tolerance_exceeded = False
        self.verification_passed = True
        self.verified_by = None
        self.verified_at = None

    def mark_completed(self, actual_date=None, verified_by=None):
        """Đánh dấu phân liều đã hoàn thành."""
        self.status = FractionStatus.COMPLETED
        self.actual_date = actual_date or datetime.datetime.now().date()
        self.completion_percentage = 100.0
        self.verified_by = verified_by
        self.verified_at = datetime.datetime.now()

    def mark_partial(self, completion_percentage: float, reason: str):
        """Đánh dấu phân liều hoàn thành một phần."""
        self.status = FractionStatus.PARTIAL
        self.completion_percentage = completion_percentage
        self.delivery_notes.append({
            'time': datetime.datetime.now().isoformat(),
            'note': f"Partial delivery: {reason}",
            'completion_percentage': completion_percentage
        })

    def mark_missed(self, reason: str):
        """Đánh dấu phân liều bị bỏ lỡ."""
        self.status = FractionStatus.MISSED
        self.delivery_notes.append({
            'time': datetime.datetime.now().isoformat(),
            'note': f"Missed fraction: {reason}"
        })

    def add_beam_record(self, beam_record: Dict[str, Any]):
        """Thêm bản ghi thực hiện chùm tia."""
        self.beam_records.append(beam_record)

    def add_note(self, note: str, author: str = "System"):
        """Thêm ghi chú về phân liều."""
        self.delivery_notes.append({
            'time': datetime.datetime.now().isoformat(),
            'author': author,
            'note': note
        })

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi đối tượng thành dictionary."""
        return {
            'fraction_number': self.fraction_number,
            'plan_id': self.plan_id,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'status': self.status,
            'beam_records': self.beam_records,
            'delivery_notes': self.delivery_notes,
            'completion_percentage': self.completion_percentage,
            'tolerance_exceeded': self.tolerance_exceeded,
            'verification_passed': self.verification_passed,
            'verified_by': self.verified_by,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentFraction':
        """Tạo đối tượng từ dictionary."""
        fraction = cls(
            fraction_number=data['fraction_number'],
            plan_id=data['plan_id']
        )

        # Khôi phục các thuộc tính
        if data.get('scheduled_date'):
            fraction.scheduled_date = datetime.datetime.fromisoformat(
                data['scheduled_date']).date()

        if data.get('actual_date'):
            fraction.actual_date = datetime.datetime.fromisoformat(
                data['actual_date']).date()

        fraction.status = data.get('status', FractionStatus.PENDING)
        fraction.beam_records = data.get('beam_records', [])
        fraction.delivery_notes = data.get('delivery_notes', [])
        fraction.completion_percentage = data.get('completion_percentage', 0.0)
        fraction.tolerance_exceeded = data.get('tolerance_exceeded', False)
        fraction.verification_passed = data.get('verification_passed', True)
        fraction.verified_by = data.get('verified_by')

        if data.get('verified_at'):
            fraction.verified_at = datetime.datetime.fromisoformat(
                data['verified_at'])
        
        return fraction


class TreatmentCourse:
    """
    Lớp đại diện cho một chuỗi điều trị.
    """

    def __init__(self, course_id: str, plan_id: str, name: str = None, fractionation_scheme=None):
        """
        Khởi tạo đối tượng TreatmentCourse.
        
        Parameters
        ----------
        course_id : str
            ID của chuỗi điều trị
        plan_id : str
            ID của kế hoạch điều trị
        name : str, optional
            Tên của chuỗi điều trị, mặc định là None
        fractionation_scheme : FractionationScheme, optional
            Phương thức phân liều, mặc định là None
        """
        self.course_id = course_id
        self.plan_id = plan_id
        self.name = name or f"Course for plan {plan_id}"
        self.fractionation_scheme = fractionation_scheme
        self.status = TreatmentStatus.PLANNED
        self.start_date = None
        self.completion_date = None
        self.machine_id = None
        self.fractions = {}  # Dict[int, TreatmentFraction]
        self.notes = []
        self.metadata = {}

        # Tạo các phân liều nếu có phương thức phân liều
        if fractionation_scheme:
            self._create_fractions()

    def _create_fractions(self):
        """Tạo các phân liều dựa trên phương thức phân liều."""
        if not self.fractionation_scheme:
            return

        num_fractions = self.fractionation_scheme.number_of_fractions
        for i in range(1, num_fractions + 1):
            self.fractions[i] = TreatmentFraction(
                fraction_number=i,
                plan_id=self.plan_id
            )

    def schedule_fractions(self, start_date, days_of_week=None):
        """
        Lập lịch cho các phân liều.
        
        Parameters
        ----------
        start_date : datetime.date
            Ngày bắt đầu điều trị
        days_of_week : List[int], optional
            Danh sách các ngày trong tuần để điều trị (0=Monday, 6=Sunday), by default None
        """
        import datetime
        
        self.start_date = start_date
        current_date = start_date

        # Mặc định điều trị các ngày trong tuần (thứ 2 - thứ 6)
        if days_of_week is None:
            days_of_week = [0, 1, 2, 3, 4]  # Monday to Friday

        for i in sorted(self.fractions.keys()):
            # Tìm ngày tiếp theo phù hợp với lịch
            while current_date.weekday() not in days_of_week:
                current_date += datetime.timedelta(days=1)

            self.fractions[i].scheduled_date = current_date
            self.fractions[i].status = FractionStatus.SCHEDULED

            # Chuyển sang ngày tiếp theo
            current_date += datetime.timedelta(days=1)
        
    def assign_machine(self, machine_id: str):
        """Gán máy điều trị cho chuỗi điều trị."""
        self.machine_id = machine_id

    def start_treatment(self, start_date=None):
        """Bắt đầu chuỗi điều trị."""
        self.status = TreatmentStatus.IN_PROGRESS
        self.start_date = start_date or datetime.datetime.now().date()

        self.notes.append({
            'date': datetime.datetime.now().isoformat(),
            'note': f"Treatment course started on {self.start_date}",
            'type': 'STATUS_CHANGE'
        })

    def update_fraction_status(self, fraction_number: int, status: FractionStatus, **kwargs):
        """Cập nhật trạng thái của phân liều."""
        if fraction_number not in self.fractions:
            return False
            
        fraction = self.fractions[fraction_number]
        fraction.status = status
        
        if status == FractionStatus.COMPLETED:
            fraction.mark_completed(**kwargs)
        elif status == FractionStatus.PARTIAL:
            fraction.mark_partial(kwargs.get('completion_percentage', 0), kwargs.get('reason', ''))
        elif status == FractionStatus.MISSED:
            fraction.mark_missed(kwargs.get('reason', ''))
            
        # Kiểm tra nếu tất cả phân liều đã hoàn thành
        self._check_course_completion()
        
        return True
    
    def _check_course_completion(self):
        """Kiểm tra và cập nhật trạng thái chuỗi điều trị."""
        # Nếu chuỗi điều trị đã hoàn thành thì bỏ qua
        if self.status == TreatmentStatus.COMPLETED:
            return

        # Kiểm tra trạng thái các phân liều
        all_completed = True
        for fraction in self.fractions.values():
            if fraction.status not in [FractionStatus.COMPLETED, FractionStatus.MISSED]:
                all_completed = False
                break

        # Cập nhật trạng thái chuỗi điều trị nếu tất cả phân liều đã hoàn thành
        if all_completed:
            self.status = TreatmentStatus.COMPLETED
            self.completion_date = datetime.datetime.now().date()

            self.notes.append({
                'date': datetime.datetime.now().isoformat(),
                'note': f"Treatment course completed on {self.completion_date}",
                'type': 'STATUS_CHANGE'
            })

    def add_note(self, note: str, author: str = "System", note_type: str = "GENERAL"):
        """Thêm ghi chú về chuỗi điều trị."""
        self.notes.append({
            'date': datetime.datetime.now().isoformat(),
            'author': author,
            'note': note,
            'type': note_type
        })

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi đối tượng thành dictionary."""
        fractions_dict = {str(k): v.to_dict()
                          for k, v in self.fractions.items()}

        result = {
            'course_id': self.course_id,
            'plan_id': self.plan_id,
            'name': self.name,
            'status': self.status,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'machine_id': self.machine_id,
            'fractions': fractions_dict,
            'notes': self.notes,
            'metadata': self.metadata
        }

        if self.fractionation_scheme:
            result['fractionation_scheme'] = self.fractionation_scheme.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentCourse':
        """Tạo đối tượng từ dictionary."""
        from quangtps.treatment.fractionation import FractionationScheme

        # Tạo đối tượng cơ bản
        course = cls(
            course_id=data['course_id'],
            plan_id=data['plan_id'],
            name=data.get('name')
        )

        # Khôi phục fractionation scheme nếu có
        if 'fractionation_scheme' in data:
            course.fractionation_scheme = FractionationScheme.from_dict(
                data['fractionation_scheme'])

        # Khôi phục các thuộc tính khác
        course.status = data.get('status', TreatmentStatus.PLANNED)

        if data.get('start_date'):
            course.start_date = datetime.datetime.fromisoformat(
                data['start_date']).date()

        if data.get('completion_date'):
            course.completion_date = datetime.datetime.fromisoformat(
                data['completion_date']).date()

        course.machine_id = data.get('machine_id')
        course.notes = data.get('notes', [])
        course.metadata = data.get('metadata', {})

        # Khôi phục các phân liều
        if 'fractions' in data:
            for fraction_num, fraction_data in data['fractions'].items():
                course.fractions[int(fraction_num)] = TreatmentFraction.from_dict(
                    fraction_data)

        return course

    def get_treatment_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt về chuỗi điều trị."""
        completed_fractions = sum(
            1 for f in self.fractions.values() if f.status == FractionStatus.COMPLETED)
        partial_fractions = sum(
            1 for f in self.fractions.values() if f.status == FractionStatus.PARTIAL)
        missed_fractions = sum(
            1 for f in self.fractions.values() if f.status == FractionStatus.MISSED)
        pending_fractions = sum(1 for f in self.fractions.values() if f.status in [
                                FractionStatus.PENDING, FractionStatus.SCHEDULED])

        total_fractions = len(self.fractions)
        progress_percentage = (
            completed_fractions / total_fractions) * 100 if total_fractions > 0 else 0

        return {
            'course_id': self.course_id,
            'name': self.name,
            'plan_id': self.plan_id,
            'status': self.status,
            'total_fractions': total_fractions,
            'completed_fractions': completed_fractions,
            'partial_fractions': partial_fractions,
            'missed_fractions': missed_fractions,
            'pending_fractions': pending_fractions,
            'progress_percentage': progress_percentage,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'machine_id': self.machine_id
        }


class TreatmentDeliveryManager:
    """
    Lớp quản lý quá trình thực hiện điều trị.
    """

    def __init__(self):
        """Khởi tạo manager"""
        self.active_sessions = {}
        self.completed_sessions = {}
        self.plan_db = PlanDB()
        self.session_history = {}

    def start_treatment_session(self, patient_id: str, plan_id: str, fraction_number: int = None) -> Optional[str]:
        """
        Bắt đầu một buổi điều trị mới.

        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        plan_id : str
            ID của kế hoạch điều trị
        fraction_number : int, optional
            Số thứ tự phân liều, nếu None thì sẽ tự động tính toán
        
        Returns
        -------
        Optional[str]
            ID của buổi điều trị nếu thành công, None nếu thất bại
        """
        try:
            # Kiểm tra kế hoạch
            plan = self.plan_db.get_plan(plan_id)
            if not plan:
                logger.error(f"Plan with ID {plan_id} not found")
                return None

            # Xác định số phân liều tiếp theo nếu không được chỉ định
            if fraction_number is None:
                fraction_history = self.get_fraction_history(plan_id)
                completed_fractions = [f['fraction_number']
                                       for f in fraction_history]
                if not completed_fractions:
                    fraction_number = 1
                else:
                    fraction_number = max(completed_fractions) + 1

            # Kiểm tra xem phân liều có vượt quá số phân liều được kế hoạch không
            if 'fraction_count' in plan and fraction_number > plan['fraction_count']:
                logger.warning(
                    f"Fraction {fraction_number} exceeds planned fraction count {plan['fraction_count']}")
                # Vẫn cho phép điều trị vượt quá (có thể là điều trị boost)

            # Tạo ID phiên điều trị mới
            session_id = f"{plan_id}_fraction_{fraction_number}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Tạo phiên điều trị mới
            session = TreatmentSession(patient_id, plan_id, fraction_number)
            success = session.start_session()

            if success:
                # Lưu phiên điều trị vào danh sách active
                self.active_sessions[session_id] = session
                logger.info(
                    f"Started treatment session {session_id} for plan {plan_id}, fraction {fraction_number}")
                return session_id
            else:
                logger.error(
                    f"Failed to start treatment session for plan {plan_id}")
                return None

        except Exception as e:
            logger.error(f"Error starting treatment session: {str(e)}")
            return None

    def get_active_session(self, session_id: str) -> Optional[TreatmentSession]:
        """
        Lấy phiên điều trị đang hoạt động.

        Parameters
        ----------
        session_id : str
            ID của phiên điều trị
        
        Returns
        -------
        Optional[TreatmentSession]
            Phiên điều trị nếu tồn tại, None nếu không
        """
        return self.active_sessions.get(session_id)

    def get_fraction_history(self, plan_id: str) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử các phân liều đã thực hiện cho một kế hoạch.

        Parameters
        ----------
        plan_id : str
            ID của kế hoạch điều trị
        
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các phân liều đã thực hiện
        """
        if plan_id not in self.session_history:
            # Thực tế phải load từ cơ sở dữ liệu
            self.session_history[plan_id] = []

        return self.session_history[plan_id]

    def complete_session(self, session_id: str) -> bool:
        """
        Hoàn thành phiên điều trị.

        Parameters
        ----------
        session_id : str
            ID của phiên điều trị
        
        Returns
        -------
        bool
            True nếu hoàn thành thành công, False nếu không
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(
                f"Session {session_id} not found in active sessions")
            return False

        success = session.complete_session()
        if success:
            # Chuyển từ active sang completed
            self.completed_sessions[session_id] = session
            self.active_sessions.pop(session_id)

            # Lưu vào lịch sử
            if session.plan_id not in self.session_history:
                self.session_history[session.plan_id] = []

            self.session_history[session.plan_id].append(
                session.get_session_record())

            logger.info(f"Completed treatment session {session_id}")
            return True
        else:
            logger.error(f"Failed to complete treatment session {session_id}")
            return False

    def abort_session(self, session_id: str, reason: str) -> bool:
        """
        Hủy bỏ phiên điều trị.

        Parameters
        ----------
        session_id : str
            ID của phiên điều trị
        reason : str
            Lý do hủy bỏ
        
        Returns
        -------
        bool
            True nếu hủy bỏ thành công, False nếu không
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.warning(
                f"Session {session_id} not found in active sessions")
            return False

        success = session.abort_session(reason)
        if success:
            # Chuyển từ active sang completed
            self.completed_sessions[session_id] = session
            self.active_sessions.pop(session_id)

            # Lưu vào lịch sử
            if session.plan_id not in self.session_history:
                self.session_history[session.plan_id] = []

            self.session_history[session.plan_id].append(
                session.get_session_record())

            logger.warning(f"Aborted treatment session {session_id}: {reason}")
            return True
        else:
            logger.error(f"Failed to abort treatment session {session_id}")
            return False

    def get_session_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy bản ghi về một phiên điều trị.
        
        Parameters
        ----------
        session_id : str
            ID của phiên điều trị
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Bản ghi chi tiết về phiên điều trị nếu tồn tại, None nếu không
        """
        # Kiểm tra trong active sessions
        session = self.active_sessions.get(session_id)
        if session:
            return session.get_session_record()

        # Kiểm tra trong completed sessions
        session = self.completed_sessions.get(session_id)
        if session:
            return session.get_session_record()

        # Không tìm thấy
        return None

    def export_treatment_record(self, session_id: str, file_path: str) -> bool:
        """
        Xuất bản ghi điều trị ra file.

        Parameters
        ----------
        session_id : str
            ID của phiên điều trị
        file_path : str
            Đường dẫn đến file xuất

        Returns
        -------
        bool
            True nếu xuất thành công, False nếu không
        """
        import json

        record = self.get_session_record(session_id)
        if not record:
            logger.error(f"No record found for session {session_id}")
            return False

        try:
            with open(file_path, 'w') as f:
                json.dump(record, f, indent=2)

            logger.info(
                f"Exported treatment record for session {session_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting treatment record: {str(e)}")
            return False
