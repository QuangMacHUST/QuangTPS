#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module lập lịch điều trị (Treatment Scheduler).

Module này cung cấp các lớp và phương thức để lập lịch điều trị xạ trị cho nhiều bệnh nhân,
tối ưu hóa việc sử dụng máy xạ trị và quản lý thời gian điều trị.
"""

import logging
import datetime
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
import calendar

from quangtps.treatment.treatment_delivery import TreatmentCourse, TreatmentStatus, FractionStatus, TreatmentFraction
from quangtps.treatment.machine.machine_specs import MachineSpecs

logger = logging.getLogger(__name__)


class TimeSlot:
    """
    Lớp đại diện cho một khoảng thời gian điều trị.
    
    Lớp này chứa thông tin về thời gian bắt đầu, thời gian kết thúc,
    và trạng thái của khoảng thời gian.
    """
    
    def __init__(
        self,
        start_time: datetime.time,
        end_time: datetime.time,
        date: datetime.date
    ):
        """
        Khởi tạo một khoảng thời gian.
        
        Parameters
        ----------
        start_time : datetime.time
            Thời gian bắt đầu
        end_time : datetime.time
            Thời gian kết thúc
        date : datetime.date
            Ngày của khoảng thời gian
        """
        self.slot_id = str(uuid.uuid4())
        self.start_time = start_time
        self.end_time = end_time
        self.date = date
        self.is_available = True
        self.course_id = None
        self.fraction_id = None
        self.patient_id = None
        self.notes = ""
    
    def reserve(self, course_id: str, fraction_id: str, patient_id: str, notes: str = ""):
        """
        Đặt chỗ khoảng thời gian.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
        fraction_id : str
            ID của phân đoạn
        patient_id : str
            ID của bệnh nhân
        notes : str, optional
            Ghi chú
        """
        self.is_available = False
        self.course_id = course_id
        self.fraction_id = fraction_id
        self.patient_id = patient_id
        self.notes = notes
    
    def release(self):
        """Giải phóng khoảng thời gian."""
        self.is_available = True
        self.course_id = None
        self.fraction_id = None
        self.patient_id = None
        self.notes = ""
    
    def overlaps(self, other: 'TimeSlot') -> bool:
        """
        Kiểm tra xem khoảng thời gian có chồng lấn với khoảng thời gian khác không.
        
        Parameters
        ----------
        other : TimeSlot
            Khoảng thời gian khác
            
        Returns
        -------
        bool
            True nếu chồng lấn, False nếu không
        """
        if self.date != other.date:
            return False
            
        return (self.start_time < other.end_time and self.end_time > other.start_time)
    
    def duration_minutes(self) -> int:
        """
        Tính thời lượng của khoảng thời gian.
        
        Returns
        -------
        int
            Thời lượng (phút)
        """
        start_minutes = self.start_time.hour * 60 + self.start_time.minute
        end_minutes = self.end_time.hour * 60 + self.end_time.minute
        return end_minutes - start_minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin khoảng thời gian thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin khoảng thời gian
        """
        return {
            "slot_id": self.slot_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "date": self.date.isoformat(),
            "is_available": self.is_available,
            "course_id": self.course_id,
            "fraction_id": self.fraction_id,
            "patient_id": self.patient_id,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeSlot':
        """
        Tạo đối tượng TimeSlot từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin khoảng thời gian
            
        Returns
        -------
        TimeSlot
            Đối tượng TimeSlot
        """
        slot = cls(
            start_time=datetime.time.fromisoformat(data["start_time"]),
            end_time=datetime.time.fromisoformat(data["end_time"]),
            date=datetime.date.fromisoformat(data["date"])
        )
        
        slot.slot_id = data["slot_id"]
        slot.is_available = data["is_available"]
        slot.course_id = data["course_id"]
        slot.fraction_id = data["fraction_id"]
        slot.patient_id = data["patient_id"]
        slot.notes = data["notes"]
        
        return slot


class BusinessHours:
    """
    Lớp đại diện cho giờ làm việc của máy xạ trị.
    
    Lớp này chứa thông tin về giờ làm việc của máy xạ trị theo từng ngày trong tuần.
    """
    
    def __init__(self):
        """Khởi tạo giờ làm việc mặc định."""
        # Mặc định: Thứ 2 - Thứ 6, 8:00 - 17:00
        self.hours = {
            0: [(datetime.time(8, 0), datetime.time(17, 0))],  # Thứ 2
            1: [(datetime.time(8, 0), datetime.time(17, 0))],  # Thứ 3
            2: [(datetime.time(8, 0), datetime.time(17, 0))],  # Thứ 4
            3: [(datetime.time(8, 0), datetime.time(17, 0))],  # Thứ 5
            4: [(datetime.time(8, 0), datetime.time(17, 0))],  # Thứ 6
            5: [],  # Thứ 7
            6: []   # Chủ nhật
        }
        
        # Break times (e.g., lunch)
        self.breaks = {
            0: [(datetime.time(12, 0), datetime.time(13, 0))],  # Thứ 2
            1: [(datetime.time(12, 0), datetime.time(13, 0))],  # Thứ 3
            2: [(datetime.time(12, 0), datetime.time(13, 0))],  # Thứ 4
            3: [(datetime.time(12, 0), datetime.time(13, 0))],  # Thứ 5
            4: [(datetime.time(12, 0), datetime.time(13, 0))],  # Thứ 6
            5: [],  # Thứ 7
            6: []   # Chủ nhật
        }
        
        # Holidays
        self.holidays = set()
    
    def set_working_hours(self, day_of_week: int, hours_list: List[Tuple[datetime.time, datetime.time]]):
        """
        Đặt giờ làm việc cho một ngày trong tuần.
        
        Parameters
        ----------
        day_of_week : int
            Ngày trong tuần (0=Monday, 6=Sunday)
        hours_list : List[Tuple[datetime.time, datetime.time]]
            Danh sách các khoảng thời gian làm việc
        """
        self.hours[day_of_week] = hours_list
    
    def set_break_times(self, day_of_week: int, break_list: List[Tuple[datetime.time, datetime.time]]):
        """
        Đặt thời gian nghỉ cho một ngày trong tuần.
        
        Parameters
        ----------
        day_of_week : int
            Ngày trong tuần (0=Monday, 6=Sunday)
        break_list : List[Tuple[datetime.time, datetime.time]]
            Danh sách các khoảng thời gian nghỉ
        """
        self.breaks[day_of_week] = break_list
    
    def add_holiday(self, date: datetime.date):
        """
        Thêm một ngày nghỉ lễ.
        
        Parameters
        ----------
        date : datetime.date
            Ngày nghỉ lễ
        """
        self.holidays.add(date)
    
    def remove_holiday(self, date: datetime.date):
        """
        Xóa một ngày nghỉ lễ.
        
        Parameters
        ----------
        date : datetime.date
            Ngày nghỉ lễ
        """
        if date in self.holidays:
            self.holidays.remove(date)
    
    def is_working_day(self, date: datetime.date) -> bool:
        """
        Kiểm tra xem một ngày có phải là ngày làm việc không.
        
        Parameters
        ----------
        date : datetime.date
            Ngày cần kiểm tra
            
        Returns
        -------
        bool
            True nếu là ngày làm việc, False nếu không
        """
        # Kiểm tra ngày nghỉ lễ
        if date in self.holidays:
            return False
            
        # Kiểm tra ngày trong tuần
        day_of_week = date.weekday()
        return len(self.hours[day_of_week]) > 0
    
    def get_working_hours(self, date: datetime.date) -> List[Tuple[datetime.time, datetime.time]]:
        """
        Lấy danh sách giờ làm việc cho một ngày cụ thể.
        
        Parameters
        ----------
        date : datetime.date
            Ngày cần lấy giờ làm việc
            
        Returns
        -------
        List[Tuple[datetime.time, datetime.time]]
            Danh sách các khoảng thời gian làm việc
        """
        if not self.is_working_day(date):
            return []
            
        day_of_week = date.weekday()
        return self.hours[day_of_week]
    
    def get_break_times(self, date: datetime.date) -> List[Tuple[datetime.time, datetime.time]]:
        """
        Lấy danh sách thời gian nghỉ cho một ngày cụ thể.
        
        Parameters
        ----------
        date : datetime.date
            Ngày cần lấy thời gian nghỉ
            
        Returns
        -------
        List[Tuple[datetime.time, datetime.time]]
            Danh sách các khoảng thời gian nghỉ
        """
        if not self.is_working_day(date):
            return []
            
        day_of_week = date.weekday()
        return self.breaks[day_of_week]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin giờ làm việc thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin giờ làm việc
        """
        hours_dict = {}
        for day, hours_list in self.hours.items():
            hours_dict[str(day)] = [
                {"start": h[0].isoformat(), "end": h[1].isoformat()}
                for h in hours_list
            ]
            
        breaks_dict = {}
        for day, break_list in self.breaks.items():
            breaks_dict[str(day)] = [
                {"start": b[0].isoformat(), "end": b[1].isoformat()}
                for b in break_list
            ]
            
        holidays_list = [date.isoformat() for date in self.holidays]
        
        return {
            "hours": hours_dict,
            "breaks": breaks_dict,
            "holidays": holidays_list
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessHours':
        """
        Tạo đối tượng BusinessHours từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin giờ làm việc
            
        Returns
        -------
        BusinessHours
            Đối tượng BusinessHours
        """
        business_hours = cls()
        
        # Giờ làm việc
        for day_str, hours_list in data["hours"].items():
            day = int(day_str)
            business_hours.hours[day] = [
                (
                    datetime.time.fromisoformat(h["start"]),
                    datetime.time.fromisoformat(h["end"])
                )
                for h in hours_list
            ]
            
        # Thời gian nghỉ
        for day_str, break_list in data["breaks"].items():
            day = int(day_str)
            business_hours.breaks[day] = [
                (
                    datetime.time.fromisoformat(b["start"]),
                    datetime.time.fromisoformat(b["end"])
                )
                for b in break_list
            ]
            
        # Ngày nghỉ lễ
        business_hours.holidays = {
            datetime.date.fromisoformat(date_str)
            for date_str in data["holidays"]
        }
        
        return business_hours


class TreatmentScheduler:
    """
    Lớp quản lý lịch điều trị xạ trị.
    
    Lớp này cung cấp các phương thức để lập lịch điều trị cho nhiều bệnh nhân,
    tối ưu hóa việc sử dụng máy xạ trị, và quản lý thời gian điều trị.
    """
    
    def __init__(
        self,
        machine_id: str,
        machine_specs: Optional[MachineSpecs] = None,
        slot_duration_minutes: int = 15
    ):
        """
        Khởi tạo lịch điều trị.
        
        Parameters
        ----------
        machine_id : str
            ID của máy xạ trị
        machine_specs : MachineSpecs, optional
            Thông số kỹ thuật của máy xạ trị
        slot_duration_minutes : int, optional
            Thời lượng mặc định của mỗi khoảng thời gian (phút)
        """
        self.machine_id = machine_id
        self.machine_specs = machine_specs
        self.slot_duration_minutes = slot_duration_minutes
        self.business_hours = BusinessHours()
        
        # Danh sách các khoảng thời gian
        self.time_slots: Dict[datetime.date, List[TimeSlot]] = {}
        
        # Danh sách các đợt điều trị đã lên lịch
        self.scheduled_courses: Dict[str, TreatmentCourse] = {}
    
    def generate_slots(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        override_existing: bool = False
    ):
        """
        Tạo các khoảng thời gian từ start_date đến end_date.
        
        Parameters
        ----------
        start_date : datetime.date
            Ngày bắt đầu
        end_date : datetime.date
            Ngày kết thúc
        override_existing : bool, optional
            Ghi đè lên các khoảng thời gian đã tồn tại
        """
        current_date = start_date
        while current_date <= end_date:
            # Bỏ qua nếu đã có và không ghi đè
            if current_date in self.time_slots and not override_existing:
                current_date += datetime.timedelta(days=1)
                continue
                
            # Kiểm tra xem ngày hiện tại có phải là ngày làm việc không
            if not self.business_hours.is_working_day(current_date):
                current_date += datetime.timedelta(days=1)
                continue
                
            # Lấy giờ làm việc và thời gian nghỉ
            working_hours = self.business_hours.get_working_hours(current_date)
            break_times = self.business_hours.get_break_times(current_date)
            
            # Tạo các khoảng thời gian
            slots = []
            for start_time, end_time in working_hours:
                # Chuyển đổi thời gian thành phút
                start_minutes = start_time.hour * 60 + start_time.minute
                end_minutes = end_time.hour * 60 + end_time.minute
                
                # Tạo các khoảng thời gian
                current_minutes = start_minutes
                while current_minutes + self.slot_duration_minutes <= end_minutes:
                    slot_start = datetime.time(
                        current_minutes // 60,
                        current_minutes % 60
                    )
                    slot_end = datetime.time(
                        (current_minutes + self.slot_duration_minutes) // 60,
                        (current_minutes + self.slot_duration_minutes) % 60
                    )
                    
                    # Kiểm tra xem khoảng thời gian có nằm trong thời gian nghỉ không
                    is_break = False
                    for break_start, break_end in break_times:
                        if (slot_start >= break_start and slot_start < break_end) or \
                           (slot_end > break_start and slot_end <= break_end):
                            is_break = True
                            break
                    
                    if not is_break:
                        slot = TimeSlot(
                            start_time=slot_start,
                            end_time=slot_end,
                            date=current_date
                        )
                        slots.append(slot)
                    
                    current_minutes += self.slot_duration_minutes
            
            # Lưu các khoảng thời gian
            self.time_slots[current_date] = slots
            
            current_date += datetime.timedelta(days=1)
    
    def get_available_slots(
        self,
        date: datetime.date,
        duration_minutes: Optional[int] = None
    ) -> List[TimeSlot]:
        """
        Lấy danh sách các khoảng thời gian trống cho một ngày.
        
        Parameters
        ----------
        date : datetime.date
            Ngày cần lấy khoảng thời gian
        duration_minutes : int, optional
            Thời lượng cần thiết (phút)
            
        Returns
        -------
        List[TimeSlot]
            Danh sách các khoảng thời gian trống
        """
        if date not in self.time_slots:
            return []
            
        available_slots = [slot for slot in self.time_slots[date] if slot.is_available]
        
        if duration_minutes is None or duration_minutes <= self.slot_duration_minutes:
            return available_slots
            
        # Tìm các khoảng thời gian liên tục
        result = []
        for i in range(len(available_slots)):
            continuous_slots = [available_slots[i]]
            total_duration = self.slot_duration_minutes
            
            for j in range(i + 1, len(available_slots)):
                if total_duration >= duration_minutes:
                    break
                    
                if available_slots[j].start_time == continuous_slots[-1].end_time:
                    continuous_slots.append(available_slots[j])
                    total_duration += self.slot_duration_minutes
                else:
                    break
            
            if total_duration >= duration_minutes:
                # Tạo một khoảng thời gian mới từ chuỗi các khoảng thời gian liên tục
                combined_slot = TimeSlot(
                    start_time=continuous_slots[0].start_time,
                    end_time=continuous_slots[-1].end_time,
                    date=date
                )
                result.append(combined_slot)
        
        return result
    
    def schedule_course(self, course: TreatmentCourse) -> bool:
        """
        Lên lịch cho một đợt điều trị.
        
        Parameters
        ----------
        course : TreatmentCourse
            Đợt điều trị cần lên lịch
            
        Returns
        -------
        bool
            True nếu lên lịch thành công, False nếu không
        """
        if not course.fractions:
            logger.warning("Đợt điều trị %s không có phân đoạn nào.", course.course_id)
            return False
            
        # Kiểm tra xem đã lên lịch cho đợt điều trị này chưa
        if course.course_id in self.scheduled_courses:
            logger.warning("Đợt điều trị %s đã được lên lịch trước đó.", course.course_id)
            return False
        
        # Lên lịch cho từng phân đoạn
        success = True
        for fraction in course.fractions:
            if not fraction.scheduled_date:
                logger.warning("Phân đoạn %s không có ngày lên lịch.", fraction.fraction_id)
                success = False
                continue
                
            # Tìm khoảng thời gian trống
            available_slots = self.get_available_slots(
                date=fraction.scheduled_date,
                duration_minutes=20  # Thời gian mặc định cho một phân đoạn
            )
            
            if not available_slots:
                logger.warning("Không tìm thấy khoảng thời gian trống cho phân đoạn %s vào ngày %s.", fraction.fraction_id, fraction.scheduled_date)
                success = False
                continue
                
            # Chọn khoảng thời gian đầu tiên
            slot = available_slots[0]
            
            # Đặt chỗ khoảng thời gian
            if isinstance(slot, TimeSlot):
                # Khoảng thời gian đơn
                slot.reserve(
                    course_id=course.course_id,
                    fraction_id=fraction.fraction_id,
                    patient_id=course.patient_id
                )
            else:
                # Chuỗi các khoảng thời gian
                for s in slot:
                    s.reserve(
                        course_id=course.course_id,
                        fraction_id=fraction.fraction_id,
                        patient_id=course.patient_id
                    )
        
        if success:
            # Lưu đợt điều trị đã lên lịch
            self.scheduled_courses[course.course_id] = course
            
        return success
    
    def cancel_course(self, course_id: str) -> bool:
        """
        Hủy lịch cho một đợt điều trị.
        
        Parameters
        ----------
        course_id : str
            ID của đợt điều trị
            
        Returns
        -------
        bool
            True nếu hủy lịch thành công, False nếu không
        """
        if course_id not in self.scheduled_courses:
            logger.warning("Đợt điều trị %s không tồn tại trong lịch.", course_id)
            return False
            
        course = self.scheduled_courses[course_id]
        
        # Hủy lịch cho từng phân đoạn
        for date, slots in self.time_slots.items():
            for slot in slots:
                if slot.course_id == course_id:
                    slot.release()
        
        # Xóa đợt điều trị khỏi danh sách đã lên lịch
        del self.scheduled_courses[course_id]
        
        logger.info("Đã hủy lịch cho đợt điều trị %s.", course_id)
        
        return True
    
    def get_schedule(self, date: datetime.date) -> List[TimeSlot]:
        """
        Lấy lịch điều trị cho một ngày.
        
        Parameters
        ----------
        date : datetime.date
            Ngày cần lấy lịch
            
        Returns
        -------
        List[TimeSlot]
            Danh sách các khoảng thời gian
        """
        if date not in self.time_slots:
            logger.warning("Không có lịch cho ngày %s.", date)
            return []
            
        return self.time_slots[date]
    
    def get_patient_schedule(self, patient_id: str) -> Dict[datetime.date, List[TimeSlot]]:
        """
        Lấy lịch điều trị cho một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        Dict[datetime.date, List[TimeSlot]]
            Lịch điều trị theo ngày
        """
        result = {}
        
        for date, slots in self.time_slots.items():
            patient_slots = [slot for slot in slots if slot.patient_id == patient_id]
            if patient_slots:
                result[date] = patient_slots
        
        if not result:
            logger.warning("Không tìm thấy lịch điều trị cho bệnh nhân %s.", patient_id)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thông tin lịch điều trị thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin lịch điều trị
        """
        time_slots_dict = {}
        for date, slots in self.time_slots.items():
            time_slots_dict[date.isoformat()] = [slot.to_dict() for slot in slots]
            
        return {
            "machine_id": self.machine_id,
            "slot_duration_minutes": self.slot_duration_minutes,
            "business_hours": self.business_hours.to_dict(),
            "time_slots": time_slots_dict,
            "scheduled_courses": {
                course_id: course.to_dict() for course_id, course in self.scheduled_courses.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreatmentScheduler':
        """
        Tạo đối tượng TreatmentScheduler từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin lịch điều trị
            
        Returns
        -------
        TreatmentScheduler
            Đối tượng TreatmentScheduler
        """
        scheduler = cls(
            machine_id=data["machine_id"],
            slot_duration_minutes=data["slot_duration_minutes"]
        )
        
        # Business hours
        scheduler.business_hours = BusinessHours.from_dict(data["business_hours"])
        
        # Time slots
        for date_str, slots_data in data["time_slots"].items():
            date = datetime.date.fromisoformat(date_str)
            scheduler.time_slots[date] = [TimeSlot.from_dict(slot_data) for slot_data in slots_data]
        
        # Scheduled courses
        for course_id, course_data in data["scheduled_courses"].items():
            scheduler.scheduled_courses[course_id] = TreatmentCourse.from_dict(course_data)
        
        return scheduler
