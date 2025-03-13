#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thư viện máy xạ trị.

Module này cung cấp các lớp và phương thức để quản lý thư viện
các loại máy xạ trị khác nhau hỗ trợ trong hệ thống.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Type
import importlib

from quangtps.treatment.machine.accelerator import Accelerator
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.proton import ProtonMachine
from quangtps.treatment.machine.carbon_ion import CarbonIonMachine
from quangtps.core.config import Config

logger = logging.getLogger(__name__)

class MachineLibrary:
    """
    Lớp quản lý thư viện máy xạ trị.
    
    Lớp này cung cấp các phương thức để quản lý thư viện các máy xạ trị,
    bao gồm việc tải, lưu, và truy cập máy.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Lấy instance duy nhất của MachineLibrary (Singleton pattern).
        
        Returns
        -------
        MachineLibrary
            Instance duy nhất của MachineLibrary
        """
        if cls._instance is None:
            cls._instance = MachineLibrary()
        return cls._instance
    
    def __init__(self):
        """
        Khởi tạo thư viện máy xạ trị.
        """
        self.machines = {}
        self.machine_types = {
            "LINAC": Linac,
            "PROTON": ProtonMachine,
            "CARBON_ION": CarbonIonMachine
        }
        
        # Đường dẫn đến thư mục dữ liệu máy
        self.data_path = os.path.join(Config.get_instance().get_data_path(), "machines")
        os.makedirs(self.data_path, exist_ok=True)
        
        # Tự động tải các máy có sẵn
        self._load_machine_definitions()
    
    def _load_machine_definitions(self):
        """
        Tải các định nghĩa máy từ thư mục dữ liệu.
        """
        if not os.path.exists(self.data_path):
            logger.warning(f"Machine data directory not found: {self.data_path}")
            return
        
        for filename in os.listdir(self.data_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        machine_data = json.load(f)
                    
                    machine_type = machine_data.get("machine_type", "LINAC")
                    if machine_type in self.machine_types:
                        machine_class = self.machine_types[machine_type]
                        machine = machine_class.from_dict(machine_data)
                        self.add_machine(machine)
                        logger.info(f"Loaded machine: {machine.machine_name}")
                    else:
                        logger.warning(f"Unknown machine type: {machine_type} in {filename}")
                        
                except Exception as e:
                    logger.error(f"Error loading machine from {filename}: {str(e)}")
    
    def add_machine(self, machine: Accelerator):
        """
        Thêm máy vào thư viện.
        
        Parameters
        ----------
        machine : Accelerator
            Máy cần thêm
        """
        if machine.machine_id in self.machines:
            logger.warning(f"Machine with ID {machine.machine_id} already exists, overwriting")
        
        self.machines[machine.machine_id] = machine
        logger.info(f"Added machine to library: {machine.machine_name} (ID: {machine.machine_id})")
    
    def get_machine(self, machine_id: str) -> Optional[Accelerator]:
        """
        Lấy máy theo ID.
        
        Parameters
        ----------
        machine_id : str
            ID của máy
            
        Returns
        -------
        Optional[Accelerator]
            Máy nếu tìm thấy, None nếu không tìm thấy
        """
        return self.machines.get(machine_id)
    
    def get_machine_by_name(self, machine_name: str) -> Optional[Accelerator]:
        """
        Lấy máy theo tên.
        
        Parameters
        ----------
        machine_name : str
            Tên của máy
            
        Returns
        -------
        Optional[Accelerator]
            Máy đầu tiên có tên trùng khớp, None nếu không tìm thấy
        """
        for machine in self.machines.values():
            if machine.machine_name == machine_name:
                return machine
        return None
    
    def get_all_machines(self) -> List[Accelerator]:
        """
        Lấy tất cả các máy trong thư viện.
        
        Returns
        -------
        List[Accelerator]
            Danh sách tất cả các máy
        """
        return list(self.machines.values())
    
    def get_machines_by_type(self, machine_type: str) -> List[Accelerator]:
        """
        Lấy danh sách máy theo loại.
        
        Parameters
        ----------
        machine_type : str
            Loại máy (LINAC, PROTON, CARBON_ION)
            
        Returns
        -------
        List[Accelerator]
            Danh sách các máy thuộc loại đã chỉ định
        """
        return [m for m in self.machines.values() if m.machine_type == machine_type]
    
    def remove_machine(self, machine_id: str) -> bool:
        """
        Xóa máy khỏi thư viện.
        
        Parameters
        ----------
        machine_id : str
            ID của máy cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if machine_id in self.machines:
            del self.machines[machine_id]
            logger.info(f"Removed machine with ID: {machine_id}")
            return True
        return False
    
    def save_machine(self, machine_id: str) -> bool:
        """
        Lưu định nghĩa máy vào file.
        
        Parameters
        ----------
        machine_id : str
            ID của máy cần lưu
            
        Returns
        -------
        bool
            True nếu lưu thành công, False nếu có lỗi
        """
        machine = self.get_machine(machine_id)
        if not machine:
            logger.warning(f"Machine with ID {machine_id} not found")
            return False
        
        try:
            file_path = os.path.join(self.data_path, f"{machine_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(machine.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved machine {machine.machine_name} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving machine {machine_id}: {str(e)}")
            return False
    
    def save_all_machines(self) -> int:
        """
        Lưu tất cả các máy vào file.
        
        Returns
        -------
        int
            Số lượng máy đã lưu thành công
        """
        success_count = 0
        for machine_id in self.machines:
            if self.save_machine(machine_id):
                success_count += 1
        
        logger.info(f"Saved {success_count}/{len(self.machines)} machines")
        return success_count
    
    def create_machine(self, machine_type: str, machine_name: str, 
                     manufacturer: str = "Generic") -> Optional[Accelerator]:
        """
        Tạo một máy mới.
        
        Parameters
        ----------
        machine_type : str
            Loại máy (LINAC, PROTON, CARBON_ION)
        machine_name : str
            Tên của máy
        manufacturer : str, optional
            Nhà sản xuất máy
            
        Returns
        -------
        Optional[Accelerator]
            Máy mới nếu tạo thành công, None nếu có lỗi
        """
        if machine_type not in self.machine_types:
            logger.warning(f"Unknown machine type: {machine_type}")
            return None
        
        try:
            machine_class = self.machine_types[machine_type]
            machine = machine_class(machine_name=machine_name, manufacturer=manufacturer)
            self.add_machine(machine)
            logger.info(f"Created new {machine_type} machine: {machine_name}")
            return machine
        except Exception as e:
            logger.error(f"Error creating machine: {str(e)}")
            return None
    
    def register_machine_type(self, type_name: str, machine_class: Type[Accelerator]):
        """
        Đăng ký một loại máy mới.
        
        Parameters
        ----------
        type_name : str
            Tên loại máy
        machine_class : Type[Accelerator]
            Lớp đại diện cho loại máy
        """
        self.machine_types[type_name] = machine_class
        logger.info(f"Registered new machine type: {type_name}")
    
    def import_machine_from_file(self, file_path: str) -> Optional[Accelerator]:
        """
        Nhập máy từ file.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file định nghĩa máy
            
        Returns
        -------
        Optional[Accelerator]
            Máy đã nhập nếu thành công, None nếu có lỗi
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                machine_data = json.load(f)
            
            machine_type = machine_data.get("machine_type", "LINAC")
            if machine_type not in self.machine_types:
                logger.warning(f"Unknown machine type: {machine_type}")
                return None
            
            machine_class = self.machine_types[machine_type]
            machine = machine_class.from_dict(machine_data)
            self.add_machine(machine)
            logger.info(f"Imported machine from {file_path}: {machine.machine_name}")
            return machine
        except Exception as e:
            logger.error(f"Error importing machine from {file_path}: {str(e)}")
            return None
    
    def export_machine_to_file(self, machine_id: str, file_path: str) -> bool:
        """
        Xuất máy ra file.
        
        Parameters
        ----------
        machine_id : str
            ID của máy cần xuất
        file_path : str
            Đường dẫn đến file xuất
            
        Returns
        -------
        bool
            True nếu xuất thành công, False nếu có lỗi
        """
        machine = self.get_machine(machine_id)
        if not machine:
            logger.warning(f"Machine with ID {machine_id} not found")
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(machine.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Exported machine {machine.machine_name} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting machine {machine_id}: {str(e)}")
            return False