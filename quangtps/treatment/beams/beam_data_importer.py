#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đọc và nhập dữ liệu chùm tia TrueBeam.

Module này cung cấp các lớp và phương thức để đọc, phân tích và nhập dữ liệu chùm tia 
từ máy TrueBeam (từ Varian) vào hệ thống QuangTPS. Dữ liệu này bao gồm các 
thông số vật lý của chùm tia như output factors, wedge factors, phân bố liều, v.v.
"""

import os
import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union

from quangtps.core.exceptions import ImportError
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.beams.beam import BeamType

logger = logging.getLogger(__name__)

class BeamDataType(str, Enum):
    """Enum đại diện cho các loại dữ liệu chùm tia."""
    OUTPUT_FACTOR = "Output Factor"
    PDD = "Percentage Depth Dose"
    PROFILE = "Profile"
    WEDGE_FACTOR = "Wedge Factor"
    TMR = "Tissue Maximum Ratio"
    TPR = "Tissue Phantom Ratio"
    MLC_TRANSMISSION = "MLC Transmission"
    JAW_TRANSMISSION = "Jaw Transmission"
    ENERGY_SPECTRUM = "Energy Spectrum"
    FFF_PROFILE = "Flattening Filter Free Profile"
    ELECTRON_ENERGY = "Electron Energy"
    

class TrueBeamDataReader:
    """
    Lớp để đọc và phân tích dữ liệu chùm tia TrueBeam.
    
    Lớp này cung cấp các phương thức để đọc dữ liệu từ các file data chùm tia TrueBeam
    và chuyển đổi chúng thành định dạng phù hợp cho hệ thống QuangTPS.
    """
    
    def __init__(self, data_directory: str):
        """
        Khởi tạo reader cho dữ liệu chùm tia TrueBeam.
        
        Parameters
        ----------
        data_directory : str
            Đường dẫn đến thư mục chứa dữ liệu chùm tia TrueBeam
        """
        self.data_directory = data_directory
        self.beam_data = {}
        self.energies = []
        self.beam_types = []
        
    def scan_data_directory(self):
        """
        Quét thư mục dữ liệu để xác định các loại chùm tia và năng lượng có sẵn.
        
        Returns
        -------
        Tuple[List[str], List[str]]
            Danh sách các năng lượng và loại chùm tia
        """
        logger.info(f"Scanning TrueBeam data directory: {self.data_directory}")
        
        # Danh sách các loại năng lượng chùm tia thường gặp
        energy_patterns = {
            "4X": "4MV", 
            "6X": "6MV", 
            "6FFF": "6MV FFF",
            "8X": "8MV", 
            "10X": "10MV", 
            "10FFF": "10MV FFF",
            "15X": "15MV", 
            "18X": "18MV",
            "20X": "20MV"
        }
        
        found_energies = set()
        found_types = set(["PHOTON", "ELECTRON"])
        
        # Quét các thư mục Excel
        excel_files = []
        for root, _, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(".xlsx") and not file.startswith("~$"):
                    lower_filename = file.lower()
                    
                    # Tìm mẫu năng lượng trong tên file
                    for pattern, energy_name in energy_patterns.items():
                        if pattern.lower() in lower_filename:
                            found_energies.add(energy_name)
                    
                    # Xác định electron vs photon
                    if "electron" in lower_filename:
                        found_types.add("ELECTRON")
                    else:
                        found_types.add("PHOTON")
                    
                    excel_files.append(os.path.join(root, file))
        
        # Quét thư mục W2CAD để tìm các file dữ liệu chi tiết
        w2cad_dir = os.path.join(self.data_directory, "W2CAD")
        if os.path.exists(w2cad_dir):
            energy_dirs = [d for d in os.listdir(w2cad_dir) 
                         if os.path.isdir(os.path.join(w2cad_dir, d))]
            
            for dir_name in energy_dirs:
                # Xử lý thư mục năng lượng
                for pattern, energy_name in energy_patterns.items():
                    if pattern in dir_name:
                        found_energies.add(energy_name)
                
                # Phân loại
                if any(name in dir_name.upper() for name in ["FFF", "6X_FFF", "10X_FFF"]):
                    found_types.add("FFF")
                    
        # Chuyển set thành list và sắp xếp
        self.energies = sorted(list(found_energies))
        self.beam_types = sorted(list(found_types))
        
        logger.info(f"Found energies: {self.energies}")
        logger.info(f"Found beam types: {self.beam_types}")
        
        return self.energies, self.beam_types
    
    def _parse_w2cad_file(self, file_path: str) -> Dict[str, Any]:
        """
        Phân tích file dữ liệu chùm tia từ định dạng W2CAD.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file dữ liệu
            
        Returns
        -------
        Dict[str, Any]
            Dữ liệu đã phân tích
        """
        try:
            data = {}
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Dòng đầu tiên thường chứa thông tin header
            if lines and '|' in lines[0]:
                header_parts = lines[0].strip().split('|')
                if len(header_parts) > 1:
                    header_info = header_parts[1].split(',')
                    if len(header_info) >= 7:
                        data['energy'] = header_info[0]
                        data['wedge_angle'] = header_info[1]
                        data['ssd'] = header_info[2]
                        data['normalization_depth'] = header_info[3]
                        data['field_size'] = header_info[4] + "x" + header_info[5]
                        data['unit'] = header_info[6]
            
            # Đọc các giá trị
            field_sizes = []
            values = []
            
            # Tìm vị trí bắt đầu của dữ liệu
            data_start = False
            for i, line in enumerate(lines[1:], 1):
                line = line.strip()
                if not data_start and line and not line.startswith('|'):
                    # Bắt đầu của dữ liệu
                    field_sizes = [float(x) for x in line.split(',') if x.strip()]
                    data_start = True
                    continue
                
                if data_start and line and not line.startswith('|'):
                    parts = line.split(',')
                    if len(parts) > 1:
                        try:
                            depth = float(parts[0])
                            row_values = [float(x) for x in parts[1:] if x.strip()]
                            values.append([depth] + row_values)
                        except ValueError:
                            pass
            
            # Chuyển đổi thành numpy array để dễ xử lý
            if field_sizes and values:
                data['field_sizes'] = field_sizes
                data['depths'] = np.array([row[0] for row in values])
                data['values'] = np.array([row[1:] for row in values])
            
            return data
        
        except Exception as e:
            logger.error(f"Error parsing W2CAD file {file_path}: {str(e)}")
            return {}
    
    def _parse_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Phân tích file Excel chứa dữ liệu chùm tia.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file Excel
            
        Returns
        -------
        Dict[str, Any]
            Dữ liệu đã phân tích
        """
        try:
            # Xác định loại dữ liệu dựa trên tên file
            basename = os.path.basename(file_path)
            data_type = None
            energy = None
            
            # Phân tích tên file để biết loại dữ liệu
            if "MV" in basename:
                if "FFF" in basename:
                    if "6FFF" in basename:
                        energy = "6MV FFF"
                    elif "10FFF" in basename:
                        energy = "10MV FFF"
                    data_type = BeamDataType.FFF_PROFILE
                else:
                    for mv in ["4MV", "6MV", "8MV", "10MV", "15MV", "18MV", "20MV"]:
                        if mv in basename:
                            energy = mv
                            break
                    data_type = BeamDataType.PROFILE
            elif "Electron" in basename:
                energy = "Electron"
                data_type = BeamDataType.ELECTRON_ENERGY
            
            # Đọc file Excel
            try:
                # Thử đọc tất cả các sheet
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                
                # Lưu dữ liệu từ các sheet
                data = {
                    "energy": energy,
                    "data_type": data_type,
                    "sheets": {}
                }
                
                for sheet_name in sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    data["sheets"][sheet_name] = df.to_dict()
                
                return data
                
            except Exception as e:
                logger.error(f"Error reading Excel file {file_path}: {str(e)}")
                return {}
        
        except Exception as e:
            logger.error(f"Error parsing Excel file {file_path}: {str(e)}")
            return {}
    
    def read_beam_data(self, energy: str = None, data_type: BeamDataType = None) -> Dict[str, Any]:
        """
        Đọc dữ liệu chùm tia cho năng lượng và loại dữ liệu cụ thể.
        
        Parameters
        ----------
        energy : str, optional
            Năng lượng chùm tia, ví dụ "6MV", "10MV FFF". Nếu None, đọc tất cả.
        data_type : BeamDataType, optional
            Loại dữ liệu cần đọc. Nếu None, đọc tất cả.
            
        Returns
        -------
        Dict[str, Any]
            Dữ liệu chùm tia đã đọc
        """
        result = {}
        
        # Quét thư mục dữ liệu
        if not self.energies:
            self.scan_data_directory()
        
        # Chuyển đổi energy pattern cần lọc
        energy_filter = energy.upper() if energy else None
        
        # Đọc dữ liệu từ W2CAD nếu có
        w2cad_dir = os.path.join(self.data_directory, "W2CAD")
        if os.path.exists(w2cad_dir):
            for root, dirs, files in os.walk(w2cad_dir):
                for file in files:
                    # Chỉ xử lý các file dữ liệu
                    if file.endswith(('.txt', '.ASC')):
                        file_path = os.path.join(root, file)
                        
                        # Lọc theo năng lượng nếu cần
                        if energy_filter and energy_filter not in root.upper():
                            continue
                        
                        # Phân tích file dữ liệu
                        file_data = self._parse_w2cad_file(file_path)
                        if file_data:
                            # Lưu dữ liệu dưới đường dẫn tương đối
                            rel_path = os.path.relpath(file_path, self.data_directory)
                            result[rel_path] = file_data
        
        # Đọc các file Excel
        for root, _, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith('.xlsx') and not file.startswith('~$'):
                    file_path = os.path.join(root, file)
                    
                    # Lọc theo năng lượng nếu cần
                    if energy_filter and energy_filter not in file.upper():
                        continue
                    
                    # Phân tích file Excel
                    excel_data = self._parse_excel_file(file_path)
                    if excel_data:
                        # Lưu dữ liệu dưới đường dẫn tương đối
                        rel_path = os.path.relpath(file_path, self.data_directory)
                        result[rel_path] = excel_data
        
        return result


class BeamDataImporter:
    """
    Lớp nhập dữ liệu chùm tia vào hệ thống QuangTPS.
    
    Lớp này cung cấp các phương thức để nhập dữ liệu chùm tia từ các nguồn khác nhau
    vào hệ thống QuangTPS và cập nhật cơ sở dữ liệu máy điều trị.
    """
    
    def __init__(self, machine_library=None):
        """
        Khởi tạo importer cho dữ liệu chùm tia.
        
        Parameters
        ----------
        machine_library : MachineLibrary, optional
            Thư viện máy điều trị để cập nhật dữ liệu. Nếu None, sẽ tải thư viện mặc định.
        """
        if machine_library is None:
            from quangtps.treatment.machine.machine_library import MachineLibrary
            self.machine_library = MachineLibrary.get_instance()
        else:
            self.machine_library = machine_library
        
        self.beam_data = {}
        
    def import_truebeam_data(self, data_directory: str, machine_id: str = None, machine_name: str = None):
        """
        Nhập dữ liệu chùm tia từ thư mục dữ liệu TrueBeam.
        
        Parameters
        ----------
        data_directory : str
            Đường dẫn đến thư mục chứa dữ liệu chùm tia TrueBeam
        machine_id : str, optional
            ID của máy cần cập nhật dữ liệu. Nếu None, sẽ tạo máy mới.
        machine_name : str, optional
            Tên của máy mới nếu cần tạo. Mặc định là "TrueBeam".
            
        Returns
        -------
        str
            ID của máy đã cập nhật/tạo mới
        
        Raises
        ------
        ImportError
            Nếu có lỗi trong quá trình nhập dữ liệu
        """
        try:
            logger.info(f"Importing TrueBeam data from {data_directory}")
            
            # Đọc dữ liệu từ thư mục
            reader = TrueBeamDataReader(data_directory)
            energies, beam_types = reader.scan_data_directory()
            
            if not energies:
                raise ImportError("No beam energies found in the data directory")
            
            # Tìm hoặc tạo máy
            machine = None
            
            if machine_id:
                # Tìm máy theo ID
                machine = self.machine_library.get_machine(machine_id)
                if not machine:
                    logger.warning(f"Machine with ID {machine_id} not found. Creating a new one.")
            
            if not machine and machine_name:
                # Tìm máy theo tên
                machine = self.machine_library.get_machine_by_name(machine_name)
            
            if not machine:
                # Tạo máy mới
                from quangtps.treatment.machine.treatment_machine import MachineType
                
                machine_name = machine_name or "TrueBeam"
                machine = self.machine_library.create_machine(
                    machine_type="LINAC",
                    machine_name=machine_name,
                    manufacturer="Varian"
                )
                
                if not machine:
                    raise ImportError("Failed to create a new machine")
                
                # Thiết lập thông số cơ bản cho máy
                machine.model = "TrueBeam"
                machine.description = "Varian TrueBeam Linear Accelerator"
            
            # Cập nhật thông tin chùm tia cho máy
            self._update_machine_beam_data(machine, reader, energies, beam_types)
            
            # Lưu máy
            self.machine_library.save_machine(machine.machine_id)
            
            logger.info(f"Successfully imported TrueBeam data for machine {machine.name} (ID: {machine.machine_id})")
            return machine.machine_id
            
        except Exception as e:
            logger.error(f"Error importing TrueBeam data: {str(e)}")
            raise ImportError(f"Failed to import TrueBeam data: {str(e)}")
    
    def _update_machine_beam_data(self, machine, reader, energies, beam_types):
        """
        Cập nhật dữ liệu chùm tia cho máy.
        
        Parameters
        ----------
        machine : TreatmentMachine
            Máy cần cập nhật
        reader : TrueBeamDataReader
            Reader chứa dữ liệu chùm tia
        energies : List[str]
            Danh sách các năng lượng
        beam_types : List[str]
            Danh sách các loại chùm tia
        """
        # Cập nhật danh sách năng lượng cho máy
        if not hasattr(machine, "available_energies") or not machine.available_energies:
            machine.available_energies = {}
        
        # Phân loại năng lượng theo loại chùm tia
        for energy in energies:
            if "FFF" in energy:
                beam_type = "PHOTON_FFF"
            elif "MV" in energy:
                beam_type = "PHOTON"
            elif "MEV" in energy:
                beam_type = "ELECTRON"
            else:
                continue
            
            if beam_type not in machine.available_energies:
                machine.available_energies[beam_type] = []
            
            if energy not in machine.available_energies[beam_type]:
                machine.available_energies[beam_type].append(energy)
        
        # Đọc dữ liệu chi tiết cho từng năng lượng
        for energy in energies:
            beam_data = reader.read_beam_data(energy=energy)
            
            # Cập nhật metadata cho máy
            machine_metadata = machine.metadata or {}
            if "beam_data" not in machine_metadata:
                machine_metadata["beam_data"] = {}
            
            # Lưu thông tin đường dẫn đến dữ liệu
            machine_metadata["beam_data"][energy] = {
                "source": "TrueBeam representative data",
                "files": list(beam_data.keys())
            }
            
            machine.metadata = machine_metadata
            
            # TODO: Xử lý và lưu dữ liệu liều chi tiết theo định dạng phù hợp
            # Điều này phụ thuộc vào cấu trúc lưu trữ dữ liệu chùm tia cụ thể của QuangTPS
        
        logger.info(f"Updated beam data for machine {machine.name} with {len(energies)} energies")
    
    def create_beam_data_from_dicom(self, dicom_rt_plan_file: str, machine_id: str = None):
        """
        Tạo dữ liệu chùm tia từ file DICOM RT Plan.
        
        Parameters
        ----------
        dicom_rt_plan_file : str
            Đường dẫn đến file DICOM RT Plan
        machine_id : str, optional
            ID của máy cần cập nhật dữ liệu. Nếu None, sẽ tìm máy phù hợp.
            
        Returns
        -------
        str
            ID của máy đã cập nhật
        """
        # TODO: Implement
        raise NotImplementedError("DICOM beam data import not yet implemented")
    
    def export_beam_data_to_json(self, machine_id: str, output_file: str):
        """
        Xuất dữ liệu chùm tia ra file JSON.
        
        Parameters
        ----------
        machine_id : str
            ID của máy cần xuất dữ liệu
        output_file : str
            Đường dẫn đến file JSON output
            
        Returns
        -------
        bool
            True nếu xuất thành công
        """
        # TODO: Implement
        raise NotImplementedError("JSON export not yet implemented") 