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
import datetime
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QComboBox, QFileDialog, QListWidget, 
                            QProgressBar, QTextEdit, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from quangtps.core.exceptions import ImportError
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.beams.beam import BeamType
from quangtps.core.config import ConfigManager
from quangtps.treatment.beams.truebeam_data_processor import TrueBeamDataProcessor
from quangtps.dose.beam_data_processor import BeamModel

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
    

class BeamDataImportWorker(QThread):
    """
    Worker thread để xử lý dữ liệu chùm tia trong nền.
    """
    progressUpdated = pyqtSignal(int, str)
    importFinished = pyqtSignal(object, object)
    
    def __init__(self, machine_type: str, source_dir: str, energies: List[str] = None):
        """
        Khởi tạo worker thread.
        
        Parameters
        ----------
        machine_type : str
            Loại máy gia tốc (ví dụ: 'TrueBeam', 'VitalBeam', v.v.)
        source_dir : str
            Thư mục chứa dữ liệu chùm tia
        energies : List[str], optional
            Danh sách các mức năng lượng cần xử lý, nếu None thì xử lý tất cả
        """
        super().__init__()
        self.machine_type = machine_type
        self.source_dir = source_dir
        self.energies = energies
        self.results = {}
    
    def run(self):
        """Thực hiện quá trình nhập dữ liệu."""
        try:
            # Cập nhật tiến trình
            self.progressUpdated.emit(10, "Khởi tạo processor...")
            
            if self.machine_type == "TrueBeam":
                processor = TrueBeamDataProcessor(data_dir=self.source_dir)
                
                # Quét tìm các file dữ liệu
                self.progressUpdated.emit(20, "Quét tìm file dữ liệu...")
                energy_files = processor.scan_for_beam_data_files()
                
                if not energy_files:
                    self.importFinished.emit(False, {
                        "error": f"Không tìm thấy file dữ liệu nào trong {self.source_dir}"
                    })
                    return
                
                available_energies = list(energy_files.keys())
                self.results["available_energies"] = available_energies
                
                # Xác định các năng lượng cần xử lý
                energies_to_process = self.energies if self.energies else available_energies
                
                # Lọc ra những năng lượng có sẵn
                energies_to_process = [e for e in energies_to_process if e in available_energies]
                
                if not energies_to_process:
                    self.importFinished.emit(False, {
                        "error": "Không có mức năng lượng nào được chọn hoặc có sẵn"
                    })
                    return
                
                # Xử lý từng mức năng lượng
                models = {}
                total_energies = len(energies_to_process)
                
                for i, energy in enumerate(energies_to_process):
                    progress = 30 + (i / total_energies) * 60
                    self.progressUpdated.emit(int(progress), f"Đang xử lý năng lượng {energy}...")
                    
                    model = processor.create_beam_model(energy)
                    if model:
                        models[energy] = model
                    else:
                        self.importFinished.emit(False, {
                            "error": f"Không thể tạo mô hình chùm tia cho năng lượng {energy}"
                        })
                        return
                
                # Lưu mô hình
                self.progressUpdated.emit(90, "Đang lưu mô hình chùm tia...")
                
                config = ConfigManager().get_config()
                output_dir = os.path.join(config.get('data_directory', 'data'), 'beam_data', 'models')
                os.makedirs(output_dir, exist_ok=True)
                
                for energy, model in models.items():
                    file_path = os.path.join(output_dir, f"TrueBeam_{energy}.json")
                    model.save(file_path)
                
                self.results["models"] = models
                self.results["output_dir"] = output_dir
                
                self.progressUpdated.emit(100, "Hoàn thành!")
                self.importFinished.emit(True, self.results)
            
            else:
                self.importFinished.emit(False, {
                    "error": f"Loại máy gia tốc không được hỗ trợ: {self.machine_type}"
                })
                
        except Exception as e:
            logger.exception("Error in beam data import worker")
            self.importFinished.emit(False, {
                "error": f"Lỗi khi nhập dữ liệu: {str(e)}"
            })


class BeamDataImporterDialog(QDialog):
    """
    Dialog để nhập dữ liệu chùm tia vào hệ thống.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Khởi tạo dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        self.setWindowTitle("Nhập Dữ Liệu Chùm Tia")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self.config = ConfigManager().get_config()
        self.init_ui()
        
        # Kết quả của quá trình nhập
        self.import_results = {}
    
    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout()
        
        # Chọn loại máy gia tốc
        machine_layout = QHBoxLayout()
        machine_layout.addWidget(QLabel("Loại máy gia tốc:"))
        
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(["TrueBeam", "VitalBeam", "Halcyon", "Khác"])
        machine_layout.addWidget(self.machine_combo)
        
        layout.addLayout(machine_layout)
        
        # Chọn thư mục nguồn
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Thư mục dữ liệu:"))
        
        self.source_edit = QTextEdit()
        self.source_edit.setMaximumHeight(60)
        self.source_edit.setReadOnly(True)
        
        # Thiết lập thư mục mặc định
        default_dir = os.path.join(self.config.get('data_directory', 'data'), 'beam_data')
        self.source_edit.setText(default_dir)
        
        source_layout.addWidget(self.source_edit)
        
        self.browse_button = QPushButton("Duyệt...")
        self.browse_button.clicked.connect(self.browse_source_dir)
        source_layout.addWidget(self.browse_button)
        
        layout.addLayout(source_layout)
        
        # Danh sách năng lượng
        layout.addWidget(QLabel("Mức năng lượng khả dụng:"))
        
        self.energy_list = QListWidget()
        self.energy_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.energy_list)
        
        # Nút quét
        self.scan_button = QPushButton("Quét Thư Mục")
        self.scan_button.clicked.connect(self.scan_directory)
        layout.addWidget(self.scan_button)
        
        # Thanh tiến trình
        layout.addWidget(QLabel("Tiến trình:"))
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Sẵn sàng")
        layout.addWidget(self.status_label)
        
        # Nút điều khiển
        button_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Nhập Dữ Liệu")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.import_data)
        button_layout.addWidget(self.import_button)
        
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def browse_source_dir(self):
        """Mở dialog để chọn thư mục nguồn."""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Chọn Thư Mục Dữ Liệu Chùm Tia",
            self.source_edit.toPlainText()
        )
        
        if dir_path:
            self.source_edit.setText(dir_path)
            self.scan_directory()
    
    def scan_directory(self):
        """Quét thư mục để tìm dữ liệu chùm tia."""
        source_dir = self.source_edit.toPlainText()
        machine_type = self.machine_combo.currentText()
        
        if not os.path.isdir(source_dir):
            QMessageBox.warning(self, "Lỗi", f"Thư mục không tồn tại: {source_dir}")
            return
        
        self.energy_list.clear()
        self.import_button.setEnabled(False)
        self.status_label.setText("Đang quét thư mục...")
        
        try:
            if machine_type == "TrueBeam":
                processor = TrueBeamDataProcessor(data_dir=source_dir)
                energy_files = processor.scan_for_beam_data_files()
                
                if energy_files:
                    for energy in energy_files.keys():
                        self.energy_list.addItem(energy)
                    
                    self.status_label.setText(f"Tìm thấy {len(energy_files)} mức năng lượng")
                    self.import_button.setEnabled(True)
                else:
                    self.status_label.setText("Không tìm thấy file dữ liệu chùm tia nào")
            else:
                QMessageBox.information(
                    self, 
                    "Chưa Hỗ Trợ", 
                    f"Loại máy gia tốc {machine_type} chưa được hỗ trợ trong phiên bản này."
                )
                self.status_label.setText("Loại máy gia tốc chưa được hỗ trợ")
        
        except Exception as e:
            logger.exception("Error scanning directory")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi quét thư mục: {str(e)}")
            self.status_label.setText("Lỗi khi quét thư mục")
    
    def import_data(self):
        """Bắt đầu quá trình nhập dữ liệu."""
        source_dir = self.source_edit.toPlainText()
        machine_type = self.machine_combo.currentText()
        
        # Lấy danh sách các mức năng lượng được chọn
        selected_energies = []
        for index in range(self.energy_list.count()):
            item = self.energy_list.item(index)
            if item.isSelected():
                selected_energies.append(item.text())
        
        if not selected_energies:
            # Nếu không có mức năng lượng nào được chọn, sử dụng tất cả
            for index in range(self.energy_list.count()):
                item = self.energy_list.item(index)
                selected_energies.append(item.text())
        
        # Vô hiệu hóa các điều khiển
        self.import_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.machine_combo.setEnabled(False)
        self.energy_list.setEnabled(False)
        
        # Cập nhật trạng thái
        self.progress_bar.setValue(0)
        self.status_label.setText("Bắt đầu nhập dữ liệu...")
        
        # Tạo và khởi động worker thread
        self.worker = BeamDataImportWorker(
            machine_type=machine_type,
            source_dir=source_dir,
            energies=selected_energies
        )
        
        self.worker.progressUpdated.connect(self.update_progress)
        self.worker.importFinished.connect(self.import_finished)
        self.worker.start()
    
    def update_progress(self, value: int, message: str):
        """
        Cập nhật thanh tiến trình và thông báo trạng thái.
        
        Parameters
        ----------
        value : int
            Giá trị tiến trình (0-100)
        message : str
            Thông báo trạng thái
        """
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def import_finished(self, success: bool, results: Dict[str, Any]):
        """
        Xử lý kết quả khi quá trình nhập hoàn tất.
        
        Parameters
        ----------
        success : bool
            True nếu nhập thành công, False nếu có lỗi
        results : Dict[str, Any]
            Kết quả của quá trình nhập
        """
        # Kích hoạt lại các điều khiển
        self.scan_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.machine_combo.setEnabled(True)
        self.energy_list.setEnabled(True)
        
        self.import_results = results
        
        if success:
            models = results.get("models", {})
            output_dir = results.get("output_dir", "")
            
            message = f"Đã nhập thành công {len(models)} mô hình chùm tia.\n"
            message += f"Dữ liệu đã được lưu vào: {output_dir}"
            
            QMessageBox.information(self, "Thành Công", message)
            self.accept()  # Đóng dialog với kết quả thành công
        else:
            error = results.get("error", "Lỗi không xác định")
            QMessageBox.critical(self, "Lỗi", f"Nhập dữ liệu thất bại: {error}")
            self.import_button.setEnabled(True)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Lấy kết quả của quá trình nhập.
        
        Returns
        -------
        Dict[str, Any]
            Kết quả của quá trình nhập
        """
        return self.import_results


class TrueBeamDataReader:
    """
    Lớp đọc và nhập dữ liệu từ máy gia tốc TrueBeam.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Khởi tạo đối tượng đọc dữ liệu TrueBeam.
        
        Parameters
        ----------
        base_dir : str, optional
            Thư mục cơ sở chứa dữ liệu TrueBeam, nếu None thì sẽ sử dụng thư mục mặc định
        """
        config = ConfigManager().get_config()
        self.base_dir = base_dir or os.path.join(config.get('data_directory', 'data'), 'beam_data', 'truebeam')
        
        # Kiểm tra và tạo thư mục nếu cần
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.processor = TrueBeamDataProcessor(data_dir=self.base_dir)
        self.available_models = []
        self.scan_existing_models()
    
    def scan_existing_models(self):
        """Quét tìm các mô hình chùm tia đã có sẵn trong hệ thống."""
        config = ConfigManager().get_config()
        models_dir = os.path.join(config.get('data_directory', 'data'), 'beam_data', 'models')
        
        # Kiểm tra thư mục mô hình
        if not os.path.isdir(models_dir):
            logger.info(f"Models directory not found, creating: {models_dir}")
            os.makedirs(models_dir, exist_ok=True)
            return
        
        # Tìm các file mô hình
        model_files = [f for f in os.listdir(models_dir) if f.startswith("TrueBeam_") and f.endswith(".json")]
        
        # Trích xuất tên năng lượng từ tên file
        for file in model_files:
            energy = file.replace("TrueBeam_", "").replace(".json", "")
            self.available_models.append(energy)
        
        logger.info(f"Found {len(self.available_models)} existing TrueBeam models: {', '.join(self.available_models)}")
    
    def get_available_models(self) -> List[str]:
        """
        Lấy danh sách các mô hình chùm tia có sẵn.
        
        Returns
        -------
        List[str]
            Danh sách các mức năng lượng có mô hình
        """
        return self.available_models
    
    def load_model(self, energy: str) -> Optional[BeamModel]:
        """
        Tải mô hình chùm tia cho một mức năng lượng cụ thể.
        
        Parameters
        ----------
        energy : str
            Mức năng lượng cần tải (ví dụ: "6MV", "10FFF")
            
        Returns
        -------
        Optional[BeamModel]
            Đối tượng mô hình chùm tia nếu tồn tại, None nếu không
        """
        # Kiểm tra xem mô hình có tồn tại không
        if energy not in self.available_models:
            logger.warning(f"No model available for energy {energy}")
            return None
        
        # Tải mô hình
        config = ConfigManager().get_config()
        models_dir = os.path.join(config.get('data_directory', 'data'), 'beam_data', 'models')
        file_path = os.path.join(models_dir, f"TrueBeam_{energy}.json")
        
        try:
            model = BeamModel.load(file_path)
            logger.info(f"Loaded beam model for energy {energy}")
            return model
        except Exception as e:
            logger.error(f"Error loading beam model for energy {energy}: {str(e)}")
            return None
    
    def import_data_from_directory(self, directory: str, energies: List[str] = None) -> Dict[str, BeamModel]:
        """
        Nhập dữ liệu từ một thư mục cụ thể.
        
        Parameters
        ----------
        directory : str
            Thư mục chứa dữ liệu chùm tia
        energies : List[str], optional
            Danh sách các mức năng lượng cần nhập, nếu None thì nhập tất cả
            
        Returns
        -------
        Dict[str, BeamModel]
            Dictionary ánh xạ từ mức năng lượng tới đối tượng mô hình
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return {}
        
        processor = TrueBeamDataProcessor(data_dir=directory)
        
        # Quét tìm các file dữ liệu
        energy_files = processor.scan_for_beam_data_files()
        if not energy_files:
            logger.warning(f"No beam data files found in {directory}")
            return {}
        
        # Xác định các năng lượng cần xử lý
        available_energies = list(energy_files.keys())
        energies_to_process = energies if energies else available_energies
        
        # Lọc ra những năng lượng có sẵn
        energies_to_process = [e for e in energies_to_process if e in available_energies]
        
        # Xử lý từng mức năng lượng
        models = {}
        for energy in energies_to_process:
            model = processor.create_beam_model(energy)
            if model:
                models[energy] = model
            else:
                logger.error(f"Failed to create beam model for energy {energy}")
        
        # Lưu các mô hình
        config = ConfigManager().get_config()
        output_dir = os.path.join(config.get('data_directory', 'data'), 'beam_data', 'models')
        os.makedirs(output_dir, exist_ok=True)
        
        for energy, model in models.items():
            file_path = os.path.join(output_dir, f"TrueBeam_{energy}.json")
            model.save(file_path)
            
            # Thêm vào danh sách mô hình có sẵn nếu chưa có
            if energy not in self.available_models:
                self.available_models.append(energy)
        
        logger.info(f"Imported {len(models)} beam models from {directory}")
        return models
    
    def show_import_dialog(self, parent: QWidget = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Hiển thị dialog nhập dữ liệu.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            Tuple gồm kết quả (True nếu thành công) và dictionary kết quả
        """
        dialog = BeamDataImporterDialog(parent)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # Cập nhật danh sách mô hình có sẵn sau khi nhập thành công
            self.scan_existing_models()
            return True, dialog.get_results()
        else:
            return False, {}


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