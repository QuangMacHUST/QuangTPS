"""
Model Manager Dialog for QuangTPS segmentation models.

This module provides a user interface for managing segmentation models,
including downloading, updating, and deleting models.
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Callable
import threading

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QFont, QColor, QPixmap
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
                           QMessageBox, QGroupBox, QComboBox, QCheckBox, QFileDialog,
                           QTabWidget, QTextEdit, QSplitter, QFrame, QSpinBox,
                           QApplication, QWidget, QStyle)

from quangtps.segmentation.model_downloader import (
    get_available_remote_models,
    download_model,
    download_models_parallel,
    compute_md5,
    ensure_default_models,
    MODELS_DIR
)

logger = logging.getLogger(__name__)


class ModelDownloaderThread(QThread):
    """Thread for downloading models without blocking the UI."""
    
    # Signal to update progress (model_name, progress_value)
    progress_updated = pyqtSignal(str, int)
    # Signal when download is complete (model_name, success)
    download_finished = pyqtSignal(str, bool)
    # Signal for overall progress when downloading multiple models
    overall_progress_updated = pyqtSignal(int)
    # Signal when all downloads are complete
    all_downloads_finished = pyqtSignal()
    
    def __init__(self, models_to_download: List[str], force: bool = False):
        """
        Initialize the downloader thread.
        
        Parameters
        ----------
        models_to_download : List[str]
            List of model names/IDs to download
        force : bool, optional
            Force download even if model exists, by default False
        """
        super().__init__()
        self.models_to_download = models_to_download
        self.force = force
        self.is_running = False
        
    def run(self):
        """Run the download process."""
        self.is_running = True
        
        # Create callback for progress updates
        def progress_callback(model_name, progress):
            if self.is_running:
                self.progress_updated.emit(model_name, progress)
        
        # Track overall progress
        total_models = len(self.models_to_download)
        completed_models = 0
        
        try:
            # Download each model sequentially with progress updates
            for model_id in self.models_to_download:
                if not self.is_running:
                    break
                
                try:
                    # Update initial progress for this model
                    self.progress_updated.emit(model_id, 0)
                    
                    # Custom progress callback for this model
                    def model_progress(count, block_size, total_size):
                        if total_size > 0:
                            progress = min(100, int(count * block_size * 100 / total_size))
                        else:
                            progress = -1  # Indeterminate if size unknown
                        progress_callback(model_id, progress)
                    
                    # Download model
                    success = download_model(model_id, self.force)
                    
                    # Emit completion signal for this model
                    self.download_finished.emit(model_id, success)
                    
                    # Update overall progress
                    completed_models += 1
                    overall_progress = int(completed_models * 100 / total_models)
                    self.overall_progress_updated.emit(overall_progress)
                    
                except Exception as e:
                    logger.error(f"Error downloading model {model_id}: {str(e)}")
                    self.download_finished.emit(model_id, False)
                    
                    # Update overall progress even on failure
                    completed_models += 1
                    overall_progress = int(completed_models * 100 / total_models)
                    self.overall_progress_updated.emit(overall_progress)
            
            # All downloads are finished
            self.all_downloads_finished.emit()
            
        except Exception as e:
            logger.error(f"Error in model download thread: {str(e)}")
        
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the download process."""
        self.is_running = False


class ModelInfoThread(QThread):
    """Thread for retrieving model information without blocking the UI."""
    
    # Signal when model info is retrieved (model_list)
    info_retrieved = pyqtSignal(list)
    # Signal when an error occurs (error_message)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        """Initialize the model info thread."""
        super().__init__()
        
    def run(self):
        """Run the model info retrieval process."""
        try:
            # Get available remote models
            remote_models = get_available_remote_models()
            
            # Enhance model information with local status
            enhanced_models = self._enhance_with_local_info(remote_models)
            
            # Emit signal with model list
            self.info_retrieved.emit(enhanced_models)
            
        except Exception as e:
            logger.error(f"Error retrieving model information: {str(e)}")
            self.error_occurred.emit(f"Error retrieving model information: {str(e)}")
    
    def _enhance_with_local_info(self, remote_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance remote model information with local availability status.
        
        Parameters
        ----------
        remote_models : List[Dict[str, Any]]
            List of remote model information
            
        Returns
        -------
        List[Dict[str, Any]]
            Enhanced model information
        """
        # Get locally available models
        local_model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.pt', '.pth'))]
        
        # Enhance remote model information
        for model in remote_models:
            model_name = model.get('name', '')
            model_filename = model.get('filename', f"{model_name}.pt")
            
            # Check if model is available locally
            model['is_local'] = model_filename in local_model_files
            
            # Add local file path if available
            if model['is_local']:
                model['local_path'] = os.path.join(MODELS_DIR, model_filename)
                
                # Add file size
                try:
                    model['local_size'] = os.path.getsize(model['local_path']) / (1024 * 1024)  # MB
                except:
                    model['local_size'] = 0
        
        return remote_models


class SegmentationModelManager(QDialog):
    """
    Dialog for managing segmentation models.
    
    This dialog allows users to:
    - View available models
    - Download models from online repositories
    - Remove downloaded models
    - Update model information
    """
    
    # Signal emitted when models are changed (downloaded or deleted)
    models_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the model manager.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)
        
        self.setWindowTitle("Quản lý mô hình phân đoạn")
        self.resize(800, 500)
        
        # List of available models
        self.available_models = []
        
        # Initialize threads
        self.info_thread = None
        self.download_thread = None
        
        # Initialize UI
        self._init_ui()
        
        # Get model information on startup
        self._refresh_model_list()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout()
        
        # Model table
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(7)
        self.model_table.setHorizontalHeaderLabels([
            "Tên",
            "Mô tả",
            "Cấu trúc hỗ trợ",
            "Phiên bản",
            "Kích thước",
            "Trạng thái",
            "Hành động"
        ])
        self.model_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.model_table.setSelectionMode(QTableWidget.SingleSelection)
        self.model_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        # Add model table to layout
        main_layout.addWidget(self.model_table)
        
        # Progress bar and status
        progress_group = QGroupBox("Tiến trình tải xuống")
        progress_layout = QVBoxLayout()
        
        # Overall progress
        self.overall_progress_label = QLabel("Tổng tiến trình:")
        self.overall_progress_bar = QProgressBar()
        progress_layout.addWidget(self.overall_progress_label)
        progress_layout.addWidget(self.overall_progress_bar)
        
        # Current model progress
        self.current_model_label = QLabel("Mô hình hiện tại:")
        self.current_model_progress_bar = QProgressBar()
        progress_layout.addWidget(self.current_model_label)
        progress_layout.addWidget(self.current_model_progress_bar)
        
        # Status label
        self.status_label = QLabel("Sẵn sàng")
        progress_layout.addWidget(self.status_label)
        
        # Set layout for progress group
        progress_group.setLayout(progress_layout)
        
        # Hide progress group initially
        progress_group.setVisible(False)
        self.progress_group = progress_group
        
        # Add progress group to main layout
        main_layout.addWidget(progress_group)
        
        # Button row
        button_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_button = QPushButton("Làm mới")
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_button.clicked.connect(self._refresh_model_list)
        button_layout.addWidget(self.refresh_button)
        
        # Download selected button
        self.download_selected_button = QPushButton("Tải xuống đã chọn")
        self.download_selected_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        self.download_selected_button.clicked.connect(self._download_selected_models)
        self.download_selected_button.setEnabled(False)
        button_layout.addWidget(self.download_selected_button)
        
        # Download all button
        self.download_all_button = QPushButton("Tải xuống tất cả")
        self.download_all_button.clicked.connect(self._download_all_models)
        button_layout.addWidget(self.download_all_button)
        
        # Cancel button
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self._cancel_download)
        self.cancel_button.setVisible(False)
        button_layout.addWidget(self.cancel_button)
        
        # Spacer
        button_layout.addStretch()
        
        # Close button
        self.close_button = QPushButton("Đóng")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        # Add button layout to main layout
        main_layout.addLayout(button_layout)
        
        # Set dialog layout
        self.setLayout(main_layout)
        
        # Connect selection changes
        self.model_table.selectionModel().selectionChanged.connect(self._update_button_states)
    
    def _refresh_model_list(self):
        """Refresh the model list from the repository."""
        # Show status
        self.status_label.setText("Đang lấy thông tin mô hình...")
        
        # Disable buttons
        self.refresh_button.setEnabled(False)
        self.download_selected_button.setEnabled(False)
        self.download_all_button.setEnabled(False)
        
        # Create and start info thread
        self.info_thread = ModelInfoThread()
        self.info_thread.info_retrieved.connect(self._update_model_table)
        self.info_thread.error_occurred.connect(self._show_info_error)
        self.info_thread.start()
    
    def _update_model_table(self, models: List[Dict[str, Any]]):
        """
        Update the model table with available models.
        
        Parameters
        ----------
        models : List[Dict[str, Any]]
            List of model information
        """
        # Store models
        self.available_models = models
        
        # Clear table
        self.model_table.setRowCount(0)
        
        # Populate table
        for i, model in enumerate(models):
            self.model_table.insertRow(i)
            
            # Model name
            name_item = QTableWidgetItem(model.get('name', 'Unknown'))
            self.model_table.setItem(i, 0, name_item)
            
            # Description
            description = model.get('description', '')
            self.model_table.setItem(i, 1, QTableWidgetItem(description))
            
            # Supported structures
            structures = model.get('structure_names', model.get('structures', []))
            structures_text = ", ".join(structures) if structures else ""
            self.model_table.setItem(i, 2, QTableWidgetItem(structures_text))
            
            # Version
            version = model.get('version', '1.0')
            self.model_table.setItem(i, 3, QTableWidgetItem(version))
            
            # Size
            size = model.get('local_size', model.get('size', 0))
            size_text = f"{size:.1f} MB" if size else ""
            self.model_table.setItem(i, 4, QTableWidgetItem(size_text))
            
            # Status
            is_local = model.get('is_local', False)
            status_text = "Đã cài đặt" if is_local else "Chưa cài đặt"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(QColor("green" if is_local else "red"))
            self.model_table.setItem(i, 5, status_item)
            
            # Action button
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(3, 0, 3, 0)
            
            if is_local:
                # Delete button
                delete_button = QPushButton()
                delete_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
                delete_button.setToolTip("Xóa mô hình")
                delete_button.clicked.connect(lambda checked, name=model.get('name'): self._delete_model(name))
                action_layout.addWidget(delete_button)
            else:
                # Download button
                download_button = QPushButton()
                download_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
                download_button.setToolTip("Tải xuống mô hình")
                download_button.clicked.connect(lambda checked, name=model.get('name'): self._download_model(name))
                action_layout.addWidget(download_button)
            
            action_layout.addStretch()
            self.model_table.setCellWidget(i, 6, action_widget)
        
        # Resize columns
        self.model_table.resizeColumnsToContents()
        
        # Update status
        self.status_label.setText(f"Đã tìm thấy {len(models)} mô hình")
        
        # Enable buttons
        self.refresh_button.setEnabled(True)
        self.download_all_button.setEnabled(len(models) > 0)
        
        # Update button states based on selection
        self._update_button_states()
    
    def _show_info_error(self, error_message: str):
        """
        Show error message when retrieving model information.
        
        Parameters
        ----------
        error_message : str
            Error message to show
        """
        self.status_label.setText("Lỗi khi lấy thông tin mô hình")
        
        # Show error message
        QMessageBox.critical(self, "Lỗi", error_message)
        
        # Enable refresh button
        self.refresh_button.setEnabled(True)
        
        # Update button states
        self._update_button_states()
    
    def _update_button_states(self):
        """Update button states based on current selection and download status."""
        # Get selected rows
        selected_rows = self.model_table.selectionModel().selectedRows()
        
        # Check if any row is selected
        if selected_rows:
            # Get selected model info
            row = selected_rows[0].row()
            model = self.available_models[row] if row < len(self.available_models) else None
            
            if model:
                # Enable download button if model is not local
                is_local = model.get('is_local', False)
                self.download_selected_button.setEnabled(not is_local and not self._is_downloading())
        else:
            # No row selected, disable download button
            self.download_selected_button.setEnabled(False)
        
        # Enable/disable download all button based on download status
        self.download_all_button.setEnabled(not self._is_downloading())
        
        # Enable/disable close button based on download status
        self.close_button.setEnabled(not self._is_downloading())
        
        # Show/hide cancel button based on download status
        self.cancel_button.setVisible(self._is_downloading())
    
    def _is_downloading(self) -> bool:
        """
        Check if download is in progress.
        
        Returns
        -------
        bool
            True if download is in progress, False otherwise
        """
        return self.download_thread is not None and self.download_thread.is_running
    
    def _download_selected_models(self):
        """Download selected models."""
        # Get selected rows
        selected_rows = self.model_table.selectionModel().selectedRows()
        
        if not selected_rows:
            return
        
        # Get selected model names
        models_to_download = []
        for row in selected_rows:
            index = row.row()
            if index < len(self.available_models):
                model = self.available_models[index]
                if not model.get('is_local', False):
                    models_to_download.append(model.get('name', ''))
        
        if not models_to_download:
            return
        
        # Start download
        self._start_download(models_to_download)
    
    def _download_all_models(self):
        """Download all models that are not already local."""
        # Get models to download
        models_to_download = []
        for model in self.available_models:
            if not model.get('is_local', False):
                models_to_download.append(model.get('name', ''))
        
        if not models_to_download:
            QMessageBox.information(self, "Thông báo", "Tất cả các mô hình đã được tải xuống")
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Tải xuống tất cả",
            f"Bạn có chắc chắn muốn tải xuống {len(models_to_download)} mô hình?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Start download
        self._start_download(models_to_download)
    
    def _download_model(self, model_name: str):
        """
        Download a single model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to download
        """
        self._start_download([model_name])
    
    def _start_download(self, models_to_download: List[str]):
        """
        Start downloading models.
        
        Parameters
        ----------
        models_to_download : List[str]
            List of model names to download
        """
        if not models_to_download:
            return
        
        # Show progress group
        self.progress_group.setVisible(True)
        
        # Reset progress bars
        self.overall_progress_bar.setValue(0)
        self.current_model_progress_bar.setValue(0)
        
        # Update labels
        self.overall_progress_label.setText(f"Tổng tiến trình ({len(models_to_download)} mô hình):")
        self.current_model_label.setText("Đang chuẩn bị...")
        
        # Update status
        self.status_label.setText(f"Đang tải xuống {len(models_to_download)} mô hình...")
        
        # Create and start download thread
        self.download_thread = ModelDownloaderThread(models_to_download)
        self.download_thread.progress_updated.connect(self._update_model_progress)
        self.download_thread.download_finished.connect(self._handle_download_finished)
        self.download_thread.overall_progress_updated.connect(self._update_overall_progress)
        self.download_thread.all_downloads_finished.connect(self._handle_all_downloads_finished)
        self.download_thread.start()
        
        # Update button states
        self._update_button_states()
    
    def _update_model_progress(self, model_name: str, progress: int):
        """
        Update progress for current model.
        
        Parameters
        ----------
        model_name : str
            Name of the model being downloaded
        progress : int
            Progress value (0-100)
        """
        # Update label
        self.current_model_label.setText(f"Mô hình hiện tại: {model_name}")
        
        # Update progress bar
        if progress >= 0:
            self.current_model_progress_bar.setRange(0, 100)
            self.current_model_progress_bar.setValue(progress)
        else:
            # Use indeterminate progress bar if progress is unknown
            self.current_model_progress_bar.setRange(0, 0)
    
    def _update_overall_progress(self, progress: int):
        """
        Update overall progress.
        
        Parameters
        ----------
        progress : int
            Progress value (0-100)
        """
        self.overall_progress_bar.setValue(progress)
    
    def _handle_download_finished(self, model_name: str, success: bool):
        """
        Handle completion of a single model download.
        
        Parameters
        ----------
        model_name : str
            Name of the downloaded model
        success : bool
            Whether download was successful
        """
        # Log completion
        if success:
            logger.info(f"Successfully downloaded model '{model_name}'")
        else:
            logger.error(f"Failed to download model '{model_name}'")
        
        # Update status for this model
        for i, model in enumerate(self.available_models):
            if model.get('name', '') == model_name:
                if success:
                    # Update model status
                    model['is_local'] = True
                    model['local_path'] = os.path.join(MODELS_DIR, model.get('filename', f"{model_name}.pt"))
                    
                    try:
                        model['local_size'] = os.path.getsize(model['local_path']) / (1024 * 1024)  # MB
                    except:
                        model['local_size'] = 0
                    
                    # Update table cell
                    status_item = QTableWidgetItem("Đã cài đặt")
                    status_item.setForeground(QColor("green"))
                    self.model_table.setItem(i, 5, status_item)
                    
                    # Update size
                    size_text = f"{model.get('local_size', 0):.1f} MB"
                    self.model_table.setItem(i, 4, QTableWidgetItem(size_text))
                    
                    # Update action button
                    action_widget = QWidget()
                    action_layout = QHBoxLayout(action_widget)
                    action_layout.setContentsMargins(3, 0, 3, 0)
                    
                    # Delete button
                    delete_button = QPushButton()
                    delete_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
                    delete_button.setToolTip("Xóa mô hình")
                    delete_button.clicked.connect(lambda checked, name=model_name: self._delete_model(name))
                    action_layout.addWidget(delete_button)
                    
                    action_layout.addStretch()
                    self.model_table.setCellWidget(i, 6, action_widget)
                    
                break
    
    def _handle_all_downloads_finished(self):
        """Handle completion of all downloads."""
        # Clear download thread
        self.download_thread = None
        
        # Hide progress group
        self.progress_group.setVisible(False)
        
        # Update status
        self.status_label.setText("Tải xuống hoàn tất")
        
        # Update button states
        self._update_button_states()
        
        # Emit models changed signal
        self.models_changed.emit()
    
    def _cancel_download(self):
        """Cancel ongoing download."""
        if self.download_thread and self.download_thread.is_running:
            # Stop download thread
            self.download_thread.stop()
            
            # Reset progress
            self.overall_progress_bar.setValue(0)
            self.current_model_progress_bar.setValue(0)
            
            # Hide progress group
            self.progress_group.setVisible(False)
            
            # Update status
            self.status_label.setText("Tải xuống đã bị hủy")
            
            # Update button states
            self._update_button_states()
    
    def _delete_model(self, model_name: str):
        """
        Delete a model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to delete
        """
        # Find model
        model = None
        model_index = -1
        for i, m in enumerate(self.available_models):
            if m.get('name', '') == model_name:
                model = m
                model_index = i
                break
        
        if model is None or model_index < 0:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Xóa mô hình",
            f"Bạn có chắc chắn muốn xóa mô hình '{model_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Get model file path
        model_path = model.get('local_path', os.path.join(MODELS_DIR, model.get('filename', f"{model_name}.pt")))
        
        # Delete model file
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                
                # Also remove info file if exists
                info_path = os.path.join(MODELS_DIR, f"{os.path.splitext(os.path.basename(model_path))[0]}_info.json")
                if os.path.exists(info_path):
                    os.remove(info_path)
                
                # Update model info
                model['is_local'] = False
                model.pop('local_path', None)
                model.pop('local_size', None)
                
                # Update table
                status_item = QTableWidgetItem("Chưa cài đặt")
                status_item.setForeground(QColor("red"))
                self.model_table.setItem(model_index, 5, status_item)
                
                # Update size
                self.model_table.setItem(model_index, 4, QTableWidgetItem(""))
                
                # Update action button
                action_widget = QWidget()
                action_layout = QHBoxLayout(action_widget)
                action_layout.setContentsMargins(3, 0, 3, 0)
                
                # Download button
                download_button = QPushButton()
                download_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
                download_button.setToolTip("Tải xuống mô hình")
                download_button.clicked.connect(lambda checked, name=model_name: self._download_model(name))
                action_layout.addWidget(download_button)
                
                action_layout.addStretch()
                self.model_table.setCellWidget(model_index, 6, action_widget)
                
                # Update status
                self.status_label.setText(f"Đã xóa mô hình '{model_name}'")
                
                # Emit models changed signal
                self.models_changed.emit()
                
            else:
                QMessageBox.warning(self, "Cảnh báo", f"Không tìm thấy tệp mô hình: {model_path}")
        
        except Exception as e:
            logger.error(f"Error deleting model '{model_name}': {str(e)}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa mô hình '{model_name}': {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = SegmentationModelManager()
    dialog.exec_() 