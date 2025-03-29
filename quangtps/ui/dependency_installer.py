#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dependency Installer

This module provides functionality to check for and install required dependencies
for advanced features like 3D visualization in the QuangTPS system.
"""

import sys
import os
import logging
import subprocess
import importlib
import threading
import time
from typing import List, Dict, Callable, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, 
    QPushButton, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

logger = logging.getLogger(__name__)

class DependencyInstaller(QObject):
    """
    A class for checking and installing dependencies.
    
    This class can:
    - Check if required packages are installed
    - Install missing packages using pip
    - Provide status updates during installation
    """
    
    # Signals for installation progress
    installation_started = pyqtSignal(str)
    installation_progress = pyqtSignal(str, int)
    installation_completed = pyqtSignal(bool, str)
    
    def __init__(self):
        """Initialize the dependency installer."""
        super().__init__()
        
        # Dictionary of feature requirements
        self.feature_requirements = {
            "3d_visualization": {
                "packages": [
                    ("pyvista", "0.38.1"),
                    ("pyvistaqt", "0.9.0"),
                    ("vtk", "9.2.6")
                ],
                "description": "3D visualization of treatment beams and patient anatomy"
            },
            "monte_carlo": {
                "packages": [
                    ("cupy-cuda11x", "12.0.0", "cupy"),
                    ("numba", "0.56.4")
                ],
                "description": "GPU-accelerated Monte Carlo dose calculation"
            },
            "dicom_rt": {
                "packages": [
                    ("pydicom", "2.3.1"),
                    ("pynetdicom", "2.0.2")
                ],
                "description": "DICOM-RT import/export capabilities"
            }
        }
    
    def check_feature_dependencies(self, feature_name: str) -> Tuple[bool, List[str]]:
        """
        Check if dependencies for a specific feature are installed.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature to check dependencies for
            
        Returns
        -------
        Tuple[bool, List[str]]
            A tuple containing:
            - Boolean indicating if all dependencies are met
            - List of missing packages
        """
        if feature_name not in self.feature_requirements:
            logger.warning(f"Unknown feature: {feature_name}")
            return False, [f"Unknown feature: {feature_name}"]
        
        requirements = self.feature_requirements[feature_name]
        missing_packages = []
        
        for package_info in requirements["packages"]:
            # Package info can be a tuple with (install_name, version, import_name)
            # If import_name is not provided, use install_name
            if len(package_info) >= 3:
                install_name, version, import_name = package_info
            else:
                install_name, version = package_info
                import_name = install_name
            
            # Try to import the package
            try:
                module = importlib.import_module(import_name)
                
                # Check version if available
                if hasattr(module, "__version__") and version:
                    installed_version = module.__version__
                    if not self._version_meets_requirement(installed_version, version):
                        missing_packages.append(f"{install_name}>={version}")
                        logger.debug(f"Package {import_name} version {installed_version} does not meet requirement {version}")
                
            except ImportError:
                missing_packages.append(f"{install_name}>={version}")
                logger.debug(f"Package {import_name} not found")
        
        return len(missing_packages) == 0, missing_packages
    
    def _version_meets_requirement(self, installed_version: str, required_version: str) -> bool:
        """
        Check if an installed version meets the required version.
        
        This is a simplified version check that just compares version numbers.
        A real implementation would use packaging.version for proper comparison.
        
        Parameters
        ----------
        installed_version : str
            The installed version string
        required_version : str
            The required version string
            
        Returns
        -------
        bool
            True if installed version meets requirement
        """
        # Simple version comparison - splits by dots and compares each component
        installed_parts = installed_version.split('.')
        required_parts = required_version.split('.')
        
        # Compare each part
        for i in range(min(len(installed_parts), len(required_parts))):
            try:
                if int(installed_parts[i]) < int(required_parts[i]):
                    return False
                elif int(installed_parts[i]) > int(required_parts[i]):
                    return True
            except ValueError:
                # Handle non-numeric version parts
                if installed_parts[i] < required_parts[i]:
                    return False
                elif installed_parts[i] > required_parts[i]:
                    return True
        
        # If we get here, the versions are equal to the depth we checked
        # If installed has more parts, it's newer
        return len(installed_parts) >= len(required_parts)
    
    def install_dependencies(self, missing_packages: List[str]) -> bool:
        """
        Install missing dependencies using pip.
        
        Parameters
        ----------
        missing_packages : List[str]
            List of packages to install
            
        Returns
        -------
        bool
            True if installation was successful
        """
        if not missing_packages:
            return True
        
        try:
            # Get path to Python executable
            python_executable = sys.executable
            
            # Start installation
            cmd = [python_executable, "-m", "pip", "install"] + missing_packages
            
            self.installation_started.emit(f"Installing packages: {', '.join(missing_packages)}")
            logger.info(f"Installing packages: {', '.join(missing_packages)}")
            
            # Run pip install
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Read output line by line
            progress = 0
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.debug(f"pip: {line.strip()}")
                    self.installation_progress.emit(line.strip(), progress)
                    progress = min(99, progress + 1)  # Increment progress, max 99%
            
            # Wait for process to complete
            return_code = process.wait()
            
            success = (return_code == 0)
            message = "Dependencies installed successfully!" if success else "Failed to install dependencies."
            
            self.installation_progress.emit(message, 100)
            self.installation_completed.emit(success, message)
            
            logger.info(message)
            return success
            
        except Exception as e:
            error_message = f"Error installing dependencies: {str(e)}"
            logger.error(error_message, exc_info=True)
            self.installation_completed.emit(False, error_message)
            return False
    
    def check_gpu_support(self) -> Tuple[bool, str]:
        """
        Check if the system has GPU support for CUDA.
        
        Returns
        -------
        Tuple[bool, str]
            A tuple containing:
            - Boolean indicating if GPU support is available
            - String with GPU information or error message
        """
        try:
            # First check if nvidia-smi is available (Windows/Linux)
            try:
                process = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=5
                )
                
                if process.returncode == 0:
                    gpu_info = process.stdout.strip()
                    return True, gpu_info
                
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            
            # If that fails, try importing CUDA packages
            try:
                import cupy
                return True, f"CuPy {cupy.__version__} detected with CUDA support"
            except ImportError:
                pass
                
            try:
                import torch
                if torch.cuda.is_available():
                    return True, f"PyTorch {torch.__version__} detected with CUDA support"
            except ImportError:
                pass
            
            return False, "No CUDA-capable GPU detected"
            
        except Exception as e:
            logger.error(f"Error checking GPU support: {str(e)}", exc_info=True)
            return False, f"Error checking GPU support: {str(e)}"


class DependencyInstallerDialog(QDialog):
    """Dialog for installing missing dependencies."""
    
    def __init__(self, feature_name: str, missing_packages: List[str], parent=None):
        """
        Initialize the dependency installer dialog.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature requiring dependencies
        missing_packages : List[str]
            List of packages to install
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        self.feature_name = feature_name
        self.missing_packages = missing_packages
        self.installer = DependencyInstaller()
        
        # Connect signals
        self.installer.installation_started.connect(self._on_installation_started)
        self.installer.installation_progress.connect(self._on_installation_progress)
        self.installer.installation_completed.connect(self._on_installation_completed)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("Install Dependencies")
        self.resize(500, 400)
        
        layout = QVBoxLayout()
        
        # Information label
        info_text = f"The following dependencies are required for the '{self.feature_name}' feature:"
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        # List of packages
        packages_text = "\n".join([f"• {pkg}" for pkg in self.missing_packages])
        packages_label = QLabel(packages_text)
        packages_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(packages_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        layout.addWidget(self.log_output)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        self.install_button = QPushButton("Install Dependencies")
        self.install_button.clicked.connect(self._on_install_clicked)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.install_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _on_install_clicked(self):
        """Handle the install button click."""
        self.install_button.setEnabled(False)
        self.close_button.setEnabled(False)
        
        # Start installation in a separate thread
        threading.Thread(
            target=self.installer.install_dependencies,
            args=(self.missing_packages,),
            daemon=True
        ).start()
    
    def _on_installation_started(self, message):
        """Handle installation start."""
        self.log_output.append(message)
        self.progress_bar.setValue(0)
    
    def _on_installation_progress(self, message, progress):
        """Handle installation progress updates."""
        self.log_output.append(message)
        self.progress_bar.setValue(progress)
        # Scroll to bottom
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
    
    def _on_installation_completed(self, success, message):
        """Handle installation completion."""
        self.log_output.append(message)
        
        if success:
            self.log_output.append("\nInstallation completed. Please restart the application.")
            QMessageBox.information(
                self,
                "Installation Complete",
                "Dependencies installed successfully. Please restart the application."
            )
            self.accept()
        else:
            self.log_output.append("\nInstallation failed. See log for details.")
            self.install_button.setEnabled(True)
        
        self.close_button.setEnabled(True)


def check_and_install_feature_dependencies(feature_name: str, parent_widget=None) -> bool:
    """
    Check and install dependencies for a specific feature.
    
    This is a convenience function to check if dependencies are installed
    and show the installation dialog if needed.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature to check dependencies for
    parent_widget : QWidget, optional
        Parent widget for the installation dialog
        
    Returns
    -------
    bool
        True if all dependencies are satisfied
    """
    installer = DependencyInstaller()
    
    # Check if dependencies are already installed
    dependencies_met, missing_packages = installer.check_feature_dependencies(feature_name)
    
    if dependencies_met:
        logger.info(f"All dependencies for {feature_name} are already installed")
        return True
    
    # Show a message asking to install
    if parent_widget:
        requirements = installer.feature_requirements.get(feature_name, {})
        description = requirements.get("description", feature_name)
        
        message = (
            f"The '{description}' feature requires additional packages "
            f"that are not currently installed.\n\n"
            f"Would you like to install them now?"
        )
        
        result = QMessageBox.question(
            parent_widget,
            "Install Dependencies",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if result == QMessageBox.Yes:
            # Show installation dialog
            dialog = DependencyInstallerDialog(feature_name, missing_packages, parent_widget)
            if dialog.exec_() == QDialog.Accepted:
                # Verify installation was successful
                dependencies_met, still_missing = installer.check_feature_dependencies(feature_name)
                return dependencies_met
    
    return False 