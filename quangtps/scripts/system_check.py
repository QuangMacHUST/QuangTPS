"""
Module for system environment checking and diagnostics in QuangTPS.
Used to verify that all required dependencies are available and properly configured.
"""

import os
import sys
import platform
import logging
import importlib
from pathlib import Path
import subprocess

from quangtps.core.logging import get_logger

logger = get_logger(__name__)

def check_system():
    """
    Perform system checks to verify environment and dependencies.
    
    Checks for:
    - Python version
    - Required packages
    - GPU availability
    - System resources
    - File permissions
    
    Returns
    -------
    bool
        True if all checks pass, False otherwise
    """
    logger.info("Running system diagnostics...")
    
    # Track overall status
    all_checks_passed = True
    
    # Check Python version
    py_version = _check_python_version()
    all_checks_passed = all_checks_passed and py_version
    
    # Check required packages
    req_packages = _check_required_packages()
    all_checks_passed = all_checks_passed and req_packages
    
    # Check GPU availability
    gpu_check = _check_gpu()
    all_checks_passed = all_checks_passed and gpu_check
    
    # Check system resources
    resource_check = _check_system_resources()
    all_checks_passed = all_checks_passed and resource_check
    
    # Check file permissions
    perm_check = _check_file_permissions()
    all_checks_passed = all_checks_passed and perm_check
    
    # Print summary
    if all_checks_passed:
        logger.info("All system checks passed. System is properly configured.")
    else:
        logger.warning("One or more system checks failed. Review the logs for details.")
    
    return all_checks_passed

def _check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    logger.info(f"Checking Python version: {platform.python_version()}")
    
    if current_version >= required_version:
        logger.info(f"Python version check passed: {current_version.major}.{current_version.minor}.{current_version.micro}")
        return True
    else:
        logger.error(f"Python version check failed. Required: {required_version[0]}.{required_version[1]}+, "
                    f"Found: {current_version.major}.{current_version.minor}.{current_version.micro}")
        return False

def _check_required_packages():
    """Check if required packages are installed and have correct versions"""
    required_packages = {
        'numpy': '1.20.0',
        'scipy': '1.6.0',
        'matplotlib': '3.3.0',
        'pydicom': '2.1.0',
        'PyQt5': '5.15.0',
        'vtk': '9.0.0',
        'pynrrd': '0.4.0',
        'pynetdicom': '1.5.0',
    }
    
    all_packages_ok = True
    
    logger.info("Checking required packages...")
    
    for package, min_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                version = module.__version__
            elif hasattr(module, 'VERSION'):
                version = module.VERSION
            else:
                version = "Unknown"
                
            logger.info(f"Package {package}: Found version {version}")
            
            # Simple version comparison - could be improved with packaging.version
            if version != "Unknown" and version < min_version:
                logger.warning(f"Package {package}: Version {version} is older than recommended {min_version}")
                
        except ImportError:
            logger.error(f"Package {package}: Not installed")
            all_packages_ok = False
    
    return all_packages_ok

def _check_gpu():
    """Check if GPU is available and compatible"""
    gpu_available = False
    
    logger.info("Checking GPU availability...")
    
    # Try to detect CUDA capability using different approaches
    try:
        # Try with torch if available
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"PyTorch detected {gpu_count} CUDA device(s)")
            for i in range(gpu_count):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            gpu_available = True
    except ImportError:
        logger.info("PyTorch not available for GPU detection")
    
    # If still no GPU detected, try querying system
    if not gpu_available:
        try:
            if platform.system() == "Windows":
                process = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                             capture_output=True, text=True, check=True)
                if process.returncode == 0:
                    gpus = [line.strip() for line in process.stdout.splitlines() if line.strip() and line.strip() != "Name"]
                    if gpus:
                        logger.info("Detected GPU(s):")
                        for gpu in gpus:
                            logger.info(f"  {gpu}")
                        gpu_available = True
            elif platform.system() == "Linux":
                process = subprocess.run(['lspci'], capture_output=True, text=True, check=True)
                if process.returncode == 0:
                    gpu_lines = [line for line in process.stdout.splitlines() if "VGA" in line or "3D" in line]
                    if gpu_lines:
                        logger.info("Detected GPU(s):")
                        for gpu in gpu_lines:
                            logger.info(f"  {gpu}")
                        gpu_available = True
        except Exception as e:
            logger.info(f"Error detecting GPU via system commands: {e}")
    
    if gpu_available:
        logger.info("GPU check: GPU(s) available")
    else:
        logger.warning("GPU check: No GPU detected. Performance may be limited.")
    
    # Always return True since GPU is optional
    return True

def _check_system_resources():
    """Check available system resources"""
    import psutil
    
    logger.info("Checking system resources...")
    
    # Memory
    virtual_memory = psutil.virtual_memory()
    total_ram_gb = virtual_memory.total / (1024 ** 3)
    available_ram_gb = virtual_memory.available / (1024 ** 3)
    
    logger.info(f"Memory: Total: {total_ram_gb:.2f} GB, Available: {available_ram_gb:.2f} GB")
    
    if total_ram_gb < 8:
        logger.warning(f"Low total RAM detected: {total_ram_gb:.2f} GB. Minimum recommended is 8 GB.")
        return False
    
    if available_ram_gb < 2:
        logger.warning(f"Low available RAM detected: {available_ram_gb:.2f} GB. Performance may be impacted.")
    
    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    logger.info(f"CPU: {physical_cores} physical cores, {cpu_count} logical cores")
    
    if physical_cores < 4:
        logger.warning(f"Low CPU core count detected: {physical_cores} physical cores. Minimum recommended is 4.")
    
    # Disk space
    try:
        app_path = Path(__file__).parent.parent.parent  # QuangTPS root directory
        disk_usage = psutil.disk_usage(app_path)
        
        free_space_gb = disk_usage.free / (1024 ** 3)
        total_space_gb = disk_usage.total / (1024 ** 3)
        
        logger.info(f"Disk: Total: {total_space_gb:.2f} GB, Free: {free_space_gb:.2f} GB on application drive")
        
        if free_space_gb < 10:
            logger.warning(f"Low disk space detected: {free_space_gb:.2f} GB free. Minimum recommended is 10 GB.")
    except Exception as e:
        logger.error(f"Error checking disk space: {e}")
    
    return True

def _check_file_permissions():
    """Check if the application has necessary file permissions"""
    app_path = Path(__file__).parent.parent.parent  # QuangTPS root directory
    
    logger.info("Checking file permissions...")
    
    # Check read permissions
    read_ok = _check_read_permission(app_path)
    
    # Check write permissions in specific directories
    data_dir = app_path / "data"
    logs_dir = app_path / "logs"
    temp_dir = app_path / "temp"
    
    # Create directories if they don't exist
    for directory in [data_dir, logs_dir, temp_dir]:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                return False
    
    # Check write permissions
    write_dirs = [data_dir, logs_dir, temp_dir]
    write_ok = all(_check_write_permission(d) for d in write_dirs)
    
    if read_ok and write_ok:
        logger.info("File permission check passed")
        return True
    else:
        if not read_ok:
            logger.error("Read permission check failed")
        if not write_ok:
            logger.error("Write permission check failed")
        return False

def _check_read_permission(path):
    """Check if the application can read from the specified path"""
    try:
        # List the directory
        list(path.iterdir())
        return True
    except Exception as e:
        logger.error(f"Read permission check failed for {path}: {e}")
        return False

def _check_write_permission(path):
    """Check if the application can write to the specified path"""
    test_file = path / f"permission_test_{os.getpid()}.tmp"
    
    try:
        # Try to create a test file
        with open(test_file, 'w') as f:
            f.write('test')
        
        # Clean up
        test_file.unlink()
        return True
    except Exception as e:
        logger.error(f"Write permission check failed for {path}: {e}")
        return False