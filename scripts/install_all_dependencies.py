#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Dependency Installer

This script installs all dependencies required for QuangTPS to run properly.
It handles platform-specific requirements and checks for existing installations.
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dependency_installer")

# Define core dependencies
CORE_DEPENDENCIES = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "pydicom>=2.3.0",
    "scikit-image>=0.19.0",
    "SimpleITK>=2.1.0",
    "h5py>=3.6.0",
    "pandas>=1.4.0",
    "PyQt5>=5.15.0",
    "vtk>=9.1.0",
    "pyvista>=0.34.0",
    "PyVistaQt>=0.6.0",
    "opencv-python>=4.5.5.0",
    "weasyprint>=54.0",
    "cython>=0.29.0",
    "pytest>=7.0.0",
    "trimesh>=3.12.0",
]

# Optional Machine Learning dependencies
ML_DEPENDENCIES = [
    "tensorflow>=2.8.0",
    "keras>=2.8.0",
    "scikit-learn>=1.0.0",
]

# Development dependencies
DEV_DEPENDENCIES = [
    "black",
    "pylint",
    "mypy",
    "pytest-cov",
    "sphinx",
    "sphinx-rtd-theme",
]

# Platform-specific dependencies
PLATFORM_DEPENDENCIES = {
    "Windows": {
        "packages": ["pywin32>=303"],
        "system_dependencies": ["Microsoft Visual C++ Redistributable for Visual Studio 2019"]
    },
    "Linux": {
        "packages": [],
        "system_dependencies": ["libgl1-mesa-glx", "libxt6", "libxrender1", "xvfb", "libcairo2", "libpango-1.0-0", "libpangocairo-1.0-0"]
    },
    "Darwin": {  # macOS
        "packages": [],
        "system_dependencies": ["xquartz"]
    }
}

def get_platform() -> str:
    """Get the current platform."""
    system = platform.system()
    if system == "Linux":
        return "Linux"
    elif system == "Windows":
        return "Windows"
    elif system == "Darwin":
        return "Darwin"
    else:
        logger.warning(f"Unknown platform: {system}")
        return "Unknown"

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error(f"Python version {python_version.major}.{python_version.minor} is not supported. Please use Python 3.8 or newer.")
        return False
    return True

def check_pip() -> bool:
    """Check if pip is installed and up to date."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL)
        # Ensure pip is up to date
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        logger.error("Pip is not installed or not working correctly.")
        return False

def get_installed_packages() -> Set[str]:
    """Get the set of installed packages."""
    try:
        result = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"]).decode("utf-8")
        installed_packages = set()
        for line in result.splitlines():
            if "==" in line:
                package_name = line.split("==")[0].lower()
                installed_packages.add(package_name)
        return installed_packages
    except subprocess.CalledProcessError:
        logger.error("Failed to get installed packages.")
        return set()

def install_package(package: str, upgrade: bool = False) -> bool:
    """Install a single package using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call(cmd)
        logger.info(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package}")
        return False

def check_system_dependencies(system_deps: List[str]) -> None:
    """Check if system dependencies are installed."""
    platform_name = get_platform()
    
    if platform_name == "Windows":
        logger.info("Please ensure you have the following installed:")
        for dep in system_deps:
            logger.info(f"  - {dep}")
    elif platform_name == "Linux":
        logger.info("You may need to install the following system packages:")
        if os.path.exists("/etc/debian_version"):
            logger.info(f"  sudo apt-get install {' '.join(system_deps)}")
        elif os.path.exists("/etc/redhat-release"):
            logger.info(f"  sudo yum install {' '.join(system_deps)}")
        else:
            logger.info(f"  {' '.join(system_deps)}")
    elif platform_name == "Darwin":
        logger.info("You may need to install the following with brew:")
        logger.info(f"  brew install {' '.join(system_deps)}")

def install_vtk_dependencies() -> bool:
    """Install VTK and related dependencies."""
    logger.info("Installing VTK and related dependencies...")
    
    # First try to install VTK directly
    if not install_package("vtk>=9.1.0", upgrade=True):
        logger.warning("Standard VTK installation failed, trying alternative approaches...")
        platform_name = get_platform()
        
        if platform_name == "Windows":
            # On Windows, try specific wheels
            if platform.architecture()[0] == '64bit':
                logger.info("Trying VTK wheel for Windows 64-bit...")
                return install_package("https://files.pythonhosted.org/packages/8e/da/684cf2d387173101abba13a4ffa42c618eb358ccb4cf8a101de9986a8b1d/vtk-9.1.0-cp39-cp39-win_amd64.whl")
            else:
                logger.error("VTK installation on 32-bit Windows is not supported.")
                return False
        elif platform_name == "Linux":
            logger.info("On Linux, please ensure you have development packages installed:")
            logger.info("  sudo apt-get install libgl1-mesa-dev libxt-dev")
            logger.info("  sudo apt-get install python3-dev cmake build-essential")
            return install_package("vtk>=9.1.0", upgrade=True)
        elif platform_name == "Darwin":
            logger.info("On macOS, consider using Homebrew:")
            logger.info("  brew install vtk")
            return install_package("vtk>=9.1.0", upgrade=True)
    
    # If we got here, VTK installed or at least we tried our best
    return True

def install_weasyprint_dependencies() -> bool:
    """Install WeasyPrint and its dependencies."""
    logger.info("Installing WeasyPrint and its dependencies...")
    
    platform_name = get_platform()
    
    if platform_name == "Windows":
        try:
            # Install the GTK runtime for Windows
            logger.info("WeasyPrint requires GTK on Windows.")
            logger.info("Please download and install the GTK runtime from:")
            logger.info("https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer")
            
            # Install WeasyPrint
            return install_package("weasyprint>=54.0", upgrade=True)
        except Exception as e:
            logger.error(f"Failed to install WeasyPrint: {e}")
            return False
    else:
        # For Linux and macOS, install system dependencies first
        system_deps = []
        if platform_name == "Linux":
            system_deps = ["libcairo2-dev", "libpango1.0-dev", "libgdk-pixbuf2.0-dev", "libffi-dev", "shared-mime-info"]
        elif platform_name == "Darwin":
            system_deps = ["cairo", "pango", "gdk-pixbuf", "libffi"]
        
        check_system_dependencies(system_deps)
        
        # Install WeasyPrint
        return install_package("weasyprint>=54.0", upgrade=True)

def install_all_dependencies(include_ml: bool = False, include_dev: bool = False, upgrade: bool = False) -> bool:
    """Install all required dependencies."""
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    platform_name = get_platform()
    logger.info(f"Platform detected: {platform_name}")
    
    # Get currently installed packages
    installed_packages = get_installed_packages()
    
    # Install platform-specific packages
    if platform_name in PLATFORM_DEPENDENCIES:
        platform_deps = PLATFORM_DEPENDENCIES[platform_name]
        system_deps = platform_deps.get("system_dependencies", [])
        check_system_dependencies(system_deps)
        
        for pkg in platform_deps.get("packages", []):
            pkg_name = pkg.split(">=")[0].split("==")[0].lower()
            if pkg_name in installed_packages and not upgrade:
                logger.info(f"Package {pkg_name} is already installed. Skipping.")
                continue
            install_package(pkg, upgrade)
    
    # Install core dependencies
    for dep in CORE_DEPENDENCIES:
        pkg_name = dep.split(">=")[0].split("==")[0].lower()
        if pkg_name in installed_packages and not upgrade:
            logger.info(f"Package {pkg_name} is already installed. Skipping.")
            continue
        install_package(dep, upgrade)
    
    # Handle special cases for VTK and WeasyPrint
    install_vtk_dependencies()
    install_weasyprint_dependencies()
    
    # Optionally install machine learning dependencies
    if include_ml:
        logger.info("Installing machine learning dependencies...")
        for dep in ML_DEPENDENCIES:
            pkg_name = dep.split(">=")[0].split("==")[0].lower()
            if pkg_name in installed_packages and not upgrade:
                logger.info(f"Package {pkg_name} is already installed. Skipping.")
                continue
            install_package(dep, upgrade)
    
    # Optionally install development dependencies
    if include_dev:
        logger.info("Installing development dependencies...")
        for dep in DEV_DEPENDENCIES:
            pkg_name = dep.split(">=")[0].split("==")[0].lower()
            if pkg_name in installed_packages and not upgrade:
                logger.info(f"Package {pkg_name} is already installed. Skipping.")
                continue
            install_package(dep, upgrade)
    
    logger.info("All dependencies have been installed.")
    return True

def main():
    """Main function for the dependency installer."""
    parser = argparse.ArgumentParser(description="Install dependencies for QuangTPS")
    parser.add_argument("--ml", action="store_true", help="Install machine learning dependencies")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade all packages")
    args = parser.parse_args()
    
    success = install_all_dependencies(
        include_ml=args.ml,
        include_dev=args.dev,
        upgrade=args.upgrade
    )
    
    if success:
        logger.info("Dependency installation completed successfully.")
    else:
        logger.error("Dependency installation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 