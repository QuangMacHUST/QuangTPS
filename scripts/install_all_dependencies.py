#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Complete Dependency Installation Script

This script installs all required dependencies for the QuangTPS system,
including proper setup for WeasyPrint and other problematic packages.

Usage:
    python install_all_dependencies.py
"""

import os
import sys
import platform
import subprocess
import logging
import shutil
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("quangtps-installer")

# All dependencies for QuangTPS by category
DEPENDENCIES = {
    "Core": [
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "scikit-image>=0.19.0",
        "scikit-learn>=1.1.0",
        "opencv-python>=4.6.0",
        "pydicom>=2.3.0",
        "SimpleITK>=2.1.0",
        "h5py>=3.7.0"
    ],
    "GUI": [
        "PyQt5>=5.15.0",
        "PyQtChart>=5.15.0",
        "pyqtgraph>=0.12.0"
    ],
    "Visualization": [
        "pyvista>=0.37.0",
        "pyvistaQt>=0.2.0",
        "vtk>=9.1.0"
    ],
    "Reporting": [
        "weasyprint>=54.0",
        "reportlab>=3.6.0",
        "jinja2>=3.1.0",
        "pypdf2>=2.10.0"
    ],
    "ML": [
        "tensorflow>=2.9.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "monai>=0.9.0"
    ],
    "Optimization": [
        "pulp>=2.6.0",
        "cvxpy>=1.2.0",
        "pyomo>=6.4.0"
    ],
    "Utilities": [
        "requests>=2.28.0",
        "tqdm>=4.64.0",
        "colorama>=0.4.5"
    ]
}

class DependencyInstaller:
    """Class to install all dependencies for QuangTPS."""
    
    def __init__(self):
        """Initialize the dependency installer."""
        self.os_name = platform.system()
        self.os_version = platform.version()
        self.python_version = platform.python_version()
        self.failed_packages = []
        
    def install_all(self):
        """Install all dependencies by category."""
        logger.info("Starting QuangTPS dependency installation")
        logger.info(f"Operating System: {self.os_name} {self.os_version}")
        logger.info(f"Python Version: {self.python_version}")
        
        # Ensure pip is available and up-to-date
        self._update_pip()
        
        # Install dependencies by category
        for category, packages in DEPENDENCIES.items():
            logger.info(f"\nInstalling {category} dependencies:")
            for package in packages:
                self._install_package(package)
        
        # Fix WeasyPrint setup
        self._setup_weasyprint()
        
        # Verify installation
        self._verify_installation()
        
        if self.failed_packages:
            logger.warning("\nThe following packages could not be installed:")
            for package in self.failed_packages:
                logger.warning(f"  - {package}")
            return False
        else:
            logger.info("\n✅ All dependencies installed successfully!")
            return True
    
    def _update_pip(self):
        """Update pip to the latest version."""
        logger.info("Updating pip...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ])
            logger.info("pip updated successfully")
        except Exception as e:
            logger.error(f"Failed to update pip: {e}")
    
    def _install_package(self, package):
        """Install a specific package."""
        logger.info(f"Installing {package}...")
        try:
            # Check if package is already installed with required version
            package_name = package.split('>=')[0] if '>=' in package else package
            try:
                spec = __import__(package_name.replace('-', '_'))
                logger.info(f"✅ {package_name} is already installed")
                return True
            except ImportError:
                pass  # Package not installed, continue with installation
            
            # Install the package
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            
            logger.info(f"✅ {package} installed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            self.failed_packages.append(package)
            return False
    
    def _setup_weasyprint(self):
        """Set up WeasyPrint properly based on the operating system."""
        logger.info("\nSetting up WeasyPrint...")
        
        if self.os_name == "Windows":
            self._setup_weasyprint_windows()
        elif self.os_name == "Linux":
            self._setup_weasyprint_linux()
        elif self.os_name == "Darwin":  # macOS
            self._setup_weasyprint_macos()
        else:
            logger.warning(f"Unsupported OS: {self.os_name}")
    
    def _setup_weasyprint_windows(self):
        """Set up WeasyPrint on Windows."""
        logger.info("Setting up WeasyPrint on Windows...")
        
        try:
            # Check if requests is installed
            try:
                import requests
                import tempfile
            except ImportError:
                logger.info("Installing requests...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "requests"
                ])
                import requests
                import tempfile
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "gtk3-runtime-installer.exe")
            
            # Latest GTK3 installer URL
            gtk_url = "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/download/2023-03-01/gtk3-runtime-3.24.38-2023-03-01-ts-win64.exe"
            
            logger.info(f"Downloading GTK3 Runtime from {gtk_url}...")
            response = requests.get(gtk_url, stream=True)
            response.raise_for_status()
            
            # Write the installer to a temporary file
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Download complete. Installing GTK3 Runtime...")
            
            # Run the installer silently with default options
            subprocess.check_call([temp_file, "/S"])
            
            # Update PATH to include GTK bin directory
            gtk_bin_path = "C:\\Program Files\\GTK3-Runtime Win64\\bin"
            
            # Check if the path exists
            if os.path.exists(gtk_bin_path):
                # Add to PATH environment variable
                current_path = os.environ.get('PATH', '')
                if gtk_bin_path not in current_path:
                    os.environ['PATH'] = f"{gtk_bin_path};{current_path}"
                    logger.info(f"Added {gtk_bin_path} to PATH environment variable")
                    
                    # Also modify system PATH for future sessions
                    try:
                        # Using PowerShell to update system PATH
                        ps_command = f'[Environment]::SetEnvironmentVariable("PATH", "{gtk_bin_path};" + [Environment]::GetEnvironmentVariable("PATH", "Machine"), "Machine")'
                        subprocess.check_call(["powershell", "-Command", ps_command])
                        logger.info("Updated system PATH environment variable")
                    except Exception as e:
                        logger.warning(f"Could not update system PATH: {e}")
                        logger.warning("You may need to manually add GTK to your PATH")
            else:
                logger.warning(f"GTK installation path {gtk_bin_path} not found. Installation may have failed or used a different path.")
            
            # Clean up temporary files
            try:
                os.remove(temp_file)
                os.rmdir(temp_dir)
            except:
                pass
                
            # Reinstall WeasyPrint
            logger.info("Reinstalling WeasyPrint...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y", "weasyprint"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--force-reinstall", "weasyprint"
            ])
            
            logger.info("WeasyPrint setup completed for Windows.")
            
        except Exception as e:
            logger.error(f"Failed to set up WeasyPrint on Windows: {e}")
            self.failed_packages.append("weasyprint-setup")
    
    def _setup_weasyprint_linux(self):
        """Set up WeasyPrint on Linux."""
        logger.info("Setting up WeasyPrint on Linux...")
        
        try:
            # Try to detect the distribution
            if shutil.which("apt"):
                logger.info("Detected Debian/Ubuntu. Installing dependencies...")
                subprocess.check_call([
                    "sudo", "apt", "install", "-y", 
                    "build-essential", "python3-dev", "python3-pip", "python3-setuptools",
                    "python3-wheel", "python3-cffi", "libcairo2", "libpango-1.0-0", 
                    "libpangocairo-1.0-0", "libgdk-pixbuf2.0-0", "libffi-dev", "shared-mime-info"
                ])
            elif shutil.which("dnf"):
                logger.info("Detected Fedora/RHEL. Installing dependencies...")
                subprocess.check_call([
                    "sudo", "dnf", "install", "-y",
                    "redhat-rpm-config", "python3-devel", "python3-pip", "python3-setuptools",
                    "python3-wheel", "python3-cffi", "cairo", "pango", "gdk-pixbuf2"
                ])
            elif shutil.which("pacman"):
                logger.info("Detected Arch Linux. Installing dependencies...")
                subprocess.check_call([
                    "sudo", "pacman", "-S", "--noconfirm",
                    "python-pip", "python-setuptools", "python-wheel", "python-cffi", 
                    "cairo", "pango", "gdk-pixbuf2"
                ])
            else:
                logger.warning("Could not detect package manager. Please install WeasyPrint dependencies manually.")
                
            # Reinstall WeasyPrint
            logger.info("Reinstalling WeasyPrint...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "weasyprint"
            ])
            
            logger.info("WeasyPrint setup completed for Linux.")
            
        except Exception as e:
            logger.error(f"Failed to set up WeasyPrint on Linux: {e}")
            self.failed_packages.append("weasyprint-setup")
    
    def _setup_weasyprint_macos(self):
        """Set up WeasyPrint on macOS."""
        logger.info("Setting up WeasyPrint on macOS...")
        
        try:
            if shutil.which("brew"):
                logger.info("Homebrew detected. Installing dependencies...")
                subprocess.check_call([
                    "brew", "install", "cairo", "pango", "gdk-pixbuf", "libffi"
                ])
                
                # Reinstall WeasyPrint
                logger.info("Reinstalling WeasyPrint...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "weasyprint"
                ])
                
                logger.info("WeasyPrint setup completed for macOS.")
            else:
                logger.warning("Homebrew is not installed. Please install it first: https://brew.sh/")
                self.failed_packages.append("homebrew-missing")
        except Exception as e:
            logger.error(f"Failed to set up WeasyPrint on macOS: {e}")
            self.failed_packages.append("weasyprint-setup")
    
    def _verify_installation(self):
        """Verify that all packages were installed correctly."""
        logger.info("\nVerifying installation...")
        
        for category, packages in DEPENDENCIES.items():
            logger.info(f"Checking {category} dependencies:")
            for package_spec in packages:
                package_name = package_spec.split('>=')[0] if '>=' in package_spec else package_spec
                package_name = package_name.replace('-', '_')  # Convert hyphens to underscores for importing
                
                try:
                    spec = __import__(package_name)
                    logger.info(f"✅ {package_name} is properly installed")
                except ImportError as e:
                    logger.warning(f"❌ {package_name} is not properly installed: {e}")
                    if package_name not in [p.split('>=')[0] for p in self.failed_packages]:
                        self.failed_packages.append(package_spec)
        
        # Special check for WeasyPrint
        try:
            import weasyprint
            try:
                # Try to create a simple PDF
                test_html = "<html><body><h1>Test</h1></body></html>"
                test_pdf = Path(os.path.expanduser("~")) / "weasyprint_test.pdf"
                weasyprint.HTML(string=test_html).write_pdf(str(test_pdf))
                
                # Check if the file was created
                if test_pdf.exists():
                    test_pdf.unlink()  # Delete the test file
                    logger.info(f"✅ WeasyPrint is fully functional")
                else:
                    logger.warning("❌ WeasyPrint could not generate a PDF")
                    if "weasyprint" not in [p.split('>=')[0] for p in self.failed_packages]:
                        self.failed_packages.append("weasyprint")
            except Exception as e:
                logger.warning(f"❌ WeasyPrint is installed but not working properly: {e}")
                if "weasyprint" not in [p.split('>=')[0] for p in self.failed_packages]:
                    self.failed_packages.append("weasyprint")
        except ImportError:
            logger.warning("❌ WeasyPrint is not installed")
            if "weasyprint" not in [p.split('>=')[0] for p in self.failed_packages]:
                self.failed_packages.append("weasyprint")


def main():
    """Main function."""
    logger.info("QuangTPS Dependency Installer")
    logger.info("============================")
    logger.info("This script will install all dependencies required for QuangTPS.")
    logger.info("This may take some time, depending on your internet connection and computer speed.")
    logger.info("Please be patient and do not close this window.")
    logger.info("============================\n")
    
    time.sleep(2)  # Give the user time to read the message
    
    installer = DependencyInstaller()
    success = installer.install_all()
    
    if success:
        logger.info("\n============================")
        logger.info("Installation completed successfully!")
        logger.info("You can now run QuangTPS.")
        logger.info("============================")
        return 0
    else:
        logger.warning("\n============================")
        logger.warning("Installation completed with some issues.")
        logger.warning("Please check the log messages above for details.")
        logger.warning("You may need to manually install some dependencies.")
        logger.warning("============================")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 