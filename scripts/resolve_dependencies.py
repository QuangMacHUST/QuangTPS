#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Dependency Resolution Script

This script checks for all required dependencies and provides guidance 
on resolving issues with problematic packages like WeasyPrint.

Usage:
    python resolve_dependencies.py [--fix] [--verbose]
    
Options:
    --fix       Attempt to fix dependency issues automatically
    --verbose   Show detailed information for each package
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
from pathlib import Path
import importlib.util
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("dependency-resolver")

# Dependency groups
CORE_DEPENDENCIES = [
    "numpy", "scipy", "matplotlib", "pandas", 
    "scikit-image", "scikit-learn", "opencv-python", 
    "pydicom", "SimpleITK", "h5py"
]

GUI_DEPENDENCIES = [
    "PyQt5", "PyQtChart", "pyqtgraph"
]

VISUALIZATION_DEPENDENCIES = [
    "pyvista", "pyvistaQt", "vtk"
]

REPORTING_DEPENDENCIES = [
    "weasyprint", "reportlab", "jinja2", "pypdf2"
]

ML_DEPENDENCIES = [
    "tensorflow", "torch", "torchvision", "monai"
]

OPTIMIZATION_DEPENDENCIES = [
    "pulp", "cvxpy", "pyomo"
]

ALL_DEPENDENCIES = {
    "Core": CORE_DEPENDENCIES,
    "GUI": GUI_DEPENDENCIES,
    "Visualization": VISUALIZATION_DEPENDENCIES,
    "Reporting": REPORTING_DEPENDENCIES,
    "Machine Learning": ML_DEPENDENCIES,
    "Optimization": OPTIMIZATION_DEPENDENCIES
}

class DependencyChecker:
    """Class to check and fix dependencies for QuangTPS."""
    
    def __init__(self, fix_issues=False, verbose=False):
        """Initialize the dependency checker.
        
        Args:
            fix_issues (bool): Whether to attempt to fix issues automatically
            verbose (bool): Whether to show detailed information for each package
        """
        self.fix_issues = fix_issues
        self.verbose = verbose
        self.os_name = platform.system()
        self.os_version = platform.version()
        self.python_version = platform.python_version()
        self.missing_packages = []
        self.problematic_packages = []
        
    def run_checks(self):
        """Run all dependency checks."""
        logger.info("Starting dependency checks for QuangTPS")
        logger.info(f"Operating System: {self.os_name} {self.os_version}")
        logger.info(f"Python Version: {self.python_version}")
        
        # Check if pip is available
        if not self._check_pip():
            logger.error("pip is not available. Cannot continue.")
            return False
        
        # Check dependencies by group
        for group_name, dependencies in ALL_DEPENDENCIES.items():
            logger.info(f"\nChecking {group_name} dependencies:")
            for package in dependencies:
                self._check_package(package, group_name)
        
        # Summary
        if not self.missing_packages and not self.problematic_packages:
            logger.info("\n✅ All dependencies are properly installed!")
            return True
        
        # Report issues
        if self.missing_packages:
            logger.warning("\nMissing packages:")
            for package in self.missing_packages:
                logger.warning(f"  - {package}")
                
        if self.problematic_packages:
            logger.warning("\nProblematic packages:")
            for package, issue in self.problematic_packages:
                logger.warning(f"  - {package}: {issue}")
        
        # Fix issues if requested
        if self.fix_issues:
            return self._fix_issues()
        
        # Provide manual fix instructions
        self._show_fix_instructions()
        return False
    
    def _check_pip(self):
        """Check if pip is available."""
        try:
            subprocess.check_output([sys.executable, "-m", "pip", "--version"])
            return True
        except Exception:
            logger.error("pip is not installed or not in PATH")
            return False
    
    def _check_package(self, package_name, group):
        """Check if a package is installed and working properly.
        
        Args:
            package_name (str): Name of the package to check
            group (str): Group the package belongs to
        """
        # Special case for weasyprint which has extra dependencies
        if package_name == "weasyprint":
            return self._check_weasyprint()
            
        # Normal package check
        try:
            # Check if package is importable
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                logger.warning(f"❌ {package_name} is not installed")
                self.missing_packages.append(package_name)
                return False
            
            # Try to import the package
            if self.verbose:
                module = importlib.import_module(package_name)
                version = getattr(module, "__version__", "unknown version")
                logger.info(f"✅ {package_name} ({version}) is installed")
            else:
                logger.info(f"✅ {package_name} is installed")
                
            return True
            
        except ImportError as e:
            logger.warning(f"❌ {package_name} is not properly installed: {str(e)}")
            self.problematic_packages.append((package_name, str(e)))
            return False
        except Exception as e:
            logger.warning(f"⚠️ {package_name} has issues: {str(e)}")
            self.problematic_packages.append((package_name, str(e)))
            return False
    
    def _check_weasyprint(self):
        """Special check for WeasyPrint which has system dependencies."""
        try:
            spec = importlib.util.find_spec("weasyprint")
            if spec is None:
                logger.warning("❌ WeasyPrint is not installed")
                self.missing_packages.append("weasyprint")
                return False
            
            # Try to import weasyprint
            import weasyprint
            version = getattr(weasyprint, "__version__", "unknown version")
            
            # Check for external dependencies
            try:
                # Try to create a simple PDF
                test_html = "<html><body><h1>Test</h1></body></html>"
                test_pdf = Path(os.path.expanduser("~")) / "weasyprint_test.pdf"
                weasyprint.HTML(string=test_html).write_pdf(str(test_pdf))
                
                # Check if the file was created
                if test_pdf.exists():
                    test_pdf.unlink()  # Delete the test file
                    logger.info(f"✅ WeasyPrint ({version}) is fully functional")
                    return True
            except Exception as e:
                # Missing system dependencies
                issue = str(e)
                logger.warning(f"⚠️ WeasyPrint is installed but missing system dependencies: {issue}")
                self.problematic_packages.append(("weasyprint", issue))
                return False
                
        except ImportError as e:
            logger.warning(f"❌ WeasyPrint is not properly installed: {str(e)}")
            self.problematic_packages.append(("weasyprint", str(e)))
            return False
        except Exception as e:
            logger.warning(f"⚠️ WeasyPrint has issues: {str(e)}")
            self.problematic_packages.append(("weasyprint", str(e)))
            return False
    
    def _fix_issues(self):
        """Attempt to fix identified issues automatically."""
        logger.info("\nAttempting to fix dependency issues...")
        
        # Install missing packages
        if self.missing_packages:
            logger.info("Installing missing packages...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade"
                ] + self.missing_packages)
                logger.info("Package installation completed.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install packages: {e}")
                return False
        
        # Fix WeasyPrint issues
        if any(p[0] == "weasyprint" for p in self.problematic_packages):
            logger.info("Fixing WeasyPrint installation...")
            if self.os_name == "Windows":
                self._fix_weasyprint_windows()
            elif self.os_name == "Linux":
                self._fix_weasyprint_linux()
            elif self.os_name == "Darwin":  # macOS
                self._fix_weasyprint_macos()
        
        # Recheck after fixes
        logger.info("\nRechecking dependencies after fixes...")
        self.missing_packages = []
        self.problematic_packages = []
        
        for group_name, dependencies in ALL_DEPENDENCIES.items():
            for package in dependencies:
                self._check_package(package, group_name)
        
        if not self.missing_packages and not self.problematic_packages:
            logger.info("\n✅ All dependency issues have been resolved!")
            return True
        else:
            logger.warning("\n⚠️ Some issues could not be fixed automatically.")
            self._show_fix_instructions()
            return False
    
    def _fix_weasyprint_windows(self):
        """Fix WeasyPrint installation on Windows."""
        logger.info("WeasyPrint requires GTK3 on Windows.")
        
        # Try to install with the appropriate options
        try:
            logger.info("Reinstalling WeasyPrint...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y", "weasyprint"
            ])
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "weasyprint"
            ])
            
            logger.info("Opening the GTK installation guide in your web browser...")
            import webbrowser
            webbrowser.open("https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows")
            
            logger.info("Please follow the instructions to install GTK3 on Windows.")
        except Exception as e:
            logger.error(f"Failed to fix WeasyPrint: {e}")
    
    def _fix_weasyprint_linux(self):
        """Fix WeasyPrint installation on Linux."""
        logger.info("WeasyPrint requires system libraries on Linux.")
        
        # Try to detect the distribution
        try:
            # Check if apt is available (Debian, Ubuntu)
            if shutil.which("apt"):
                logger.info("Installing WeasyPrint dependencies with apt...")
                subprocess.check_call([
                    "sudo", "apt", "install", "-y", 
                    "build-essential", "python3-dev", "python3-pip", "python3-setuptools",
                    "python3-wheel", "python3-cffi", "libcairo2", "libpango-1.0-0", 
                    "libpangocairo-1.0-0", "libgdk-pixbuf2.0-0", "libffi-dev", "shared-mime-info"
                ])
            # Check if dnf is available (Fedora, RHEL)
            elif shutil.which("dnf"):
                logger.info("Installing WeasyPrint dependencies with dnf...")
                subprocess.check_call([
                    "sudo", "dnf", "install", "-y",
                    "redhat-rpm-config", "python3-devel", "python3-pip", "python3-setuptools",
                    "python3-wheel", "python3-cffi", "cairo", "pango", "gdk-pixbuf2"
                ])
            # Check if pacman is available (Arch Linux)
            elif shutil.which("pacman"):
                logger.info("Installing WeasyPrint dependencies with pacman...")
                subprocess.check_call([
                    "sudo", "pacman", "-S", "--noconfirm",
                    "python-pip", "python-setuptools", "python-wheel", "python-cffi", 
                    "cairo", "pango", "gdk-pixbuf2"
                ])
            else:
                logger.warning("Could not detect package manager. Please install WeasyPrint dependencies manually.")
                return
                
            # Reinstall WeasyPrint
            logger.info("Reinstalling WeasyPrint...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "weasyprint"
            ])
        except Exception as e:
            logger.error(f"Failed to fix WeasyPrint: {e}")
    
    def _fix_weasyprint_macos(self):
        """Fix WeasyPrint installation on macOS."""
        logger.info("WeasyPrint requires system libraries on macOS.")
        
        # Try to detect if Homebrew is installed
        try:
            if shutil.which("brew"):
                logger.info("Installing WeasyPrint dependencies with Homebrew...")
                subprocess.check_call([
                    "brew", "install", "cairo", "pango", "gdk-pixbuf", "libffi"
                ])
                
                # Reinstall WeasyPrint
                logger.info("Reinstalling WeasyPrint...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "weasyprint"
                ])
            else:
                logger.warning("Homebrew is not installed. Please install it first: https://brew.sh/")
        except Exception as e:
            logger.error(f"Failed to fix WeasyPrint: {e}")
    
    def _show_fix_instructions(self):
        """Show instructions for manually fixing dependency issues."""
        logger.info("\n============= MANUAL FIX INSTRUCTIONS =============")
        
        # Instructions for missing packages
        if self.missing_packages:
            logger.info("\n1. Install missing packages:")
            cmd = f"{sys.executable} -m pip install {' '.join(self.missing_packages)}"
            logger.info(f"   {cmd}")
        
        # Special instructions for WeasyPrint
        if any(p[0] == "weasyprint" for p in self.problematic_packages):
            logger.info("\n2. Fix WeasyPrint installation:")
            logger.info("   WeasyPrint requires system dependencies to function properly.")
            
            if self.os_name == "Windows":
                logger.info("   For Windows:")
                logger.info("   1. Install GTK3 runtime from: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases")
                logger.info("   2. Add the GTK bin directory to your PATH environment variable (e.g., C:\\Program Files\\GTK3-Runtime Win64\\bin)")
                logger.info("   3. Restart your terminal and try again")
                
            elif self.os_name == "Linux":
                logger.info("   For Linux (Debian/Ubuntu):")
                logger.info("   sudo apt install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info")
                
            elif self.os_name == "Darwin":  # macOS
                logger.info("   For macOS:")
                logger.info("   1. Install Homebrew from https://brew.sh/ if not already installed")
                logger.info("   2. Run: brew install cairo pango gdk-pixbuf libffi")
                logger.info("   3. Reinstall WeasyPrint: pip install --force-reinstall weasyprint")
        
        logger.info("\nFor more detailed instructions, visit:")
        logger.info("https://github.com/username/quangtps/wiki/Dependency-Troubleshooting")
        logger.info("\n===================================================")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="QuangTPS Dependency Resolution Tool")
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Attempt to fix dependency issues automatically"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed information for each package"
    )
    
    args = parser.parse_args()
    checker = DependencyChecker(fix_issues=args.fix, verbose=args.verbose)
    
    success = checker.run_checks()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 