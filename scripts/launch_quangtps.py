#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Launcher

This script provides a simplified way to launch the QuangTPS treatment planning system
with appropriate settings for 3D visualization.
"""

import os
import sys
import argparse
import logging
import importlib.util
import subprocess
from pathlib import Path

def setup_logging():
    """Set up logging for the launcher."""
    log_dir = Path.home() / "quangtps_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "launcher.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_core = ["numpy", "pydicom", "PyQt5", "matplotlib"]
    required_3d = ["pyvista", "pyvistaqt", "vtk"]
    
    missing = []
    
    # Check core dependencies
    for package in required_core:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    
    # Check 3D visualization dependencies
    missing_3d = []
    for package in required_3d:
        if importlib.util.find_spec(package) is None:
            missing_3d.append(package)
    
    return missing, missing_3d

def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return True
    
    print(f"Installing missing dependencies: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install them manually.")
        return False

def find_quangtps_module():
    """Find the QuangTPS module directory."""
    # First check if we're running from the repository
    current_dir = Path(__file__).parent
    repo_root = current_dir.parent
    
    if (repo_root / "quangtps").exists():
        return repo_root
    
    # Check if installed as a package
    spec = importlib.util.find_spec("quangtps")
    if spec is not None:
        return Path(spec.origin).parent.parent
    
    return None

def launch_quangtps(with_3d=True, debug=False):
    """Launch QuangTPS."""
    quangtps_dir = find_quangtps_module()
    
    if quangtps_dir is None:
        print("QuangTPS module not found. Please install it or run from the repository.")
        return False
    
    # Add QuangTPS directory to Python path
    sys.path.insert(0, str(quangtps_dir))
    
    # Import the main module
    try:
        from quangtps.ui.main_window import main as quangtps_main
        
        # Set environment variable for 3D visualization
        if with_3d:
            os.environ["QUANGTPS_ENABLE_3D"] = "1"
        else:
            os.environ["QUANGTPS_ENABLE_3D"] = "0"
            
        # Set debug mode
        if debug:
            os.environ["QUANGTPS_DEBUG"] = "1"
            
        # Launch QuangTPS
        quangtps_main()
        return True
    except ImportError as e:
        print(f"Error importing QuangTPS: {e}")
        return False

def main():
    """Main function for the launcher."""
    parser = argparse.ArgumentParser(description="Launch QuangTPS Treatment Planning System")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D visualization")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--skip-dependency-check", action="store_true", help="Skip dependency check")
    args = parser.parse_args()
    
    setup_logging()
    logging.info("Starting QuangTPS launcher")
    
    # Check dependencies
    if not args.skip_dependency_check:
        missing, missing_3d = check_dependencies()
        
        if missing:
            print(f"The following core dependencies are missing: {', '.join(missing)}")
            if not install_dependencies(missing):
                return 1
        
        if not args.no_3d and missing_3d:
            print(f"3D visualization dependencies are missing: {', '.join(missing_3d)}")
            print("Would you like to install them? (y/n)")
            
            response = input().strip().lower()
            if response in ("y", "yes"):
                if not install_dependencies(missing_3d):
                    print("Continuing without 3D visualization...")
                    args.no_3d = True
            else:
                print("Continuing without 3D visualization...")
                args.no_3d = True
    
    # Launch QuangTPS
    with_3d = not args.no_3d
    if launch_quangtps(with_3d=with_3d, debug=args.debug):
        logging.info("QuangTPS started successfully")
        return 0
    else:
        logging.error("Failed to start QuangTPS")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 