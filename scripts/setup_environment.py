#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script thiết lập môi trường cho QuangTPS.

Script này thiết lập và kiểm tra môi trường cần thiết để chạy QuangTPS,
bao gồm thư viện, thư mục, cơ sở dữ liệu, và cấu hình khác.
"""

import os
import sys
import platform
import locale
import subprocess
import importlib
import importlib.metadata as metadata
from pathlib import Path

def setup_utf8_console():
    """Thiết lập UTF-8 cho console."""
    if sys.platform == 'win32':
        try:
            # Thử đặt locale cho tiếng Việt
            locale.setlocale(locale.LC_ALL, 'Vietnamese_Vietnam.65001')
        except locale.Error:
            try:
                # Nếu không có locale tiếng Việt, sử dụng UTF-8 chung
                locale.setlocale(locale.LC_ALL, '.65001')
            except locale.Error:
                try:
                    # Thử set mặc định
                    locale.setlocale(locale.LC_ALL, '')
                except:
                    # Nếu không thể, bỏ qua
                    pass
        
        # Đặt các biến môi trường cho Python
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
        
        try:
            # Thiết lập utf-8 cho stdout và stderr
            import codecs
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except Exception as e:
            print(f"Cảnh báo: Không thể thiết lập UTF-8 cho console: {str(e)}")

def create_directories():
    """Tạo các thư mục cần thiết."""
    # Lấy thư mục gốc
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Các thư mục cần tạo
    directories = [
        'data',
        'data/beam_data',
        'data/dicom',
        'data/database',
        'data/images',
        'data/structures',
        'data/clinical_protocols',
        'data/machine_data',
        'data/models',
        'data/templates',
        'data/patient_data',
        'logs',
        'temp'
    ]
    
    # Tạo các thư mục
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Thư mục đã tạo: {dir_path}")
    
    # Thiết lập biến môi trường QUANGTPS_ROOT
    os.environ['QUANGTPS_ROOT'] = root_dir
    
    return root_dir

def check_dependencies():
    """Kiểm tra các thư viện phụ thuộc."""
    # Dictionary ánh xạ tên gói với tên module và phiên bản tối thiểu
    dependencies = {
        "numpy": ("numpy", "1.20.0"),
        "scipy": ("scipy", "1.7.0"),
        "pandas": ("pandas", "1.3.0"),
        "pydicom": ("pydicom", "2.2.0"),
        "PyQt5": ("PyQt5", "5.15.0"),
        "matplotlib": ("matplotlib", "3.4.0"),
        "scikit-image": ("skimage", "0.18.0"),
        "dicompyler-core": ("dicompylercore", "0.5.5")
    }
    
    missing_packages = []
    outdated_packages = []
    installed_packages = {}
    
    print("\nKiểm tra các thư viện phụ thuộc:")
    print("-" * 60)
    print(f"{'Thư viện':<20} {'Phiên bản yêu cầu':<20} {'Phiên bản hiện tại':<20} {'Trạng thái':<10}")
    print("-" * 60)
    
    for package, (module, min_version) in dependencies.items():
        try:
            # Thử import module
            imported_module = importlib.import_module(module)
            
            # Lấy phiên bản
            try:
                if hasattr(imported_module, '__version__'):
                    version = imported_module.__version__
                else:
                    try:
                        version = metadata.version(package)
                    except:
                        version = "Không xác định"
            except:
                version = "Không xác định"
            
            installed_packages[package] = version
            
            # Kiểm tra phiên bản
            if version != "Không xác định":
                try:
                    # Chuyển đổi các chuỗi phiên bản thành tuple để so sánh
                    current_version_parts = [int(x) for x in version.split('.')]
                    min_version_parts = [int(x) for x in min_version.split('.')]
                    
                    # So sánh từng phần của phiên bản
                    is_outdated = False
                    for i in range(min(len(current_version_parts), len(min_version_parts))):
                        if current_version_parts[i] < min_version_parts[i]:
                            is_outdated = True
                            break
                        elif current_version_parts[i] > min_version_parts[i]:
                            break
                    
                    if is_outdated:
                        status = "Cũ"
                        outdated_packages.append((package, version, min_version))
                    else:
                        status = "OK"
                except:
                    status = "Không kiểm tra được"
            else:
                status = "Không xác định"
            
            print(f"{package:<20} {min_version:<20} {version:<20} {status:<10}")
            
        except ImportError:
            missing_packages.append((package, min_version))
            print(f"{package:<20} {min_version:<20} {'Chưa cài đặt':<20} {'Thiếu':<10}")
    
    print("-" * 60)
    
    if missing_packages:
        print("\nCác thư viện còn thiếu:")
        for package, min_version in missing_packages:
            print(f"  - {package} (>= {min_version})")
        
        pip_command = "pip install " + " ".join([f"{pkg}>={ver}" for pkg, ver in missing_packages])
        print(f"\nCài đặt bằng lệnh: {pip_command}")
    
    if outdated_packages:
        print("\nCác thư viện cần cập nhật:")
        for package, current_version, required_version in outdated_packages:
            print(f"  - {package}: {current_version} -> {required_version}")
        
        pip_command = "pip install --upgrade " + " ".join([f"{pkg}>={ver}" for pkg, ver, _ in outdated_packages])
        print(f"\nCập nhật bằng lệnh: {pip_command}")
    
    if not missing_packages and not outdated_packages:
        print("\nTất cả thư viện phụ thuộc đã được cài đặt đúng phiên bản.")
    
    return installed_packages, missing_packages, outdated_packages

def collect_system_info():
    """Thu thập thông tin hệ thống."""
    system_info = {}
    
    # Thông tin hệ điều hành
    system_info['OS'] = platform.system()
    system_info['OS Version'] = platform.version()
    system_info['OS Release'] = platform.release()
    system_info['Machine'] = platform.machine()
    system_info['Processor'] = platform.processor()
    
    # Thông tin Python
    system_info['Python Version'] = platform.python_version()
    system_info['Python Implementation'] = platform.python_implementation()
    system_info['Python Compiler'] = platform.python_compiler()
    
    # Thông tin locale
    system_info['Default Encoding'] = sys.getdefaultencoding()
    system_info['Filesystem Encoding'] = sys.getfilesystemencoding()
    try:
        system_info['Locale'] = '.'.join(locale.getlocale())
    except:
        system_info['Locale'] = 'Không xác định'
    
    return system_info

def print_system_info(system_info):
    """In thông tin hệ thống."""
    print("\nThông tin hệ thống:")
    print("-" * 60)
    for key, value in system_info.items():
        print(f"{key:<25}: {value}")
    print("-" * 60)

def initialize_database():
    """Khởi tạo cơ sở dữ liệu."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    try:
        # Thêm đường dẫn gốc vào sys.path
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        # Import module cơ sở dữ liệu
        from quangtps.database.db_connector import initialize_database
        
        # Khởi tạo cơ sở dữ liệu
        db_path = os.path.join(root_dir, "data", "database", "quangtps.db")
        initialize_database(db_path)
        print(f"\nĐã khởi tạo cơ sở dữ liệu tại: {db_path}")
        return True
    except ImportError:
        print("\nKhông thể import module cơ sở dữ liệu.")
        return False
    except Exception as e:
        print(f"\nLỗi khi khởi tạo cơ sở dữ liệu: {str(e)}")
        return False

def check_quangtps_imports():
    """Kiểm tra các module QuangTPS."""
    quangtps_modules = [
        "quangtps.core",
        "quangtps.database",
        "quangtps.dicom",
        "quangtps.dose",
        "quangtps.evaluation",
        "quangtps.imaging",
        "quangtps.planning",
        "quangtps.reporting",
        "quangtps.segmentation",
        "quangtps.treatment",
        "quangtps.ui",
        "quangtps.adaptive"
    ]
    
    print("\nKiểm tra các module QuangTPS:")
    print("-" * 60)
    
    import_errors = []
    
    for module in quangtps_modules:
        try:
            importlib.import_module(module)
            print(f"{module:<30}: OK")
        except ImportError as e:
            print(f"{module:<30}: Lỗi ({str(e)})")
            import_errors.append((module, str(e)))
    
    print("-" * 60)
    
    if import_errors:
        print("\nCác module còn lỗi:")
        for module, error in import_errors:
            print(f"  - {module}: {error}")
    else:
        print("\nTất cả module QuangTPS đã được import thành công.")
    
    return import_errors

def check_gpu():
    """Kiểm tra GPU và hỗ trợ CUDA."""
    print("\nKiểm tra GPU và CUDA:")
    print("-" * 60)
    
    # Kiểm tra TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Đã tìm thấy {len(gpus)} GPU:")
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i+1}: {gpu}")
            print(f"TensorFlow version: {tf.__version__}")
            print(f"CUDA available: {tf.test.is_built_with_cuda()}")
        else:
            print("Không tìm thấy GPU hỗ trợ TensorFlow.")
    except ImportError:
        print("TensorFlow chưa được cài đặt.")
    except Exception as e:
        print(f"Lỗi khi kiểm tra GPU TensorFlow: {str(e)}")
    
    # Kiểm tra PyTorch
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Số lượng GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i+1}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("\nPyTorch chưa được cài đặt.")
    except Exception as e:
        print(f"\nLỗi khi kiểm tra GPU PyTorch: {str(e)}")
    
    print("-" * 60)

def generate_diagnostic_report():
    """Tạo báo cáo chẩn đoán."""
    # Thiết lập UTF-8
    setup_utf8_console()
    
    print("=" * 80)
    print("BÁO CÁO CHẨN ĐOÁN MÔI TRƯỜNG QUANGTPS")
    print("=" * 80)
    
    # Thu thập và in thông tin hệ thống
    system_info = collect_system_info()
    print_system_info(system_info)
    
    # Tạo các thư mục
    root_dir = create_directories()
    
    # Kiểm tra các thư viện phụ thuộc
    check_dependencies()
    
    # Kiểm tra các module QuangTPS
    check_quangtps_imports()
    
    # Kiểm tra GPU
    check_gpu()
    
    # Khởi tạo cơ sở dữ liệu
    initialize_database()
    
    # Lưu báo cáo vào file
    report_path = os.path.join(root_dir, "logs", "diagnostic_report.txt")
    try:
        # Chuyển hướng đầu ra sang file
        import sys
        from contextlib import redirect_stdout
        
        with open(report_path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print("=" * 80)
                print("BÁO CÁO CHẨN ĐOÁN MÔI TRƯỜNG QUANGTPS")
                print("=" * 80)
                
                print_system_info(system_info)
                check_dependencies()
                check_quangtps_imports()
                check_gpu()
        
        print(f"\nĐã lưu báo cáo chẩn đoán vào: {report_path}")
    except Exception as e:
        print(f"\nLỗi khi lưu báo cáo: {str(e)}")
    
    print("\nQuá trình thiết lập môi trường đã hoàn tất.")

if __name__ == "__main__":
    generate_diagnostic_report()
 