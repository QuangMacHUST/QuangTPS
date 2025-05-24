#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module kiểm tra sức khỏe hệ thống QuangTPS.

Module này cung cấp các công cụ để kiểm tra tình trạng hoạt động
của tất cả các thành phần trong hệ thống QuangTPS, phát hiện lỗi
và đề xuất giải pháp khắc phục.
"""

import logging
import sys
import os
import importlib
import traceback
import time
import gc
import psutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Kết quả kiểm tra sức khỏe một component."""

    component_name: str
    status: str  # 'healthy', 'warning', 'error', 'critical'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    exception: Optional[Exception] = None


@dataclass
class SystemHealthReport:
    """Báo cáo sức khỏe tổng thể hệ thống."""

    timestamp: datetime = field(default_factory=datetime.now)
    overall_status: str = "healthy"
    total_checks: int = 0
    passed_checks: int = 0
    warning_checks: int = 0
    failed_checks: int = 0
    critical_checks: int = 0

    core_modules: List[HealthCheckResult] = field(default_factory=list)
    ui_modules: List[HealthCheckResult] = field(default_factory=list)
    dose_modules: List[HealthCheckResult] = field(default_factory=list)
    evaluation_modules: List[HealthCheckResult] = field(default_factory=list)
    optimization_modules: List[HealthCheckResult] = field(default_factory=list)

    system_resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)

    recommendations: List[str] = field(default_factory=list)


class SystemHealthChecker:
    """Lớp kiểm tra sức khỏe hệ thống toàn diện."""

    def __init__(self):
        self.results = []
        self.start_time = None

    def run_comprehensive_check(
        self,
        check_dependencies: bool = True,
        check_performance: bool = True,
        check_modules: bool = True,
    ) -> SystemHealthReport:
        """
        Chạy kiểm tra sức khỏe toàn diện.

        Args:
            check_dependencies: Kiểm tra dependencies
            check_performance: Kiểm tra hiệu suất
            check_modules: Kiểm tra modules

        Returns:
            SystemHealthReport: Báo cáo sức khỏe
        """
        self.start_time = time.time()
        self.results = []

        logger.info("Bắt đầu kiểm tra sức khỏe hệ thống...")

        # Kiểm tra system resources
        system_resources = self._check_system_resources()

        # Kiểm tra dependencies
        dependencies = {}
        if check_dependencies:
            dependencies = self._check_dependencies()

        # Kiểm tra modules
        if check_modules:
            self._check_core_modules()
            self._check_ui_modules()
            self._check_dose_modules()
            self._check_evaluation_modules()
            self._check_optimization_modules()

        # Kiểm tra hiệu suất
        if check_performance:
            self._check_performance()

        # Tạo báo cáo
        report = self._generate_report(system_resources, dependencies)

        logger.info(
            f"Hoàn thành kiểm tra sức khỏe trong {time.time() - self.start_time:.2f}s"
        )

        return report

    def _check_component(self, component_name: str, check_func) -> HealthCheckResult:
        """Kiểm tra một component cụ thể."""
        start_time = time.time()

        try:
            status, message, details = check_func()
            execution_time = time.time() - start_time

            result = HealthCheckResult(
                component_name=component_name,
                status=status,
                message=message,
                details=details or {},
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            result = HealthCheckResult(
                component_name=component_name,
                status="error",
                message=f"Lỗi kiểm tra: {str(e)}",
                details={"exception": str(e), "traceback": traceback.format_exc()},
                execution_time=execution_time,
                exception=e,
            )

        self.results.append(result)
        return result

    def _check_system_resources(self) -> Dict[str, Any]:
        """Kiểm tra tài nguyên hệ thống."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_count = psutil.cpu_count()

            return {
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_percent": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 1),
                "cpu_count": cpu_count,
                "python_version": sys.version,
            }
        except Exception as e:
            logger.error(f"Lỗi kiểm tra system resources: {e}")
            return {}

    def _check_dependencies(self) -> Dict[str, str]:
        """Kiểm tra các dependencies chính."""
        dependencies = {
            "numpy": None,
            "scipy": None,
            "matplotlib": None,
            "PyQt5": None,
            "pydicom": None,
            "SimpleITK": None,
            "vtk": None,
            "numba": None,
            "tensorflow": None,
            "torch": None,
            "pandas": None,
            "scikit-learn": None,
        }

        for dep_name in dependencies.keys():
            try:
                module = importlib.import_module(dep_name)
                version = getattr(module, "__version__", "unknown")
                dependencies[dep_name] = version

                self._check_component(
                    f"dependency_{dep_name}",
                    lambda: ("healthy", f"{dep_name} v{version} available", None),
                )

            except ImportError:
                dependencies[dep_name] = "not_available"
                self._check_component(
                    f"dependency_{dep_name}",
                    lambda: ("warning", f"{dep_name} not available", None),
                )

        return dependencies

    def _check_core_modules(self):
        """Kiểm tra core modules."""
        core_modules = [
            "quangtps.core.exceptions",
            "quangtps.core.config",
            "quangtps.core.logging",
            "quangtps.core.patient",
        ]

        for module_name in core_modules:
            self._check_component(
                f"core_{module_name.split('.')[-1]}",
                lambda mn=module_name: self._try_import_module(mn),
            )

    def _check_ui_modules(self):
        """Kiểm tra UI modules."""
        ui_modules = [
            "quangtps.ui.main_window",
            "quangtps.ui.external_beam_planning_tab",
            "quangtps.ui.structure_tab",
            "quangtps.ui.evaluation.plan_qa_widget",
            "quangtps.ui.dose_3d_viewer",
        ]

        for module_name in ui_modules:
            self._check_component(
                f"ui_{module_name.split('.')[-1]}",
                lambda mn=module_name: self._try_import_module(mn),
            )

    def _check_dose_modules(self):
        """Kiểm tra dose calculation modules."""
        dose_modules = [
            "quangtps.dose.dose_engine",
            "quangtps.dose.dose_grid",
            "quangtps.dose.algorithms.collapsed_cone",
            "quangtps.dose.algorithms.pencil_beam",
            "quangtps.dose.algorithms.monte_carlo",
        ]

        for module_name in dose_modules:
            self._check_component(
                f"dose_{module_name.split('.')[-1]}",
                lambda mn=module_name: self._try_import_module(mn),
            )

    def _check_evaluation_modules(self):
        """Kiểm tra evaluation modules."""
        eval_modules = [
            "quangtps.evaluation.metrics.gamma_analysis",
            "quangtps.evaluation.metrics.dose_metrics",
            "quangtps.evaluation.qa.comprehensive_qa_engine",
            "quangtps.evaluation.qa.statistical_analysis",
        ]

        for module_name in eval_modules:
            self._check_component(
                f"eval_{module_name.split('.')[-1]}",
                lambda mn=module_name: self._try_import_module(mn),
            )

    def _check_optimization_modules(self):
        """Kiểm tra optimization modules."""
        opt_modules = [
            "quangtps.optimization.optimizer",
            "quangtps.optimization.objectives",
            "quangtps.optimization.mco.mco_optimizer",
        ]

        for module_name in opt_modules:
            self._check_component(
                f"opt_{module_name.split('.')[-1]}",
                lambda mn=module_name: self._try_import_module(mn),
            )

    def _check_performance(self):
        """Kiểm tra hiệu suất hệ thống."""
        # Test numpy performance
        self._check_component("performance_numpy", self._test_numpy_performance)

        # Test memory usage
        self._check_component("performance_memory", self._test_memory_usage)

    def _try_import_module(self, module_name: str) -> Tuple[str, str, Dict]:
        """Thử import một module."""
        try:
            module = importlib.import_module(module_name)
            return (
                "healthy",
                f"Module {module_name} imported successfully",
                {
                    "module_file": getattr(module, "__file__", "unknown"),
                    "module_version": getattr(module, "__version__", "unknown"),
                },
            )
        except ImportError as e:
            return (
                "error",
                f"Failed to import {module_name}: {str(e)}",
                {"error_type": "ImportError", "error_message": str(e)},
            )
        except Exception as e:
            return (
                "error",
                f"Error importing {module_name}: {str(e)}",
                {"error_type": type(e).__name__, "error_message": str(e)},
            )

    def _test_numpy_performance(self) -> Tuple[str, str, Dict]:
        """Test numpy performance."""
        try:
            import numpy as np

            # Test matrix multiplication
            size = 1000
            start_time = time.time()
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            c = np.dot(a, b)
            execution_time = time.time() - start_time

            # Performance thresholds
            if execution_time < 1.0:
                status = "healthy"
                message = f"Numpy performance excellent: {execution_time:.3f}s"
            elif execution_time < 5.0:
                status = "warning"
                message = f"Numpy performance acceptable: {execution_time:.3f}s"
            else:
                status = "error"
                message = f"Numpy performance poor: {execution_time:.3f}s"

            return (
                status,
                message,
                {
                    "matrix_size": size,
                    "execution_time": execution_time,
                    "operations_per_second": (size * size * size) / execution_time,
                },
            )

        except Exception as e:
            return "error", f"Numpy performance test failed: {str(e)}", {}

    def _test_memory_usage(self) -> Tuple[str, str, Dict]:
        """Test memory usage."""
        try:
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

            # Force garbage collection
            gc.collect()

            # Get memory after GC
            gc_memory = process.memory_info().rss / (1024 * 1024)  # MB

            memory_freed = initial_memory - gc_memory

            if initial_memory < 500:  # < 500 MB
                status = "healthy"
                message = f"Memory usage normal: {initial_memory:.1f} MB"
            elif initial_memory < 1000:  # < 1 GB
                status = "warning"
                message = f"Memory usage elevated: {initial_memory:.1f} MB"
            else:
                status = "error"
                message = f"Memory usage high: {initial_memory:.1f} MB"

            return (
                status,
                message,
                {
                    "initial_memory_mb": initial_memory,
                    "post_gc_memory_mb": gc_memory,
                    "memory_freed_mb": memory_freed,
                },
            )

        except Exception as e:
            return "error", f"Memory test failed: {str(e)}", {}

    def _generate_report(
        self, system_resources: Dict, dependencies: Dict
    ) -> SystemHealthReport:
        """Tạo báo cáo sức khỏe."""
        report = SystemHealthReport()
        report.system_resources = system_resources
        report.dependencies = dependencies

        # Phân loại kết quả
        for result in self.results:
            if "core_" in result.component_name:
                report.core_modules.append(result)
            elif "ui_" in result.component_name:
                report.ui_modules.append(result)
            elif "dose_" in result.component_name:
                report.dose_modules.append(result)
            elif "eval_" in result.component_name:
                report.evaluation_modules.append(result)
            elif "opt_" in result.component_name:
                report.optimization_modules.append(result)

        # Tính toán thống kê
        report.total_checks = len(self.results)
        for result in self.results:
            if result.status == "healthy":
                report.passed_checks += 1
            elif result.status == "warning":
                report.warning_checks += 1
            elif result.status == "error":
                report.failed_checks += 1
            elif result.status == "critical":
                report.critical_checks += 1

        # Xác định trạng thái tổng thể
        if report.critical_checks > 0:
            report.overall_status = "critical"
        elif report.failed_checks > report.passed_checks:
            report.overall_status = "error"
        elif report.warning_checks > 0:
            report.overall_status = "warning"
        else:
            report.overall_status = "healthy"

        # Tạo recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: SystemHealthReport) -> List[str]:
        """Tạo đề xuất cải thiện."""
        recommendations = []

        # Memory recommendations
        if report.system_resources.get("memory_percent", 0) > 80:
            recommendations.append(
                "Bộ nhớ sử dụng cao (>80%). Khuyến nghị đóng các ứng dụng không cần thiết."
            )

        # Disk recommendations
        if report.system_resources.get("disk_percent", 0) > 90:
            recommendations.append(
                "Dung lượng đĩa thấp (<10% còn trống). Khuyến nghị dọn dẹp tệp tin."
            )

        # Dependencies recommendations
        critical_deps = ["numpy", "PyQt5", "pydicom"]
        for dep in critical_deps:
            if report.dependencies.get(dep) == "not_available":
                recommendations.append(
                    f"Dependency quan trọng {dep} không có. Khuyến nghị cài đặt: pip install {dep}"
                )

        # Module recommendations
        if report.failed_checks > 0:
            recommendations.append(
                f"Có {report.failed_checks} module lỗi. Kiểm tra log để biết chi tiết."
            )

        if not recommendations:
            recommendations.append(
                "Hệ thống hoạt động tốt, không có đề xuất cải thiện."
            )

        return recommendations


def run_system_health_check() -> SystemHealthReport:
    """Chạy kiểm tra sức khỏe hệ thống nhanh."""
    checker = SystemHealthChecker()
    return checker.run_comprehensive_check()


def print_health_report(report: SystemHealthReport):
    """In báo cáo sức khỏe hệ thống."""
    print("\n" + "=" * 80)
    print("QUANGTPS SYSTEM HEALTH REPORT")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall Status: {report.overall_status.upper()}")
    print(f"Total Checks: {report.total_checks}")
    print(f"  ✓ Passed: {report.passed_checks}")
    print(f"  ⚠ Warning: {report.warning_checks}")
    print(f"  ✗ Failed: {report.failed_checks}")
    print(f"  ⚡ Critical: {report.critical_checks}")

    # System resources
    print("\nSYSTEM RESOURCES:")
    print("-" * 40)
    if report.system_resources:
        print(
            f"Memory: {report.system_resources.get('memory_available_gb', 0):.1f}GB / "
            f"{report.system_resources.get('memory_total_gb', 0):.1f}GB "
            f"({report.system_resources.get('memory_percent', 0):.1f}% used)"
        )
        print(
            f"Disk: {report.system_resources.get('disk_free_gb', 0):.1f}GB free "
            f"({report.system_resources.get('disk_percent', 0):.1f}% used)"
        )
        print(f"CPU: {report.system_resources.get('cpu_count', 0)} cores")

    # Dependencies
    print("\nDEPENDENCIES:")
    print("-" * 40)
    for dep, version in report.dependencies.items():
        status = "✓" if version != "not_available" else "✗"
        version_text = version if version != "not_available" else "NOT AVAILABLE"
        print(f"{status} {dep:<15} {version_text}")

    # Module summary
    def print_module_summary(modules, title):
        if modules:
            print(f"\n{title}:")
            print("-" * 40)
            for module in modules:
                status_icon = {
                    "healthy": "✓",
                    "warning": "⚠",
                    "error": "✗",
                    "critical": "⚡",
                }
                icon = status_icon.get(module.status, "?")
                print(f"{icon} {module.component_name:<25} {module.message}")

    print_module_summary(report.core_modules, "CORE MODULES")
    print_module_summary(report.ui_modules, "UI MODULES")
    print_module_summary(report.dose_modules, "DOSE MODULES")
    print_module_summary(report.evaluation_modules, "EVALUATION MODULES")
    print_module_summary(report.optimization_modules, "OPTIMIZATION MODULES")

    # Recommendations
    if report.recommendations:
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    report = run_system_health_check()
    print_health_report(report)
