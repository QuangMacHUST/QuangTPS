#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module kiểm tra và xác nhận log file máy điều trị.

Module này cung cấp các lớp và phương thức để định nghĩa các quy tắc
kiểm tra log file máy điều trị và tự động phát hiện sai lệch.
"""

import os
import logging
import json
import re
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from quangtps.evaluation.qa.machine_log_analyzer import (
    LogFileAnalyzer,
    DeviationSeverity,
)

logger = logging.getLogger(__name__)


class ValidationRuleType(Enum):
    """Loại quy tắc kiểm tra log file."""

    PARAMETER_LIMIT = auto()  # Giới hạn tham số (min/max)
    PARAMETER_DEVIATION = auto()  # Độ lệch tham số so với kế hoạch
    PATTERN_MATCH = auto()  # Khớp mẫu nội dung
    CUSTOM_FUNCTION = auto()  # Hàm tùy chỉnh


class ValidationRule:
    """Quy tắc kiểm tra log file."""

    def __init__(
        self,
        name: str,
        rule_type: ValidationRuleType,
        parameter: str = None,
        min_value: float = None,
        max_value: float = None,
        tolerance: float = None,
        pattern: str = None,
        custom_function: Callable = None,
        message: str = None,
        severity: DeviationSeverity = DeviationSeverity.MINOR,
    ):
        """
        Khởi tạo quy tắc kiểm tra.

        Parameters
        ----------
        name : str
            Tên quy tắc
        rule_type : ValidationRuleType
            Loại quy tắc
        parameter : str, optional
            Tên tham số cần kiểm tra
        min_value : float, optional
            Giá trị tối thiểu cho phép
        max_value : float, optional
            Giá trị tối đa cho phép
        tolerance : float, optional
            Dung sai cho phép
        pattern : str, optional
            Mẫu regex để khớp
        custom_function : Callable, optional
            Hàm tùy chỉnh để kiểm tra
        message : str, optional
            Thông báo lỗi
        severity : DeviationSeverity, optional
            Mức độ nghiêm trọng
        """
        self.name = name
        self.rule_type = rule_type
        self.parameter = parameter
        self.min_value = min_value
        self.max_value = max_value
        self.tolerance = tolerance
        self.pattern = pattern
        self.custom_function = custom_function
        self.message = message
        self.severity = severity

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi quy tắc thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin quy tắc
        """
        rule_dict = {
            "name": self.name,
            "rule_type": self.rule_type.name,
            "severity": self.severity.name,
        }

        # Thêm các tham số tùy theo loại quy tắc
        if self.rule_type == ValidationRuleType.PARAMETER_LIMIT:
            rule_dict.update(
                {
                    "parameter": self.parameter,
                    "min_value": self.min_value,
                    "max_value": self.max_value,
                }
            )
        elif self.rule_type == ValidationRuleType.PARAMETER_DEVIATION:
            rule_dict.update({"parameter": self.parameter, "tolerance": self.tolerance})
        elif self.rule_type == ValidationRuleType.PATTERN_MATCH:
            rule_dict.update({"pattern": self.pattern})

        if self.message:
            rule_dict["message"] = self.message

        return rule_dict

    @classmethod
    def from_dict(cls, rule_dict: Dict[str, Any]) -> "ValidationRule":
        """
        Tạo quy tắc từ dictionary.

        Parameters
        ----------
        rule_dict : Dict[str, Any]
            Dictionary chứa thông tin quy tắc

        Returns
        -------
        ValidationRule
            Đối tượng ValidationRule
        """
        rule_type = ValidationRuleType[rule_dict["rule_type"]]
        severity = DeviationSeverity[rule_dict.get("severity", "MINOR")]

        return cls(
            name=rule_dict["name"],
            rule_type=rule_type,
            parameter=rule_dict.get("parameter"),
            min_value=rule_dict.get("min_value"),
            max_value=rule_dict.get("max_value"),
            tolerance=rule_dict.get("tolerance"),
            pattern=rule_dict.get("pattern"),
            message=rule_dict.get("message"),
            severity=severity,
        )


class ValidationResult:
    """Kết quả kiểm tra quy tắc."""

    def __init__(
        self,
        rule: ValidationRule,
        passed: bool,
        value: Any = None,
        expected: Any = None,
        message: str = None,
        timestamp: Union[str, datetime] = None,
    ):
        """
        Khởi tạo kết quả kiểm tra.

        Parameters
        ----------
        rule : ValidationRule
            Quy tắc đã kiểm tra
        passed : bool
            Kết quả kiểm tra (True nếu đạt)
        value : Any, optional
            Giá trị thực tế
        expected : Any, optional
            Giá trị mong đợi
        message : str, optional
            Thông báo kết quả
        timestamp : Union[str, datetime], optional
            Thời điểm kiểm tra
        """
        self.rule = rule
        self.passed = passed
        self.value = value
        self.expected = expected
        self.message = message or rule.message
        self.timestamp = timestamp or datetime.now()

        # Chuyển đổi timestamp thành datetime nếu cần
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp)
            except ValueError:
                # Thử định dạng khác
                try:
                    self.timestamp = datetime.strptime(
                        self.timestamp, "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    # Giữ nguyên giá trị chuỗi
                    pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kết quả thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kết quả
        """
        # Chuyển đổi timestamp thành chuỗi nếu là datetime
        timestamp = (
            self.timestamp.isoformat()
            if isinstance(self.timestamp, datetime)
            else self.timestamp
        )

        return {
            "rule": self.rule.to_dict(),
            "passed": self.passed,
            "value": self.value,
            "expected": self.expected,
            "message": self.message,
            "timestamp": timestamp,
            "severity": self.rule.severity.name,
        }


class LogFileValidator:
    """Lớp thực hiện kiểm tra log file máy điều trị."""

    def __init__(self, rules: List[ValidationRule] = None):
        """
        Khởi tạo validator với danh sách quy tắc.

        Parameters
        ----------
        rules : List[ValidationRule], optional
            Danh sách quy tắc kiểm tra
        """
        self.rules = rules or []
        # Dictionary ánh xạ tên tham số sang tên cột trong log data
        self.parameter_map = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """
        Thêm quy tắc kiểm tra.

        Parameters
        ----------
        rule : ValidationRule
            Quy tắc cần thêm
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """
        Xóa quy tắc kiểm tra.

        Parameters
        ----------
        rule_name : str
            Tên quy tắc cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False

    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """
        Lấy quy tắc kiểm tra theo tên.

        Parameters
        ----------
        rule_name : str
            Tên quy tắc cần lấy

        Returns
        -------
        Optional[ValidationRule]
            Quy tắc tìm thấy hoặc None
        """
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None

    def set_parameter_map(self, param_map: Dict[str, str]) -> None:
        """
        Thiết lập ánh xạ tham số sang tên cột.

        Parameters
        ----------
        param_map : Dict[str, str]
            Dictionary ánh xạ tên tham số sang tên cột
        """
        self.parameter_map = param_map

    def validate(
        self, log_data: pd.DataFrame, analyzer: LogFileAnalyzer = None
    ) -> List[ValidationResult]:
        """
        Kiểm tra log data với các quy tắc định nghĩa.

        Parameters
        ----------
        log_data : pd.DataFrame
            Dữ liệu log đã xử lý
        analyzer : LogFileAnalyzer, optional
            Phân tích log file đã thực hiện

        Returns
        -------
        List[ValidationResult]
            Danh sách kết quả kiểm tra
        """
        results = []

        for rule in self.rules:
            try:
                if rule.rule_type == ValidationRuleType.PARAMETER_LIMIT:
                    # Kiểm tra giới hạn tham số
                    result = self._validate_parameter_limit(rule, log_data)
                    results.extend(result)

                elif rule.rule_type == ValidationRuleType.PARAMETER_DEVIATION:
                    # Kiểm tra độ lệch tham số
                    if analyzer is not None:
                        result = self._validate_parameter_deviation(
                            rule, log_data, analyzer
                        )
                        results.extend(result)
                    else:
                        logger.warning(
                            f"Không thể kiểm tra độ lệch cho quy tắc '{rule.name}' vì không có analyzer"
                        )

                elif rule.rule_type == ValidationRuleType.PATTERN_MATCH:
                    # Kiểm tra khớp mẫu
                    result = self._validate_pattern_match(rule, log_data)
                    results.append(result)

                elif rule.rule_type == ValidationRuleType.CUSTOM_FUNCTION:
                    # Thực thi hàm tùy chỉnh
                    if rule.custom_function:
                        result = rule.custom_function(rule, log_data, analyzer)
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    else:
                        logger.warning(
                            f"Quy tắc '{rule.name}' là CUSTOM_FUNCTION nhưng không có hàm tùy chỉnh"
                        )

            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra quy tắc '{rule.name}': {str(e)}")
                import traceback

                traceback.print_exc()

        return results

    def _validate_parameter_limit(
        self, rule: ValidationRule, log_data: pd.DataFrame
    ) -> List[ValidationResult]:
        """
        Kiểm tra giới hạn tham số.

        Parameters
        ----------
        rule : ValidationRule
            Quy tắc kiểm tra
        log_data : pd.DataFrame
            Dữ liệu log

        Returns
        -------
        List[ValidationResult]
            Các kết quả kiểm tra
        """
        results = []

        # Chuyển đổi tên tham số sang tên cột nếu có
        column_name = self.parameter_map.get(rule.parameter, rule.parameter)

        if column_name not in log_data.columns:
            # Không tìm thấy cột tương ứng
            result = ValidationResult(
                rule=rule,
                passed=False,
                message=f"Không tìm thấy tham số '{rule.parameter}' trong dữ liệu log",
            )
            return [result]

        # Lấy dữ liệu tham số
        param_data = log_data[column_name]

        # Kiểm tra giới hạn
        if rule.min_value is not None:
            min_violations = param_data < rule.min_value
            if any(min_violations):
                for idx in min_violations[min_violations].index:
                    value = param_data[idx]
                    timestamp = (
                        log_data["timestamp"][idx]
                        if "timestamp" in log_data.columns
                        else idx
                    )

                    result = ValidationResult(
                        rule=rule,
                        passed=False,
                        value=value,
                        expected=f">= {rule.min_value}",
                        message=f"Giá trị {rule.parameter} ({value}) nhỏ hơn giới hạn tối thiểu ({rule.min_value})",
                        timestamp=timestamp,
                    )
                    results.append(result)

        if rule.max_value is not None:
            max_violations = param_data > rule.max_value
            if any(max_violations):
                for idx in max_violations[max_violations].index:
                    value = param_data[idx]
                    timestamp = (
                        log_data["timestamp"][idx]
                        if "timestamp" in log_data.columns
                        else idx
                    )

                    result = ValidationResult(
                        rule=rule,
                        passed=False,
                        value=value,
                        expected=f"<= {rule.max_value}",
                        message=f"Giá trị {rule.parameter} ({value}) lớn hơn giới hạn tối đa ({rule.max_value})",
                        timestamp=timestamp,
                    )
                    results.append(result)

        # Nếu không có lỗi nào
        if not results:
            result = ValidationResult(
                rule=rule,
                passed=True,
                message=f"Tham số '{rule.parameter}' nằm trong giới hạn cho phép",
            )
            results.append(result)

        return results

    def _validate_parameter_deviation(
        self, rule: ValidationRule, log_data: pd.DataFrame, analyzer: LogFileAnalyzer
    ) -> List[ValidationResult]:
        """
        Kiểm tra độ lệch tham số so với kế hoạch.

        Parameters
        ----------
        rule : ValidationRule
            Quy tắc kiểm tra
        log_data : pd.DataFrame
            Dữ liệu log
        analyzer : LogFileAnalyzer
            Phân tích log file

        Returns
        -------
        List[ValidationResult]
            Các kết quả kiểm tra
        """
        results = []

        # Lấy thông tin độ lệch từ analyzer
        deviations = analyzer.deviations if hasattr(analyzer, "deviations") else []

        # Tìm sai lệch tương ứng với tham số
        param_deviations = [
            d for d in deviations if d.get("parameter") == rule.parameter
        ]

        if not param_deviations:
            # Không tìm thấy thông tin sai lệch
            result = ValidationResult(
                rule=rule,
                passed=True,
                message=f"Không tìm thấy thông tin sai lệch cho tham số '{rule.parameter}'",
            )
            return [result]

        for deviation in param_deviations:
            value = deviation.get("value", 0)
            tolerance = rule.tolerance

            if value > tolerance:
                result = ValidationResult(
                    rule=rule,
                    passed=False,
                    value=value,
                    expected=f"<= {tolerance}",
                    message=f"Sai lệch {rule.parameter} ({value}) vượt quá dung sai cho phép ({tolerance})",
                    timestamp=deviation.get("timestamp"),
                )
            else:
                result = ValidationResult(
                    rule=rule,
                    passed=True,
                    value=value,
                    expected=f"<= {tolerance}",
                    message=f"Sai lệch {rule.parameter} ({value}) nằm trong dung sai cho phép ({tolerance})",
                    timestamp=deviation.get("timestamp"),
                )

            results.append(result)

        return results

    def _validate_pattern_match(
        self, rule: ValidationRule, log_data: pd.DataFrame
    ) -> ValidationResult:
        """
        Kiểm tra khớp mẫu trong nội dung log.

        Parameters
        ----------
        rule : ValidationRule
            Quy tắc kiểm tra
        log_data : pd.DataFrame
            Dữ liệu log

        Returns
        -------
        ValidationResult
            Kết quả kiểm tra
        """
        # Chuyển đổi DataFrame thành chuỗi để tìm kiếm mẫu
        log_str = log_data.to_string() if not log_data.empty else ""

        if not rule.pattern:
            return ValidationResult(
                rule=rule,
                passed=False,
                message="Quy tắc không có mẫu regex để kiểm tra",
            )

        try:
            pattern = re.compile(rule.pattern)
            matches = pattern.findall(log_str)

            if matches:
                return ValidationResult(
                    rule=rule,
                    passed=True,
                    value=len(matches),
                    message=f"Tìm thấy {len(matches)} kết quả khớp với mẫu '{rule.pattern}'",
                )
            else:
                return ValidationResult(
                    rule=rule,
                    passed=False,
                    value=0,
                    message=f"Không tìm thấy kết quả khớp với mẫu '{rule.pattern}'",
                )

        except re.error as e:
            return ValidationResult(
                rule=rule,
                passed=False,
                message=f"Lỗi regex: {str(e)}",
            )

    def save_rules(self, filepath: str) -> bool:
        """
        Lưu danh sách quy tắc vào file.

        Parameters
        ----------
        filepath : str
            Đường dẫn file lưu

        Returns
        -------
        bool
            True nếu lưu thành công
        """
        try:
            rules_data = [rule.to_dict() for rule in self.rules]

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu quy tắc vào file {filepath}: {str(e)}")
            return False

    @classmethod
    def load_rules(cls, filepath: str) -> "LogFileValidator":
        """
        Tạo validator từ file quy tắc.

        Parameters
        ----------
        filepath : str
            Đường dẫn file quy tắc

        Returns
        -------
        LogFileValidator
            Đối tượng validator với quy tắc đã tải
        """
        validator = cls()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                rules_data = json.load(f)

            for rule_dict in rules_data:
                rule = ValidationRule.from_dict(rule_dict)
                validator.add_rule(rule)

            return validator
        except Exception as e:
            logger.error(f"Lỗi khi tải quy tắc từ file {filepath}: {str(e)}")
            return validator

    @staticmethod
    def create_default_rules() -> List[ValidationRule]:
        """
        Tạo bộ quy tắc kiểm tra mặc định.

        Returns
        -------
        List[ValidationRule]
            Danh sách quy tắc mặc định
        """
        rules = []

        # Quy tắc kiểm tra góc gantry
        rules.append(
            ValidationRule(
                name="gantry_angle_limit",
                rule_type=ValidationRuleType.PARAMETER_LIMIT,
                parameter="gantry_angle",
                min_value=0,
                max_value=360,
                message="Góc gantry phải từ 0 đến 360 độ",
                severity=DeviationSeverity.MAJOR,
            )
        )

        # Quy tắc kiểm tra độ lệch góc gantry
        rules.append(
            ValidationRule(
                name="gantry_angle_deviation",
                rule_type=ValidationRuleType.PARAMETER_DEVIATION,
                parameter="gantry_angle",
                tolerance=1.0,  # Độ lệch cho phép 1 độ
                message="Độ lệch góc gantry vượt quá 1 độ",
                severity=DeviationSeverity.MAJOR,
            )
        )

        # Quy tắc kiểm tra góc collimator
        rules.append(
            ValidationRule(
                name="collimator_angle_limit",
                rule_type=ValidationRuleType.PARAMETER_LIMIT,
                parameter="collimator_angle",
                min_value=0,
                max_value=360,
                message="Góc collimator phải từ 0 đến 360 độ",
                severity=DeviationSeverity.MAJOR,
            )
        )

        # Quy tắc kiểm tra độ lệch góc collimator
        rules.append(
            ValidationRule(
                name="collimator_angle_deviation",
                rule_type=ValidationRuleType.PARAMETER_DEVIATION,
                parameter="collimator_angle",
                tolerance=1.0,  # Độ lệch cho phép 1 độ
                message="Độ lệch góc collimator vượt quá 1 độ",
                severity=DeviationSeverity.MODERATE,
            )
        )

        # Quy tắc kiểm tra vị trí MLC
        rules.append(
            ValidationRule(
                name="mlc_position_deviation",
                rule_type=ValidationRuleType.PARAMETER_DEVIATION,
                parameter="mlc_position",
                tolerance=2.0,  # Độ lệch cho phép 2mm
                message="Độ lệch vị trí MLC vượt quá 2mm",
                severity=DeviationSeverity.CRITICAL,
            )
        )

        # Quy tắc kiểm tra dose rate
        rules.append(
            ValidationRule(
                name="dose_rate_limit",
                rule_type=ValidationRuleType.PARAMETER_LIMIT,
                parameter="dose_rate",
                min_value=0,
                max_value=1000,  # MU/min
                message="Dose rate phải từ 0 đến 1000 MU/min",
                severity=DeviationSeverity.MAJOR,
            )
        )

        # Quy tắc kiểm tra độ lệch dose rate
        rules.append(
            ValidationRule(
                name="dose_rate_deviation",
                rule_type=ValidationRuleType.PARAMETER_DEVIATION,
                parameter="dose_rate",
                tolerance=20.0,  # Độ lệch cho phép 20 MU/min
                message="Độ lệch dose rate vượt quá 20 MU/min",
                severity=DeviationSeverity.MODERATE,
            )
        )

        return rules


# Hàm tiện ích để tạo validator với quy tắc mặc định
def create_default_validator() -> LogFileValidator:
    """
    Tạo validator với bộ quy tắc mặc định.

    Returns
    -------
    LogFileValidator
        Validator với quy tắc mặc định
    """
    validator = LogFileValidator()
    default_rules = LogFileValidator.create_default_rules()

    for rule in default_rules:
        validator.add_rule(rule)

    return validator
