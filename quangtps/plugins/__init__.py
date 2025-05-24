#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plugin system cho QuangTPS.

Module này cung cấp hệ thống plugin để mở rộng chức năng của QuangTPS
với các module bên ngoài hoặc tùy chỉnh.
"""

import logging
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Registry lưu trữ các plugins đã đăng ký
_plugin_registry: Dict[str, Dict[str, Any]] = {
    "dose_algorithms": {},
    "optimization_methods": {},
    "auto_segmentation": {},
    "evaluation_metrics": {},
    "io_handlers": {},
    "ui_extensions": {},
}

# Hook callbacks
_plugin_hooks: Dict[str, List[Callable]] = {
    "pre_dose_calculation": [],
    "post_dose_calculation": [],
    "pre_optimization": [],
    "post_optimization": [],
    "patient_loaded": [],
    "plan_created": [],
}


def init_plugins(plugin_dir: Optional[str] = None) -> bool:
    """
    Khởi tạo hệ thống plugins.

    Parameters:
        plugin_dir: Thư mục chứa plugins (optional)

    Returns:
        bool: True nếu khởi tạo thành công
    """
    try:
        logger.info("Đang khởi tạo hệ thống plugins...")

        # Đăng ký các plugins cơ bản
        _register_core_plugins()

        # Load plugins từ thư mục nếu có
        if plugin_dir:
            _load_plugins_from_directory(plugin_dir)

        # Load plugins từ thư mục mặc định
        default_plugin_dir = Path(__file__).parent / "external"
        if default_plugin_dir.exists():
            _load_plugins_from_directory(str(default_plugin_dir))

        logger.info(
            f"Đã khởi tạo hệ thống plugins thành công. "
            f"Đăng ký: {sum(len(p) for p in _plugin_registry.values())} plugins"
        )
        return True

    except Exception as e:
        logger.error(f"Lỗi khởi tạo plugins: {e}")
        return False


def register_plugin(
    category: str,
    name: str,
    plugin_object: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Đăng ký một plugin.

    Parameters:
        category: Loại plugin (dose_algorithms, optimization_methods, etc.)
        name: Tên plugin
        plugin_object: Object của plugin
        metadata: Metadata bổ sung

    Returns:
        bool: True nếu đăng ký thành công
    """
    try:
        if category not in _plugin_registry:
            logger.warning(f"Category không hỗ trợ: {category}")
            return False

        _plugin_registry[category][name] = {
            "object": plugin_object,
            "metadata": metadata or {},
            "enabled": True,
        }

        logger.info(f"Đã đăng ký plugin '{name}' trong category '{category}'")
        return True

    except Exception as e:
        logger.error(f"Lỗi đăng ký plugin {name}: {e}")
        return False


def get_plugin(category: str, name: str) -> Optional[Any]:
    """
    Lấy plugin theo category và tên.

    Parameters:
        category: Loại plugin
        name: Tên plugin

    Returns:
        Plugin object hoặc None nếu không tìm thấy
    """
    try:
        plugin_info = _plugin_registry.get(category, {}).get(name)
        if plugin_info and plugin_info["enabled"]:
            return plugin_info["object"]
        return None
    except Exception as e:
        logger.error(f"Lỗi lấy plugin {category}.{name}: {e}")
        return None


def list_plugins(category: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Liệt kê các plugins có sẵn.

    Parameters:
        category: Loại plugin cụ thể (optional)

    Returns:
        Dictionary chứa danh sách plugins
    """
    if category:
        if category in _plugin_registry:
            return {category: list(_plugin_registry[category].keys())}
        else:
            return {}

    return {cat: list(plugins.keys()) for cat, plugins in _plugin_registry.items()}


def add_hook(hook_name: str, callback: Callable) -> bool:
    """
    Thêm hook callback.

    Parameters:
        hook_name: Tên hook
        callback: Function callback

    Returns:
        bool: True nếu thêm thành công
    """
    try:
        if hook_name not in _plugin_hooks:
            _plugin_hooks[hook_name] = []

        _plugin_hooks[hook_name].append(callback)
        logger.debug(f"Đã thêm hook callback cho '{hook_name}'")
        return True

    except Exception as e:
        logger.error(f"Lỗi thêm hook {hook_name}: {e}")
        return False


def trigger_hook(hook_name: str, *args, **kwargs) -> List[Any]:
    """
    Kích hoạt các hook callbacks.

    Parameters:
        hook_name: Tên hook
        *args, **kwargs: Tham số cho callbacks

    Returns:
        List kết quả từ các callbacks
    """
    results = []

    try:
        callbacks = _plugin_hooks.get(hook_name, [])
        for callback in callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Lỗi callback hook {hook_name}: {e}")

        return results

    except Exception as e:
        logger.error(f"Lỗi trigger hook {hook_name}: {e}")
        return results


def _register_core_plugins():
    """Đăng ký các plugins cơ bản."""
    try:
        # Đăng ký dose algorithms
        from quangtps.dose.algorithms import get_available_algorithms

        algorithms = get_available_algorithms()
        for alg_name in algorithms:
            try:
                # Fallback registration
                register_plugin(
                    "dose_algorithms",
                    alg_name,
                    None,
                    {"type": "core", "description": f"Core dose algorithm: {alg_name}"},
                )
            except Exception:
                pass

        # Đăng ký optimization methods
        optimization_methods = [
            "gradient_descent",
            "simulated_annealing",
            "genetic_algorithm",
        ]
        for method in optimization_methods:
            register_plugin(
                "optimization_methods",
                method,
                None,
                {"type": "core", "description": f"Core optimization method: {method}"},
            )

        # Đăng ký auto segmentation models
        segmentation_models = ["unet", "deeplab", "mask_rcnn"]
        for model in segmentation_models:
            register_plugin(
                "auto_segmentation",
                model,
                None,
                {"type": "core", "description": f"Core segmentation model: {model}"},
            )

        logger.debug("Đã đăng ký core plugins")

    except Exception as e:
        logger.warning(f"Không thể đăng ký đầy đủ core plugins: {e}")


def _load_plugins_from_directory(plugin_dir: str):
    """Load plugins từ thư mục."""
    try:
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return

        # Thêm thư mục vào sys.path
        if str(plugin_path) not in sys.path:
            sys.path.insert(0, str(plugin_path))

        # Tìm và load các Python files
        for plugin_file in plugin_path.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Tìm và đăng ký plugin functions
                if hasattr(module, "register_plugin_functions"):
                    module.register_plugin_functions()
                    logger.debug(f"Loaded plugin từ {plugin_file}")

            except Exception as e:
                logger.warning(f"Không thể load plugin từ {plugin_file}: {e}")

    except Exception as e:
        logger.error(f"Lỗi load plugins từ directory {plugin_dir}: {e}")


def get_plugin_info(category: str, name: str) -> Optional[Dict[str, Any]]:
    """
    Lấy thông tin về plugin.

    Parameters:
        category: Loại plugin
        name: Tên plugin

    Returns:
        Dictionary chứa thông tin plugin
    """
    try:
        return _plugin_registry.get(category, {}).get(name)
    except Exception:
        return None


def enable_plugin(category: str, name: str) -> bool:
    """Enable plugin."""
    try:
        plugin_info = _plugin_registry.get(category, {}).get(name)
        if plugin_info:
            plugin_info["enabled"] = True
            logger.info(f"Enabled plugin {category}.{name}")
            return True
        return False
    except Exception as e:
        logger.error(f"Lỗi enable plugin {category}.{name}: {e}")
        return False


def disable_plugin(category: str, name: str) -> bool:
    """Disable plugin."""
    try:
        plugin_info = _plugin_registry.get(category, {}).get(name)
        if plugin_info:
            plugin_info["enabled"] = False
            logger.info(f"Disabled plugin {category}.{name}")
            return True
        return False
    except Exception as e:
        logger.error(f"Lỗi disable plugin {category}.{name}: {e}")
        return False


# Compatibility functions
def load_plugins():
    """Compatibility function cho backward compatibility."""
    return init_plugins()


def get_available_plugins():
    """Lấy danh sách plugins khả dụng."""
    return list_plugins()


# Initialize plugins khi import module
try:
    init_plugins()
except Exception as e:
    logger.warning(f"Không thể auto-initialize plugins: {e}")


__all__ = [
    "init_plugins",
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "add_hook",
    "trigger_hook",
    "get_plugin_info",
    "enable_plugin",
    "disable_plugin",
    "load_plugins",
    "get_available_plugins",
]
