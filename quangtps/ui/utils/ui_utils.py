def create_eclipse_icon(icon_name):
    """Tạo biểu tượng theo phong cách Eclipse."""
    from PyQt5.QtGui import QIcon

    icon_map = {
        "dose": "dose_calculation.png",
        "plan": "plan.png",
        "beam": "beam.png",
        "structure": "structure.png",
        "optimize": "optimize.png",
        "evaluate": "evaluate.png",
        "report": "report.png",
        "save": "save.png",
        "open": "open.png",
        "new": "new.png",
        "delete": "delete.png",
        "edit": "edit.png",
        "copy": "copy.png",
        "dvh": "dvh.png",
        "isodose": "isodose.png",
        "3d": "3d_view.png",
        "apply": "apply.png",
        "cancel": "cancel.png",
        "calculate": "calculate.png",
        "generate": "generate.png",
        "analysis": "analysis.png",
        "export": "export.png",
        "import": "import.png",
        "settings": "settings.png",
        "kbp": "kbp.png",
        "ai": "ai.png",
    }

    # Đường dẫn đến thư mục chứa biểu tượng
    icon_dir = os.path.join("quangtps", "ui", "icons", "new_icons")

    if icon_name in icon_map:
        icon_path = os.path.join(icon_dir, icon_map[icon_name])
        if os.path.exists(icon_path):
            return QIcon(icon_path)

    # Biểu tượng mặc định nếu không tìm thấy
    return QIcon()
