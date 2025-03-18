#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để tạo các biểu tượng cơ bản cho QuangTPS.
Tạo các biểu tượng màu đơn giản cho các chức năng khác nhau.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Đường dẫn thư mục biểu tượng
ICONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'quangtps', 'ui', 'icons')

# Đảm bảo thư mục tồn tại
os.makedirs(ICONS_DIR, exist_ok=True)

# Màu sắc cho các loại biểu tượng khác nhau
COLORS = {
    'workflow': '#3498db',      # Xanh dương
    'patient': '#2ecc71',       # Xanh lá
    'imaging': '#9b59b6',       # Tím
    'planning': '#e74c3c',      # Đỏ
    'dose': '#f39c12',          # Cam
    'evaluation': '#1abc9c',    # Ngọc lam
    'treatment': '#d35400',     # Nâu đỏ
    'qa': '#8e44ad',            # Tím đậm
    'report': '#34495e',        # Xám đen
    'action': '#3498db',        # Xanh dương (cho các hành động)
    'tool': '#7f8c8d',          # Xám (cho các công cụ)
    'logo': '#2980b9',          # Xanh dương đậm (cho logo)
}


def generate_text_icon(text, color, size=(32, 32), bg_color=None, border_radius=5):
    """
    Tạo biểu tượng văn bản với màu nền và văn bản cho trước.
    
    Args:
        text (str): Văn bản hiển thị trên biểu tượng
        color (str): Mã màu HEX cho nền
        size (tuple): Kích thước biểu tượng (chiều rộng, chiều cao)
        bg_color (str, optional): Màu nền. Nếu None, sẽ dùng màu trong suốt.
        border_radius (int): Bán kính bo góc
        
    Returns:
        PIL.Image: Đối tượng hình ảnh
    """
    # Tạo hình ảnh với alpha channel (RGBA)
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Vẽ hình nền hình chữ nhật bo góc
    if bg_color:
        # Vẽ hình chữ nhật đầy
        draw.rectangle([(0, 0), size], fill=bg_color)
    else:
        # Vẽ viền
        draw.rounded_rectangle([(0, 0), (size[0]-1, size[1]-1)], border_radius, fill=None, outline=color, width=2)
    
    # Thêm text
    try:
        # Thử dùng font hệ thống
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Lấy kích thước text
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    
    # Vẽ text ở giữa
    position = ((size[0] - text_width) / 2, (size[1] - text_height) / 2)
    draw.text(position, text, fill=color, font=font)
    
    return img


def generate_shape_icon(shape_type, color, size=(32, 32), bg_color=None, border_radius=5):
    """
    Tạo biểu tượng hình dạng cơ bản.
    
    Args:
        shape_type (str): Loại hình ('circle', 'square', 'triangle', etc.)
        color (str): Mã màu HEX
        size (tuple): Kích thước biểu tượng
        bg_color (str, optional): Màu nền
        border_radius (int): Bán kính bo góc (cho hình vuông)
        
    Returns:
        PIL.Image: Đối tượng hình ảnh
    """
    # Tạo hình ảnh với alpha channel
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Vẽ hình nền nếu có
    if bg_color:
        draw.rectangle([(0, 0), size], fill=bg_color)
    
    # Margin
    margin = 4
    shape_bounds = [margin, margin, size[0] - margin, size[1] - margin]
    
    # Vẽ hình theo loại
    if shape_type == 'circle':
        draw.ellipse(shape_bounds, fill=color)
    elif shape_type == 'square':
        draw.rounded_rectangle(shape_bounds, border_radius, fill=color)
    elif shape_type == 'triangle':
        points = [
            (size[0] // 2, margin),
            (margin, size[1] - margin),
            (size[0] - margin, size[1] - margin)
        ]
        draw.polygon(points, fill=color)
    elif shape_type == 'star':
        # Vẽ ngôi sao 5 cánh đơn giản
        center_x, center_y = size[0] // 2, size[1] // 2
        radius_outer = min(center_x, center_y) - margin
        radius_inner = radius_outer * 0.4
        points = []
        
        for i in range(10):
            angle = i * 36 * 3.14159 / 180
            radius = radius_outer if i % 2 == 0 else radius_inner
            x = center_x + radius * 0.9 * -1 * (1 if i > 5 and i < 9 else -1) + radius * 0.1
            y = center_y + radius * -1 * (1 if i > 7 or i < 3 else -1)
            points.append((x, y))
            
        draw.polygon(points, fill=color)
    
    return img


def create_all_icons():
    """Tạo tất cả các biểu tượng cần thiết."""
    # Danh sách các biểu tượng cần tạo
    icons_to_create = [
        # Biểu tượng cho các tab
        ('workflow', 'W', 'tab'),
        ('patient', 'P', 'tab'),
        ('imaging', 'I', 'tab'),
        ('planning', 'PL', 'tab'),
        ('dose', 'D', 'tab'),
        ('evaluation', 'E', 'tab'),
        ('treatment', 'T', 'tab'),
        ('qa', 'QA', 'tab'),
        ('report', 'R', 'tab'),
        
        # Biểu tượng cho các hành động
        ('new_patient', 'NP', 'action'),
        ('open_patient', 'OP', 'action'),
        ('new_plan', 'NP', 'action'),
        ('calculate_dose', 'CD', 'action'),
        ('optimize', 'O', 'action'),
        ('evaluate', 'EV', 'action'),
        ('import_dicom', 'ID', 'action'),
        ('export', 'EX', 'action'),
        ('exit', 'X', 'action'),
        
        # Biểu tượng cho menu trợ giúp
        ('help', '?', 'action'),
        ('about', 'i', 'action'),
        
        # Biểu tượng cho công cụ
        ('measure', 'M', 'tool'),
        ('roi', 'ROI', 'tool'),
        ('3d_view', '3D', 'tool'),
    ]
    
    # Tạo biểu tượng cho mỗi mục
    for icon_name, text, icon_type in icons_to_create:
        # Xác định màu
        if icon_type == 'tab':
            color = COLORS.get(icon_name, '#3498db')
        elif icon_type == 'action':
            color = COLORS['action']
        else:
            color = COLORS['tool']
        
        # Tạo biểu tượng
        icon = generate_text_icon(text, color, size=(32, 32), border_radius=5)
        
        # Lưu biểu tượng
        icon_path = os.path.join(ICONS_DIR, f"{icon_name}.png")
        icon.save(icon_path)
        print(f"Created icon: {icon_path}")
    
    # Tạo logo đặc biệt
    logo = generate_shape_icon('square', COLORS['logo'], size=(64, 64), border_radius=10)
    draw = ImageDraw.Draw(logo)
    
    # Thêm text "QTPS" cho logo
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    draw.text((14, 18), "QTPS", fill='white', font=font)
    
    # Lưu logo
    logo_path = os.path.join(ICONS_DIR, "logo.png")
    logo.save(logo_path)
    print(f"Created logo: {logo_path}")


if __name__ == "__main__":
    create_all_icons()
    print("Icon generation complete!")
