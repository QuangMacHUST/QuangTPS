#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý mẫu báo cáo cho QuangTPS.

Module này cung cấp các lớp và hàm để quản lý các mẫu báo cáo
điều trị trong hệ thống lập kế hoạch xạ trị QuangTPS.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportTemplate:
    """Lớp biểu diễn một mẫu báo cáo điều trị."""
    
    def __init__(self, 
                id: str,
                name: str,
                description: str = "",
                file_path: Optional[str] = None,
                created_by: str = "QuangTPS",
                creation_date: Optional[datetime] = None):
        """
        Khởi tạo một mẫu báo cáo.
        
        Parameters
        ----------
        id : str
            ID duy nhất của mẫu
        name : str
            Tên mô tả của mẫu
        description : str, optional
            Mô tả chi tiết, mặc định là chuỗi rỗng
        file_path : Optional[str], optional
            Đường dẫn đến tệp mẫu, mặc định là None
        created_by : str, optional
            Tên người tạo, mặc định là "QuangTPS"
        creation_date : Optional[datetime], optional
            Ngày tạo, mặc định là thời điểm hiện tại
        """
        self.id = id
        self.name = name
        self.description = description
        self.file_path = file_path
        self.created_by = created_by
        self.creation_date = creation_date or datetime.now()
        self.modified_date = self.creation_date
        self.version = "1.0"
        self.tags = []
        self.institution = ""
        self.template_type = "html"  # html, docx, pdf
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mẫu thành từ điển.
        
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa thông tin mẫu
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "created_by": self.created_by,
            "creation_date": self.creation_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "version": self.version,
            "tags": self.tags,
            "institution": self.institution,
            "template_type": self.template_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportTemplate':
        """
        Tạo mẫu từ từ điển.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển chứa thông tin mẫu
            
        Returns
        -------
        ReportTemplate
            Đối tượng mẫu báo cáo
        """
        template = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            file_path=data.get("file_path"),
            created_by=data.get("created_by", "QuangTPS")
        )
        
        if "creation_date" in data:
            template.creation_date = datetime.fromisoformat(data["creation_date"])
        if "modified_date" in data:
            template.modified_date = datetime.fromisoformat(data["modified_date"])
        
        template.version = data.get("version", "1.0")
        template.tags = data.get("tags", [])
        template.institution = data.get("institution", "")
        template.template_type = data.get("template_type", "html")
        template.metadata = data.get("metadata", {})
        
        return template
    
    def __str__(self) -> str:
        """Biểu diễn chuỗi của mẫu."""
        return f"ReportTemplate(id='{self.id}', name='{self.name}')"


class TemplateManager:
    """
    Quản lý các mẫu báo cáo cho QuangTPS.
    
    Lớp này cung cấp các phương thức để quản lý việc tạo, cập nhật,
    xóa và sử dụng các mẫu báo cáo điều trị.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Khởi tạo trình quản lý mẫu.
        
        Parameters
        ----------
        template_dir : Optional[str], optional
            Thư mục chứa các tệp mẫu, mặc định là None
        """
        # Xác định thư mục mẫu
        if template_dir is None:
            # Tìm thư mục gốc của dự án
            project_root = self._find_project_root()
            self.template_dir = os.path.join(project_root, "data", "report_templates")
        else:
            self.template_dir = template_dir
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Đường dẫn đến tệp cấu hình
        self.config_file = os.path.join(self.template_dir, "templates.json")
        
        # Tải danh sách mẫu
        self.templates: Dict[str, ReportTemplate] = {}
        self._load_templates()
        
        logger.debug(f"Đã khởi tạo TemplateManager với {len(self.templates)} mẫu từ {self.template_dir}")
    
    def _find_project_root(self) -> str:
        """
        Tìm thư mục gốc của dự án QuangTPS.
        
        Returns
        -------
        str
            Đường dẫn đến thư mục gốc
        """
        # Bắt đầu từ thư mục hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Đi lên cho đến khi tìm thấy thư mục gốc
        while True:
            # Kiểm tra nếu đây là thư mục gốc của QuangTPS
            if os.path.exists(os.path.join(current_dir, "quangtps")) and \
               os.path.isdir(os.path.join(current_dir, "quangtps")):
                return current_dir
            
            # Đi lên một cấp
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Đã đến thư mục gốc
                # Nếu không tìm thấy, sử dụng thư mục hiện tại
                return os.path.dirname(os.path.abspath(__file__))
            
            current_dir = parent_dir
    
    def _load_templates(self) -> None:
        """Tải danh sách mẫu từ tệp cấu hình."""
        if not os.path.exists(self.config_file):
            # Tạo tệp cấu hình mặc định nếu không tồn tại
            self._save_templates()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for template_data in data.get("templates", []):
                template = ReportTemplate.from_dict(template_data)
                self.templates[template.id] = template
            
            logger.debug(f"Đã tải {len(self.templates)} mẫu từ {self.config_file}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mẫu từ {self.config_file}: {str(e)}")
    
    def load_default_templates(self) -> None:
        """
        Tải các mẫu mặc định từ thư mục mẫu hệ thống.
        
        Phương thức này sẽ sao chép các mẫu mặc định từ thư mục cài đặt
        vào thư mục mẫu người dùng nếu chúng chưa tồn tại.
        """
        # Đường dẫn đến thư mục mẫu hệ thống (cài đặt)
        system_templates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "report_templates"
        )
        
        if not os.path.exists(system_templates_dir):
            logger.warning(f"Thư mục mẫu hệ thống không tồn tại: {system_templates_dir}")
            return
        
        # Đếm số mẫu được thêm vào
        added_count = 0
        
        try:
            # Lấy danh sách mẫu từ thư mục hệ thống
            for filename in os.listdir(system_templates_dir):
                if filename.endswith('.html') or filename.endswith('.jinja2'):
                    template_id = os.path.splitext(filename)[0]
                    
                    # Kiểm tra nếu mẫu đã tồn tại
                    if template_id in self.templates:
                        logger.debug(f"Mẫu {template_id} đã tồn tại, bỏ qua")
                        continue
                    
                    # Đường dẫn tệp mẫu
                    src_path = os.path.join(system_templates_dir, filename)
                    dst_path = os.path.join(self.template_dir, filename)
                    
                    # Sao chép tệp mẫu nếu chưa tồn tại
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                    
                    # Tạo mẫu mới
                    template = ReportTemplate(
                        id=template_id,
                        name=template_id.replace('_', ' ').title(),
                        description=f"Mẫu báo cáo mặc định: {template_id}",
                        file_path=dst_path,
                        created_by="QuangTPS System"
                    )
                    
                    # Thêm vào danh sách
                    self.templates[template_id] = template
                    added_count += 1
            
            if added_count > 0:
                logger.info(f"Đã thêm {added_count} mẫu mặc định")
                # Lưu cấu hình sau khi thêm mẫu mặc định
                self._save_templates()
            else:
                logger.debug("Không thêm mẫu mặc định mới nào")
                
        except Exception as e:
            logger.error(f"Lỗi khi tải mẫu mặc định: {str(e)}")
    
    def _save_templates(self) -> None:
        """Lưu danh sách mẫu vào tệp cấu hình."""
        try:
            data = {
                "templates": [template.to_dict() for template in self.templates.values()]
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            logger.debug(f"Đã lưu {len(self.templates)} mẫu vào {self.config_file}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu mẫu vào {self.config_file}: {str(e)}")
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """
        Lấy mẫu theo ID.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu cần lấy
            
        Returns
        -------
        Optional[ReportTemplate]
            Mẫu nếu tìm thấy, None nếu không
        """
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[ReportTemplate]:
        """
        Lấy tất cả các mẫu.
        
        Returns
        -------
        List[ReportTemplate]
            Danh sách các mẫu
        """
        return list(self.templates.values())
    
    def add_template(self, template: ReportTemplate) -> bool:
        """
        Thêm mẫu mới.
        
        Parameters
        ----------
        template : ReportTemplate
            Mẫu cần thêm
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu không
        """
        if template.id in self.templates:
            logger.warning(f"Mẫu có ID '{template.id}' đã tồn tại")
            return False
        
        self.templates[template.id] = template
        self._save_templates()
        logger.info(f"Đã thêm mẫu mới: {template.name} (ID: {template.id})")
        return True
    
    def update_template(self, template: ReportTemplate) -> bool:
        """
        Cập nhật mẫu.
        
        Parameters
        ----------
        template : ReportTemplate
            Mẫu cần cập nhật
            
        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu không
        """
        if template.id not in self.templates:
            logger.warning(f"Không tìm thấy mẫu có ID '{template.id}' để cập nhật")
            return False
        
        template.modified_date = datetime.now()
        self.templates[template.id] = template
        self._save_templates()
        logger.info(f"Đã cập nhật mẫu: {template.name} (ID: {template.id})")
        return True
    
    def delete_template(self, template_id: str) -> bool:
        """
        Xóa mẫu.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu không
        """
        if template_id not in self.templates:
            logger.warning(f"Không tìm thấy mẫu có ID '{template_id}' để xóa")
            return False
        
        template = self.templates[template_id]
        
        # Xóa tệp mẫu nếu có
        if template.file_path and os.path.exists(template.file_path):
            try:
                os.remove(template.file_path)
            except Exception as e:
                logger.error(f"Lỗi khi xóa tệp mẫu {template.file_path}: {str(e)}")
        
        # Xóa khỏi danh sách
        del self.templates[template_id]
        self._save_templates()
        logger.info(f"Đã xóa mẫu: {template.name} (ID: {template_id})")
        return True
    
    def create_template(self, 
                       name: str, 
                       description: str = "", 
                       template_type: str = "html",
                       content: Optional[str] = None) -> Optional[ReportTemplate]:
        """
        Tạo mẫu mới.
        
        Parameters
        ----------
        name : str
            Tên mẫu
        description : str, optional
            Mô tả mẫu, mặc định là chuỗi rỗng
        template_type : str, optional
            Loại mẫu (html, docx, pdf), mặc định là "html"
        content : Optional[str], optional
            Nội dung của mẫu, mặc định là None
            
        Returns
        -------
        Optional[ReportTemplate]
            Mẫu mới nếu tạo thành công, None nếu không
        """
        # Tạo ID duy nhất
        import uuid
        template_id = str(uuid.uuid4())
        
        # Tạo đường dẫn tệp
        template_filename = f"{template_id}.{template_type}"
        file_path = os.path.join(self.template_dir, template_filename)
        
        # Tạo mẫu mới
        template = ReportTemplate(
            id=template_id,
            name=name,
            description=description,
            file_path=file_path
        )
        
        template.template_type = template_type
        
        # Lưu nội dung mẫu nếu có
        if content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Lỗi khi lưu nội dung mẫu: {str(e)}")
                return None
        
        # Thêm vào danh sách
        self.add_template(template)
        
        return template
    
    def import_template(self, source_path: str, name: Optional[str] = None) -> Optional[ReportTemplate]:
        """
        Nhập mẫu từ tệp ngoài.
        
        Parameters
        ----------
        source_path : str
            Đường dẫn đến tệp mẫu nguồn
        name : Optional[str], optional
            Tên mẫu, mặc định là tên tệp
            
        Returns
        -------
        Optional[ReportTemplate]
            Mẫu mới nếu nhập thành công, None nếu không
        """
        if not os.path.exists(source_path):
            logger.error(f"Không tìm thấy tệp nguồn: {source_path}")
            return None
        
        try:
            # Tạo ID duy nhất
            import uuid
            template_id = str(uuid.uuid4())
            
            # Xác định tên và loại mẫu
            if name is None:
                name = os.path.basename(source_path)
            
            file_ext = os.path.splitext(source_path)[1].lower().replace(".", "")
            if file_ext not in ["html", "docx", "pdf"]:
                file_ext = "html"  # Mặc định là HTML
            
            # Tạo đường dẫn đích
            dest_filename = f"{template_id}.{file_ext}"
            dest_path = os.path.join(self.template_dir, dest_filename)
            
            # Sao chép tệp
            shutil.copy2(source_path, dest_path)
            
            # Tạo mẫu mới
            template = ReportTemplate(
                id=template_id,
                name=name,
                file_path=dest_path
            )
            
            template.template_type = file_ext
            
            # Thêm vào danh sách
            self.add_template(template)
            
            return template
            
        except Exception as e:
            logger.error(f"Lỗi khi nhập mẫu từ {source_path}: {str(e)}")
            return None
    
    def get_template_content(self, template_id: str) -> Optional[str]:
        """
        Lấy nội dung của mẫu.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu
            
        Returns
        -------
        Optional[str]
            Nội dung mẫu nếu tìm thấy, None nếu không
        """
        template = self.get_template(template_id)
        if not template or not template.file_path or not os.path.exists(template.file_path):
            return None
        
        try:
            with open(template.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Lỗi khi đọc nội dung mẫu {template_id}: {str(e)}")
            return None
    
    def set_template_content(self, template_id: str, content: str) -> bool:
        """
        Thiết lập nội dung cho mẫu.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu
        content : str
            Nội dung mới
            
        Returns
        -------
        bool
            True nếu thành công, False nếu không
        """
        template = self.get_template(template_id)
        if not template or not template.file_path:
            return False
        
        try:
            with open(template.file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Cập nhật ngày sửa đổi
            template.modified_date = datetime.now()
            self._save_templates()
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi ghi nội dung mẫu {template_id}: {str(e)}")
            return False
    
    def search_templates(self, query: str) -> List[ReportTemplate]:
        """
        Tìm kiếm mẫu theo từ khóa.
        
        Parameters
        ----------
        query : str
            Từ khóa tìm kiếm
            
        Returns
        -------
        List[ReportTemplate]
            Danh sách các mẫu phù hợp
        """
        query = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query in template.name.lower() or
                query in template.description.lower() or
                query in template.institution.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def filter_templates_by_type(self, template_type: str) -> List[ReportTemplate]:
        """
        Lọc mẫu theo loại.
        
        Parameters
        ----------
        template_type : str
            Loại mẫu (html, docx, pdf)
            
        Returns
        -------
        List[ReportTemplate]
            Danh sách các mẫu phù hợp
        """
        return [t for t in self.templates.values() if t.template_type.lower() == template_type.lower()]
    
    def get_default_template(self, template_type: str = "html") -> Optional[ReportTemplate]:
        """
        Lấy mẫu mặc định cho loại mẫu cụ thể.
        
        Parameters
        ----------
        template_type : str, optional
            Loại mẫu (html, docx, pdf), mặc định là "html"
            
        Returns
        -------
        Optional[ReportTemplate]
            Mẫu mặc định nếu có, None nếu không
        """
        templates = self.filter_templates_by_type(template_type)
        if not templates:
            return None
        
        # Tìm mẫu đánh dấu là mặc định
        for template in templates:
            if template.metadata.get("is_default", False):
                return template
        
        # Nếu không có mẫu mặc định, trả về mẫu đầu tiên
        return templates[0]
    
    def set_default_template(self, template_id: str) -> bool:
        """
        Đặt mẫu làm mẫu mặc định.
        
        Parameters
        ----------
        template_id : str
            ID của mẫu
            
        Returns
        -------
        bool
            True nếu thành công, False nếu không
        """
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        template_type = template.template_type
        
        # Xóa trạng thái mặc định của các mẫu khác cùng loại
        for t in self.templates.values():
            if t.template_type == template_type:
                if "is_default" in t.metadata:
                    t.metadata["is_default"] = False
        
        # Đặt mẫu này làm mặc định
        template.metadata["is_default"] = True
        
        # Lưu thay đổi
        self._save_templates()
        
        return True


# Tạo thể hiện singleton
_template_manager_instance = None

def get_template_manager(template_dir: Optional[str] = None) -> TemplateManager:
    """
    Lấy thể hiện trình quản lý mẫu.
    
    Parameters
    ----------
    template_dir : Optional[str], optional
        Thư mục chứa các tệp mẫu, mặc định là None
        
    Returns
    -------
    TemplateManager
        Thể hiện trình quản lý mẫu
    """
    global _template_manager_instance
    
    if _template_manager_instance is None:
        _template_manager_instance = TemplateManager(template_dir)
    
    return _template_manager_instance 