#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các hộp thoại (dialogs) cho giao diện người dùng.

Module này xuất các lớp hộp thoại được sử dụng trong giao diện người dùng
của hệ thống lập kế hoạch xạ trị QuangTPS.
"""

from quangtps.ui.dialogs.beam_dialog import BeamDialog
from quangtps.ui.dialogs.model_download_dialog import ModelDownloadDialog

__all__ = ['BeamDialog', 'ModelDownloadDialog']
