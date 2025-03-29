#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple implementation of the ImagingTab class
"""

import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

logger = logging.getLogger(__name__)

class ImagingTab(QWidget):
    """
    Simplified implementation of the ImagingTab for QuangTPS.
    This is a placeholder implementation to get the application running.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the imaging tab.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.info("Initializing ImagingTab (simple implementation)")
        
        layout = QVBoxLayout(self)
        
        # Add a placeholder label
        label = QLabel("Imaging Tab - Simplified Version")
        label.setStyleSheet("font-size: 16pt; color: #555;")
        layout.addWidget(label)
        
        # Add an explanation
        info = QLabel(
            "This is a simplified implementation of the ImagingTab.\n"
            "The original implementation likely has errors that prevent it from loading."
        )
        layout.addWidget(info)
        
        layout.addStretch(1)
