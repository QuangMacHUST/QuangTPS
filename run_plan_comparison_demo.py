#!/usr/bin/env python
"""
Run Plan Comparison Demo

This script launches the plan comparison demo application.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the demo
from quangtps.examples.plan_comparison_demo import main

if __name__ == "__main__":
    main() 