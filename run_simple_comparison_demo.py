#!/usr/bin/env python
"""
Run Simple Plan Comparison Demo

This script launches the simplified plan comparison demo that doesn't
depend on the full QuangTPS codebase.
"""

import sys
import os

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the demo
from quangtps.examples.simple_plan_comparison_demo import main

if __name__ == "__main__":
    main() 