#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test đơn giản để kiểm tra import modules.
"""


def test_core_imports():
    """Test import các module core."""
    try:
        from quangtps.core import exceptions

        print("✓ Core exceptions: OK")
    except Exception as e:
        print(f"✗ Core exceptions: {e}")

    try:
        from quangtps.optimization.objectives import (
            ObjectiveType,
            ObjectiveBase,
            ObjectiveCollection,
        )

        print("✓ Objectives: OK")
    except Exception as e:
        print(f"✗ Objectives: {e}")

    try:
        from quangtps.dose.dose_engine import DoseEngine

        print("✓ Dose engine: OK")
    except Exception as e:
        print(f"✗ Dose engine: {e}")

    try:
        from quangtps.optimization.optimizers import OptimizerFactory

        print("✓ Optimizers: OK")
    except Exception as e:
        print(f"✗ Optimizers: {e}")


def test_algorithm_creation():
    """Test tạo algorithms."""
    try:
        from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm

        algorithm = PencilBeamAlgorithm()
        print(f"✓ PencilBeam created: {algorithm.name}")
    except Exception as e:
        print(f"✗ PencilBeam creation: {e}")


if __name__ == "__main__":
    print("=== SIMPLE QUANGTPS TEST ===")
    print("\n--- Core Imports ---")
    test_core_imports()
    print("\n--- Algorithm Creation ---")
    test_algorithm_creation()
    print("\n=== TEST COMPLETE ===")
