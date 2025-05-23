#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prescription Module

Module này cung cấp các class và utility functions để quản lý
prescription (kê đơn xạ trị) trong hệ thống QuangTPS.
"""

from .prescription import (
    # Core classes
    Prescription,
    TargetPrescription,
    DoseConstraint,
    PrescriptionMetadata,
    # Enums
    DoseUnit,
    FractionationScheme,
    TreatmentIntent,
    # Factory functions
    create_prescription,
    create_standard_prescription,
    get_standard_constraints,
    # Constants
    STANDARD_CONSTRAINTS,
)

__all__ = [
    "Prescription",
    "TargetPrescription",
    "DoseConstraint",
    "PrescriptionMetadata",
    "DoseUnit",
    "FractionationScheme",
    "TreatmentIntent",
    "create_prescription",
    "create_standard_prescription",
    "get_standard_constraints",
    "STANDARD_CONSTRAINTS",
]
