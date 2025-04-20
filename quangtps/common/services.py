#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for managing common services in QuangTPS.

This module imports core services and extends them with application-specific
functionality, such as patient management.
"""

import logging
import os
import json
import datetime
from typing import Dict, Any, List, Optional, Union, Type, TypeVar

from quangtps.core.services import ServiceBase, ServiceRegistry
from quangtps.common.models import Patient

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Get the singleton instance of the service registry
service_registry = ServiceRegistry.get_instance()


class PatientService(ServiceBase):
    """Service for managing patient data."""

    def __init__(self):
        """Initialize the patient service."""
        super().__init__()
        self._patients = {}
        self._data_directory = None

    def _initialize(self) -> bool:
        """Initialize the patient service.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        logger.info("Initializing patient service")
        try:
            # Set up data directory
            self._data_directory = os.path.join(
                os.path.expanduser("~"), ".quangtps", "patients"
            )
            os.makedirs(self._data_directory, exist_ok=True)

            # Load existing patients
            self._load_patients()

            logger.info("Patient service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing patient service: {str(e)}")
            return False

    def _shutdown(self) -> bool:
        """Shut down the patient service.

        Returns:
            bool: True if shutdown was successful, False otherwise.
        """
        logger.info("Shutting down patient service")
        try:
            # Save all patient data
            self._save_patients()

            logger.info("Patient service shut down successfully")
            return True
        except Exception as e:
            logger.error(f"Error shutting down patient service: {str(e)}")
            return False

    def _load_patients(self) -> None:
        """Load patients from the data directory."""
        try:
            if not os.path.exists(self._data_directory):
                logger.warning(
                    f"Patient data directory does not exist: {self._data_directory}"
                )
                return

            for filename in os.listdir(self._data_directory):
                if filename.endswith(".json"):
                    try:
                        filepath = os.path.join(self._data_directory, filename)
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            patient = Patient.from_dict(data)
                            self._patients[patient.id] = patient
                            logger.debug(f"Loaded patient: {patient.id}")
                    except Exception as e:
                        logger.error(
                            f"Error loading patient data from {filename}: {str(e)}"
                        )
        except Exception as e:
            logger.error(f"Error loading patients: {str(e)}")

    def _save_patients(self) -> None:
        """Save all patients to the data directory."""
        try:
            if not os.path.exists(self._data_directory):
                os.makedirs(self._data_directory, exist_ok=True)

            for patient_id, patient in self._patients.items():
                try:
                    filepath = os.path.join(self._data_directory, f"{patient_id}.json")
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(patient.to_dict(), f, indent=2)
                        logger.debug(f"Saved patient: {patient_id}")
                except Exception as e:
                    logger.error(
                        f"Error saving patient data for {patient_id}: {str(e)}"
                    )
        except Exception as e:
            logger.error(f"Error saving patients: {str(e)}")

    def add_patient(self, patient: Patient) -> bool:
        """Add a new patient.

        Args:
            patient (Patient): Patient object to add.

        Returns:
            bool: True if the patient was added successfully, False otherwise.
        """
        try:
            if patient.id in self._patients:
                logger.warning(f"Patient already exists: {patient.id}")
                return False

            self._patients[patient.id] = patient
            self._save_patient(patient)
            logger.info(f"Added patient: {patient.id}")
            return True
        except Exception as e:
            logger.error(f"Error adding patient: {str(e)}")
            return False

    def _save_patient(self, patient: Patient) -> bool:
        """Save a single patient to the data directory.

        Args:
            patient (Patient): Patient object to save.

        Returns:
            bool: True if the patient was saved successfully, False otherwise.
        """
        try:
            if not os.path.exists(self._data_directory):
                os.makedirs(self._data_directory, exist_ok=True)

            filepath = os.path.join(self._data_directory, f"{patient.id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(patient.to_dict(), f, indent=2)
                logger.debug(f"Saved patient: {patient.id}")
            return True
        except Exception as e:
            logger.error(f"Error saving patient data for {patient.id}: {str(e)}")
            return False

    def update_patient(self, patient: Patient) -> bool:
        """Update an existing patient.

        Args:
            patient (Patient): Patient object with updated data.

        Returns:
            bool: True if the patient was updated successfully, False otherwise.
        """
        try:
            if patient.id not in self._patients:
                logger.warning(f"Patient does not exist: {patient.id}")
                return False

            self._patients[patient.id] = patient
            self._save_patient(patient)
            logger.info(f"Updated patient: {patient.id}")
            return True
        except Exception as e:
            logger.error(f"Error updating patient: {str(e)}")
            return False

    def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient.

        Args:
            patient_id (str): ID of the patient to delete.

        Returns:
            bool: True if the patient was deleted successfully, False otherwise.
        """
        try:
            if patient_id not in self._patients:
                logger.warning(f"Patient does not exist: {patient_id}")
                return False

            del self._patients[patient_id]

            # Delete patient file
            filepath = os.path.join(self._data_directory, f"{patient_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Deleted patient file: {filepath}")

            logger.info(f"Deleted patient: {patient_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting patient: {str(e)}")
            return False

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get a patient by ID.

        Args:
            patient_id (str): ID of the patient to retrieve.

        Returns:
            Optional[Patient]: Patient object or None if not found.
        """
        return self._patients.get(patient_id)

    def get_all_patients(self) -> List[Patient]:
        """Get all patients.

        Returns:
            List[Patient]: List of all patient objects.
        """
        return list(self._patients.values())

    def search_patients(self, query: str) -> List[Patient]:
        """Search for patients by name, ID, or other attributes.

        Args:
            query (str): Search query.

        Returns:
            List[Patient]: List of matching patient objects.
        """
        query_lower = query.lower()
        results = []

        for patient in self._patients.values():
            # Check if the query matches patient ID, name, or MRN
            if (
                query_lower in patient.id.lower()
                or query_lower in patient.first_name.lower()
                or query_lower in patient.last_name.lower()
                or query_lower in patient.mrn.lower()
            ):
                results.append(patient)

        return results

    def get_patient_count(self) -> int:
        """Get the total number of patients.

        Returns:
            int: Total number of patients.
        """
        return len(self._patients)
