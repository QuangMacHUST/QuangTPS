import os
import datetime
from typing import Dict, List, Optional, Any, Tuple

from quangtps.core.logging import get_logger
from quangtps.core.services import ServiceRegistry
from quangtps.database.patient_db import PatientDB
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.core.config import Config

logger = get_logger(__name__)

class RTAdministration:
    """
    Eclipse-like RT Administration tools for managing radiotherapy department resources,
    treatment machines, user access, and system settings.
    """
    
    def __init__(self):
        """Initialize the RT Administration module."""
        try:
            self.patient_db = ServiceRegistry.get('patient_database')
            self.config = ServiceRegistry.get('config')
        except Exception as e:
            logger.warning(f"Error accessing ServiceRegistry: {e}")
            from quangtps.database.patient_db import PatientDatabase
            from quangtps.core.config import Config
            self.patient_db = PatientDatabase()
            self.config = Config()
        self._load_resources()
        
    def _load_resources(self):
        """Load treatment machines and other resources."""
        self.treatment_machines = []
        self.users = []
        self.user_groups = []
        self.approval_templates = []
        
        # This would be loaded from configuration files
        # For now, just placeholder data
        
    def get_machine_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all treatment machines in the department.
        
        Returns:
            List of treatment machine dictionaries
        """
        machines = []
        
        # Mock data - would be loaded from database in real implementation
        machines.append({
            "id": "TB1",
            "name": "TrueBeam 1",
            "type": "TrueBeam",
            "serial_number": "TB12345",
            "institution": "Main Hospital",
            "room": "L1-01",
            "status": "Available",
            "energies": ["6X", "10X", "15X", "6FFF", "10FFF", "6E", "9E", "12E", "15E", "20E"],
            "has_cbct": True,
            "has_6dof_couch": True
        })
        
        machines.append({
            "id": "TB2",
            "name": "TrueBeam 2",
            "type": "TrueBeam",
            "serial_number": "TB12346",
            "institution": "Main Hospital",
            "room": "L1-02",
            "status": "Available",
            "energies": ["6X", "10X", "15X", "6FFF", "10FFF", "6E", "9E", "12E", "15E"],
            "has_cbct": True,
            "has_6dof_couch": True
        })
        
        machines.append({
            "id": "HAL1",
            "name": "Halcyon 1",
            "type": "Halcyon",
            "serial_number": "HAL5678",
            "institution": "Main Hospital",
            "room": "L1-03",
            "status": "Maintenance",
            "energies": ["6X", "6FFF"],
            "has_cbct": True,
            "has_6dof_couch": False
        })
        
        return machines
    
    def get_machine_details(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific treatment machine.
        
        Args:
            machine_id: ID of the treatment machine
            
        Returns:
            Dictionary with machine details or None if not found
        """
        machines = self.get_machine_list()
        for machine in machines:
            if machine["id"] == machine_id:
                return machine
        return None
    
    def set_machine_status(self, machine_id: str, status: str) -> bool:
        """
        Update the status of a treatment machine.
        
        Args:
            machine_id: ID of the treatment machine
            status: New status ("Available", "Maintenance", "Offline")
            
        Returns:
            True if successful, False otherwise
        """
        # This would update the status in the database
        logger.info(f"Setting machine {machine_id} status to {status}")
        return True
    
    def get_user_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all system users.
        
        Returns:
            List of user dictionaries
        """
        users = []
        
        # Mock data - would be loaded from database in real implementation
        users.append({
            "id": "user1",
            "username": "jsmith",
            "full_name": "John Smith",
            "role": "Radiation Oncologist",
            "email": "jsmith@hospital.org",
            "groups": ["Physicians", "Plan Approvers"],
            "is_active": True,
            "last_login": datetime.datetime(2023, 3, 15, 9, 30, 0)
        })
        
        users.append({
            "id": "user2",
            "username": "bjones",
            "full_name": "Barbara Jones",
            "role": "Medical Physicist",
            "email": "bjones@hospital.org",
            "groups": ["Physicists", "Plan Approvers", "QA Approvers"],
            "is_active": True,
            "last_login": datetime.datetime(2023, 3, 20, 14, 15, 0)
        })
        
        users.append({
            "id": "user3",
            "username": "rwilson",
            "full_name": "Robert Wilson",
            "role": "Dosimetrist",
            "email": "rwilson@hospital.org",
            "groups": ["Dosimetrists", "Plan Creators"],
            "is_active": True,
            "last_login": datetime.datetime(2023, 3, 22, 11, 45, 0)
        })
        
        users.append({
            "id": "user4",
            "username": "lthomas",
            "full_name": "Lisa Thomas",
            "role": "Radiation Therapist",
            "email": "lthomas@hospital.org",
            "groups": ["Therapists", "Image Reviewers"],
            "is_active": False,
            "last_login": datetime.datetime(2023, 1, 5, 10, 0, 0)
        })
        
        return users
    
    def get_user_details(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with user details or None if not found
        """
        users = self.get_user_list()
        for user in users:
            if user["id"] == user_id:
                return user
        return None
    
    def set_user_active_status(self, user_id: str, is_active: bool) -> bool:
        """
        Update the active status of a user.
        
        Args:
            user_id: ID of the user
            is_active: Whether the user should be active
            
        Returns:
            True if successful, False otherwise
        """
        # This would update the status in the database
        logger.info(f"Setting user {user_id} active status to {is_active}")
        return True
    
    def get_user_group_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all user groups.
        
        Returns:
            List of user group dictionaries
        """
        groups = []
        
        # Mock data - would be loaded from database in real implementation
        groups.append({
            "id": "group1",
            "name": "Physicians",
            "description": "Radiation Oncologists",
            "permissions": ["view_patients", "approve_plans", "create_prescriptions"]
        })
        
        groups.append({
            "id": "group2",
            "name": "Physicists",
            "description": "Medical Physicists",
            "permissions": ["view_patients", "approve_plans", "approve_qa", "machine_calibration"]
        })
        
        groups.append({
            "id": "group3",
            "name": "Dosimetrists",
            "description": "Treatment Planners",
            "permissions": ["view_patients", "create_plans", "create_fields"]
        })
        
        groups.append({
            "id": "group4",
            "name": "Therapists",
            "description": "Radiation Therapists",
            "permissions": ["view_patients", "view_treatments", "approve_images"]
        })
        
        groups.append({
            "id": "group5",
            "name": "Plan Approvers",
            "description": "Users who can approve treatment plans",
            "permissions": ["approve_plans"]
        })
        
        return groups
    
    def get_approval_template_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all approval templates.
        
        Returns:
            List of approval template dictionaries
        """
        templates = []
        
        # Mock data - would be loaded from database in real implementation
        templates.append({
            "id": "template1",
            "name": "Standard Treatment Plan Approval",
            "description": "Standard approval process for treatment plans",
            "steps": [
                {"role": "Dosimetrist", "action": "Create", "required": True},
                {"role": "Physicist", "action": "Physics Check", "required": True},
                {"role": "Physician", "action": "Final Approval", "required": True}
            ]
        })
        
        templates.append({
            "id": "template2",
            "name": "SBRT Plan Approval",
            "description": "Approval process for stereotactic body radiation therapy plans",
            "steps": [
                {"role": "Dosimetrist", "action": "Create", "required": True},
                {"role": "Physicist", "action": "Physics Check", "required": True},
                {"role": "Physicist", "action": "SBRT QA Check", "required": True},
                {"role": "Physician", "action": "Final Approval", "required": True}
            ]
        })
        
        templates.append({
            "id": "template3",
            "name": "Weekly Chart Check",
            "description": "Weekly chart checking process",
            "steps": [
                {"role": "Physicist", "action": "Chart Check", "required": True},
                {"role": "Physicist", "action": "Document", "required": True}
            ]
        })
        
        return templates
    
    def get_department_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about department activity.
        
        Returns:
            Dictionary with department statistics
        """
        stats = {}
        
        # This would be calculated from actual data in the database
        # For now, just placeholder data
        
        # Treatment statistics
        stats["treatments"] = {
            "total_patients": 120,
            "active_patients": 75,
            "completed_patients": 45,
            "treatments_today": 42,
            "treatments_this_week": 210,
            "treatments_this_month": 840
        }
        
        # Plan statistics
        stats["plans"] = {
            "total_plans": 150,
            "approved_plans": 130,
            "draft_plans": 20,
            "plans_created_this_week": 15,
            "plans_by_type": {
                "IMRT": 65,
                "VMAT": 45,
                "3D Conformal": 25,
                "SBRT": 10,
                "SRS": 5
            }
        }
        
        # Machine statistics
        stats["machines"] = {
            "total_machines": 3,
            "available_machines": 2,
            "maintenance_machines": 1,
            "utilization": {
                "TB1": 85.0,  # percentage
                "TB2": 92.0,
                "HAL1": 0.0
            },
            "treatments_by_machine": {
                "TB1": 22,
                "TB2": 20,
                "HAL1": 0
            }
        }
        
        # QA statistics
        stats["qa"] = {
            "qa_completed_this_week": 12,
            "qa_pending": 3,
            "qa_failures": 1,
            "machine_qa_status": {
                "TB1": "Pass",
                "TB2": "Pass",
                "HAL1": "In Progress"
            }
        }
        
        return stats
    
    def backup_data(self, backup_path: str) -> bool:
        """
        Create a backup of all clinical data.
        
        Args:
            backup_path: Path to store the backup
            
        Returns:
            True if successful, False otherwise
        """
        # This would create a backup of the database and configuration
        logger.info(f"Creating backup at {backup_path}")
        
        # Placeholder implementation
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Create a dummy backup file
            with open(backup_path, 'w') as f:
                f.write(f"Backup created at {datetime.datetime.now()}")
                
            return True
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return False
    
    def restore_data(self, backup_path: str) -> bool:
        """
        Restore clinical data from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if successful, False otherwise
        """
        # This would restore the database and configuration from a backup
        logger.info(f"Restoring from backup at {backup_path}")
        
        # Placeholder implementation
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
                
            # In a real implementation, this would restore the data
            return True
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False
    
    def get_system_logs(self, start_date: datetime.datetime = None, 
                       end_date: datetime.datetime = None,
                       log_level: str = "INFO",
                       max_entries: int = 100) -> List[Dict[str, Any]]:
        """
        Get system logs within a specified time range.
        
        Args:
            start_date: Start date for log entries (None for no limit)
            end_date: End date for log entries (None for no limit)
            log_level: Minimum log level to include ("DEBUG", "INFO", "WARNING", "ERROR")
            max_entries: Maximum number of log entries to return
            
        Returns:
            List of log entry dictionaries
        """
        logs = []
        
        # This would query the logs from the database or log files
        # For now, just placeholder data
        
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        min_level_index = log_levels.index(log_level)
        
        # Generate some sample log entries
        now = datetime.datetime.now()
        for i in range(20):
            timestamp = now - datetime.timedelta(hours=i)
            
            if start_date and timestamp < start_date:
                continue
                
            if end_date and timestamp > end_date:
                continue
                
            level = log_levels[min(min_level_index + i % 4, 3)]
            
            if log_levels.index(level) < min_level_index:
                continue
                
            logs.append({
                "timestamp": timestamp,
                "level": level,
                "source": "RTAdministration" if i % 3 == 0 else "PatientDB" if i % 3 == 1 else "TreatmentPlanner",
                "message": f"Sample log message {i}",
                "user": f"user{(i % 4) + 1}"
            })
            
            if len(logs) >= max_entries:
                break
                
        # Sort logs by timestamp (newest first)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return logs
    
    def get_license_status(self) -> Dict[str, Any]:
        """
        Get information about system licenses.
        
        Returns:
            Dictionary with license information
        """
        # This would query the license server or configuration
        # For now, just placeholder data
        
        return {
            "product": "QuangTPS Treatment Planning System",
            "version": "1.2.0",
            "license_type": "Enterprise",
            "licensed_to": "Main Hospital",
            "expiration_date": datetime.date(2024, 12, 31),
            "features": {
                "IMRT": True,
                "VMAT": True,
                "SRS/SBRT": True,
                "4D Planning": True,
                "Monte Carlo": True,
                "Scripting": True,
                "API Access": True
            },
            "concurrent_users": {
                "licensed": 10,
                "currently_active": 3
            }
        }


class QAManagement:
    """
    Tools for managing quality assurance procedures and results.
    """
    
    def __init__(self):
        """Initialize the QA Management module."""
        pass
        
    def get_machine_qa_schedule(self, machine_id: str = None) -> List[Dict[str, Any]]:
        """
        Get the QA schedule for treatment machines.
        
        Args:
            machine_id: ID of a specific machine (None for all machines)
            
        Returns:
            List of QA schedule dictionaries
        """
        qa_schedule = []
        
        # Mock data - would be loaded from database in real implementation
        qa_schedule.append({
            "id": "qa1",
            "machine_id": "TB1",
            "type": "Daily",
            "name": "Morning QA",
            "description": "Daily morning QA checks",
            "due_date": datetime.date.today(),
            "assigned_to": "user2",
            "status": "Pending"
        })
        
        qa_schedule.append({
            "id": "qa2",
            "machine_id": "TB1",
            "type": "Monthly",
            "name": "Output Calibration",
            "description": "Monthly output calibration",
            "due_date": datetime.date.today() + datetime.timedelta(days=10),
            "assigned_to": "user2",
            "status": "Scheduled"
        })
        
        qa_schedule.append({
            "id": "qa3",
            "machine_id": "TB2",
            "type": "Daily",
            "name": "Morning QA",
            "description": "Daily morning QA checks",
            "due_date": datetime.date.today(),
            "assigned_to": "user2",
            "status": "Completed",
            "completion_date": datetime.datetime.now() - datetime.timedelta(hours=2),
            "result": "Pass"
        })
        
        qa_schedule.append({
            "id": "qa4",
            "machine_id": "HAL1",
            "type": "Annual",
            "name": "Annual Calibration",
            "description": "Annual full calibration",
            "due_date": datetime.date.today() + datetime.timedelta(days=30),
            "assigned_to": "user2",
            "status": "Scheduled"
        })
        
        # Filter by machine_id if specified
        if machine_id:
            qa_schedule = [qa for qa in qa_schedule if qa["machine_id"] == machine_id]
            
        return qa_schedule
    
    def get_patient_qa_schedule(self, patient_id: str = None) -> List[Dict[str, Any]]:
        """
        Get the QA schedule for patient-specific QA.
        
        Args:
            patient_id: ID of a specific patient (None for all patients)
            
        Returns:
            List of QA schedule dictionaries
        """
        qa_schedule = []
        
        # Mock data - would be loaded from database in real implementation
        qa_schedule.append({
            "id": "pqa1",
            "patient_id": "patient1",
            "plan_id": "plan1",
            "type": "Patient-Specific",
            "name": "IMRT QA",
            "description": "IMRT plan verification",
            "due_date": datetime.date.today(),
            "assigned_to": "user2",
            "status": "Pending"
        })
        
        qa_schedule.append({
            "id": "pqa2",
            "patient_id": "patient2",
            "plan_id": "plan2",
            "type": "Patient-Specific",
            "name": "VMAT QA",
            "description": "VMAT plan verification",
            "due_date": datetime.date.today() - datetime.timedelta(days=1),
            "assigned_to": "user2",
            "status": "Completed",
            "completion_date": datetime.datetime.now() - datetime.timedelta(days=1, hours=3),
            "result": "Pass",
            "gamma_index": 98.7
        })
        
        qa_schedule.append({
            "id": "pqa3",
            "patient_id": "patient3",
            "plan_id": "plan3",
            "type": "Patient-Specific",
            "name": "SBRT QA",
            "description": "SBRT plan verification",
            "due_date": datetime.date.today() + datetime.timedelta(days=2),
            "assigned_to": "user2",
            "status": "Scheduled"
        })
        
        # Filter by patient_id if specified
        if patient_id:
            qa_schedule = [qa for qa in qa_schedule if qa["patient_id"] == patient_id]
            
        return qa_schedule
    
    def complete_qa_task(self, qa_id: str, result: str, 
                        completion_date: datetime.datetime = None,
                        notes: str = "",
                        measurements: Dict[str, Any] = None) -> bool:
        """
        Mark a QA task as completed.
        
        Args:
            qa_id: ID of the QA task
            result: Result of the QA ("Pass", "Fail", "Conditional Pass")
            completion_date: Date of completion (None for current datetime)
            notes: Additional notes about the QA
            measurements: Dictionary of measurement values
            
        Returns:
            True if successful, False otherwise
        """
        # This would update the QA status in the database
        logger.info(f"Completing QA task {qa_id} with result {result}")
        
        # Use current datetime if not specified
        if completion_date is None:
            completion_date = datetime.datetime.now()
            
        # Placeholder implementation
        return True
    
    def get_qa_templates(self) -> List[Dict[str, Any]]:
        """
        Get a list of QA templates.
        
        Returns:
            List of QA template dictionaries
        """
        templates = []
        
        # Mock data - would be loaded from database in real implementation
        templates.append({
            "id": "qatemplate1",
            "name": "Daily Linac QA",
            "description": "Daily quality assurance for linear accelerators",
            "machine_types": ["TrueBeam", "VitalBeam", "Clinac"],
            "frequency": "Daily",
            "tasks": [
                {
                    "name": "Output Constancy",
                    "description": "Check output constancy",
                    "expected_value": 100.0,
                    "tolerance": 2.0,
                    "unit": "%"
                },
                {
                    "name": "Laser Alignment",
                    "description": "Check laser alignment",
                    "expected_value": 0.0,
                    "tolerance": 1.0,
                    "unit": "mm"
                },
                {
                    "name": "Optical Distance Indicator",
                    "description": "Check ODI accuracy",
                    "expected_value": 100.0,
                    "tolerance": 2.0,
                    "unit": "cm"
                }
            ]
        })
        
        templates.append({
            "id": "qatemplate2",
            "name": "Monthly Linac QA",
            "description": "Monthly quality assurance for linear accelerators",
            "machine_types": ["TrueBeam", "VitalBeam", "Clinac"],
            "frequency": "Monthly",
            "tasks": [
                {
                    "name": "Output Calibration",
                    "description": "Check and calibrate output",
                    "expected_value": 1.0,
                    "tolerance": 0.01,
                    "unit": "cGy/MU"
                },
                {
                    "name": "Field Symmetry",
                    "description": "Check field symmetry",
                    "expected_value": 0.0,
                    "tolerance": 2.0,
                    "unit": "%"
                },
                {
                    "name": "Field Flatness",
                    "description": "Check field flatness",
                    "expected_value": 0.0,
                    "tolerance": 3.0,
                    "unit": "%"
                },
                {
                    "name": "MLC Position",
                    "description": "Check MLC positioning accuracy",
                    "expected_value": 0.0,
                    "tolerance": 1.0,
                    "unit": "mm"
                }
            ]
        })
        
        templates.append({
            "id": "qatemplate3",
            "name": "IMRT Patient QA",
            "description": "Patient-specific QA for IMRT plans",
            "plan_types": ["IMRT"],
            "frequency": "Per Patient",
            "tasks": [
                {
                    "name": "Gamma Analysis",
                    "description": "Gamma index analysis",
                    "expected_value": 95.0,
                    "tolerance": 5.0,
                    "unit": "%",
                    "criteria": "3%/3mm"
                },
                {
                    "name": "Point Dose",
                    "description": "High dose point comparison",
                    "expected_value": 0.0,
                    "tolerance": 3.0,
                    "unit": "%"
                }
            ]
        })
        
        return templates 