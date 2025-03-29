#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the plan templates module.

This module contains tests for the treatment plan template functionality,
including creation, saving, and loading of templates.
"""

import os
import shutil
import tempfile
import unittest
import json
from unittest.mock import MagicMock, patch

import numpy as np

from quangtps.ui.templates.rt_plan_templates import (
    AnatomicalSite, TreatmentIntent, TreatmentTechnique,
    BeamArrangement, Prescription, PlanningObjective,
    get_beam_arrangement, get_prescription, get_planning_objectives,
    create_plan_from_template
)
from quangtps.ui.templates.template_manager import (
    TemplateManager, get_template_manager
)

class TestPlanTemplates(unittest.TestCase):
    """Test case for plan template functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for template storage
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock planning interface
        self.mock_interface = MagicMock()
        self.mock_interface.get_current_patient_id.return_value = "test_patient"
        self.mock_interface.get_current_ct_id.return_value = "test_ct"
        self.mock_interface.get_current_structure_set_id.return_value = "test_struct"
        
        # Patch the template manager's directory path
        patcher = patch.object(TemplateManager, 'user_templates_dir', 
                              new_callable=lambda: self.temp_dir)
        self.addCleanup(patcher.stop)
        patcher.start()
        
        # Create a template manager for testing
        self.template_manager = get_template_manager(self.mock_interface)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_beam_arrangement_creation(self):
        """Test creation of beam arrangements."""
        # Create a simple beam arrangement
        arrangement = BeamArrangement(
            name="Test Arrangement",
            technique=TreatmentTechnique.THREE_D_CRT,
            gantry_angles=[0, 90, 180, 270],
            collimator_angles=[0, 0, 0, 0],
            couch_angles=[0, 0, 0, 0],
            energies=["6X", "6X", "10X", "10X"],
            field_sizes=[(10, 10), (10, 10), (8, 12), (8, 12)]
        )
        
        # Verify arrangement properties
        self.assertEqual(arrangement.name, "Test Arrangement")
        self.assertEqual(arrangement.technique, TreatmentTechnique.THREE_D_CRT)
        self.assertEqual(len(arrangement.gantry_angles), 4)
        
        # Test beam parameter extraction
        params = arrangement.get_beam_parameters()
        self.assertEqual(len(params), 4)
        self.assertEqual(params[0]['gantry_angle'], 0)
        self.assertEqual(params[2]['energy'], "10X")
        self.assertEqual(params[3]['field_size'], (8, 12))
    
    def test_prescription_creation(self):
        """Test creation of prescription templates."""
        # Create a simple prescription
        prescription = Prescription(
            target_volume="PTV",
            total_dose=60.0,
            fractions=30,
            description="Test prescription"
        )
        
        # Verify prescription properties
        self.assertEqual(prescription.target_volume, "PTV")
        self.assertEqual(prescription.total_dose, 60.0)
        self.assertEqual(prescription.fractions, 30)
        self.assertEqual(prescription.dose_per_fraction, 2.0)
        
        # Test dictionary conversion
        prescription_dict = prescription.get_prescription_dict()
        self.assertEqual(prescription_dict['total_dose'], 60.0)
        self.assertEqual(prescription_dict['fractions'], 30)
    
    def test_planning_objective_creation(self):
        """Test creation of planning objectives."""
        # Create different types of objectives
        min_dose_obj = PlanningObjective(
            structure="PTV",
            objective_type=PlanningObjective.Type.MIN_DOSE,
            dose=57.0,
            priority=100.0
        )
        
        max_dvh_obj = PlanningObjective(
            structure="Rectum",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=50.0,
            volume=30.0,
            priority=80.0
        )
        
        # Verify objective properties
        self.assertEqual(min_dose_obj.structure, "PTV")
        self.assertEqual(min_dose_obj.objective_type, PlanningObjective.Type.MIN_DOSE)
        self.assertEqual(min_dose_obj.dose, 57.0)
        self.assertIsNone(min_dose_obj.volume)
        
        self.assertEqual(max_dvh_obj.structure, "Rectum")
        self.assertEqual(max_dvh_obj.objective_type, PlanningObjective.Type.MAX_DVH)
        self.assertEqual(max_dvh_obj.dose, 50.0)
        self.assertEqual(max_dvh_obj.volume, 30.0)
        
        # Test exception for missing volume
        with self.assertRaises(ValueError):
            PlanningObjective(
                structure="Rectum",
                objective_type=PlanningObjective.Type.MAX_DVH,
                dose=50.0,
                priority=80.0
            )
    
    def test_get_predefined_templates(self):
        """Test retrieving predefined templates."""
        # Test getting beam arrangements
        vmat_dual_arc = get_beam_arrangement("VMAT Dual Arc")
        self.assertIsNotNone(vmat_dual_arc)
        self.assertEqual(vmat_dual_arc.technique, TreatmentTechnique.VMAT)
        self.assertEqual(len(vmat_dual_arc.gantry_angles), 2)
        
        # Test getting prescriptions
        prostate_imrt = get_prescription("Prostate IMRT")
        self.assertIsNotNone(prostate_imrt)
        self.assertEqual(prostate_imrt.total_dose, 78.0)
        self.assertEqual(prostate_imrt.fractions, 39)
        
        # Test getting objectives
        prostate_objectives = get_planning_objectives("Prostate")
        self.assertIsNotNone(prostate_objectives)
        self.assertGreater(len(prostate_objectives), 0)
        
        # Test invalid names
        self.assertIsNone(get_beam_arrangement("NonExistentArrangement"))
        self.assertIsNone(get_prescription("NonExistentPrescription"))
        self.assertIsNone(get_planning_objectives("NonExistentSite"))
    
    def test_create_plan_from_template(self):
        """Test creating a plan from a template."""
        # Create a plan from a template
        with patch('quangtps.ui.templates.rt_plan_templates.Plan') as MockPlan:
            # Configure the mock
            mock_plan_instance = MockPlan.return_value
            mock_plan_instance.to_dict.return_value = {
                'name': 'Test Plan',
                'patient_id': 'test_patient',
                'ct_dataset_id': 'test_ct',
                'structure_set_id': 'test_struct',
                'prescription': {
                    'target_volume': 'PTV',
                    'total_dose': 78.0,
                    'fractions': 39
                },
                'beams': [
                    {'name': 'Beam 1', 'gantry_angle': 0.0},
                    {'name': 'Beam 2', 'gantry_angle': 51.0}
                ]
            }
            
            # Call the function
            plan_data = create_plan_from_template(
                "Prostate IMRT", "test_patient", "test_ct", "test_struct")
            
            # Verify that the plan was created correctly
            self.assertEqual(plan_data['name'], 'Test Plan')
            self.assertEqual(plan_data['patient_id'], 'test_patient')
            
            # Check that the necessary methods were called
            mock_plan_instance.add_prescription.assert_called_once()
            self.assertGreater(mock_plan_instance.add_beam.call_count, 0)
            mock_plan_instance.to_dict.assert_called_once()
    
    def test_template_manager_save_load(self):
        """Test saving and loading templates with the manager."""
        # Create a sample plan data
        plan_data = {
            'name': 'Sample Plan',
            'technique': 'IMRT',
            'prescription': {
                'target_volume': 'PTV',
                'total_dose': 70.0,
                'fractions': 35
            },
            'beams': [
                {
                    'name': 'Beam 1',
                    'gantry_angle': 0.0,
                    'collimator_angle': 0.0,
                    'couch_angle': 0.0,
                    'energy': '6X',
                    'field_size_x': 10.0,
                    'field_size_y': 10.0
                }
            ]
        }
        
        # Mock the UI interaction for template name input
        with patch('PyQt5.QtWidgets.QInputDialog.getText') as mock_get_text:
            mock_get_text.return_value = ("Test Template", True)
            
            # Save the template
            result = self.template_manager.save_as_template(plan_data)
            self.assertTrue(result)
            
            # Check if file was created
            template_file = os.path.join(self.temp_dir, "Test_Template.json")
            self.assertTrue(os.path.exists(template_file))
            
            # Verify file contents
            with open(template_file, 'r') as f:
                template_data = json.load(f)
                self.assertEqual(template_data['name'], 'Sample Plan')
                self.assertEqual(
                    template_data['prescription']['total_dose'], 70.0)
    
    def test_template_manager_apply_template(self):
        """Test applying a template to create a plan."""
        # Set up mock for create_plan_from_template
        with patch('quangtps.ui.templates.template_manager.create_plan_from_template') as mock_create:
            mock_create.return_value = {
                'name': 'Test Plan',
                'prescription': {'total_dose': 78.0}
            }
            
            # Set up mock for interface.load_plan_from_data
            self.mock_interface.load_plan_from_data.return_value = True
            
            # Apply the template
            result = self.template_manager.apply_template_to_plan("Prostate IMRT")
            
            # Verify results
            self.assertTrue(result)
            mock_create.assert_called_once_with(
                "Prostate IMRT", "test_patient", "test_ct", "test_struct")
            self.mock_interface.load_plan_from_data.assert_called_once()


if __name__ == '__main__':
    unittest.main() 