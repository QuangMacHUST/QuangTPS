#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collision detection module for QuangTPS.

This module provides functionality for detecting and visualizing
potential collisions between the gantry, couch, and patient during
treatment delivery, similar to the Eclipse collision detection tool.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
import json
import math
from scipy.spatial.transform import Rotation as R

from quangtps.core.types import Plan, Patient, Structure, Beam, ControlPoint
from quangtps.core.geometry import Point3D, Vector3D, Transform3D
from quangtps.geometry.isocenter import get_isocenter
from quangtps.core.logging import get_logger
from quangtps.core.config import Config

logger = get_logger(__name__)

# Machine geometry constants for TrueBeam
# These values are approximations and should be calibrated for each specific machine
MACHINE_GEOMETRIES = {
    "TrueBeam": {
        "gantry_radius": 1000.0,  # mm
        "gantry_width": 470.0,    # mm
        "collimator_length": 400.0, # mm
        "min_couch_height": -400.0, # mm
        "max_couch_height": 200.0,  # mm
        "couch_width": 530.0,      # mm
        "couch_length": 2400.0,    # mm
        "couch_thickness": 70.0,   # mm
        "detector_radius": 800.0,  # mm
        "detector_width": 300.0,   # mm
        "detector_height": 300.0,  # mm
    },
    "VitalBeam": {
        "gantry_radius": 1000.0,  # mm
        "gantry_width": 470.0,    # mm
        "collimator_length": 400.0, # mm
        "min_couch_height": -400.0, # mm
        "max_couch_height": 200.0,  # mm
        "couch_width": 530.0,      # mm
        "couch_length": 2400.0,    # mm
        "couch_thickness": 70.0,   # mm
        "detector_radius": 800.0,  # mm
        "detector_width": 300.0,   # mm
        "detector_height": 300.0,  # mm
    },
    "Halcyon": {
        "gantry_radius": 900.0,   # mm
        "gantry_width": 1000.0,   # mm (Halcyon has a wider bore)
        "collimator_length": 350.0, # mm
        "min_couch_height": -300.0, # mm
        "max_couch_height": 200.0,  # mm
        "couch_width": 530.0,      # mm
        "couch_length": 2400.0,    # mm
        "couch_thickness": 70.0,   # mm
        "detector_radius": 800.0,  # mm
        "detector_width": 300.0,   # mm
        "detector_height": 300.0,  # mm
    }
}

class CollisionVolume:
    """
    Represents a 3D volume used for collision detection.
    
    This could represent the patient body, gantry, couch, etc.
    """
    
    def __init__(self, name: str, type_name: str):
        """
        Initialize a collision volume.
        
        Args:
            name: Name of the volume (e.g., "Gantry", "Patient", "Couch")
            type_name: Type of volume (e.g., "cylinder", "box", "mesh")
        """
        self.name = name
        self.type = type_name
        self.transform = Transform3D()  # Initial transform (identity)
        
        # Specific parameters depending on the type
        self.parameters = {}
    
    def set_transform(self, transform: Transform3D):
        """
        Set the transformation matrix for this volume.
        
        Args:
            transform: The transformation to apply
        """
        self.transform = transform
    
    def translate(self, dx: float, dy: float, dz: float):
        """
        Translate the volume.
        
        Args:
            dx: Translation along X axis
            dy: Translation along Y axis
            dz: Translation along Z axis
        """
        translation = Transform3D()
        translation.translate(dx, dy, dz)
        self.transform = translation * self.transform
    
    def rotate(self, axis: str, angle_deg: float):
        """
        Rotate the volume around an axis.
        
        Args:
            axis: Axis of rotation ("x", "y", or "z")
            angle_deg: Rotation angle in degrees
        """
        rotation = Transform3D()
        if axis.lower() == "x":
            rotation.rotate_x(math.radians(angle_deg))
        elif axis.lower() == "y":
            rotation.rotate_y(math.radians(angle_deg))
        elif axis.lower() == "z":
            rotation.rotate_z(math.radians(angle_deg))
        else:
            logger.warning(f"Invalid rotation axis: {axis}")
            return
        
        self.transform = rotation * self.transform
    
    def to_dict(self) -> Dict:
        """
        Convert the volume to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the volume
        """
        return {
            "name": self.name,
            "type": self.type,
            "transform": self.transform.to_list(),
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CollisionVolume':
        """
        Create a collision volume from a dictionary.
        
        Args:
            data: Dictionary containing volume data
        
        Returns:
            CollisionVolume object
        """
        volume = cls(data["name"], data["type"])
        volume.transform = Transform3D.from_list(data["transform"])
        volume.parameters = data["parameters"]
        return volume
    
    @classmethod
    def create_cylinder(cls, name: str, radius: float, height: float, 
                       center: Tuple[float, float, float] = (0, 0, 0),
                       axis: str = "z") -> 'CollisionVolume':
        """
        Create a cylindrical collision volume.
        
        Args:
            name: Name of the volume
            radius: Radius of the cylinder
            height: Height of the cylinder
            center: Center point of the cylinder
            axis: Axis along which the cylinder extends ("x", "y", or "z")
        
        Returns:
            CollisionVolume object representing a cylinder
        """
        volume = cls(name, "cylinder")
        volume.parameters = {
            "radius": radius,
            "height": height,
            "axis": axis
        }
        
        # Position the cylinder at the specified center
        volume.translate(center[0], center[1], center[2])
        
        # Rotate the cylinder if needed (default is along z-axis)
        if axis.lower() == "x":
            volume.rotate("y", 90)
        elif axis.lower() == "y":
            volume.rotate("x", 90)
        
        return volume
    
    @classmethod
    def create_box(cls, name: str, width: float, height: float, depth: float,
                  center: Tuple[float, float, float] = (0, 0, 0)) -> 'CollisionVolume':
        """
        Create a box collision volume.
        
        Args:
            name: Name of the volume
            width: Width of the box (x dimension)
            height: Height of the box (y dimension)
            depth: Depth of the box (z dimension)
            center: Center point of the box
        
        Returns:
            CollisionVolume object representing a box
        """
        volume = cls(name, "box")
        volume.parameters = {
            "width": width,
            "height": height,
            "depth": depth
        }
        
        # Position the box at the specified center
        volume.translate(center[0], center[1], center[2])
        
        return volume
    
    @classmethod
    def create_from_structure(cls, name: str, structure: Structure) -> 'CollisionVolume':
        """
        Create a collision volume from a patient structure.
        
        Args:
            name: Name of the volume
            structure: Structure object to use
        
        Returns:
            CollisionVolume object representing the structure
        """
        volume = cls(name, "structure")
        volume.parameters = {
            "structure_id": structure.id,
            "structure_name": structure.name,
            # Store a simplified representation for quick collision checks
            "bounds": structure.get_bounds(),
            # We could also store a simplified mesh or other representation
        }
        
        return volume

class CollisionDetector:
    """
    Collision detection system for radiotherapy treatment planning.
    
    This class provides methods for detecting and visualizing potential
    collisions between the gantry, couch, and patient during treatment delivery.
    """
    
    def __init__(self, machine_type: str = "TrueBeam"):
        """
        Initialize the collision detector.
        
        Args:
            machine_type: Type of treatment machine (e.g., "TrueBeam", "VitalBeam")
        """
        self.machine_type = machine_type
        self.machine_geometry = MACHINE_GEOMETRIES.get(machine_type, MACHINE_GEOMETRIES["TrueBeam"])
        
        # Create machine components
        self.create_machine_components()
        
        # Patient volume (will be set later)
        self.patient_volume = None
        
        # Isocenter position (will be set later)
        self.isocenter = Point3D(0, 0, 0)
        
        # Threshold distance for collision warnings (mm)
        self.collision_threshold = 50.0
        self.warning_threshold = 100.0
    
    def create_machine_components(self):
        """Create the collision volumes for machine components."""
        self.volumes = {}
        
        # Gantry head (cylindrical approximation)
        self.volumes["gantry"] = CollisionVolume.create_cylinder(
            "Gantry", 
            self.machine_geometry["gantry_radius"],
            self.machine_geometry["gantry_width"],
            (0, 0, 0),
            "x"  # Aligned with x-axis
        )
        
        # Collimator (box approximation)
        collimator_width = 200.0
        self.volumes["collimator"] = CollisionVolume.create_box(
            "Collimator",
            collimator_width,
            collimator_width,
            self.machine_geometry["collimator_length"],
            (0, 0, 0)
        )
        
        # Treatment couch (box approximation)
        self.volumes["couch"] = CollisionVolume.create_box(
            "Couch",
            self.machine_geometry["couch_width"],
            self.machine_geometry["couch_thickness"],
            self.machine_geometry["couch_length"],
            (0, -self.machine_geometry["couch_thickness"]/2, 0)
        )
        
        # Detector (if present, e.g., for imaging)
        self.volumes["detector"] = CollisionVolume.create_box(
            "Detector",
            self.machine_geometry["detector_width"],
            self.machine_geometry["detector_height"],
            50.0,  # Detector thickness
            (0, 0, -self.machine_geometry["detector_radius"])
        )
    
    def set_patient(self, patient: Patient, structure_name: str = "BODY"):
        """
        Set the patient for collision detection.
        
        Args:
            patient: Patient object
            structure_name: Name of structure to use for patient volume (default: "BODY")
        
        Returns:
            True if successful, False otherwise
        """
        # Find the requested structure
        structure = None
        for s in patient.get_structures():
            if s.name.upper() == structure_name.upper():
                structure = s
                break
        
        if not structure:
            logger.warning(f"Structure '{structure_name}' not found for patient")
            return False
        
        # Create a collision volume from the structure
        self.patient_volume = CollisionVolume.create_from_structure("Patient", structure)
        self.volumes["patient"] = self.patient_volume
        
        return True
    
    def set_plan(self, plan: Plan):
        """
        Set the treatment plan for collision detection.
        
        Args:
            plan: Treatment plan
        """
        # Get the isocenter from the plan
        self.isocenter = get_isocenter(plan)
        
        # Update machine components based on isocenter
        self.update_machine_components()
    
    def update_machine_components(self):
        """Update the positions of machine components based on isocenter."""
        # Position all components relative to the isocenter
        iso_x, iso_y, iso_z = self.isocenter.x, self.isocenter.y, self.isocenter.z
        
        # Reset transforms
        self.create_machine_components()
        
        # Translate all components to isocenter
        for name, volume in self.volumes.items():
            if name != "patient":  # Don't move the patient
                volume.translate(iso_x, iso_y, iso_z)
    
    def position_gantry_and_couch(self, gantry_angle: float, couch_angle: float, 
                                 collimator_angle: float = 0.0):
        """
        Position the gantry and couch for collision detection.
        
        Args:
            gantry_angle: Gantry angle in degrees (IEC standard)
            couch_angle: Couch angle in degrees (IEC standard)
            collimator_angle: Collimator angle in degrees (IEC standard)
        """
        # Reset positions first
        self.update_machine_components()
        
        # Rotate the gantry
        # In IEC standard, gantry angle 0 is at the top, increasing clockwise
        # when viewed from the foot of the table
        self.volumes["gantry"].rotate("z", gantry_angle)
        
        # Position collimator at the end of the gantry and rotate it
        collimator = self.volumes["collimator"]
        gantry_radius = self.machine_geometry["gantry_radius"]
        
        # Calculate collimator position based on gantry angle
        rad_angle = math.radians(gantry_angle)
        collimator_x = gantry_radius * math.sin(rad_angle)
        collimator_y = -gantry_radius * math.cos(rad_angle)
        
        # Reset collimator position
        self.volumes["collimator"] = CollisionVolume.create_box(
            "Collimator",
            200.0,  # Width
            200.0,  # Height
            self.machine_geometry["collimator_length"],
            (0, 0, 0)
        )
        
        # Position and rotate the collimator
        collimator = self.volumes["collimator"]
        collimator.translate(collimator_x, collimator_y, 0)
        collimator.rotate("z", gantry_angle)
        collimator.rotate("x", collimator_angle)
        
        # Rotate the couch
        # In IEC standard, couch angle 0 is aligned with the gantry,
        # increasing counterclockwise when viewed from above
        self.volumes["couch"].rotate("y", couch_angle)
        
        # Position the detector opposite to the gantry (if imaging is performed)
        detector_x = -gantry_radius * math.sin(rad_angle)
        detector_y = gantry_radius * math.cos(rad_angle)
        
        # Reset detector position
        self.volumes["detector"] = CollisionVolume.create_box(
            "Detector",
            self.machine_geometry["detector_width"],
            self.machine_geometry["detector_height"],
            50.0,  # Detector thickness
            (0, 0, 0)
        )
        
        # Position and rotate the detector
        detector = self.volumes["detector"]
        detector.translate(detector_x, detector_y, 0)
        detector.rotate("z", gantry_angle + 180)  # Opposite to gantry
    
    def check_collision(self, gantry_angle: float, couch_angle: float, 
                       collimator_angle: float = 0.0) -> Dict:
        """
        Check for potential collisions at the given angles.
        
        Args:
            gantry_angle: Gantry angle in degrees
            couch_angle: Couch angle in degrees
            collimator_angle: Collimator angle in degrees
        
        Returns:
            Dictionary with collision information:
            {
                "collision_detected": bool,
                "warning": bool,
                "min_distance": float,
                "colliding_components": List[Tuple[str, str, float]]
            }
        """
        # Position the components
        self.position_gantry_and_couch(gantry_angle, couch_angle, collimator_angle)
        
        # Check for collisions
        result = {
            "collision_detected": False,
            "warning": False,
            "min_distance": float('inf'),
            "colliding_components": []
        }
        
        if not self.patient_volume:
            # Can't check collisions without patient data
            result["min_distance"] = -1
            return result
        
        # Check each machine component against the patient
        for name, volume in self.volumes.items():
            if name == "patient":
                continue
            
            distance = self._calculate_distance(volume, self.patient_volume)
            
            if distance < result["min_distance"]:
                result["min_distance"] = distance
            
            if distance < self.collision_threshold:
                result["collision_detected"] = True
                result["colliding_components"].append((name, "patient", distance))
            elif distance < self.warning_threshold:
                result["warning"] = True
                result["colliding_components"].append((name, "patient", distance))
        
        # Also check gantry-couch collision
        distance = self._calculate_distance(self.volumes["gantry"], self.volumes["couch"])
        if distance < result["min_distance"]:
            result["min_distance"] = distance
        
        if distance < self.collision_threshold:
            result["collision_detected"] = True
            result["colliding_components"].append(("gantry", "couch", distance))
        elif distance < self.warning_threshold:
            result["warning"] = True
            result["colliding_components"].append(("gantry", "couch", distance))
        
        return result
    
    def _calculate_distance(self, volume1: CollisionVolume, volume2: CollisionVolume) -> float:
        """
        Calculate the approximate distance between two collision volumes.
        
        This is a simplified implementation that uses bounding boxes/spheres.
        A more accurate implementation would use detailed collision detection algorithms.
        
        Args:
            volume1: First collision volume
            volume2: Second collision volume
        
        Returns:
            Approximate distance between the volumes in mm
        """
        # For simplicity, we'll use a conservative approximation
        # For real implementation, use proper collision detection libraries
        
        # For cylinders, approximate as a sphere at center with radius
        if volume1.type == "cylinder" and volume2.type == "cylinder":
            center1 = np.array(volume1.transform.get_translation())
            center2 = np.array(volume2.transform.get_translation())
            radius1 = volume1.parameters["radius"]
            radius2 = volume2.parameters["radius"]
            
            distance = np.linalg.norm(center2 - center1) - radius1 - radius2
            return max(0, distance)
        
        # For boxes, use a simplified check based on centers and dimensions
        elif volume1.type == "box" and volume2.type == "box":
            center1 = np.array(volume1.transform.get_translation())
            center2 = np.array(volume2.transform.get_translation())
            
            # This is very approximate - doesn't account for rotation properly
            half_dim1 = np.array([
                volume1.parameters["width"] / 2,
                volume1.parameters["height"] / 2,
                volume1.parameters["depth"] / 2
            ])
            half_dim2 = np.array([
                volume2.parameters["width"] / 2,
                volume2.parameters["height"] / 2,
                volume2.parameters["depth"] / 2
            ])
            
            # Conservative estimate - half the distance between centers minus dimensions
            center_distance = np.linalg.norm(center2 - center1)
            dimension_sum = np.linalg.norm(half_dim1) + np.linalg.norm(half_dim2)
            
            distance = center_distance - dimension_sum
            return max(0, distance)
        
        # For structure-based volumes, use the bounding box
        elif volume1.type == "structure" or volume2.type == "structure":
            if volume1.type == "structure":
                struct_volume = volume1
                other_volume = volume2
            else:
                struct_volume = volume2
                other_volume = volume1
            
            # Get structure bounds
            bounds = struct_volume.parameters["bounds"]
            center = np.array([
                (bounds[0] + bounds[1]) / 2,  # Center X
                (bounds[2] + bounds[3]) / 2,  # Center Y
                (bounds[4] + bounds[5]) / 2   # Center Z
            ])
            
            # Calculate half-dimensions of structure bounding box
            half_dims = np.array([
                (bounds[1] - bounds[0]) / 2,  # Half width
                (bounds[3] - bounds[2]) / 2,  # Half height
                (bounds[5] - bounds[4]) / 2   # Half depth
            ])
            
            # Get other volume center
            other_center = np.array(other_volume.transform.get_translation())
            
            # Calculate a conservative distance estimate
            center_distance = np.linalg.norm(other_center - center)
            
            # Use appropriate dimension based on other volume type
            if other_volume.type == "cylinder":
                other_radius = other_volume.parameters["radius"]
                distance = center_distance - np.linalg.norm(half_dims) - other_radius
            elif other_volume.type == "box":
                other_half_dims = np.array([
                    other_volume.parameters["width"] / 2,
                    other_volume.parameters["height"] / 2,
                    other_volume.parameters["depth"] / 2
                ])
                distance = center_distance - np.linalg.norm(half_dims) - np.linalg.norm(other_half_dims)
            else:
                # Default conservative estimate
                distance = center_distance - np.linalg.norm(half_dims) - 100.0
            
            return max(0, distance)
        
        # Default conservative estimate
        else:
            center1 = np.array(volume1.transform.get_translation())
            center2 = np.array(volume2.transform.get_translation())
            
            center_distance = np.linalg.norm(center2 - center1)
            conservative_radius = 500.0  # A conservative default radius
            
            distance = center_distance - (2 * conservative_radius)
            return max(0, distance)
    
    def check_plan_collisions(self, plan: Plan) -> Dict:
        """
        Check all beams in a plan for potential collisions.
        
        Args:
            plan: Treatment plan
        
        Returns:
            Dictionary with collision information for each beam
        """
        self.set_plan(plan)
        
        results = {}
        
        for beam in plan.get_beams():
            beam_results = []
            
            # For static beams
            if beam.technique == "STATIC":
                result = self.check_collision(
                    beam.gantry_angle,
                    beam.couch_angle,
                    beam.collimator_angle
                )
                beam_results.append({
                    "control_point": 0,
                    "gantry_angle": beam.gantry_angle,
                    "couch_angle": beam.couch_angle,
                    "collimator_angle": beam.collimator_angle,
                    "result": result
                })
            
            # For arc beams, check multiple points along the arc
            elif beam.technique in ["ARC", "VMAT"]:
                control_points = beam.control_points
                if not control_points:
                    # Default to checking just start and end angles
                    start_angle = beam.gantry_start_angle
                    stop_angle = beam.gantry_stop_angle
                    
                    # Check 10 points along the arc
                    num_points = 10
                    direction = 1 if stop_angle > start_angle else -1
                    if abs(stop_angle - start_angle) > 180:
                        direction *= -1
                    
                    # Handle wraparound (e.g., 350 to 10 degrees)
                    if direction > 0 and stop_angle < start_angle:
                        stop_angle += 360
                    elif direction < 0 and stop_angle > start_angle:
                        stop_angle -= 360
                    
                    angles = np.linspace(start_angle, stop_angle, num_points)
                    
                    for i, angle in enumerate(angles):
                        # Keep angle in range [0, 360)
                        angle_normalized = angle % 360
                        
                        result = self.check_collision(
                            angle_normalized,
                            beam.couch_angle,
                            beam.collimator_angle
                        )
                        beam_results.append({
                            "control_point": i,
                            "gantry_angle": angle_normalized,
                            "couch_angle": beam.couch_angle,
                            "collimator_angle": beam.collimator_angle,
                            "result": result
                        })
                else:
                    # Check each control point
                    for i, cp in enumerate(control_points):
                        result = self.check_collision(
                            cp.gantry_angle,
                            cp.couch_angle if hasattr(cp, 'couch_angle') else beam.couch_angle,
                            cp.collimator_angle if hasattr(cp, 'collimator_angle') else beam.collimator_angle
                        )
                        beam_results.append({
                            "control_point": i,
                            "gantry_angle": cp.gantry_angle,
                            "couch_angle": cp.couch_angle if hasattr(cp, 'couch_angle') else beam.couch_angle,
                            "collimator_angle": cp.collimator_angle if hasattr(cp, 'collimator_angle') else beam.collimator_angle,
                            "result": result
                        })
            
            results[beam.id] = beam_results
        
        return results
    
    def get_collision_free_angles(self, couch_angle: float = 0.0, 
                                 step: float = 10.0) -> List[float]:
        """
        Get a list of gantry angles that are free from collisions.
        
        Args:
            couch_angle: Fixed couch angle
            step: Angle step size for checking
        
        Returns:
            List of collision-free gantry angles
        """
        collision_free_angles = []
        
        for angle in np.arange(0, 360, step):
            result = self.check_collision(angle, couch_angle)
            if not result["collision_detected"] and not result["warning"]:
                collision_free_angles.append(angle)
        
        return collision_free_angles
    
    def export_to_json(self, filename: str):
        """
        Export the collision detector state to a JSON file.
        
        Args:
            filename: Output filename
        """
        data = {
            "machine_type": self.machine_type,
            "isocenter": [self.isocenter.x, self.isocenter.y, self.isocenter.z],
            "collision_threshold": self.collision_threshold,
            "warning_threshold": self.warning_threshold,
            "volumes": {name: vol.to_dict() for name, vol in self.volumes.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, filename: str):
        """
        Import collision detector state from a JSON file.
        
        Args:
            filename: Input filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.machine_type = data["machine_type"]
            self.isocenter = Point3D(*data["isocenter"])
            self.collision_threshold = data["collision_threshold"]
            self.warning_threshold = data["warning_threshold"]
            
            self.volumes = {}
            for name, vol_data in data["volumes"].items():
                self.volumes[name] = CollisionVolume.from_dict(vol_data)
            
            return True
        except Exception as e:
            logger.error(f"Error importing collision detector state: {e}")
            return False


def create_collision_detector(plan: Plan, patient: Patient, 
                             machine_type: str = "TrueBeam") -> CollisionDetector:
    """
    Create and initialize a collision detector for a treatment plan.
    
    Args:
        plan: Treatment plan
        patient: Patient data
        machine_type: Treatment machine type
    
    Returns:
        Initialized CollisionDetector object
    """
    detector = CollisionDetector(machine_type)
    detector.set_patient(patient)
    detector.set_plan(plan)
    return detector 