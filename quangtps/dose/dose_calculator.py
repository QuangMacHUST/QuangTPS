import os
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
import traceback

from quangtps.core.exceptions import DoseCalculationError
from quangtps.core.utils import ensure_directory
from quangtps.imaging.image import Image
from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm
from quangtps.dose.algorithms.collapsed_cone import CollapsedConeAlgorithm 
from quangtps.dose.algorithms.monte_carlo import MonteCarloAlgorithm
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.dose.physics.truebeam_models import TruebeamModelManager

logger = logging.getLogger(__name__)

class DoseCalculator:
    """
    Class responsible for dose calculation using various algorithms.
    """
    
    ALGORITHMS = {
        "PENCIL_BEAM": PencilBeamAlgorithm,
        "COLLAPSED_CONE": CollapsedConeAlgorithm,
        "MONTE_CARLO": MonteCarloAlgorithm
    }
    
    def __init__(self, algorithm: str = "PENCIL_BEAM", beam_model_dir: Optional[str] = None):
        """
        Initialize the dose calculator with the specified algorithm.
        
        Parameters
        ----------
        algorithm : str
            The dose calculation algorithm to use
        beam_model_dir : str, optional
            Directory containing beam models data
        """
        self.algorithm_name = algorithm
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {list(self.ALGORITHMS.keys())}")
        
        self.algorithm = self.ALGORITHMS[algorithm]()
        self.beam_models = {}  # Cache for beam models
        
        # Set up beam model directory
        if beam_model_dir is None:
            # Use default directory in package
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            beam_model_dir = os.path.join(package_dir, "data", "beam_models")
        
        self.beam_model_dir = beam_model_dir
        ensure_directory(self.beam_model_dir)
        
        try:
            # Initialize Truebeam model manager if data is available
            truebeam_data_dir = os.path.join(package_dir, "data", "truebeam")
            if os.path.exists(truebeam_data_dir):
                truebeam_model_dir = os.path.join(self.beam_model_dir, "truebeam")
                ensure_directory(truebeam_model_dir)
                self.truebeam_manager = TruebeamModelManager(
                    data_directory=truebeam_data_dir,
                    model_directory=truebeam_model_dir
                )
                logger.info(f"Initialized Truebeam model manager with {len(self.truebeam_manager.get_available_energies())} available energies")
            else:
                self.truebeam_manager = None
                logger.warning("Truebeam data directory not found. Truebeam models will not be available.")
        except Exception as e:
            self.truebeam_manager = None
            logger.error(f"Error initializing Truebeam model manager: {str(e)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    def validate_inputs(self, plan: Plan, ct_image: Image) -> None:
        """
        Validate inputs for dose calculation.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan
        ct_image : Image
            The CT image
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        # Validate plan
        if not plan:
            raise ValueError("Plan cannot be None")
            
        if not hasattr(plan, 'beams') or not plan.beams:
            raise ValueError(f"Plan {plan.name if hasattr(plan, 'name') else '<unnamed>'} has no beams")
            
        # Validate CT image
        if not ct_image:
            raise ValueError("CT image cannot be None")
            
        if not hasattr(ct_image, 'data') or ct_image.data is None:
            raise ValueError("CT image has no data")
            
        if not isinstance(ct_image.data, np.ndarray) or ct_image.data.size == 0:
            raise ValueError("CT image data is empty or not a NumPy array")
            
        # Check image dimensions
        if len(ct_image.data.shape) != 3:
            raise ValueError(f"CT image must be 3D, got shape {ct_image.data.shape}")
            
        # Check CT numbers are in valid range
        if np.min(ct_image.data) < -1024 or np.max(ct_image.data) > 3071:
            logger.warning(f"CT numbers outside typical range [-1024, 3071]: min={np.min(ct_image.data)}, max={np.max(ct_image.data)}")
            
        # Validate each beam
        for i, beam in enumerate(plan.beams):
            if not hasattr(beam, 'isocenter') or beam.isocenter is None:
                raise ValueError(f"Beam {i} ({beam.name if hasattr(beam, 'name') else '<unnamed>'}) has no isocenter")
                
            if not hasattr(beam, 'gantry_angle') or (beam.gantry_angle is None and beam.gantry_angle != 0):
                raise ValueError(f"Beam {i} ({beam.name if hasattr(beam, 'name') else '<unnamed>'}) has no gantry angle")
                
            if not hasattr(beam, 'field_size') or beam.field_size is None or any(s <= 0 for s in beam.field_size if s is not None):
                raise ValueError(f"Beam {i} ({beam.name if hasattr(beam, 'name') else '<unnamed>'}) has invalid field size: {beam.field_size if hasattr(beam, 'field_size') else 'None'}")
    
    def calculate_dose_for_beam(self, beam: Beam, ct_image: Image) -> Image:
        """
        Calculate dose for a single beam.
        
        Parameters
        ----------
        beam : Beam
            The beam to calculate dose for
        ct_image : Image
            The CT image for dose calculation
            
        Returns
        -------
        Image
            The calculated dose image
        """
        try:
            beam_name = beam.name if hasattr(beam, 'name') else "<unnamed>"
            logger.info(f"Calculating dose for beam {beam_name} using {self.algorithm_name} algorithm")
            
            # Load beam model based on beam energy and type
            beam_model = self._get_beam_model(beam)
            
            # Set the beam model in the algorithm
            self.algorithm.set_beam_model(beam_model)
            
            # Calculate dose
            dose_image = self.algorithm.calculate_beam_dose(beam, ct_image)
            
            # Validate returned dose image
            if dose_image is None or not hasattr(dose_image, 'data') or dose_image.data is None:
                raise DoseCalculationError(f"Algorithm returned empty dose for beam {beam_name}")
                
            if not isinstance(dose_image.data, np.ndarray) or dose_image.data.size == 0:
                raise DoseCalculationError(f"Algorithm returned invalid dose data for beam {beam_name}")
            
            # Set dose image properties
            dose_image.modality = "RTDOSE"
            dose_image.description = f"Dose for beam {beam_name} calculated with {self.algorithm_name}"
            
            return dose_image
            
        except Exception as e:
            error_msg = f"Error calculating dose for beam {beam.name if hasattr(beam, 'name') else '<unnamed>'}: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e
    
    def calculate_dose_for_plan(self, plan: Plan, ct_image: Image) -> Optional[Image]:
        """
        Calculate total dose for a treatment plan.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan to calculate dose for
        ct_image : Image
            The CT image for dose calculation
            
        Returns
        -------
        Image or None
            The calculated total dose image, or None if calculation fails
            
        Raises
        ------
        DoseCalculationError
            If dose calculation fails critically
        """
        try:
            plan_name = plan.name if hasattr(plan, 'name') else "<unnamed>"
            logger.info(f"Calculating total dose for plan {plan_name} using {self.algorithm_name} algorithm")
            
            # Validate inputs
            self.validate_inputs(plan, ct_image)
            
            # Calculate dose for each beam
            beam_doses = []
            total_mu = 0
            
            # Calculate total MU (with validation)
            for beam in plan.beams:
                if hasattr(beam, 'monitor_units') and isinstance(beam.monitor_units, (int, float)) and beam.monitor_units > 0:
                    total_mu += beam.monitor_units
            
            # If no valid MUs, use equal weights
            use_equal_weights = total_mu <= 0
            if use_equal_weights:
                logger.warning(f"No valid monitor units found in plan. Using equal weights for all beams.")
            
            # Calculate dose for each beam
            for i, beam in enumerate(plan.beams):
                beam_name = beam.name if hasattr(beam, 'name') else f"Beam_{i+1}"
                logger.info(f"Calculating dose for beam {i+1}/{len(plan.beams)}: {beam_name}")
                
                try:
                    beam_dose = self.calculate_dose_for_beam(beam, ct_image)
                    
                    # Apply beam weight/MU
                    if not use_equal_weights and hasattr(beam, 'monitor_units') and isinstance(beam.monitor_units, (int, float)) and beam.monitor_units > 0:
                        weight = beam.monitor_units / total_mu
                        logger.info(f"Applied weight {weight:.4f} to beam {beam_name} based on {beam.monitor_units} MU")
                    else:
                        weight = 1.0 / len(plan.beams)  # Equal weight if no MU
                        logger.info(f"Applied equal weight {weight:.4f} to beam {beam_name}")
                    
                    if hasattr(beam_dose, 'data') and beam_dose.data is not None and beam_dose.data.size > 0:
                        beam_dose.data = beam_dose.data * weight
                        beam_doses.append(beam_dose)
                    else:
                        logger.warning(f"Skipping beam {beam_name} as it has invalid dose data")
                        
                except Exception as e:
                    logger.error(f"Error calculating dose for beam {beam_name}: {str(e)}")
                    logger.info(f"Continuing with other beams...")
                    continue
            
            # Check if we have any valid doses
            if not beam_doses:
                logger.error("No valid beam doses calculated. Cannot create plan dose.")
                return None
                
            # Create a total dose array matching the first beam dose
            template_dose = beam_doses[0]
            total_dose_data = np.zeros_like(template_dose.data)
            
            # Sum all doses
            for beam_dose in beam_doses:
                if beam_dose.data.shape == total_dose_data.shape:
                    total_dose_data += beam_dose.data
                else:
                    logger.warning(f"Skipping a beam dose with incompatible shape: {beam_dose.data.shape} vs {total_dose_data.shape}")
            
            # Create total dose image
            total_dose = Image(
                data=total_dose_data,
                metadata={
                    **template_dose.metadata,
                    'plan_name': plan_name,
                    'plan_id': plan.plan_id if hasattr(plan, 'plan_id') else '',
                    'algorithm': self.algorithm_name,
                    'num_beams': len(beam_doses),
                    'total_mu': total_mu,
                    'calculation_time': time.time()
                }
            )
            
            # Set image properties
            total_dose.modality = "RTDOSE"
            total_dose.description = f"Total dose for plan {plan_name} calculated with {self.algorithm_name}"
            
            logger.info(f"Successfully calculated total dose for plan {plan_name}")
            return total_dose
            
        except Exception as e:
            error_msg = f"Error calculating dose for plan {plan.name if hasattr(plan, 'name') else '<unnamed>'}: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e
    
    def _get_beam_model(self, beam: Beam):
        """
        Get appropriate beam model for the given beam.
        
        Parameters
        ----------
        beam : Beam
            The beam to get model for
            
        Returns
        -------
        object
            The beam model object
        
        Raises
        ------
        DoseCalculationError
            If beam model cannot be found or loaded
        """
        try:
            machine_name = beam.machine.lower()
            energy = beam.energy
            
            # Check if this is a Truebeam beam
            if "truebeam" in machine_name and self.truebeam_manager is not None:
                logger.info(f"Loading Truebeam model for energy {energy}")
                beam_model = self.truebeam_manager.load_model(energy)
                if beam_model:
                    return beam_model
                else:
                    logger.warning(f"Truebeam model for energy {energy} not found. Using generic model.")
            
            # If not Truebeam or model not found, use generic model
            logger.info(f"Using generic beam model for {machine_name} with energy {energy}")
            return self.algorithm.create_generic_beam_model(energy)
            
        except Exception as e:
            error_msg = f"Error loading beam model for beam {beam.name} with energy {beam.energy}: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e
    
    def get_available_beam_models(self):
        """
        Get a list of available beam models.
        
        Returns
        -------
        Dict
            Dictionary of available beam models grouped by machine type
        """
        available_models = {}
        
        # Generic models
        available_models["generic"] = {
            "photon": ["6MV", "10MV", "15MV"],
            "electron": ["6MeV", "9MeV", "12MeV", "15MeV", "18MeV"]
        }
        
        # Truebeam models
        if self.truebeam_manager is not None:
            available_models["truebeam"] = {
                "photon": self.truebeam_manager.get_available_energies()
            }
        
        return available_models

    def calculate_biological_metrics(self, physical_dose: Image, fractionation: int, 
                                  alpha_beta: float = None) -> Dict[str, Image]:
        """
        Calculate biological dose metrics (BED, EQD2).
        
        Parameters
        ----------
        physical_dose : Image
            Physical dose distribution
        fractionation : int
            Number of fractions
        alpha_beta : float, optional
            Alpha/beta ratio for tissue
            
        Returns
        -------
        Dict[str, Image]
            Dictionary containing BED and EQD2 distributions
        """
        try:
            if not alpha_beta:
                logger.warning("No alpha/beta ratio provided, using default value of 10")
                alpha_beta = 10.0
                
            # Calculate dose per fraction
            dose_per_fraction = physical_dose.data / fractionation
            
            # Calculate BED
            bed_data = physical_dose.data * (1 + dose_per_fraction / alpha_beta)
            bed_image = Image(
                data=bed_data,
                spacing=physical_dose.spacing,
                origin=physical_dose.origin,
                direction=physical_dose.direction
            )
            bed_image.modality = "RTDOSE"
            bed_image.description = f"BED distribution (α/β = {alpha_beta})"
            
            # Calculate EQD2
            eqd2_data = physical_dose.data * ((dose_per_fraction + alpha_beta) / (2 + alpha_beta))
            eqd2_image = Image(
                data=eqd2_data,
                spacing=physical_dose.spacing,
                origin=physical_dose.origin,
                direction=physical_dose.direction
            )
            eqd2_image.modality = "RTDOSE"
            eqd2_image.description = f"EQD2 distribution (α/β = {alpha_beta})"
            
            return {
                "BED": bed_image,
                "EQD2": eqd2_image
            }
            
        except Exception as e:
            error_msg = f"Error calculating biological metrics: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e 