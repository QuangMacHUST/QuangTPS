import os
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

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
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {list(self.ALGORITHMS.keys())}")
        
        self.algorithm_name = algorithm
        self.algorithm = self.ALGORITHMS[algorithm]()
        
        # Set up beam model directory
        if beam_model_dir is None:
            # Use default directory in package
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            beam_model_dir = os.path.join(package_dir, "data", "beam_models")
        
        self.beam_model_dir = beam_model_dir
        ensure_directory(self.beam_model_dir)
        
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
    
    def set_algorithm(self, algorithm: str):
        """
        Change the dose calculation algorithm.
        
        Parameters
        ----------
        algorithm : str
            The dose calculation algorithm to use
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {list(self.ALGORITHMS.keys())}")
        
        self.algorithm_name = algorithm
        self.algorithm = self.ALGORITHMS[algorithm]()
        logger.info(f"Switched dose calculation algorithm to {algorithm}")
    
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
            logger.info(f"Calculating dose for beam {beam.name} using {self.algorithm_name} algorithm")
            
            # Load beam model based on beam energy and type
            beam_model = self._get_beam_model(beam)
            
            # Set the beam model in the algorithm
            self.algorithm.set_beam_model(beam_model)
            
            # Calculate dose
            dose_image = self.algorithm.calculate_beam_dose(beam, ct_image)
            
            # Set dose image properties
            dose_image.modality = "RTDOSE"
            dose_image.description = f"Dose for beam {beam.name} calculated with {self.algorithm_name}"
            
            return dose_image
            
        except Exception as e:
            error_msg = f"Error calculating dose for beam {beam.name}: {str(e)}"
            logger.error(error_msg)
            raise DoseCalculationError(error_msg) from e
    
    def calculate_dose_for_plan(self, plan: Plan, ct_image: Image) -> Image:
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
        Image
            The calculated total dose image
        """
        try:
            logger.info(f"Calculating total dose for plan {plan.name} using {self.algorithm_name} algorithm")
            
            if not plan.beams:
                raise DoseCalculationError(f"No beams found in plan {plan.name}")
            
            # Calculate dose for each beam
            beam_doses = []
            for i, beam in enumerate(plan.beams):
                logger.info(f"Calculating dose for beam {i+1}/{len(plan.beams)}: {beam.name}")
                beam_dose = self.calculate_dose_for_beam(beam, ct_image)
                
                # Apply beam weight
                beam_dose.data *= beam.weight
                beam_doses.append(beam_dose)
            
            # Create total dose by summing all beam doses
            total_dose = Image(
                data=np.zeros_like(beam_doses[0].data),
                spacing=beam_doses[0].spacing,
                origin=beam_doses[0].origin,
                direction=beam_doses[0].direction
            )
            
            # Sum all beam doses
            for dose in beam_doses:
                total_dose.data += dose.data
            
            # Set dose image properties
            total_dose.modality = "RTDOSE"
            total_dose.description = f"Total dose for plan {plan.name} calculated with {self.algorithm_name}"
            
            # Apply plan normalization if specified
            if plan.normalization_value is not None:
                normalization_factor = plan.normalization_value / np.max(total_dose.data)
                total_dose.data *= normalization_factor
                logger.info(f"Applied plan normalization factor: {normalization_factor:.4f}")
            
            return total_dose
            
        except Exception as e:
            error_msg = f"Error calculating dose for plan {plan.name}: {str(e)}"
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