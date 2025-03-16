
import numpy as np
import logging
import SimpleITK as sitk
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import random
from dataclasses import dataclass
import math

from quangtps.core.constants import ELECTRON_REST_MASS_ENERGY
from quangtps.core.types import DoseGrid, BeamParameters
from quangtps.physics.interaction import PhotonInteraction, ElectronInteraction
from quangtps.physics.material import MaterialProperties, create_material_map_from_ct
from quangtps.physics.particle import Particle, ParticleType, ParticleHistory
from quangtps.physics.source import PhotonSource, ElectronSource
from quangtps.dose.base import DoseCalculationAlgorithm, DoseCalculationResult


logger = logging.getLogger(__name__)


@dataclass
class MonteCarloParameters:
    """Parameters for Monte Carlo dose calculation."""
    number_of_histories: int = 10000000  # Number of particle histories to simulate
    uncertainty_threshold: float = 0.02  # Target statistical uncertainty (relative)
    electron_cutoff_energy: float = 0.2  # MeV, energy below which electrons are locally absorbed
    photon_cutoff_energy: float = 0.01  # MeV, energy below which photons are locally absorbed
    use_variance_reduction: bool = True  # Enable variance reduction techniques
    number_of_threads: int = 8  # Number of parallel threads to use
    voxel_grid_resolution: Optional[Tuple[float, float, float]] = None  # Optional dose grid resolution override (mm)
    random_seed: Optional[int] = None  # Random seed for reproducibility
    transport_mechanics: str = "condensed"  # "condensed" or "detailed" electron transport
    # Physics parameters
    use_delta_scattering: bool = True  # Use delta scattering for photon transport
    use_mott_correction: bool = True  # Use Mott correction for electron scattering
    use_bremsstrahlung: bool = True  # Simulate bremsstrahlung photon production
    report_progress: bool = True  # Report progress during calculation
    progress_interval: int = 1000000  # Number of histories between progress reports


class MonteCarloEngine:
    """Core Monte Carlo particle transport engine."""
    
    def __init__(self, parameters: MonteCarloParameters):
        """Initialize the Monte Carlo engine.
        
        Args:
            parameters: Configuration parameters for the simulation
        """
        self.parameters = parameters
        self.random_generator = random.Random(parameters.random_seed)
        self.photon_interaction = PhotonInteraction()
        self.electron_interaction = ElectronInteraction()
        self.material_properties = MaterialProperties()
        
    def transport_particle(self, particle: Particle, dose_grid: np.ndarray, 
                          materials: np.ndarray, voxel_sizes: Tuple[float, float, float]) -> List[Particle]:
        """Transport a single particle through the geometry and score dose.
        
        Args:
            particle: The particle to transport
            dose_grid: The dose grid to score to (modified in-place)
            materials: Material index grid with same dimensions as dose_grid
            voxel_sizes: Size of each voxel in mm
        
        Returns:
            List of secondary particles generated during transport
        """
        secondaries = []
        
        # Continue until particle is absorbed or leaves the geometry
        while particle.is_alive:
            # Get current voxel indices
            ix, iy, iz = self._get_voxel_indices(particle.position, voxel_sizes)
            
            # Check if particle is within dose grid boundaries
            if not self._is_in_geometry(ix, iy, iz, dose_grid.shape):
                particle.is_alive = False
                continue
            
            # Get material at current position
            material_index = materials[ix, iy, iz]
            material = self.material_properties.get_material(material_index)
            
            # Handle transport based on particle type
            if particle.type == ParticleType.PHOTON:
                # Determine distance to next interaction
                mfp = self.photon_interaction.get_mean_free_path(material, particle.energy)
                distance = -math.log(self.random_generator.random()) * mfp
                
                # Check if interaction occurs within current voxel
                voxel_path = self._track_through_voxel(particle, ix, iy, iz, voxel_sizes)
                
                if distance <= voxel_path:
                    # Move particle to interaction site
                    self._move_particle(particle, distance)
                    
                    # Score energy deposition from photon interaction (if any)
                    energy_dep, new_particles = self.photon_interaction.interact(
                        particle, material, self.random_generator)
                    
                    if energy_dep > 0:
                        self._score_dose(dose_grid, ix, iy, iz, energy_dep, material)
                    
                    # Add secondary particles to tracking list
                    for new_particle in new_particles:
                        if ((new_particle.type == ParticleType.ELECTRON and 
                            new_particle.energy > self.parameters.electron_cutoff_energy) or
                            (new_particle.type == ParticleType.PHOTON and 
                            new_particle.energy > self.parameters.photon_cutoff_energy)):
                            secondaries.append(new_particle)
                    
                    # Terminate current particle if it was absorbed
                    if particle.energy <= self.parameters.photon_cutoff_energy:
                        # Deposit remaining energy locally
                        self._score_dose(dose_grid, ix, iy, iz, particle.energy, material)
                        particle.is_alive = False
                else:
                    # Move to voxel boundary
                    self._move_particle(particle, voxel_path)
            
            elif particle.type == ParticleType.ELECTRON:
                if self.parameters.transport_mechanics == "condensed":
                    # Condensed history approach - take larger steps
                    step_length = min(0.1 * material.radiation_length, 
                                    self._get_max_voxel_dimension(voxel_sizes) * 0.5)
                    
                    # Calculate energy loss for this step (continuous slowing down approximation)
                    energy_loss = self.electron_interaction.calculate_energy_loss(
                        particle.energy, material, step_length)
                    
                    # Score energy deposition
                    self._score_dose(dose_grid, ix, iy, iz, energy_loss, material)
                    particle.energy -= energy_loss
                    
                    # Handle multiple scattering (change direction)
                    self.electron_interaction.apply_multiple_scattering(
                        particle, material, step_length, self.random_generator)
                    
                    # Move particle
                    self._move_particle(particle, step_length)
                    
                    # Check for discrete interactions (delta ray, bremsstrahlung)
                    if self.parameters.use_bremsstrahlung and self.random_generator.random() < 0.05:
                        # Simplified bremsstrahlung production probability
                        photon_energy = self.electron_interaction.sample_bremsstrahlung_energy(
                            particle.energy, self.random_generator)
                        
                        if photon_energy > self.parameters.photon_cutoff_energy:
                            # Create bremsstrahlung photon
                            photon = Particle(
                                position=particle.position.copy(),
                                direction=self._sample_bremsstrahlung_direction(particle, self.random_generator),
                                energy=photon_energy,
                                type=ParticleType.PHOTON
                            )
                            secondaries.append(photon)
                            particle.energy -= photon_energy
                else:
                    # Detailed history approach - shorter steps with discrete interactions
                    # This would be a more detailed implementation
                    pass
                
                # Check if electron energy is below cutoff
                if particle.energy <= self.parameters.electron_cutoff_energy:
                    # Deposit remaining energy locally
                    self._score_dose(dose_grid, ix, iy, iz, particle.energy, material)
                    particle.is_alive = False
            
        return secondaries
    
    def _get_voxel_indices(self, position: np.ndarray, voxel_sizes: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert position coordinates to voxel indices."""
        ix = int(position[0] / voxel_sizes[0])
        iy = int(position[1] / voxel_sizes[1])
        iz = int(position[2] / voxel_sizes[2])
        return ix, iy, iz
    
    def _is_in_geometry(self, ix: int, iy: int, iz: int, shape: Tuple[int, int, int]) -> bool:
        """Check if the given indices are within the geometry bounds."""
        return 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]
    
    def _track_through_voxel(self, particle: Particle, ix: int, iy: int, iz: int, 
                            voxel_sizes: Tuple[float, float, float]) -> float:
        """Calculate distance to voxel boundary along particle direction."""
        # Position within voxel (relative to voxel lower corner)
        pos_in_voxel = [
            particle.position[0] - ix * voxel_sizes[0],
            particle.position[1] - iy * voxel_sizes[1],
            particle.position[2] - iz * voxel_sizes[2]
        ]
        
        # Distance to boundaries in each direction
        dist_to_boundary = [float('inf')] * 3
        
        for i in range(3):
            if particle.direction[i] > 1e-6:  # Moving in positive direction
                dist_to_boundary[i] = (voxel_sizes[i] - pos_in_voxel[i]) / particle.direction[i]
            elif particle.direction[i] < -1e-6:  # Moving in negative direction
                dist_to_boundary[i] = -pos_in_voxel[i] / particle.direction[i]
        
        # Return minimum positive distance to boundary
        return min(dist_to_boundary)
    
    def _move_particle(self, particle: Particle, distance: float) -> None:
        """Move particle along its direction by the specified distance."""
        for i in range(3):
            particle.position[i] += particle.direction[i] * distance
    
    def _score_dose(self, dose_grid: np.ndarray, ix: int, iy: int, iz: int, 
                   energy: float, material: Any) -> None:
        """Score energy deposition as dose in the specified voxel."""
        if 0 <= ix < dose_grid.shape[0] and 0 <= iy < dose_grid.shape[1] and 0 <= iz < dose_grid.shape[2]:
            # Convert from energy to dose (energy/mass)
            dose = energy / material.density
            dose_grid[ix, iy, iz] += dose
    
    def _get_max_voxel_dimension(self, voxel_sizes: Tuple[float, float, float]) -> float:
        """Return the maximum voxel dimension."""
        return max(voxel_sizes)
    
    def _sample_bremsstrahlung_direction(self, particle: Particle, random_generator: random.Random) -> np.ndarray:
        """Sample direction for bremsstrahlung photon relative to electron direction."""
        # Simplified model - bremsstrahlung photons are emitted in a narrow cone around electron direction
        theta = 1.0 / (particle.energy + 1.0) * random_generator.random()  # Higher energy = narrower cone
        phi = 2 * math.pi * random_generator.random()
        
        # Create coordinate system around electron direction
        e1 = particle.direction
        if abs(e1[0]) < 0.9:
            e2 = np.cross(e1, [1, 0, 0])
        else:
            e2 = np.cross(e1, [0, 1, 0])
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        
        # Direction in local coordinate system
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        
        # Transform to global coordinates
        direction = z * e1 + x * e2 + y * e3
        return direction / np.linalg.norm(direction)


class MonteCarloAlgorithm(DoseCalculationAlgorithm):
    """
    Monte Carlo dose calculation algorithm implementation.
    
    This algorithm simulates individual particle transport through matter 
    to calculate dose distributions with high accuracy, especially in 
    heterogeneous tissues.
    """
    
    def __init__(self, parameters: Optional[MonteCarloParameters] = None):
        """
        Initialize Monte Carlo algorithm with specified parameters.
        
        Args:
            parameters: Configuration parameters for Monte Carlo simulation.
                        If None, default parameters will be used.
        """
        super().__init__("Monte Carlo")
        self.parameters = parameters or MonteCarloParameters()
        self.engine = MonteCarloEngine(self.parameters)
        
    def calculate(self, ct_image: sitk.Image, structures: Dict[str, sitk.Image],
                 beam_parameters: BeamParameters) -> DoseCalculationResult:
        """
        Calculate dose distribution using Monte Carlo simulation.
        
        Args:
            ct_image: CT image used for material and density information
            structures: Dictionary of structure masks (target, OARs)
            beam_parameters: Parameters describing the beam setup
            
        Returns:
            Calculated dose distribution and additional information
        """
        logger.info("Starting Monte Carlo dose calculation")
        start_time = time.time()
        
        # Convert CT image to numpy array
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        # Get image geometry information
        spacing = ct_image.GetSpacing()
        origin = ct_image.GetOrigin()
        size = ct_image.GetSize()
        
        # Set up dose grid (with same dimensions as CT image)
        dose_grid = np.zeros_like(ct_array, dtype=np.float32)
        
        # Set up variance grid to track statistical uncertainty
        variance_grid = np.zeros_like(dose_grid)
        
        # Initialize material map from CT
        material_map = create_material_map_from_ct(ct_array)
        
        # Create source based on beam parameters
        if beam_parameters.beam_type.lower() == "photon":
            source = PhotonSource(beam_parameters)
        elif beam_parameters.beam_type.lower() == "electron":
            source = ElectronSource(beam_parameters)
        else:
            raise ValueError(f"Unsupported beam type: {beam_parameters.beam_type}")
        
        # Set up batch processing
        batch_size = min(1000000, self.parameters.number_of_histories)
        num_batches = (self.parameters.number_of_histories + batch_size - 1) // batch_size
        
        # Set up parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parameters.number_of_threads) as executor:
            for batch in range(num_batches):
                batch_histories = min(batch_size, self.parameters.number_of_histories - batch * batch_size)
                
                # Submit batch calculations to thread pool
                futures = []
                for i in range(self.parameters.number_of_threads):
                    # Distribute histories among threads
                    thread_histories = batch_histories // self.parameters.number_of_threads
                    if i < batch_histories % self.parameters.number_of_threads:
                        thread_histories += 1
                    
                    if thread_histories > 0:
                        futures.append(executor.submit(
                            self._simulate_batch, 
                            thread_histories, 
                            source, 
                            ct_array, 
                            material_map, 
                            spacing,
                            batch * batch_size + i * (batch_histories // self.parameters.number_of_threads)
                        ))
                
                # Collect results from all threads
                batch_dose_grids = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        thread_dose_grid = future.result()
                        batch_dose_grids.append(thread_dose_grid)
                    except Exception as e:
                        logger.error(f"Thread error: {e}")
                
                # Combine batch results
                for thread_dose_grid in batch_dose_grids:
                    dose_grid += thread_dose_grid
                
                # Report progress
                if self.parameters.report_progress:
                    histories_completed = min((batch + 1) * batch_size, self.parameters.number_of_histories)
                    progress = histories_completed / self.parameters.number_of_histories * 100
                    logger.info(f"Monte Carlo progress: {progress:.1f}% ({histories_completed:,} / {self.parameters.number_of_histories:,} histories)")
        
        # Normalize dose to prescribed dose or MU
        if beam_parameters.dose_grid_normalization:
            normalization_factor = beam_parameters.dose_grid_normalization / np.max(dose_grid)
            dose_grid *= normalization_factor
        
        # Calculate uncertainty
        uncertainty_grid = np.sqrt(variance_grid) / dose_grid
        uncertainty_grid[dose_grid < 0.1 * np.max(dose_grid)] = 0  # Ignore low dose regions
        
        # Create SimpleITK image from dose grid
        dose_image = sitk.GetImageFromArray(dose_grid)
        dose_image.SetSpacing(spacing)
        dose_image.SetOrigin(origin)
        
        # Create uncertainty image
        uncertainty_image = sitk.GetImageFromArray(uncertainty_grid)
        uncertainty_image.SetSpacing(spacing)
        uncertainty_image.SetOrigin(origin)
        
        calc_time = time.time() - start_time
        logger.info(f"Monte Carlo dose calculation completed in {calc_time:.2f} seconds")
        
        # Return results
        result = DoseCalculationResult(
            dose=dose_image,
            algorithm_name="Monte Carlo",
            calculation_time=calc_time,
            additional_data={
                "uncertainty": uncertainty_image,
                "max_uncertainty": np.max(uncertainty_grid),
                "number_of_histories": self.parameters.number_of_histories
            }
        )
        
        return result
    
    def _simulate_batch(self, num_histories: int, source: Union[PhotonSource, ElectronSource], 
                       ct_array: np.ndarray, material_map: np.ndarray, 
                       voxel_sizes: Tuple[float, float, float], seed_offset: int) -> np.ndarray:
        """
        Simulate a batch of particle histories.
        
        Args:
            num_histories: Number of particle histories to simulate
            source: Particle source (photon or electron)
            ct_array: CT image data
            material_map: Material mapping for each voxel
            voxel_sizes: Size of voxels in mm
            seed_offset: Offset for random seed
            
        Returns:
            Dose grid from this batch of histories
        """
        # Create thread-local random generator with different seed
        seed = None if self.parameters.random_seed is None else self.parameters.random_seed + seed_offset
        rng = random.Random(seed)
        
        # Create local dose grid
        local_dose_grid = np.zeros_like(ct_array, dtype=np.float32)
        
        # Set up local engine
        engine = MonteCarloEngine(self.parameters)
        engine.random_generator = rng
        
        # Start particle histories
        for i in range(num_histories):
            # Sample primary particle from source
            primary_particle = source.sample_particle(rng)
            
            # Create particle history for tracking
            history = ParticleHistory(primary_particle)
            
            # Transport primary particle
            secondary_particles = engine.transport_particle(
                primary_particle, local_dose_grid, material_map, voxel_sizes)
            
            # Add secondaries to history
            for secondary in secondary_particles:
                history.add_secondary(secondary)
            
            # Transport all secondary particles
            while history.has_active_secondaries():
                secondary = history.get_next_secondary()
                new_secondaries = engine.transport_particle(
                    secondary, local_dose_grid, material_map, voxel_sizes)
                
                for new_secondary in new_secondaries:
                    history.add_secondary(new_secondary)
        
        # Normalize by number of histories
        local_dose_grid /= num_histories
        
        return local_dose_grid
    
    def get_name(self) -> str:
        """Get algorithm name."""
        return "Monte Carlo"
    
    def get_description(self) -> str:
        """Get algorithm description."""
        return (
            "Monte Carlo algorithm simulates the transport of individual particles (photons, electrons) "
            "through matter. It tracks random interactions using probability distributions based on physics "
            "principles to achieve the highest possible accuracy in heterogeneous tissues."
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return {
            "number_of_histories": self.parameters.number_of_histories,
            "uncertainty_threshold": self.parameters.uncertainty_threshold,
            "electron_cutoff_energy": self.parameters.electron_cutoff_energy,
            "photon_cutoff_energy": self.parameters.photon_cutoff_energy,
            "use_variance_reduction": self.parameters.use_variance_reduction,
            "number_of_threads": self.parameters.number_of_threads,
            "voxel_grid_resolution": self.parameters.voxel_grid_resolution,
            "transport_mechanics": self.parameters.transport_mechanics
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set algorithm parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        if "number_of_histories" in parameters:
            self.parameters.number_of_histories = int(parameters["number_of_histories"])
            
        if "uncertainty_threshold" in parameters:
            self.parameters.uncertainty_threshold = float(parameters["uncertainty_threshold"])
            
        if "electron_cutoff_energy" in parameters:
            self.parameters.electron_cutoff_energy = float(parameters["electron_cutoff_energy"])
            
        if "photon_cutoff_energy" in parameters:
            self.parameters.photon_cutoff_energy = float(parameters["photon_cutoff_energy"])
            
        if "use_variance_reduction" in parameters:
            self.parameters.use_variance_reduction = bool(parameters["use_variance_reduction"])
            
        if "number_of_threads" in parameters:
            self.parameters.number_of_threads = int(parameters["number_of_threads"])
            
        if "voxel_grid_resolution" in parameters:
            self.parameters.voxel_grid_resolution = parameters["voxel_grid_resolution"]
            
        if "transport_mechanics" in parameters:
            self.parameters.transport_mechanics = parameters["transport_mechanics"]
            
        # Update engine with new parameters
        self.engine = MonteCarloEngine(self.parameters)