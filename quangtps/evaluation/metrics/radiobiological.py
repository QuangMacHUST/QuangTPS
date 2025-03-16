"""
Radiobiological metrics for treatment plan evaluation.

This module provides classes and functions for calculating radiobiological
metrics such as TCP (Tumor Control Probability), NTCP (Normal Tissue Complication
Probability), EUD (Equivalent Uniform Dose) and BED (Biologically Effective Dose).
"""

import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any

import SimpleITK as sitk

from quangtps.core.types import DoseGrid
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)


class TCP:
    """
    Tumor Control Probability models.
    
    This class implements various TCP models including Poisson-based,
    Logistic, and LQ-based models for predicting tumor control based on
    dose distribution.
    """
    
    @staticmethod
    def poisson_model(dose: np.ndarray, d50: float, gamma50: float, 
                      voxel_volume: Union[float, np.ndarray] = 1.0, 
                      rho: float = 1e7) -> float:
        """
        Calculate TCP using the Poisson model.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        d50 : float
            Dose that gives 50% TCP in Gy
        gamma50 : float
            Normalized slope at D50
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
        rho : float
            Clonogenic cell density (cells/cm³)
            
        Returns
        -------
        float
            TCP value (0-1)
        """
        # Convert voxel_volume to array if it's a scalar
        if isinstance(voxel_volume, float):
            voxel_volume = np.ones_like(dose) * voxel_volume
        
        # Calculate survival fraction in each voxel
        alpha = 0.693 / d50
        survival = np.exp(-alpha * dose)
        
        # Calculate number of surviving cells in each voxel
        n_surviving = rho * voxel_volume * survival
        
        # Sum up total surviving cells
        total_surviving = np.sum(n_surviving)
        
        # Calculate TCP
        tcp = np.exp(-total_surviving)
        
        return tcp
    
    @staticmethod
    def logistic_model(dose: np.ndarray, d50: float, gamma50: float, 
                       voxel_volume: Union[float, np.ndarray] = 1.0) -> float:
        """
        Calculate TCP using the logistic model.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        d50 : float
            Dose that gives 50% TCP in Gy
        gamma50 : float
            Normalized slope at D50
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
            
        Returns
        -------
        float
            TCP value (0-1)
        """
        # Calculate EUD
        eud_calculator = EUD()
        eud = eud_calculator.calculate(dose, 1, voxel_volume)
        
        # Apply logistic function
        k = 4 * gamma50 / d50
        tcp = 1.0 / (1.0 + np.exp(-k * (eud - d50)))
        
        return tcp
    
    @staticmethod
    def lq_model(dose: np.ndarray, alpha: float, beta: float, 
                 fractions: int, voxel_volume: Union[float, np.ndarray] = 1.0, 
                 rho: float = 1e7) -> float:
        """
        Calculate TCP using the Linear-Quadratic model.
        
        Parameters
        ----------
        dose : np.ndarray
            Total dose distribution in Gy
        alpha : float
            Linear component of cell killing (Gy⁻¹)
        beta : float
            Quadratic component of cell killing (Gy⁻²)
        fractions : int
            Number of fractions
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
        rho : float
            Clonogenic cell density (cells/cm³)
            
        Returns
        -------
        float
            TCP value (0-1)
        """
        # Convert voxel_volume to array if it's a scalar
        if isinstance(voxel_volume, float):
            voxel_volume = np.ones_like(dose) * voxel_volume
        
        # Calculate dose per fraction
        dose_per_fraction = dose / fractions
        
        # Calculate survival fraction based on LQ model
        sf = np.exp(-(alpha * dose + beta * dose * dose_per_fraction))
        
        # Calculate number of surviving cells in each voxel
        n_surviving = rho * voxel_volume * sf
        
        # Sum up total surviving cells
        total_surviving = np.sum(n_surviving)
        
        # Calculate TCP
        tcp = np.exp(-total_surviving)
        
        return tcp


class NTCP:
    """
    Normal Tissue Complication Probability models.
    
    This class implements various NTCP models including Lyman-Kutcher-Burman (LKB),
    Critical volume, and Relative Seriality models for predicting normal tissue
    complications based on dose distribution.
    """
    
    @staticmethod
    def lkb_model(dose: np.ndarray, td50: float, m: float, n: float, 
                  voxel_volume: Union[float, np.ndarray] = 1.0) -> float:
        """
        Calculate NTCP using the Lyman-Kutcher-Burman (LKB) model.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        td50 : float
            Dose that gives 50% NTCP in Gy for uniform dose
        m : float
            Steepness parameter
        n : float
            Volume effect parameter
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
            
        Returns
        -------
        float
            NTCP value (0-1)
        """
        # Calculate generalized EUD with volume effect parameter n
        eud_calculator = EUD()
        eud = eud_calculator.calculate(dose, 1/n, voxel_volume)
        
        # Calculate the t-value
        t = (eud - td50) / (m * td50)
        
        # Calculate the NTCP using the probit function
        ntcp = 0.5 * (1 + math.erf(t / math.sqrt(2)))
        
        return ntcp
    
    @staticmethod
    def relative_seriality_model(dose: np.ndarray, d50: float, gamma50: float, s: float,
                                voxel_volume: Union[float, np.ndarray] = 1.0) -> float:
        """
        Calculate NTCP using the Relative Seriality model.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        d50 : float
            Dose that gives 50% NTCP in Gy for uniform dose
        gamma50 : float
            Maximum relative slope of dose-response curve
        s : float
            Relative seriality parameter
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
            
        Returns
        -------
        float
            NTCP value (0-1)
        """
        # Convert voxel_volume to array if it's a scalar
        if isinstance(voxel_volume, float):
            voxel_volume = np.ones_like(dose) * voxel_volume
        
        # Calculate total volume
        total_volume = np.sum(voxel_volume)
        
        # Calculate volume fraction of each voxel
        dv = voxel_volume / total_volume
        
        # Calculate Poisson response for each voxel
        k = 4 * gamma50
        p = 2**(k * (dose/d50 - 1))
        p = p / (1 + p)
        
        # Apply seriality formula
        product = 1.0
        for i in range(len(dose.flat)):
            product *= (1 - p.flat[i]**s)**dv.flat[i]
        
        ntcp = 1 - product
        
        return ntcp
    
    @staticmethod
    def critical_volume_model(dose: np.ndarray, threshold_dose: float, critical_volume_fraction: float,
                             voxel_volume: Union[float, np.ndarray] = 1.0) -> float:
        """
        Calculate NTCP using the Critical Volume model.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        threshold_dose : float
            Threshold dose for tissue damage in Gy
        critical_volume_fraction : float
            Fraction of organ that can be damaged without complication
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
            
        Returns
        -------
        float
            NTCP value (0-1)
        """
        # Convert voxel_volume to array if it's a scalar
        if isinstance(voxel_volume, float):
            voxel_volume = np.ones_like(dose) * voxel_volume
            
        # Calculate total volume
        total_volume = np.sum(voxel_volume)
        
        # Calculate volume receiving dose above threshold
        high_dose_volume = np.sum(voxel_volume[dose >= threshold_dose])
        
        # Calculate fraction of volume receiving high dose
        volume_fraction = high_dose_volume / total_volume
        
        # Apply critical volume model
        fsv = 1.0  # Fraction of surviving volume
        if volume_fraction > critical_volume_fraction:
            fsv = 1.0 - (volume_fraction - critical_volume_fraction) / (1.0 - critical_volume_fraction)
        
        # Convert to NTCP with sigmoidal response
        # This is a simplified approach; more complex models exist
        ntcp = 1.0 - fsv
        
        return ntcp


class EUD:
    """
    Equivalent Uniform Dose (EUD) calculator.
    
    This class implements methods for calculating EUD, which represents the dose
    that would give the same biological effect if uniformly delivered to the 
    entire structure.
    """
    
    def calculate(self, dose: np.ndarray, a: float, 
                 voxel_volume: Union[float, np.ndarray] = 1.0) -> float:
        """
        Calculate the Equivalent Uniform Dose (EUD).
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        a : float
            Volume parameter (a < 0 for tumors, a > 0 for normal tissues)
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
            
        Returns
        -------
        float
            EUD value in Gy
        """
        # Convert voxel_volume to array if it's a scalar
        if isinstance(voxel_volume, float):
            voxel_volume = np.ones_like(dose) * voxel_volume
            
        # Calculate total volume
        total_volume = np.sum(voxel_volume)
        
        # Calculate volume fraction of each voxel
        dv = voxel_volume / total_volume
        
        # Handle zero doses to avoid math errors
        dose = np.maximum(dose, 1e-10)
        
        # Calculate EUD based on generalized mean
        if abs(a) < 1e-10:  # For a ≈ 0, use geometric mean
            log_sum = np.sum(dv.flat * np.log(dose.flat))
            eud = np.exp(log_sum)
        else:
            # For other values of a, use generalized mean
            dose_a = dose**a
            dose_a_mean = np.sum(dv.flat * dose_a.flat)
            eud = dose_a_mean**(1/a)
        
        return eud
    
    def calculate_from_dvh(self, dose_bins: np.ndarray, volume_bins: np.ndarray, a: float) -> float:
        """
        Calculate EUD from a dose-volume histogram.
        
        Parameters
        ----------
        dose_bins : np.ndarray
            Dose bins in Gy
        volume_bins : np.ndarray
            Differential volume histogram (fraction of volume per dose bin)
        a : float
            Volume parameter
            
        Returns
        -------
        float
            EUD value in Gy
        """
        # Handle zero doses to avoid math errors
        dose_bins = np.maximum(dose_bins, 1e-10)
        
        # Calculate EUD based on generalized mean
        if abs(a) < 1e-10:  # For a ≈ 0, use geometric mean
            log_sum = np.sum(volume_bins * np.log(dose_bins))
            eud = np.exp(log_sum)
        else:
            # For other values of a, use generalized mean
            dose_a = dose_bins**a
            dose_a_mean = np.sum(volume_bins * dose_a)
            eud = dose_a_mean**(1/a)
        
        return eud


class BED:
    """
    Biologically Effective Dose (BED) calculator.
    
    This class implements methods for calculating BED, which accounts for
    the biological effect of different fractionation schemes using the
    Linear-Quadratic model.
    """
    
    @staticmethod
    def calculate(total_dose: float, dose_per_fraction: float, alpha_beta: float) -> float:
        """
        Calculate the Biologically Effective Dose (BED).
        
        Parameters
        ----------
        total_dose : float
            Total physical dose in Gy
        dose_per_fraction : float
            Dose per fraction in Gy
        alpha_beta : float
            Alpha/beta ratio in Gy
            
        Returns
        -------
        float
            BED value in Gy
        """
        bed = total_dose * (1 + dose_per_fraction / alpha_beta)
        return bed
    
    @staticmethod
    def calculate_eqd2(total_dose: float, dose_per_fraction: float, alpha_beta: float) -> float:
        """
        Calculate the EQD2 (equivalent dose in 2 Gy fractions).
        
        Parameters
        ----------
        total_dose : float
            Total physical dose in Gy
        dose_per_fraction : float
            Dose per fraction in Gy
        alpha_beta : float
            Alpha/beta ratio in Gy
            
        Returns
        -------
        float
            EQD2 value in Gy
        """
        eqd2 = total_dose * ((dose_per_fraction + alpha_beta) / (2 + alpha_beta))
        return eqd2
    
    @staticmethod
    def convert_bed_to_physical(bed: float, dose_per_fraction: float, alpha_beta: float) -> float:
        """
        Convert BED to physical dose.
        
        Parameters
        ----------
        bed : float
            Biologically Effective Dose in Gy
        dose_per_fraction : float
            Dose per fraction in Gy
        alpha_beta : float
            Alpha/beta ratio in Gy
            
        Returns
        -------
        float
            Total physical dose in Gy
        """
        physical_dose = bed / (1 + dose_per_fraction / alpha_beta)
        return physical_dose
    
    @staticmethod
    def calculate_bed_distribution(dose: np.ndarray, fractionation: Fractionation, 
                                  alpha_beta: float) -> np.ndarray:
        """
        Calculate the BED distribution from a dose distribution.
        
        Parameters
        ----------
        dose : np.ndarray
            Total dose distribution in Gy
        fractionation : Fractionation
            Fractionation scheme
        alpha_beta : float
            Alpha/beta ratio in Gy
            
        Returns
        -------
        np.ndarray
            BED distribution in Gy
        """
        dose_per_fraction = dose / fractionation.num_fractions
        bed = dose * (1 + dose_per_fraction / alpha_beta)
        return bed
    
    @staticmethod
    def calculate_eqd2_distribution(dose: np.ndarray, fractionation: Fractionation, 
                                   alpha_beta: float) -> np.ndarray:
        """
        Calculate the EQD2 distribution from a dose distribution.
        
        Parameters
        ----------
        dose : np.ndarray
            Total dose distribution in Gy
        fractionation : Fractionation
            Fractionation scheme
        alpha_beta : float
            Alpha/beta ratio in Gy
            
        Returns
        -------
        np.ndarray
            EQD2 distribution in Gy
        """
        dose_per_fraction = dose / fractionation.num_fractions
        eqd2 = dose * ((dose_per_fraction + alpha_beta) / (2 + alpha_beta))
        return eqd2
