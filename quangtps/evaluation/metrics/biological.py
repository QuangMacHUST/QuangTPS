"""
Biological models for treatment plan evaluation.

This module provides Lyman-Kutcher-Burman (LKB) models for calculating 
TCP (Tumor Control Probability) and NTCP (Normal Tissue Complication
Probability) for radiotherapy treatment plan evaluation.
"""

import numpy as np
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any

from quangtps.core.types import DoseGrid
from quangtps.treatment.fractionation import Fractionation

logger = logging.getLogger(__name__)


class TCP_LKB:
    """
    Tumor Control Probability model using Lyman-Kutcher-Burman approach.
    
    This class implements the LKB model for predicting tumor control based on
    dose distribution, specifically tailored for evaluating treatment plans.
    """
    
    @staticmethod
    def calculate(dose: np.ndarray, td50: float, m: float, n: float, 
                  voxel_volume: Union[float, np.ndarray] = 1.0) -> float:
        """
        Calculate TCP using the Lyman-Kutcher-Burman model.
        
        Parameters
        ----------
        dose : np.ndarray
            Dose distribution in Gy
        td50 : float
            Dose that gives 50% TCP in Gy for uniform dose
        m : float
            Steepness parameter
        n : float
            Volume effect parameter
        voxel_volume : float or np.ndarray
            Volume of each voxel in cm³
            
        Returns
        -------
        float
            TCP value (0-1)
        """
        # Convert dose array to flattened array if it's not already
        if len(dose.shape) > 1:
            dose_flat = dose.flatten()
        else:
            dose_flat = dose
            
        # Handle voxel volume
        if isinstance(voxel_volume, (int, float)):
            vol_flat = np.ones_like(dose_flat) * voxel_volume
        else:
            if len(voxel_volume.shape) > 1:
                vol_flat = voxel_volume.flatten()
            else:
                vol_flat = voxel_volume
                
        # Calculate total volume
        total_volume = np.sum(vol_flat)
        
        # Calculate generalized equivalent uniform dose (gEUD)
        if n != 1:
            # For n != 1, use power-law relationship
            dose_vol_powers = np.power(dose_flat, 1/n) * vol_flat
            gEUD = np.power(np.sum(dose_vol_powers) / total_volume, n)
        else:
            # For n = 1, geometric mean (volume weighted)
            log_dose = np.log(dose_flat)
            gEUD = np.exp(np.sum(log_dose * vol_flat) / total_volume)
            
        # Calculate TCP using probit function
        t = (gEUD - td50) / (m * td50)
        tcp = 0.5 * (1 + math.erf(t / np.sqrt(2)))
        
        return tcp
    
    @staticmethod
    def calculate_from_dvh(dose_bins: np.ndarray, volume_bins: np.ndarray, 
                           td50: float, m: float, n: float) -> float:
        """
        Calculate TCP from a dose-volume histogram using the LKB model.
        
        Parameters
        ----------
        dose_bins : np.ndarray
            Dose bins in Gy
        volume_bins : np.ndarray
            Differential volume histogram (fraction of volume per dose bin)
        td50 : float
            Dose that gives 50% TCP in Gy for uniform dose
        m : float
            Steepness parameter
        n : float
            Volume effect parameter
            
        Returns
        -------
        float
            TCP value (0-1)
        """
        # Ensure volume bins are normalized
        volume_bins_norm = volume_bins / np.sum(volume_bins)
        
        # Calculate generalized equivalent uniform dose (gEUD)
        if n != 1:
            # For n != 1, use power-law relationship
            dose_vol_powers = np.power(dose_bins, 1/n) * volume_bins_norm
            gEUD = np.power(np.sum(dose_vol_powers), n)
        else:
            # For n = 1, geometric mean (volume weighted)
            log_dose = np.log(dose_bins)
            gEUD = np.exp(np.sum(log_dose * volume_bins_norm))
            
        # Calculate TCP using probit function
        t = (gEUD - td50) / (m * td50)
        tcp = 0.5 * (1 + math.erf(t / np.sqrt(2)))
        
        return tcp


class NTCP_LKB:
    """
    Normal Tissue Complication Probability model using Lyman-Kutcher-Burman approach.
    
    This class implements the LKB model for predicting normal tissue complications
    based on dose distribution, specifically tailored for evaluating treatment plans.
    """
    
    @staticmethod
    def calculate(dose: np.ndarray, td50: float, m: float, n: float, 
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
        # Convert dose array to flattened array if it's not already
        if len(dose.shape) > 1:
            dose_flat = dose.flatten()
        else:
            dose_flat = dose
            
        # Handle voxel volume
        if isinstance(voxel_volume, (int, float)):
            vol_flat = np.ones_like(dose_flat) * voxel_volume
        else:
            if len(voxel_volume.shape) > 1:
                vol_flat = voxel_volume.flatten()
            else:
                vol_flat = voxel_volume
                
        # Calculate total volume
        total_volume = np.sum(vol_flat)
        
        # Calculate generalized equivalent uniform dose (gEUD)
        if n != 1:
            # For n != 1, use power-law relationship
            dose_vol_powers = np.power(dose_flat, 1/n) * vol_flat
            gEUD = np.power(np.sum(dose_vol_powers) / total_volume, n)
        else:
            # For n = 1, geometric mean (volume weighted)
            log_dose = np.log(dose_flat)
            gEUD = np.exp(np.sum(log_dose * vol_flat) / total_volume)
            
        # Calculate NTCP using probit function
        t = (gEUD - td50) / (m * td50)
        ntcp = 0.5 * (1 + math.erf(t / np.sqrt(2)))
        
        return ntcp
    
    @staticmethod
    def calculate_from_dvh(dose_bins: np.ndarray, volume_bins: np.ndarray, 
                           td50: float, m: float, n: float) -> float:
        """
        Calculate NTCP from a dose-volume histogram using the LKB model.
        
        Parameters
        ----------
        dose_bins : np.ndarray
            Dose bins in Gy
        volume_bins : np.ndarray
            Differential volume histogram (fraction of volume per dose bin)
        td50 : float
            Dose that gives 50% NTCP in Gy for uniform dose
        m : float
            Steepness parameter
        n : float
            Volume effect parameter
            
        Returns
        -------
        float
            NTCP value (0-1)
        """
        # Ensure volume bins are normalized
        volume_bins_norm = volume_bins / np.sum(volume_bins)
        
        # Calculate generalized equivalent uniform dose (gEUD)
        if n != 1:
            # For n != 1, use power-law relationship
            dose_vol_powers = np.power(dose_bins, 1/n) * volume_bins_norm
            gEUD = np.power(np.sum(dose_vol_powers), n)
        else:
            # For n = 1, geometric mean (volume weighted)
            log_dose = np.log(dose_bins)
            gEUD = np.exp(np.sum(log_dose * volume_bins_norm))
            
        # Calculate NTCP using probit function
        t = (gEUD - td50) / (m * td50)
        ntcp = 0.5 * (1 + math.erf(t / np.sqrt(2)))
        
        return ntcp
