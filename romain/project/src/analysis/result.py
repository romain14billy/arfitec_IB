"""Structures de données utilisées pour représenter les résultats d'analyse."""

from dataclasses import dataclass
import numpy as np


# =====================================================
# BASE OBJECT
# =====================================================

@dataclass
class Spectrum:
    """
    1D physical spectrum

    x = axis (energy, time, etc.)
    y = value (counts, sigma, error, etc.)
    """
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray


# =====================================================
# PAIR SMAT / MAT
# =====================================================

@dataclass
class DualSpectrum:
    """
    Same observable:
    - smat = sans matériau
    - mat  = avec matériau
    """
    smat: Spectrum
    mat: Spectrum


# =====================================================
# TIME OF FLIGHT DOMAIN
# =====================================================

@dataclass
class TOFDomain:
    """
    Time-of-flight measurements (raw MCSTAS output)
    """
    tof: DualSpectrum

    unit_x: str = "s"
    unit_y: str = "counts"


# =====================================================
# ENERGY DOMAIN
# =====================================================

@dataclass
class EnergyDomain:
    """
    Energy representations:
    - monitor = direct MCSTAS energy detector
    - tof     = reconstructed from TOF
    """
    monitor: DualSpectrum
    tof: DualSpectrum

    unit_x: str = "eV"
    unit_y: str = "counts"


# =====================================================
# CROSS SECTION DOMAIN
# =====================================================

@dataclass
class CrossSection:
    """
    Sigma(E)
    All curves expressed as function of energy.
    """
    tof: Spectrum
    monitor: Spectrum
    ncrystal: Spectrum

    unit_x: str = "eV"
    unit_y: str = "barn"


# =====================================================
# RELATIVE ERROR DOMAIN
# =====================================================

@dataclass
class ErrorEnergy:
    """
    Relative errors in energy domain (%)
    """
    energy: DualSpectrum

    cross_section_tof_vs_Ncrystal: Spectrum
    cross_section_etof_and_emonitor: Spectrum
    cross_section_monitor_vs_Ncrystal: Spectrum

    unit_x: str = "eV"
    unit_y: str = "%"


@dataclass
class ErrorTime:
    """
    Relative errors in TOF domain (%)
    """
    tof: DualSpectrum

    unit_x: str = "s"
    unit_y: str = "%"


# =====================================================
# FINAL RESULT
# =====================================================

@dataclass
class Result:
    name: str

    tof: TOFDomain
    energy: EnergyDomain
    sigma: CrossSection

    error_energy: ErrorEnergy
    error_time: ErrorTime