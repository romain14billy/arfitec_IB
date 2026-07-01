import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.constants as cst
from config import PARAMS
    
""" Constantes """

Na = cst.Avogadro 
R_gaz = cst.R
Temp = 293
k_b = cst.Boltzmann
masse_n = 1.67492750056 * 10**(-27)
R_tube = 1.25
angle = 0 * np.pi / 180
atom_dens = 10**6 * Na / (R_gaz * Temp) / 1e6
eV = 1.602176634e-19
ED = 11e-6

t_min = PARAMS['t_min']
t_max = PARAMS['t_max']
E_min = PARAMS['E_min']
E_max = PARAMS['E_max']


def load_metadata(fichier):
    """Extrait les paramètres d'en-tête du fichier de données."""
    props = {}
    with open(fichier, "r") as f:
        for ligne in f:
            if "Dead time" in ligne: props['dead_time'] = float(ligne.split(":")[1])
            if "Channel width" in ligne: props['channel_width'] = float(ligne.split(":")[1])
            if "Number of frames" in ligne: props['nbr_frames'] = float(ligne.split(":")[1])
            if "Path length" in ligne: props['path_length'] = float(ligne.split(":")[1]) * 1e-2
    return props


def apply_dead_time_correction(counts, dead_time, nbr_frames, channel_width):
    """Corrige le déficit de comptage dû au temps mort du détecteur."""
    return counts / (1 - counts * dead_time / (nbr_frames * channel_width))


def remove_background(counts, n_pts=100):
    """Calcule et soustrait le bruit de fond moyen mesuré sur les derniers canaux."""
    bg = np.mean(counts[-n_pts:])
    return counts - bg


def apply_grouping_methode1(flux, M=20):
    """Lissage par moyenne glissante (Méthode 1)."""
    window = np.ones(M) / M
    return np.convolve(flux, window, mode='same')


def apply_grouping_methode2(data, N=10):
    """Regroupement par paquets fixes (Méthode 2)."""
    n_ptn = (len(data) // N) * N
    return np.mean(data[:n_ptn].reshape(-1, N), axis=1)


def compute_efficiency_tof(ToF, path_length):
    """Calcule le vecteur d'efficacité du détecteur en fonction du ToF."""
    def integrand(x, t):
        sigma = 848.24 * t / path_length * np.sqrt(2 * 1.6e-19 / masse_n) * 1e-24
        return np.exp(- (2 * atom_dens * sigma / np.cos(angle)) * np.sqrt(R_tube**2 - x**2))
    
    calcul_eff = lambda t: 1 - quad(integrand, 0, R_tube, args=(t, ))[0] / R_tube
    return np.vectorize(calcul_eff)(ToF)


def compute_efficiency_energy(E):
    def integrand(x, e):
        sigma = 848.24 / np.sqrt(e) * 1e-24
        return np.exp(- (2 * atom_dens * sigma / np.cos(angle)) * np.sqrt(R_tube**2 - x**2))
    calcul_eff = lambda e: 1 - quad(integrand, 0, R_tube, args=(e, ))[0] / R_tube
    return np.vectorize(calcul_eff)(E)


def convert_to_energy_scale(flux_tof, ToF, path_length):
    """Transforme le flux temporel en flux énergétique via le Jacobien."""
    E = 0.5 * masse_n * (path_length / ToF)**2 / eV
    E_joules = E * eV
    
    # Calcul du Jacobien |dt/dE|
    jacobian = 0.5 * path_length * np.sqrt(masse_n / (2 * E_joules**3))
    
    flux_E = flux_tof * jacobian
    flux_E2 = flux_E * E
    return E, flux_E, flux_E2

def fit_maxwellian_grid_search(ToF, flux, path_length):
    """Détermine la meilleure température par moindres carrés (incréments de 5K)."""
    Temp_K = np.arange(260, 400, 5)
    LSM = []

    for T in Temp_K:
        maxwellian = 0.5 * (masse_n / (k_b * T))**2 * path_length**4 / ToF**5 * np.exp(-masse_n * path_length**2 / (2 * k_b * T * ToF**2))
        
        fact_amplitude = np.max(flux) / np.max(maxwellian)
        maxwellian_norm = maxwellian * fact_amplitude
        
        # Calcul de l'erreur des moindres carrés (à partir du canal 20)
        err_global = np.sum((maxwellian_norm[20:] - flux[20:])**2)
        LSM.append(err_global)
        
    idx_best = np.argmin(LSM)
    return Temp_K[idx_best], LSM[idx_best]

def maxwell_model_tof(t, a0, a1):
    """Modèle maxwellien pur dans l'espace temporel ToF."""
    return a0 / t**5 * np.exp(-a1 / (t * 1e6)**2)


def model_tof_epi(t, a0, a1, a2, Ed, b, beta, E_array):
    """Modèle global combinant la contribution thermique (Maxwell) et épithermique."""
    F_M = (a0 / (t * 1e6)**5) * np.exp(-a1 / (t * 1e6)**2)
    F_E = a2 * (1 - np.exp(-(E_array / Ed)**2)) * E_array**(b - 1) * np.exp(-E_array / beta)
    return F_M + F_E


def calculate_r_squared(y_true, y_pred):
    """Calcule le coefficient de détermination R² pour évaluer la qualité d'un fit."""
    residus = y_true - y_pred
    ss_res = np.sum(residus**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def model_epi_pure(t, a2, a3, a4, a5):
    return a2 * (1 - np.exp(-a3 / (1e6 * t)**2)) * (1e6 * t)**a4 * np.exp(-a5 / (1e6 * t)**2)

def maxwell_model_E(E, a0, a1):
    """Modèle maxwellien standard dans l'espace énergétique."""
    return a0 * E * np.exp(-a1 * E)


def maxwell_model_E_corr(E, a0, a1):
    """Modèle maxwellien corrigé (multiplié par E) dans l'espace énergétique."""
    return a0 * E**2 * np.exp(-a1 * E)


def maxwell_epi_analytique_E(E, a0, a1, a2, Ed, b_param, beta_param):
    """Modèle global (Thermique + Épithermique) pour le spectre de flux en énergie Flux(E)."""
    F_M = a0 * E * np.exp(-a1 * E)
    F_E = a2 * (1 - np.exp(-(E / Ed)**2)) * E**(b_param - 1) * np.exp(-E / beta_param)
    return F_M + F_E


def maxwell_epi_analytique_E_corr(E, a0, a1, a2, Ed, b_param, beta_param):
    """Modèle global (Thermique + Épithermique) pour le spectre corrigé Flux(E) * E."""
    F_M = a0 * E**2 * np.exp(-a1 * E)
    F_E = a2 * (1 - np.exp(-(E / Ed)**2)) * E**b_param * np.exp(-E / beta_param)
    return F_M + F_E



def transmission_coeff(flux_sample, flux0):
    return (flux_sample / flux0)

def cross_section(Tr, d, n):
    return -1/(n*d)*np.log(Tr)*1e24


def process_neutron_data(fichier):
    """Exécute l'intégralité de la chaîne de traitement pour un fichier donné."""
    # 1. Chargement du fichier et des métadonnées
    meta = load_metadata(fichier)
    channels, counts = np.loadtxt(fichier, skiprows=15, unpack=True)
    
    # 2. Corrections primaires (Temps mort, bruit de fond)
    counts_dt = apply_dead_time_correction(counts, meta['dead_time'], meta['nbr_frames'], meta['channel_width'])
    counts_bg_corr = remove_background(counts_dt)
    
    therm_counts = remove_background(counts)
    
    # Normalisation par pulses et incertitudes
    flux_normalise = counts_bg_corr / meta['nbr_frames']
    therm_norm_flux = therm_counts / meta['nbr_frames']
    unc_normalisee = np.sqrt(counts) / meta['nbr_frames']
    
    # 3. Calcul de la cinétique temporelle (ToF)
    ToF = (meta['channel_width'] * channels * 1e-6) - ED
    
    # 4. Lissage et correction d'efficacité (Espace Temporel)
    flux_lisse = apply_grouping_methode1(flux_normalise)
    mean_therm_norm_flux = apply_grouping_methode1(therm_norm_flux)
    eff_ToF = compute_efficiency_tof(ToF, meta['path_length'])
    flux_tof_ungrouped = flux_normalise / eff_ToF
    flux_final_tof = flux_lisse / eff_ToF
    unc_flux_reelle = unc_normalisee / eff_ToF
    
    # 5. Méthode de regroupement 2 (Paquets de 10)
    flux_grouped = apply_grouping_methode2(flux_normalise)
    channels_grouped = apply_grouping_methode2(channels)
    ToF_grouped = apply_grouping_methode2(ToF)
    unc_grouped = apply_grouping_methode2(unc_normalisee) / np.sqrt(10)
    
    flux_tof_grouped = apply_grouping_methode2(flux_final_tof)
    
    # 6. Changement de variable vers l'Espace Énergétique
    E, flux_E, flux_E2 = convert_to_energy_scale(flux_final_tof, ToF, meta['path_length'])
    eff_E = compute_efficiency_energy(E)
    
    # Dictionnaire de sortie nettoyé
    return {
        'meta': meta,
        'channels': channels,
        'channels_grouped': channels_grouped,
        'ToF': ToF,
        'ToF_grouped': ToF_grouped,
        'flux_normalise': flux_normalise,
        'flux_lisse': flux_lisse,
        'mean_therm_norm_flux': mean_therm_norm_flux,
        'flux_grouped': flux_grouped,
        'flux_tof_ungrouped':flux_tof_ungrouped,
        'flux_tof': flux_final_tof,
        'flux_tof_grouped': flux_tof_grouped,
        'unc_tof': unc_normalisee,
        'unc_tof_grouped': unc_grouped,
        'unc_flux_reelle':unc_flux_reelle,
        'E': E,
        'eff_E': eff_E,
        'eff_ToF': eff_ToF,
        'flux_E': flux_E,
        'flux_E2': flux_E2
    }


Temp_K = np.arange(260,400,5)
N_temp = len(Temp_K)
cmap = plt.cm.coolwarm  