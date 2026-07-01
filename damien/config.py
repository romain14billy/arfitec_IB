
import numpy as np

# Dictionnaire global des paramètres physiques modifiables par l'IHM
PARAMS = {
    "thickness": 0.8,         # Épaisseur de l'échantillon (cm)
    "atom_density": 8.49e22,   # Densité atomique (cm^-3)
    "Length": 1.9, 

    "E_min": 3e-3,            # Énergie minimale (eV)
    "E_max": 200e-3,          # Énergie maximale (eV)
    "t_min": 200e-6,          # Temps de vol minimal (s)
    "t_max": 3700e-6,         # Temps de vol maximal (s)
    "y_min": 0.0,
    "y_max": 20.0,
}

# Grille de température pour les tracés théoriques (Maxwelliens)
Temp_K = np.arange(260, 400, 5)
N_temp = len(Temp_K)