import numpy as np

def get_lambda(t_half=2.5785):
    """Calcule la constante de déchéance (s^-1). t_half: période en heures."""
    return np.log(2) / (t_half * 3600)

def get_R(counts, m, t_i, t_d, t_m, M=54.938, C=1.0, y=0.989, eps=1.0, eta=1.0):
    """
    Calcule le taux de réaction R[cite: 1].
    counts: comptes nets, m: masse (g), t_x: temps d'irradiation/décroissance/mesure (s)[cite: 1]
    M: masse molaire, C: concentration, y: rendement, eps: efficacité, eta: abondance[cite: 1]
    """
    N_A = 6.02214076e23
    lmbda = get_lambda()
    
    num = counts * M * lmbda
    den = (C * y * eps * eta * m * N_A * 
           (1 - np.exp(-lmbda * t_i)) * 
           np.exp(-lmbda * t_d) * 
           (1 - np.exp(-lmbda * t_m)))
    
    return num / den

def get_flux(R, R_Cd, F_Cd, G_th, G_epi, sig_th, sig_epi):
    """
    Calcule phi_th (flux thermique) et phi_epi (flux épithermique)[cite: 1].
    R/R_Cd: taux sans/avec Cd, F_Cd: facteur Cd[cite: 1]
    G_th/G_epi: auto-protection, sig_th/sig_epi: sections efficaces[cite: 1]
    """
    R_epi = F_Cd * R_Cd # Taux d'activation épithermique corrigé[cite: 1]
    phi_epi = R_epi / (G_epi * sig_epi)
    phi_th = (R - R_epi) / (G_th * sig_th)
    
    return phi_th, phi_epi