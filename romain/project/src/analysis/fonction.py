import numpy as np
import NCrystal as NC
from scipy.signal import savgol_filter

from src.analysis.result import (
    Result,
    Spectrum,
    DualSpectrum,
    TOFDomain,
    EnergyDomain,
    CrossSection,
    ErrorEnergy,
    ErrorTime,
)

# =====================================================
# CONSTANTES PHYSIQUES ET SEUILS NUMÉRIQUES
# =====================================================

# seuil minimal pour éviter log(0) ou divisions instables
INTENSITY_MIN = 1e-8

# constantes physiques
E_CHARGE = 1.602176634e-19  # charge élémentaire (C)
M_NEUTRON = 1.674927498e-27  # masse neutron (kg)

# densité atomique (atomes/cm³)
ATOM_DENSITY = {
    "Bi": 2.82621e22,
    "Cu": 8.49e22,
    "Al": 6.03e22,
    "Fe": 8.49e22,
}


# =====================================================
# LISSAGE DE SIGNAL
# =====================================================

def smooth(y, window, polyorder):
    """
    Lisse un signal avec Savitzky-Golay après nettoyage NaN.

    Étapes :
    - conversion numpy
    - vérification taille minimale
    - interpolation des NaN
    - application filtre SG
    """

    y = np.asarray(y)

    # signal trop court → pas de lissage
    if len(y) < window:
        return y

    # masque valeurs valides
    mask = np.isfinite(y)

    # trop peu de données valides
    if np.sum(mask) < window:
        return y

    # interpolation des trous
    y_interp = np.interp(
        np.arange(len(y)),
        np.where(mask)[0],
        y[mask],
    )

    # lissage final
    return savgol_filter(y_interp, window, polyorder)


# =====================================================
# CHARGEMENT DES SPECTRES EXPÉRIMENTAUX
# =====================================================

def load_spectrum(folder):
    """
    Charge les fichiers e.dat et tof.dat.
    Format attendu :
        colonne 0 = axe (E ou TOF)
        colonne 1 = intensité
        colonne 2 = erreur sur l'intensité
    """

    e = np.loadtxt(folder / "e.dat", comments="#")
    tof = np.loadtxt(folder / "tof.dat", comments="#")

    return {
        "E": e[:, 0],
        "IE": e[:, 1],
        "IEERR": e[:, 2],
        "TOF": tof[:, 0],
        "ITOF": tof[:, 1],
        "ITOFERR": tof[:, 2],
    }


# =====================================================
# CONVERSION D’UNITÉS ET NORMALISATION
# =====================================================

def convert_units(data):
    """
    Convertit unités expérimentales :
    - énergie : keV → MeV
    - TOF : µs → s

    Puis convertit intensités et leurs erreurs en densité spectrale.
    """

    E = data["E"] * 1e-3
    TOF = data["TOF"] * 1e-6

    # dérivées pour normalisation spectrale
    dE = np.gradient(E)
    dTOF = np.gradient(TOF)

    IE = data["IE"] / np.where(dE == 0, 1, dE)
    IEERR = data["IEERR"] / np.where(dE == 0, 1, dE)
    ITOF = data["ITOF"] / np.where(dTOF == 0, 1, dTOF)
    ITOFERR = data["ITOFERR"] / np.where(dTOF == 0, 1, dTOF)

    return E, IE, IEERR, TOF, ITOF, ITOFERR


# =====================================================
# CONVERSION TOF → ÉNERGIE PHYSIQUE
# =====================================================

def tof_to_energy(tof, distance):
    """
    Conversion temps de vol → énergie neutron.
    """

    tof = np.asarray(tof)

    return (
        M_NEUTRON
        * distance**2
        / (2 * E_CHARGE * tof**2)
    )


# =====================================================
# NETTOYAGE DE SPECTRES
# =====================================================

def make_spectrum(x, y, yerr=None, name=None):
    """
    Crée un Spectrum propre :
    - supprime NaN
    - vérifie cohérence
    - trie par axe x
    """

    x = np.asarray(x)
    y = np.asarray(y)

    try:
        x = x.astype(float)
    except (TypeError, ValueError):
        x = np.full_like(x, np.nan, dtype=float)

    try:
        y = y.astype(float)
    except (TypeError, ValueError):
        y = np.full_like(y, np.nan, dtype=float)

    if yerr is None:
        yerr = np.zeros_like(y, dtype=float)
    else:
        yerr = np.asarray(yerr)
        try:
            yerr = yerr.astype(float)
        except (TypeError, ValueError):
            yerr = np.full_like(y, np.nan, dtype=float)

    # suppression valeurs invalides
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]

    # sécurité structurelle
    if not (len(x) == len(y) == len(yerr)):
        raise ValueError(f"Inconsistent spectrum {name}")

    # tri
    idx = np.argsort(x)

    return Spectrum(x[idx], y[idx], yerr[idx])


def clean_spectrum(x, y, yerr=None):
    """
    Nettoyage brut sans encapsulation Spectrum :
    - NaN removal
    - tri
    - suppression doublons
    """

    x = np.asarray(x)
    y = np.asarray(y)

    try:
        x = x.astype(float)
    except (TypeError, ValueError):
        x = np.full_like(x, np.nan, dtype=float)

    try:
        y = y.astype(float)
    except (TypeError, ValueError):
        y = np.full_like(y, np.nan, dtype=float)

    if yerr is None:
        yerr = np.zeros_like(y, dtype=float)
    else:
        yerr = np.asarray(yerr)
        try:
            yerr = yerr.astype(float)
        except (TypeError, ValueError):
            yerr = np.full_like(y, np.nan, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    x = x[mask]
    y = y[mask]
    yerr = yerr[mask]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    yerr = yerr[idx]

    # suppression valeurs x dupliquées
    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]
    yerr_unique = yerr[unique_idx]

    return x_unique, y_unique, yerr_unique


# =====================================================
# INTERPOLATION ROBUSTE
# =====================================================

def interp1d_safe(x, y, x_new):
    """
    Interpolation sécurisée :
    - gère NaN
    - vérifie suffisance données
    - fallback NaN si impossible
    """

    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    if len(x_unique) < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    return np.interp(x_new, x_unique, y_unique)


# =====================================================
# TOF → ÉNERGIE + INTENSITÉ PHYSIQUE
# =====================================================

def tof_spectrum_to_energy(tof, intensity, intensity_err, distance):
    """
    Convertit un spectre TOF en spectre énergie et propage l'erreur.
    """

    tof = np.asarray(tof)
    intensity = np.asarray(intensity)
    intensity_err = np.asarray(intensity_err)

    # filtrage initial
    mask = (
        np.isfinite(tof)
        & np.isfinite(intensity)
        & np.isfinite(intensity_err)
        & (tof > 0)
        & (intensity > INTENSITY_MIN)
    )

    tof = tof[mask]
    intensity = intensity[mask]
    intensity_err = intensity_err[mask]

    # conversion énergie
    E = tof_to_energy(tof, distance)

    # filtrage énergie
    mask = np.isfinite(E) & (E > 0)

    E = E[mask]
    tof = tof[mask]
    intensity = intensity[mask]
    intensity_err = intensity_err[mask]

    # correction jacobien physique
    jacobian = tof / (2 * E)
    I = intensity * jacobian
    I_err = intensity_err * jacobian

    # nettoyage final
    E, I, I_err = clean_spectrum(E, I, I_err)

    return make_spectrum(E, I, I_err, "energy_from_tof")


# =====================================================
# SECTION EFFICACE EXPÉRIMENTALE
# =====================================================

def compute_cross_section(
    e_sample,
    i_sample,
    i_sample_err,
    e_ref,
    i_ref,
    i_ref_err,
    cg,
    thickness_cm,
    atom_density,
):
    """
    Calcule section efficace et son incertitude :
    σ = -ln(I/I0) / (n * t)
    """

    # interpolation sur grille commune
    i_sample_cg = np.interp(cg, e_sample, i_sample, left=np.nan, right=np.nan)
    i_ref_cg = np.interp(cg, e_ref, i_ref, left=np.nan, right=np.nan)
    i_sample_err_cg = np.interp(cg, e_sample, i_sample_err, left=np.nan, right=np.nan)
    i_ref_err_cg = np.interp(cg, e_ref, i_ref_err, left=np.nan, right=np.nan)

    mask = (
        np.isfinite(i_sample_cg)
        & np.isfinite(i_ref_cg)
        & np.isfinite(i_sample_err_cg)
        & np.isfinite(i_ref_err_cg)
        & (i_sample_cg > 0)
        & (i_ref_cg > 0)
    )

    ratio = np.full_like(cg, np.nan, dtype=float)
    sigma = np.full_like(cg, np.nan, dtype=float)
    sigma_err = np.full_like(cg, np.nan, dtype=float)

    ratio[mask] = i_sample_cg[mask] / i_ref_cg[mask]

    valid = ratio > 0
    sigma[valid] = -np.log(ratio[valid]) / thickness_cm

    relative_uncertainty = np.full_like(cg, np.nan, dtype=float)
    relative_uncertainty[mask] = np.sqrt(
        (i_sample_err_cg[mask] / i_sample_cg[mask]) ** 2
        + (i_ref_err_cg[mask] / i_ref_cg[mask]) ** 2
    )

    sigma_err[valid] = relative_uncertainty[valid] / thickness_cm

    factor = 1e24 / atom_density
    return sigma * factor, sigma_err * factor


# =====================================================
# ERREUR RELATIVE
# =====================================================

def relative_error(x_ref, y_ref, x_test, y_test, cg):
    """
    Erreur relative en %.
    """

    y_ref_cg = np.interp(cg, x_ref, y_ref, left=np.nan, right=np.nan)
    y_test_cg = np.interp(cg, x_test, y_test, left=np.nan, right=np.nan)

    out = np.full_like(cg, np.nan, dtype=float)

    mask = np.isfinite(y_ref_cg) & np.isfinite(y_test_cg) & (np.abs(y_ref_cg) > 0)

    out[mask] = (y_test_cg[mask] - y_ref_cg[mask]) / np.abs(y_ref_cg[mask]) * 100.0

    return out


# =====================================================
# NCRYSTAL
# =====================================================

def ncrystal_sigma(ncmat_file, energies):
    """
    Section efficace théorique via NCrystal.
    """

    mat = NC.load(ncmat_file)
    scatter = mat.scatter

    return np.array([
        scatter.crossSection(E, (0, 0, 1))
        for E in energies
    ])


def _resolve_ncmat_name(config, material):
    """Return the NCrystal file name from config['physics'] or a default one."""
    physics = config.get("physics", {})
    ncmat_name = physics.get("NCrystal")
    if ncmat_name:
        return str(ncmat_name)

    return f"{material}_sg166.ncmat"


def compute_ncrystal_grid_and_sigma(config, material, monitor_x, npoints=1000):
    """Return (E_ref, sigma_nc).

    The first call for a given material computes the NCrystal curve on a log-spaced
    energy grid. Subsequent calls reuse the cached curve and interpolate it onto the
    requested grid, so the expensive NCrystal evaluation is done only once.
    """
    ncmat_name = _resolve_ncmat_name(config, material)

    finite = np.isfinite(monitor_x)
    if not np.any(finite):
        return np.copy(monitor_x), np.full_like(monitor_x, np.nan, dtype=float)

    emin = float(np.nanmin(monitor_x[finite]))
    emax = float(np.nanmax(monitor_x[finite]))
    emin = max(emin, 1e-12)

    if emin <= 0 or emin == emax:
        E_ref = np.copy(monitor_x)
    else:
        E_ref = np.logspace(np.log10(emin), np.log10(emax), npoints)

    sigma_nc = ncrystal_sigma(ncmat_name, E_ref)
    return E_ref, sigma_nc

# =====================================================
# NORMALIZE 
# =====================================================

def normalize_to_target_counts(sm, am, target_counts=10000):
    """
    Normalise tous les histogrammes à un nombre cible de neutrons
    détectés dans le SMAT, tout en conservant les rapports MAT/SMAT.
    """

    n_smat = np.sum(sm["IE"])

    if n_smat <= 0:
        raise ValueError("SMAT contains no counts")

    scale = target_counts / n_smat

    for key in ("IE", "ITOF", "IEERR", "ITOFERR"):
        sm[key] *= scale
        am[key] *= scale

    return scale


# =====================================================
# PIPELINE EXPÉRIMENTAL PRINCIPAL
# =====================================================

def process_experiment(exp, config):
    """
    Pipeline complet :
    - chargement
    - conversion
    - nettoyage
    - TOF → énergie
    - smoothing
    - cross sections
    - erreurs
    - packaging Result
    """

    print(f"Traitement {exp.name}")

    thickness_cm = config["physics"]["thickness_cm"]
    material = config["physics"]["material"]
    atom_density = ATOM_DENSITY[material]
    distance = config["physics"]["distance_tof_m"]

    smoothing_window = config["smoothing"]["window"]
    polyorder = config["smoothing"]["polyorder"]

    # =================================================
    # LOAD
    # =================================================
    sm = load_spectrum(exp.smat)
    am = load_spectrum(exp.mat)

    # ==============================================
    # NORMALISATION ABSOLUE
    # ==============================================

    scale = normalize_to_target_counts(
        sm,
        am,
        target_counts=10000,
    )

    print(f"Normalisation appliquée : facteur = {scale:.6e}")

    e_sm, ie_sm, ie_sm_err, tof_sm, itof_sm, itof_sm_err = convert_units(sm)
    e_am, ie_am, ie_am_err, tof_am, itof_am, itof_am_err = convert_units(am)

    # =================================================
    # CLEAN MONITORS
    # =================================================
    e_sm, ie_sm, ie_sm_err = clean_spectrum(e_sm, ie_sm, ie_sm_err)
    e_am, ie_am, ie_am_err = clean_spectrum(e_am, ie_am, ie_am_err)

    monitor_smat = make_spectrum(e_sm, ie_sm, ie_sm_err, "energy_monitor_smat")
    monitor_mat = make_spectrum(e_am, ie_am, ie_am_err, "energy_monitor_mat")

    # =================================================
    # TOF SPECTRA
    # =================================================
    tof_smat = make_spectrum(*clean_spectrum(tof_sm, itof_sm, itof_sm_err), "tof_smat")
    tof_mat = make_spectrum(*clean_spectrum(tof_am, itof_am, itof_am_err), "tof_mat")

    # =================================================
    # TOF → ENERGY
    # =================================================
    energy_tof_smat = tof_spectrum_to_energy(tof_sm, itof_sm, itof_sm_err, distance)
    energy_tof_mat = tof_spectrum_to_energy(tof_am, itof_am, itof_am_err, distance)

    # =================================================
    # SMOOTHING
    # =================================================
    monitor_smat_smooth = smooth(monitor_smat.y, smoothing_window, polyorder)
    monitor_smat_smooth_err = smooth(monitor_smat.yerr, smoothing_window, polyorder)
    monitor_mat_smooth = smooth(monitor_mat.y, smoothing_window, polyorder)
    monitor_mat_smooth_err = smooth(monitor_mat.yerr, smoothing_window, polyorder)
    tof_smat_smooth = smooth(energy_tof_smat.y, smoothing_window, polyorder)
    tof_smat_smooth_err = smooth(energy_tof_smat.yerr, smoothing_window, polyorder)
    tof_mat_smooth = smooth(energy_tof_mat.y, smoothing_window, polyorder)
    tof_mat_smooth_err = smooth(energy_tof_mat.yerr, smoothing_window, polyorder)

    # =================================================
    # CROSS SECTIONS (simplified)
    # =================================================
    E_ref, sigma_nc = compute_ncrystal_grid_and_sigma(
        config,
        material,
        monitor_smat.x,
        npoints=1000,
    )

    sigma_monitor, sigma_monitor_err = compute_cross_section(
        monitor_mat.x,
        monitor_mat_smooth,
        monitor_mat_smooth_err,
        monitor_smat.x,
        monitor_smat_smooth,
        monitor_smat_smooth_err,
        monitor_smat.x,
        thickness_cm,
        atom_density,
    )

    sigma_tof, sigma_tof_err = compute_cross_section(
        energy_tof_mat.x,
        tof_mat_smooth,
        tof_mat_smooth_err,
        energy_tof_smat.x,
        tof_smat_smooth,
        tof_smat_smooth_err,
        energy_tof_smat.x,
        thickness_cm,
        atom_density,
    )

    # =================================================
    # ERREURS
    # =================================================
    err_energy_smat = relative_error(
        monitor_smat.x,
        monitor_smat.y,
        energy_tof_smat.x,
        energy_tof_smat.y,
        energy_tof_smat.x,
    )

    err_energy_mat = relative_error(
        monitor_mat.x,
        monitor_mat.y,
        energy_tof_mat.x,
        energy_tof_mat.y,
        energy_tof_mat.x,
    )

    sigma_monitor_on_tof = interp1d_safe(
        monitor_smat.x,
        sigma_monitor,
        energy_tof_smat.x,
    )
    sigma_nc_on_tof = interp1d_safe(
        E_ref,
        sigma_nc,
        energy_tof_smat.x,
    )

    err_sigma_tof_monitor = relative_error(
        energy_tof_smat.x,
        sigma_monitor_on_tof,
        energy_tof_smat.x,
        sigma_tof,
        energy_tof_smat.x,
    )

    err_sigma_tof_ncrystal = relative_error(
        energy_tof_smat.x,
        sigma_nc_on_tof,
        energy_tof_smat.x,
        sigma_tof,
        energy_tof_smat.x,
    )

    sigma_nc_on_monitor = interp1d_safe(
        E_ref,
        sigma_nc,
        monitor_smat.x,
    )
    err_sigma_monitor_ncrystal = relative_error(
        monitor_smat.x,
        sigma_nc_on_monitor,
        monitor_smat.x,
        sigma_monitor,
        monitor_smat.x,
    )


    # =================================================
    # RESULT FINAL
    # =================================================
    return Result(
        name=exp.name,
        tof=TOFDomain(
            tof=DualSpectrum(
                smat=tof_smat,
                mat=tof_mat,
            )
        ),
        energy=EnergyDomain(
            monitor=DualSpectrum(
                smat=monitor_smat,
                mat=monitor_mat,
            ),
            tof=DualSpectrum(
                smat=energy_tof_smat,
                mat=energy_tof_mat,
            ),
        ),
        sigma=CrossSection(
            tof=make_spectrum(
                energy_tof_smat.x,
                sigma_tof,
                sigma_tof_err,
                "sigma_tof",
            ),
            monitor=make_spectrum(
                monitor_smat.x,
                sigma_monitor,
                sigma_monitor_err,
                "sigma_monitor",
            ),
            ncrystal=make_spectrum(E_ref, sigma_nc, np.zeros_like(E_ref), "sigma_ncrystal"),
        ),
        error_energy=ErrorEnergy(
            energy=DualSpectrum(
                smat=make_spectrum(
                    energy_tof_smat.x,
                    err_energy_smat,
                    np.zeros_like(err_energy_smat),
                    "error_energy_smat",
                ),
                mat=make_spectrum(
                    energy_tof_mat.x,
                    err_energy_mat,
                    np.zeros_like(err_energy_mat),
                    "error_energy_mat",
                ),
            ),
            cross_section_tof_vs_Ncrystal=make_spectrum(
                energy_tof_smat.x,
                err_sigma_tof_ncrystal,
                np.zeros_like(err_sigma_tof_ncrystal),
                "error_sigma_tof_ncrystal",
            ),
            cross_section_etof_and_emonitor=make_spectrum(
                energy_tof_smat.x,
                err_sigma_tof_monitor,
                np.zeros_like(err_sigma_tof_monitor),
                "error_sigma_tof_monitor",
            ),
            cross_section_monitor_vs_Ncrystal=make_spectrum(
                monitor_smat.x,
                err_sigma_monitor_ncrystal,
                np.zeros_like(err_sigma_monitor_ncrystal),
                "error_sigma_monitor_ncrystal",
            ),
        ),
        error_time=ErrorTime(
            tof=DualSpectrum(
                smat=make_spectrum(
                    tof_smat.x,
                    np.zeros_like(tof_smat.x),
                    np.zeros_like(tof_smat.x),
                    "error_time_smat",
                ),
                mat=make_spectrum(
                    tof_mat.x,
                    np.zeros_like(tof_mat.x),
                    np.zeros_like(tof_mat.x),
                    "error_time_mat",
                ),
            )
        ),
    )