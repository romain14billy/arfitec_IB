import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button,CheckButtons
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk

from physics import Na, k_b, R_gaz, R_tube, masse_n, angle, atom_dens, eV
from physics import model_epi_pure, fit_maxwellian_grid_search, maxwell_model_tof, model_tof_epi, calculate_r_squared
from physics import maxwell_model_E, maxwell_model_E_corr, maxwell_epi_analytique_E, maxwell_epi_analytique_E_corr
from physics import cross_section, transmission_coeff, apply_grouping_methode1, apply_grouping_methode2
from physics import compute_cross_section_uncertainty, read_reference_file, compute_amp_init_from_ref, compute_grouping_cross_section_m2
from physics import compute_fit_results_from_dataset, compute_plot8_models

from config import PARAMS

Temp_K = np.arange(260, 400, 5)
N_temp = len(Temp_K)
cmap = plt.cm.coolwarm  


def _integrer_canvas(fig, frame):
    """Gère l'affichage de la figure dans un conteneur Tkinter."""
    if frame is not None:
        # Nettoyage des anciens widgets dans la frame pour éviter la superposition
        for widget in frame.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    else:
        plt.show()


def plot_1(fichiers, datasets, frame=None):
    print('Calculating please wait...')
    fig, ax = plt.subplots(figsize=(12, 5))
    unc_color = (0.0, 1.0, 0.0, 0.5)
    
    for i, nom in enumerate(fichiers):
        data = datasets[nom]
        
        label_exp = "experimental distribution" if i == 0 else None
        label_g1 = "grouping methode 1" if i == 0 else None
        label_g2 = "grouping methode 2" if i == 0 else None
        
        ax.plot(data['channels'], data['flux_normalise'], '.-', markersize=8, linewidth=0.3, color=unc_color, label=label_exp)
        ax.plot(data['channels'], data['flux_lisse'], '.-', markersize=6, linewidth=0.3, color="red", label=label_g1)
        ax.plot(data['channels_grouped'], data['flux_grouped'], '.-', markersize=6, linewidth=0.3, color="blue", label=label_g2)
        
        idx_max = np.argmax(data['flux_lisse'])
        ch_max = data['channels'][idx_max]
        val_max = data['flux_lisse'][idx_max]
        ax.text(ch_max, val_max * 1.02, nom, fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    ax.set_xlabel('channels')
    ax.set_ylabel('counts')
    ax.set_title('Time of flight spectrum')    
    ax.legend(labelcolor=['green', 'red', 'blue'], markerscale=3.0, handlelength=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    _integrer_canvas(fig, frame)
    return fig

def plot_2(fichiers, datasets, frame=None):
    print('Calculating please wait...')
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, nom in enumerate(fichiers):
        data = datasets[nom]
        
        p_exp = ax.errorbar(data['channels'], data['mean_therm_norm_flux'], yerr=data['unc_tof'], fmt='.', markersize=5, capsize=2, label=f"{nom} - experimental")
        
        base_color = p_exp[0].get_color()
        rgb = mcolors.to_rgb(base_color)
        dark_color = [c * 0.6 for c in rgb] 
        
        p_exp[2][0].set_color((rgb[0], rgb[1], rgb[2], 0.2))
        if p_exp[1]:
            for cap in p_exp[1]:
                cap.set_color((rgb[0], rgb[1], rgb[2], 0.2))
        
        p_corr = ax.errorbar(data['channels'], data['flux_lisse'], yerr=data['unc_tof'], fmt='x', markersize=4, color=dark_color, capsize=2, label=f"{nom} - corrected")
        p_corr[2][0].set_color((dark_color[0], dark_color[1], dark_color[2], 0.2))
        if p_corr[1]:
            for cap in p_corr[1]:
                cap.set_color((dark_color[0], dark_color[1], dark_color[2], 0.2))
    
    ax.set_xlabel('channels')
    ax.set_ylabel('counts')
    ax.set_title('Time of flight spectrum')    
    ax.legend(markerscale=3.0, handlelength=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    _integrer_canvas(fig, frame)
    return fig

def plot_3(fichiers, datasets, frame=None):
    print('Calculating please wait...')   
    fig, ax = plt.subplots(figsize=(12, 5))
    for nom in fichiers:
        data = datasets[nom]
        ax.plot(data['E'], data['eff_E'], '.', label=nom)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('log(E) (eV)')
    ax.set_ylabel('log (efficiency)')
    ax.set_title('Evolution of efficiency in function of energy')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    
    _integrer_canvas(fig, frame)
    
def plot_4(fichiers, datasets, frame=None):
    print('Calculating please wait...')    
    fig, ax = plt.subplots(figsize=(12, 5))
    for nom in fichiers:
        data = datasets[nom]
        ax.plot(data['ToF'] * 1e6, data['eff_ToF'], '.', label=nom)
    ax.set_xlabel('time (us)')
    ax.set_ylabel('efficiency')
    ax.set_title('Evolution of efficiency in function of time')    
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    
    _integrer_canvas(fig, frame)
    return fig
    
def plot_5(fichiers, datasets, frame=None):
    print('Calculating please wait...')        
    fig, ax = plt.subplots(figsize=(12, 5)) 
    t_min = PARAMS['t_min']
    t_max = PARAMS['t_max']
    
    for nom in fichiers:
        data = datasets[nom]
        mask = (data['ToF'] >= t_min) & (data['ToF'] <= t_max)
        for i, T in enumerate(Temp_K):
            maxwellian = 1 / 2 * (masse_n / (k_b * T))**2 * data['meta']['path_length']**4 / data['ToF']**5 * np.exp(-masse_n * data['meta']['path_length']**2 /(2*k_b*T*data['ToF']**2))
            Y_exp = data['flux_tof'][mask]
            X_theo = maxwellian[mask]
            fact_amplitude = np.sum(X_theo * Y_exp) / np.sum(X_theo**2)
            maxwellian_norm = maxwellian * fact_amplitude
            color = cmap(i /(N_temp-1))
            ax.plot(data['ToF'] * 1e6, maxwellian_norm, '-', color=color, markersize=1)
        
        ax.errorbar(data['ToF'][10:] * 1e6, data['flux_tof'][10:], yerr=data['unc_tof'][10:], fmt='.', markersize=4, color='purple', ecolor=(0.5, 0, 1, 0.2), capsize=2, label=f'{nom} - corrected')
    
    ax.set_xlabel('time (us)')
    ax.set_ylabel('counts')
    ax.set_title('Time of flight spectrum')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    norm = plt.Normalize(vmin=Temp_K.min(), vmax=Temp_K.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) 
    
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, ticks=[Temp_K.min(), (Temp_K.max()+3*Temp_K.min())/4, Temp_K.mean(), 3*(Temp_K.max()+Temp_K.min()/3)/4, Temp_K.max()])
    cbar.set_label('Temperature T (K)', fontsize=9)
    
    ax.legend(loc='upper right', fontsize=8)
    
    _integrer_canvas(fig, frame)
    return fig
    
def plot_6(fichiers, datasets, frame=None):

    print('Calculating please wait...')

    fig, ax = plt.subplots(figsize=(12, 5))

    t_min = PARAMS['t_min']
    t_max = PARAMS['t_max']

    for nom in fichiers:
        data = datasets[nom]
        mask = (
            (data['ToF'] >= t_min)
            & (data['ToF'] <= t_max)
        )

        ToF_fit = data['ToF'][mask]
        flux_fit = data['flux_tof'][mask]

        T_best, erreur_min = fit_maxwellian_grid_search(
            ToF_fit,
            flux_fit,
            data['meta']['path_length']
        )

        maxwellian = (
            0.5
            * (masse_n / (k_b * T_best))**2
            * data['meta']['path_length']**4
            / ToF_fit**5
            * np.exp(
                -masse_n
                * data['meta']['path_length']**2
                / (2 * k_b * T_best * ToF_fit**2)
            )
        )

        fact_amplitude = (np.sum(maxwellian * flux_fit) / np.sum(maxwellian**2))

        maxwellian_norm = maxwellian * fact_amplitude

        print(
            f" --> Best temperature fitting for {nom} : "
            f"{T_best} K (Erreur min : {erreur_min:.2e})"
        )

        p = ax.errorbar(
            data['ToF'][10:] * 1e6,
            data['flux_tof'][10:],
            yerr=data['unc_tof'][10:],
            fmt='.',
            markersize=4,
            ecolor=(0.5, 0, 1, 0.1),
            capsize=2,
            label=f'{nom} - corrected'
        )

        couleur_courante = p[0].get_color()

        ax.plot(
            ToF_fit * 1e6,
            maxwellian_norm,
            '--',
            color=couleur_courante,
            label=f'{nom} - Maxwellian T = {T_best} K'
        )

    ax.set_xlabel('time (us)')
    ax.set_ylabel('counts')
    ax.set_title('Time of flight spectrum - Multi-files Comparison')
    ax.legend(fontsize=8)
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.7
    )

    _integrer_canvas(fig, frame)
    return fig
    
def plot_7(fichiers, datasets, choice_sub=7.1, frame=None):
        """Prend désormais choice_sub en paramètre pour éviter l'utilisation de input()"""
        print('Calculating please wait...')
        t_min = PARAMS['t_min']
        t_max = PARAMS['t_max']
        
        data = datasets[fichiers[0]]
        fit = compute_fit_results_from_dataset(data)

        # map fit results to local variables used by plotting logic
        fit_results = fit
        flux_modele_1 = fit['flux_modele_1']
        flux_modele_2 = fit['flux_modele_2']
        flux_modele_1_epi = fit['flux_modele_1_epi']
        flux_epi_pure = fit['flux_epi_pure']
        flux_luis = fit['flux_luis']
        mask_1 = fit['mask_1']
        mask_2 = fit['mask_2']
        a1_tof_pure_1 = fit['a1_tof_pure_1']
        a1_tof_pure_2 = fit['a1_tof_pure_2']
        T_1 = fit['T_1']
        T_1_epi = fit['T_1_epi']
        r_squared_1 = fit['r_squared_1']
        r_squared_2 = fit['r_squared_2']
        r_squared_1_epi = fit['r_squared_1_epi']
        
        if choice_sub == 7.1:            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.errorbar(data['ToF'][mask_1] * 1e6, data['flux_tof'][mask_1], yerr=data['unc_tof'][mask_1], fmt='.', markersize=4, color='purple', ecolor=(0.5, 0, 1, 0.2), capsize=2, label='Experimental corrected')
            ax.plot(data['ToF'][mask_1] * 1e6, flux_modele_1[mask_1], '-', color='black', linewidth=2, label=f'Fit Maxwellian 1 (a1 = {a1_tof_pure_1:.1f}, R² = {r_squared_1:.2f})')
            ax.errorbar(data['ToF_grouped'][mask_2] * 1e6, data['flux_tof_grouped'][mask_2], yerr=data['unc_tof_grouped'][mask_2], fmt='.', markersize=4, color='blue', ecolor=(0.5, 0, 1, 0.2), capsize=2, label='Experimental corrected 2')
            ax.plot(data['ToF_grouped'][mask_2] * 1e6, flux_modele_2[mask_2], '-', color='red', linewidth=2, label=f'Fit Maxwellian 2 (a1 = {a1_tof_pure_2:.1f}, R² = {r_squared_2:.2f})')
            
            ax.set_xlabel('time (us)')
            ax.set_ylabel('counts')
            ax.set_title('Time of flight spectrum with optimal Maxwellian fit') 
            ax.legend(labelcolor=['black', 'red', 'purple', 'blue'], markerscale=2.0, fontsize=8)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            _integrer_canvas(fig, frame)
            
        elif choice_sub == 7.2:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.errorbar(data['ToF'][mask_1] * 1e6, data['flux_tof'][mask_1], yerr=data['unc_tof'][mask_1], fmt='.', markersize=4, color='purple', ecolor=(0.5, 0, 1, 0.2), capsize=2, label='Experimental corrected')
            ax.plot(data['ToF'][mask_1] * 1e6, flux_modele_1[mask_1], '--', color='blue', linewidth=1.5, label=f'Fit Maxwellian pure (T = {T_1:.1f} K, R² = {r_squared_1:.2f})')
            ax.plot(data['ToF'][mask_1] * 1e6, flux_modele_1_epi[mask_1], '--', color='red', linewidth=2, label=f'Fit Maxwellian + Epi (T = {T_1_epi:.1f} K, R² = {r_squared_1_epi:.2f})')
            ax.plot(data['ToF'][mask_1] * 1e6, flux_epi_pure[mask_1], "--", color='green', label='Epithermal contribution')
            ax.plot(data['ToF'][mask_1] * 1e6, flux_luis[mask_1], "--", color='orange', label='Luis Fit')
            
            ax.set_xlabel('time (us)')
            ax.set_ylabel('counts')
            ax.set_title('Time of flight spectrum with pure and epithermal Maxwellian fits')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            _integrer_canvas(fig, frame)

        return fig, fit_results         
            
def plot_8(fichiers, datasets, fit_results, choice_sub=8.1, frame=None):
    print('Calculating please wait...')
    data = datasets[fichiers[0]]
    models = compute_plot8_models(data, fit_results)

    flux_modele_1 = models['flux_modele_1']
    flux_modele_2 = models['flux_modele_2']
    jacobian = models['jacobian']
    mask_E = models['mask_E']
    mask_epi = models['mask_epi']
    a0_best_1 = models.get('a0_best_1')
    a1_best_1 = models.get('a1_best_1')
    a0_best_2 = models.get('a0_best_2')
    a1_best_2 = models.get('a1_best_2')
    r_squared_1 = models.get('r_squared_1')
    r_squared_2 = models.get('r_squared_2')
    T_1 = models.get('T_1')
    T_2 = models.get('T_2')
    flux_tof_pure_converted = models.get('flux_tof_pure_converted')
    flux_tof_epi_converted = models.get('flux_tof_epi_converted')
        
    if choice_sub == 8.1:
        a0_tof_pure_1 = fit_results.get('a0_tof_pure_1')
        a1_tof_pure_1 = fit_results.get('a1_tof_pure_1')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharex=True)
        ax1.errorbar(data['E'][mask_E], data['flux_E'][mask_E], yerr=data['unc_tof'][mask_E], fmt='.', markersize=4, color='purple', ecolor=(0.5, 0, 1, 0.2), capsize=2, label='Experimental corrected 1')
        ax1.plot(data['E'][mask_E], flux_modele_1[mask_E], '-', color='black', linewidth=2, label=f'Fit Maxwellian 1 (T = {T_1:.1f} K, R² = {r_squared_1:.2f})')

        if a0_tof_pure_1 is not None and a1_tof_pure_1 is not None:
            flux_tof_pure_poly = maxwell_model_tof(data['ToF'], a0_tof_pure_1, a1_tof_pure_1)
            flux_tof_pure_converted = flux_tof_pure_poly * jacobian
            r2_tof_conv_1 = calculate_r_squared(data['flux_E'][mask_E], flux_tof_pure_converted[mask_E])
            r2_tof_conv_2 = calculate_r_squared(data['flux_E2'][mask_E], (flux_tof_pure_converted * data['E'])[mask_E])
            ax1.plot(data['E'][mask_E], flux_tof_pure_converted[mask_E], ':', color='orange', linewidth=2.5, label=f'Fit ToF 7.1 converti (R² = {r2_tof_conv_1:.2f})')

        ax1.set_xlabel('Energy (eV)'); ax1.set_ylabel('Flux (E)'); ax1.set_title('Energy spectrum with optimal Maxwellian fit')
        ax1.legend(loc='upper right', fontsize=8); ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7); ax1.set_xscale('log')

        ax2.errorbar(data['E'][mask_E], data['flux_E2'][mask_E], yerr=data['unc_tof'][mask_E], fmt='.', markersize=4, color='purple', ecolor=(0.5, 0, 1, 0.2), capsize=2, label='Energy flux * E')
        ax2.plot(data['E'][mask_E], flux_modele_2[mask_E], '-', color='black', linewidth=2, label=f'Fit Maxwellian 2 (T = {T_2:.1f} K, R² = {r_squared_2:.2f})')
        if a0_tof_pure_1 is not None and a1_tof_pure_1 is not None:
            ax2.plot(data['E'][mask_E], (flux_tof_pure_converted * data['E'])[mask_E], ':', color='orange', linewidth=2.5, label=f'Fit ToF 7.1 converti * E (R² = {r2_tof_conv_2:.2f})')

        ax2.set_xlabel('Energy (eV)'); ax2.set_ylabel('Flux (E) * E '); ax2.set_title('Corrected energy spectrum with optimal Maxwellian fit')
        ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7); ax2.set_xscale('log')
        plt.tight_layout()
        _integrer_canvas(fig, frame)
        return fig

    elif choice_sub == 8.2:
        try:
            a0_from_tof = fit_results['a0_epi_1']
            a1_from_tof = fit_results['a1_epi_1']
            a2_from_tof = fit_results['a2_epi_1']
            Ed_from_tof = fit_results['Ed_epi_1']
            T_depuis_tof = fit_results['T_1_epi']
            b_from_tof = fit_results['b_epi_1']
            beta_from_tof = fit_results['beta_epi_1']
        except KeyError:
            print("ERROR: Temporal epithermal parameters not found in fit_results.")
            return

        b_manuel, beta_manuel = 0.27, 1.921
        b_corr_manuel, beta_corr_manuel = 0.5, 1.921

        E_pic_1 = 1 / a1_from_tof
        hauteur_maxw_pure_1 = a0_best_1 * E_pic_1 * np.exp(-a1_best_1 * E_pic_1)
        a0_epi_energy_1 = hauteur_maxw_pure_1 / E_pic_1
        a1_epi_energy_1 = 1 / (k_b / eV * T_depuis_tof)

        forme_epi_1 = (1 - np.exp(-(data['E'][mask_epi] / Ed_from_tof)**2)) * (data['E'][mask_epi]**(b_manuel - 1)) * np.exp(-data['E'][mask_epi] / beta_manuel)
        a2_epi_energy_1 = np.mean(data['flux_E'][mask_epi] / forme_epi_1)

        flux_modele_1_epi = maxwell_epi_analytique_E(data['E'], a0_epi_energy_1, a1_epi_energy_1, a2_epi_energy_1, Ed_from_tof, b_manuel, beta_manuel)

        flux_tof_epi_pure = model_tof_epi(data['ToF'], a0_from_tof, a1_from_tof, a2_from_tof, Ed_from_tof, b_from_tof, beta_from_tof, data['E'])
        flux_tof_epi_converted = flux_tof_epi_pure * jacobian
        r2_tof_epi_conv_1 = calculate_r_squared(data['flux_E'][mask_E], flux_tof_epi_converted[mask_E])
        r2_tof_epi_conv_2 = calculate_r_squared(data['flux_E2'][mask_E], (flux_tof_epi_converted * data['E'])[mask_E])

        E_pic_2 = 2 / a1_from_tof
        hauteur_maxw_pure_2 = a0_best_2 * (E_pic_2**2) * np.exp(-a1_best_2 * E_pic_2)
        a0_epi_energy_2 = hauteur_maxw_pure_2 / (E_pic_2**2)
        a1_epi_energy_2 = 1 / (k_b / eV * T_depuis_tof)

        forme_epi_2 = (1 - np.exp(-(data['E'][mask_epi] / Ed_from_tof)**2)) * (data['E'][mask_epi]**b_corr_manuel) * np.exp(-data['E'][mask_epi] / beta_corr_manuel)
        a2_epi_energy_2 = np.mean(data['flux_E2'][mask_epi] / forme_epi_2)

        flux_modele_2_epi = maxwell_epi_analytique_E_corr(data['E'], a0_epi_energy_2, a1_epi_energy_2, a2_epi_energy_2, Ed_from_tof, b_corr_manuel, beta_corr_manuel)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

        ax1.plot(data['E'][mask_E], data['flux_E'][mask_E], '.', markersize=4, color='purple', label='Experimental corrected 1')
        ax1.plot(data['E'][mask_E], flux_modele_1[mask_E], '--', color='blue', linewidth=1.5, label='Fit Maxwellian 1 pure')
        ax1.plot(data['E'][mask_E], flux_modele_1_epi[mask_E], '--', color='red', linewidth=2, label='Fit Maxwellian 1 + Epi')
        ax1.plot(data['E'][mask_E], flux_tof_epi_converted[mask_E], ':', color='orange', linewidth=2.5, label=f'Fit ToF 7.2 converti (R² = {r2_tof_epi_conv_1:.2f})')
        ax1.set_ylabel('Flux (E)'); ax1.set_title('Energy spectrum (Linear Y)'); ax1.legend(loc='upper right', fontsize=8); ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7); ax1.set_xscale('log')

        ax2.plot(data['E'][mask_E], data['flux_E2'][mask_E], '.', markersize=4, color='purple', label='Energy flux * E')
        ax2.plot(data['E'][mask_E], flux_modele_2[mask_E], '--', color='blue', linewidth=1.5, label='Fit Maxwellian 2 pure')
        ax2.plot(data['E'][mask_E], flux_modele_2_epi[mask_E], '--', color='red', linewidth=2, label='Fit Maxwellian 2 + Epi')
        ax2.plot(data['E'][mask_E], (flux_tof_epi_converted * data['E'])[mask_E], ':', color='orange', linewidth=2.5, label=f'Fit ToF 7.2 converti * E (R² = {r2_tof_epi_conv_2:.2f})')
        ax2.set_ylabel('Flux (E) * E'); ax2.set_title('Corrected energy spectrum (Linear Y)'); ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7); ax2.set_xscale('log')

        ax3.plot(data['E'][mask_E], data['flux_E'][mask_E], '.', markersize=4, color='purple', label='Experimental corrected 1')
        ax3.plot(data['E'][mask_E], flux_modele_1[mask_E], '--', color='blue', linewidth=1.5, label='Fit Maxwellian 1 pure')
        ax3.plot(data['E'][mask_E], flux_modele_1_epi[mask_E], '--', color='red', linewidth=2, label='Fit Maxwellian 1 + Epi')
        ax3.plot(data['E'][mask_E], flux_tof_epi_converted[mask_E], ':', color='orange', linewidth=2.5, label='Fit ToF 7.2 converti')
        ax3.set_xlabel('Energy (eV)'); ax3.set_ylabel('Flux (E)'); ax3.set_title('Energy spectrum (Log Y)'); ax3.legend(loc='upper right', fontsize=8); ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7); ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.set_ylim(bottom=np.max(data['flux_E'][mask_E]) * 1e-6)

        ax4.plot(data['E'][mask_E], data['flux_E2'][mask_E], '.', markersize=4, color='purple', label='Energy flux * E')
        ax4.plot(data['E'][mask_E], flux_modele_2[mask_E], '--', color='blue', linewidth=1.5, label='Fit Maxwellian 2 pure')
        ax4.plot(data['E'][mask_E], flux_modele_2_epi[mask_E], '--', color='red', linewidth=2, label='Fit Maxwellian 2 + Epi')
        ax4.plot(data['E'][mask_E], (flux_tof_epi_converted * data['E'])[mask_E], ':', color='orange', linewidth=2.5, label='Fit ToF 7.2 converti * E')
        ax4.set_xlabel('Energy (eV)'); ax4.set_ylabel('Flux (E) * E'); ax4.set_title('Corrected energy spectrum (Log Y)'); ax4.legend(loc='upper right', fontsize=8); ax4.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7); ax4.set_xscale('log'); ax4.set_yscale('log'); ax4.set_ylim(bottom=np.max(data['flux_E2'][mask_E]) * 1e-6)

        plt.tight_layout()
        _integrer_canvas(fig, frame)
        return fig

def plot_9(fichiers, datasets, frame=None):
    print('Calculating please wait...')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    t_min = PARAMS['t_min']
    t_max = PARAMS['t_max']
    
    for nom in fichiers:
        data = datasets[nom]
        _, counts_bruts = np.loadtxt(os.path.join("data", nom), skiprows=15, unpack=True)
        flux_brut = counts_bruts / data['meta']['nbr_frames']
        
        dt = data['meta']['channel_width'] * 1e-6
        integral = np.sum(data['flux_tof']) * dt
        flux_normalise_integral = data['flux_tof'] / integral
        
        ax1.plot(data['ToF'] * 1e6, flux_brut, '.', markersize=4, label=f"{nom} - Brut")
        ax2.plot(data['ToF'] * 1e6, data['flux_tof'], '.', markersize=4, label=f"{nom} - Corrigé")
        ax3.plot(data['ToF'] * 1e6, flux_normalise_integral, '.', markersize=4, label=f"{nom} - Area Norm")
        
    for ax, title, ylabel in zip([ax1, ax2, ax3], ['Raw Flux Comparison', 'Fully Corrected ToF Flux Comparison', 'Shape Comparison'], ['Raw Flux', 'Corrected Flux', 'Normalized Flux']):
        ax.set_xlabel('Time (us)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=8)
        
    plt.tight_layout()
    _integrer_canvas(fig, frame)
    return fig

def plot_10(fichiers, datasets, frame=None):
    print('Calculating flux integral vs power...')
    puissances_disponibles = [250, 500, 1000]
    puissances = puissances_disponibles[:len(fichiers)]
    integrales = []
    unc = []
    
    for f in fichiers:
        data = datasets[f]
        dt = data['meta']['channel_width'] * 1e-6
        integrales.append(np.sum(data['flux_tof']) * dt)
        unc.append(dt * np.sqrt(np.sum(data['unc_flux_reelle']**2)))
        
    x_data, y_data, y_err = np.array(puissances), np.array(integrales), np.array(unc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='purple', ecolor=(0.5, 0, 1, 0.4), capsize=4, markersize=4, label='Données expérimentales')
    
    if len(x_data) > 1:
        modele_lineaire = lambda P, a: a * P
        popt, _ = curve_fit(modele_lineaire, x_data, y_data)
        pente_a = popt[0]
        
        x_trace = np.linspace(0, np.max(x_data), 100)
        ax.plot(x_trace, modele_lineaire(x_trace, pente_a), '-', color='black', alpha=0.6, label=f'Fit linéaire : I = {pente_a:.3e} * P')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
    ax.set_xlabel('Puissance [kW]'); ax.set_ylabel('Intégrale du flux'); ax.grid(True, linestyle="--")
    ax.legend()
    _integrer_canvas(fig, frame)
    return fig

def plot_11(fichiers, datasets, fichier_ref="", frame=None):
    """Calcule la section efficace pour un échantillon à partir du rapport ToF pur.
    Le calcul de la transmission est réalisé sur les spectres temporels (ToF) corrigés non groupés.
    Intègre des options pour masquer indépendamment M1 et M2, ainsi qu'un bouton d'accumulation.
    """
    if len(fichiers) < 2:
        print("Erreur : Vous devez avoir au moins un fichier de référence et un échantillon.")
        return
    thickness    = PARAMS['thickness']
    atom_density = PARAMS['atom_density']
    E_min        = PARAMS['E_min']
    E_max        = PARAMS['E_max']    
    
    nom_flux0 = fichiers[0]
    nom_sample = fichiers[1]
    
    data_flux0 = datasets[nom_flux0]
    data_sample = datasets[nom_sample]
    
    if frame is not None:
        for widget in frame.winfo_children():
            widget.destroy()
            
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plt.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.20)
    
    couleurs_cycle = list(mcolors.TABLEAU_COLORS.values())
    color_index = 0
    
    lignes_m2 = []
    courbe_m1 = None      
    courbe_active = None  
    
    # 2. Tracé du(s) fichier(s) de référence externe si configuré (délégué à physics.read_reference_file)
    amps_from_refs = []
    if fichier_ref:
        # Normalize to iterable of paths
        refs = fichier_ref if isinstance(fichier_ref, (list, tuple, np.ndarray)) else [fichier_ref]
        try:
            for i_ref, ref_path in enumerate(refs):
                try:
                    nom_fichier_base, E_ref, sigma_ref, unc_ref, mask_ref, E_plot = read_reference_file(ref_path, E_min, E_max)
                    # choose color: first ref in black, others from cycle
                    import random
                    color = 'black' if i_ref == 0 else random.choice(['red', 'green', 'purple', 'brown', 'pink', 'olive', 'darkgray'])

                    if unc_ref is not None:
                        p_ref = ax.errorbar(E_plot, sigma_ref[mask_ref], yerr=unc_ref[mask_ref], 
                                             fmt='-', color=color, linewidth=1.5, label=f"Ref: {nom_fichier_base}")
                        # soften errorbars alpha if black
                        if color == 'black':
                            p_ref[2][0].set_color((0, 0, 0, 0.2))
                    else:
                        ax.plot(E_plot, sigma_ref[mask_ref], '-', color=color, linewidth=1.5, label=f"Ref: {nom_fichier_base}")

                    # compute amplitude estimate for this reference
                    try:
                        amp_i = compute_amp_init_from_ref(sigma_ref, mask_ref, cross_sec_m1_raw, mask_E0)
                        amps_from_refs.append(amp_i)
                    except Exception:
                        pass

                except Exception as e_ref:
                    print(f"Impossible de lire le fichier de référence '{ref_path}': {e_ref}")
        except Exception as e:
            print(f"Erreur lors du traitement des fichiers de référence : {e}")

    # If we collected amplitudes from one or more references, take their mean as initial amplitude
    if amps_from_refs:
        amp_init = float(np.mean(amps_from_refs))
    elif not fichier_ref:
        amp_init = 1.0
            
    flux0_tof = data_flux0['flux_tof_ungrouped']
    sample_tof = data_sample['flux_tof_ungrouped']
    ToF_canal = data_flux0['ToF']
    
    unc0_tof = data_flux0.get('unc_tof', np.zeros_like(flux0_tof))
    uncS_tof = data_sample.get('unc_tof', np.zeros_like(sample_tof))
    
    E_direct = 0.5 * masse_n * (data_flux0['meta']['path_length'] / ToF_canal)**2 / eV
    mask_E0 = (E_direct >= E_min) & (E_direct <= E_max)
    
    flux0_tof_lisse_m1 = apply_grouping_methode1(flux0_tof, M=30)
    sample_tof_lisse_m1 = apply_grouping_methode1(sample_tof, M=30)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Tr_m1_tof = sample_tof_lisse_m1 / flux0_tof_lisse_m1
        cross_sec_m1_raw = cross_section(Tr_m1_tof, thickness, atom_density)
        
    cross_sec_m1_raw = np.clip(cross_sec_m1_raw, 0, None)
    
    if fichier_ref and 'sigma_ref' in locals():
        mask_amp_ref = mask_ref
        amp_init = compute_amp_init_from_ref(sigma_ref, mask_amp_ref, cross_sec_m1_raw, mask_E0)
    else:
        amp_init = 1.0
        
    N_initial = 20
        
    ax.set_xscale('log')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Cross section (barns)')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    ax_slider_N = plt.axes([0.20, 0.11, 0.55, 0.025])
    slider_N = Slider(ax_slider_N, 'Grouping N', 2, 60, valinit=N_initial, valfmt='%d')
    
    ax_slider_amp = plt.axes([0.20, 0.06, 0.55, 0.025])
    slider_amp = Slider(ax_slider_amp, 'Amplitude Fact', 0.1, 3.0, valinit=amp_init, valfmt='%.3f')
    
    ax_check = plt.axes([0.82, 0.11, 0.12, 0.06])
    check_box = CheckButtons(ax_check, labels=['Hide M1', 'Hide M2'], actives=[True, False])
    
    for patch in ax_check.patches:
        patch.set_width(0.55)
        patch.set_height(0.55)
        
    for line in ax_check.lines:
        line.set_markersize(8)
        
    ax_check.patch.set_facecolor('#f0f0f0')
    for text in check_box.labels:
        text.set_fontsize(8)
    
    ax_button = plt.axes([0.82, 0.05, 0.12, 0.04])
    btn_accumuler = Button(ax_button, 'Keep Curve', color='#4CAF50', hovercolor='#45a049')
    
    def update(val):
        nonlocal courbe_active, courbe_m1, color_index
        N_actuel = int(slider_N.val)
        amp_actuelle = slider_amp.val
        
        etats = check_box.get_status()
        cache_m1, cache_m2 = etats[0], etats[1]
        
        if courbe_m1 is not None:
            courbe_m1.remove()
            courbe_m1 = None
            
        if not cache_m1:
            unc0_m1 = np.sqrt(np.abs(apply_grouping_methode1(unc0_tof**2, M=30) / 30))
            uncS_m1 = np.sqrt(np.abs(apply_grouping_methode1(uncS_tof**2, M=30) / 30))
            
            unc_m1 = compute_cross_section_uncertainty(
                flux0_tof_lisse_m1, unc0_m1, 
                sample_tof_lisse_m1, uncS_m1, 
                thickness, atom_density
            )
            
            courbe_m1 = ax.errorbar(E_direct[mask_E0], cross_sec_m1_raw[mask_E0] * amp_actuelle, 
                                    yerr=unc_m1[mask_E0] * amp_actuelle, fmt='.', color='red', 
                                    markersize=3, alpha=0.3, elinewidth=0.5, capsize=1, label="Méthode 1 (Lissage M=30)")
        
        if courbe_active is not None:
            courbe_active.remove()
            courbe_active = None
            
        if not cache_m2:
            ToF_g, E_g, cross_sec_g, unc_g, mask_g = compute_grouping_cross_section_m2(
                ToF_canal, flux0_tof, sample_tof, unc0_tof, uncS_tof,
                data_flux0['meta']['path_length'], N_actuel, thickness, atom_density, E_min, E_max
            )

            courbe_active = ax.errorbar(E_g[mask_g], cross_sec_g[mask_g] * amp_actuelle, 
                                        yerr=unc_g[mask_g] * amp_actuelle, fmt='.-', 
                                        color=couleurs_cycle[color_index], markersize=5.5, linewidth=1,
                                        elinewidth=0.7, capsize=1.5, label=f"Grouping Actuel (N={N_actuel}, Amp={amp_actuelle:.2f})")
        
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.set_title(f'Influence of Grouping Size N = {N_actuel} on {nom_sample}')
        ax.legend(fontsize=8, loc='upper right')
        fig.canvas.draw_idle()
        
    def accumuler_courbe(event):
        nonlocal color_index, courbe_active
        if courbe_active is None:
            return
            
        N_actuel = int(slider_N.val)
        amp_actuelle = slider_amp.val
        
        ToF_g, E_g, cross_sec_g, unc_g, mask_g = compute_grouping_cross_section_m2(
            ToF_canal, flux0_tof, sample_tof, unc0_tof, uncS_tof,
            data_flux0['meta']['path_length'], N_actuel, thickness, atom_density, E_min, E_max
        )

        nouvelle_ligne = ax.errorbar(E_g[mask_g], cross_sec_g[mask_g] * amp_actuelle, 
                                     yerr=unc_g[mask_g] * amp_actuelle, fmt='.-', 
                                     color=courbe_active[0].get_color() if courbe_active is not None else couleurs_cycle[color_index], markersize=3.5, linewidth=1, alpha=0.4,
                                     elinewidth=0.5, capsize=1, label=f"Saved Grouping N={N_actuel} (Amp={amp_actuelle:.2f})")
        
        lignes_m2.append(nouvelle_ligne)
        color_index = (color_index + 1) % len(couleurs_cycle)
        
        update(None)
        
    def gestion_affichage(label):
        etats = check_box.get_status()
        cache_m2 = etats[1]
        
        for ligne in lignes_m2:
            for artist in [ligne[0]] + list(ligne[1]) + list(ligne[2]):
                artist.set_visible(not cache_m2)
                
        update(None)
        
    slider_N.on_changed(update)
    slider_amp.on_changed(update)
    btn_accumuler.on_clicked(accumuler_courbe)
    check_box.on_clicked(gestion_affichage)
    
    ax._slider_N_ref = slider_N
    ax._slider_amp_ref = slider_amp
    ax._btn_ref = btn_accumuler
    ax._check_ref = check_box
    
    update(None)
    _integrer_canvas(fig, frame)
    return fig

def plot_12(fichiers, datasets, fichier_ref="", frame=None):
    """Prend les variables interactives directement en arguments."""
    thickness = PARAMS['thickness']
    atom_density = PARAMS['atom_density']
    E_min = PARAMS['E_min']
    E_max = PARAMS['E_max']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    chemin_complet_ref = os.path.join("data", fichier_ref)
    E_ref, sigma_ref, unc_ref = np.loadtxt(chemin_complet_ref, unpack=True)
    mask_ref = (E_ref * 1e-3 >= E_min) & (E_ref * 1e-3 <= E_max)
    
    p_ref = ax.errorbar(E_ref[mask_ref] * 1e-3, sigma_ref[mask_ref], yerr=unc_ref[mask_ref], 
                         fmt='-', color='black', linewidth=1.5, label=f"Ref: {fichier_ref}")
    p_ref[2][0].set_color((0, 0, 0, 0.2))
        
    ax.set_xscale('log'); ax.set_xlabel('Energy (eV)'); ax.set_ylabel('Cross section (barns)')
    ax.legend(); ax.grid(True, which="both", linestyle="--")
    
    _integrer_canvas(fig, frame)
    return fig
