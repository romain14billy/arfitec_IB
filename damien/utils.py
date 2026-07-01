import os
import numpy as np


def grouped_file(fichiers_chemins):
    """Prend une liste de chemins de fichiers d'acquisition,
    additionne les 'counts' canal par canal et les 'number of frames',
    puis exporte un nouveau fichier en mettant à jour la ligne de métadonnée.
    """
    if not fichiers_chemins:
        print("Erreur : La liste de fichiers est vide.")
        return

    premier_fichier = fichiers_chemins[0]
    nbr_frames_total = 0
    idx_frame_line = None
    cle_separateur = ":"
    
    # 1. Lecture de l'en-tête du premier fichier
    with open(premier_fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()
    
    # On isole les 15 premières lignes d'en-tête
    entete_lignes = lignes[:15]
    
    # Extraction des frames pour le premier fichier
    for idx, ligne in enumerate(entete_lignes):
        if "number of frame" in ligne.lower() or "number of frames" in ligne.lower():
            cle_separateur = ":" if ":" in ligne else "="
            try:
                valeur = ligne.split(cle_separateur)[-1].strip()
                nbr_frames_total += int(valeur)
                idx_frame_line = idx  
            except ValueError:
                pass

    # Chargement des données numériques du premier fichier
    donnees_premiers = np.loadtxt(premier_fichier, skiprows=15)
    tof_axe = donnees_premiers[:, 0]
    counts_cumules = donnees_premiers[:, 1]

    # 2. Boucle sur les fichiers restants (extraction globale de l'en-tête complet)
    for chemin in fichiers_chemins[1:]:
        with open(chemin, 'r', encoding='utf-8') as f:
            lignes_courantes = f.readlines()[:15]
            for ligne in lignes_courantes:
                if "number of frame" in ligne.lower() or "number of frames" in ligne.lower():
                    sep = ":" if ":" in ligne else "="
                    try:
                        valeur = ligne.split(sep)[-1].strip()
                        nbr_frames_total += int(valeur)
                    except ValueError:
                        pass
        
        # Somme des coups canal par canal
        donnees_courantes = np.loadtxt(chemin, skiprows=15)
        counts_cumules += donnees_courantes[:, 1]

    # 3. Remplacement propre sur la même ligne (sans saut de ligne parasite)
    if idx_frame_line is not None:
        ligne_origine = entete_lignes[idx_frame_line]
        # On extrait la clé propre sans espaces ni retours à la ligne
        cle = ligne_origine.split(cle_separateur)[0].strip()
        # On reconstruit la ligne de manière unifiée
        entete_lignes[idx_frame_line] = f"{cle}{cle_separateur} {nbr_frames_total}\n"
    else:
        print("Attention : La ligne 'Number of frames' n'a pas été détectée.")

    # 4. Exportation
    base_path, ext = os.path.splitext(premier_fichier)
    chemin_export = f"{base_path}_grp{ext}"

    with open(chemin_export, 'w', encoding='utf-8') as f_out:
        f_out.writelines(entete_lignes)
        for t, c in zip(tof_axe, counts_cumules):
            f_out.write(f"  {t:<13}  {int(c)}\n")

    print(f"Succès : Fichier groupé exporté -> {chemin_export}")
    print(f"Nombre total de frames cumulées réécrit : {nbr_frames_total}")
    