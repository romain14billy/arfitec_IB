import tkinter as tk
import numpy as np
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Menu

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import os

from physics import process_neutron_data

from plot import (
    plot_1,
    plot_2,
    plot_3,
    plot_4,
    plot_5,
    plot_6,
    plot_7,
    plot_8,
    plot_9,
    plot_10,
    plot_11,
    plot_12
)

# Correction de l'import (retrait du .py)
from utils import (
    grouped_file
)

from config import PARAMS


# ==========================================================
# GUI APPLICATION
# ==========================================================

class NeutronApp:

    # ======================================================
    # INIT
    # ======================================================

    def __init__(self, root):
        
        # --- STATE FLAGS ---
        self.is_loading = False  # Blocks user inputs during heavy I/O operations
        self.apply_y_limits = False

        self.root = root
        self.root.title("Neutron Spectrum Analysis (Chopper Experiment)")
        self.root.state("zoomed")
        
        # --- MENU BAR CREATION ---
        barre_menu = Menu(self.root)
        self.root.config(menu=barre_menu)

        # 1. "File" Dropdown Menu
        menu_fichier = Menu(barre_menu, tearoff=0)
        barre_menu.add_cascade(label="File", menu=menu_fichier)
        menu_fichier.add_command(label="Save Plot", command=self.save_current_plot)
        menu_fichier.add_command(label="Export Data", command=self.export_current_data)
        menu_fichier.add_separator()
        menu_fichier.add_command(label="Exit", command=self.root.quit)

        # 2. "Tools" Dropdown Menu
        menu_outils = Menu(barre_menu, tearoff=0)
        barre_menu.add_cascade(label="Tools", menu=menu_outils)
        menu_outils.add_command(label="Group Files", command=self.action_grouper_fichiers)
        
        # ==================================================
        # DESIGN & STYLES (TABS & FONTS)
        # ==================================================
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Couleurs de notre charte graphique moderne
        BG_DARK = "#2c3e50"      # Bleu ardoise foncé pour le panneau gauche
        TEXT_LIGHT = "#ffffff"   # Blanc pur pour les textes du panneau gauche
        FONT_MAIN = ("Segoe UI", 10)
        FONT_BOLD = ("Segoe UI", 11, "bold")
        
        # Configuration des onglets
        self.style.configure("TNotebook.Tab", background="#95a5a6", foreground="black", padding=[18, 6], font=("Segoe UI", 10))
        self.style.map("TNotebook.Tab", background=[("selected", "#34495e")], foreground=[("selected", "white")], font=[("selected", ("Segoe UI", 10, "bold"))])

        # --------------------------------------------------
        # DATA STORAGE & CACHE
        # --------------------------------------------------
        self.datasets = {}
        self.fit_results = None
        self.plot_history = []  
        self.is_replaying = False

        # ==================================================
        # MAIN LAYOUT WITH NOTEBOOK (TABS)
        # ==================================================
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_analysis = tk.Frame(self.notebook)
        self.tab_config = tk.Frame(self.notebook)

        self.notebook.add(self.tab_analysis, text=" Analysis & Control ")
        self.notebook.add(self.tab_config, text=" Physical Parameters ")
        
        # --- STATUS BAR --- (Correction apportée ici avec tk.Label)
        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

        # ==================================================
        # TAB 1: ANALYSIS & CONTROL LAYOUT
        # ==================================================
        self.tab_analysis.grid_columnconfigure(0, weight=0) 
        self.tab_analysis.grid_columnconfigure(1, weight=1) 
        self.tab_analysis.grid_rowconfigure(0, weight=1)

        # 1. Le panneau de contrôle gauche (Style Sombre & Contrasté)
        self.control_frame = tk.Frame(
            self.tab_analysis,
            width=280,
            bg=BG_DARK
        )
        self.control_frame.pack_propagate(False)
        self.control_frame.grid(row=0, column=0, sticky="nsw")

        # 2. La zone graphique droite
        self.plot_frame_container = tk.Frame(self.tab_analysis, bg="#ffffff")
        self.plot_frame_container.grid(row=0, column=1, sticky="nsew")
        
        # Découpe de la zone droite pour intégrer la barre d'historique en haut
        self.plot_frame_container.grid_rowconfigure(0, weight=0) # Barre du haut (Historique)
        self.plot_frame_container.grid_rowconfigure(1, weight=1) # Graphique
        self.plot_frame_container.grid_columnconfigure(0, weight=1)

        # BARRE SUPÉRIEURE DROITE : Menu déroulant de l'historique
        self.top_bar = tk.Frame(self.plot_frame_container, bg="#f8f9fa", height=40, bd=1, relief=tk.RIDGE)
        self.top_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.history_label = tk.Label(self.top_bar, text="Plot History :", bg="#f8f9fa", font=("Segoe UI", 10, "bold"), fg="#333333")
        self.history_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.history_combobox = ttk.Combobox(self.top_bar, state="readonly", width=45, font=("Segoe UI", 9))
        self.history_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.history_combobox.set("No history yet")
        self.history_combobox.bind("<<ComboboxSelected>>", self.replay_plot_from_history)

        # La zone réelle où matplotlib va dessiner
        self.plot_frame = tk.Frame(self.plot_frame_container, bg="#ffffff")
        self.plot_frame.grid(row=1, column=0, sticky="nsew")

        # ==================================================
        # PANNEAU GAUCHE : TITRE
        # ==================================================
        self.title_label = tk.Label(
            self.control_frame,
            text="NEUTRON ANALYSIS",
            font=("Segoe UI", 14, "bold"),
            bg=BG_DARK,
            fg="#f1c40f" # Jaune lumineux pour un contraste parfait
        )
        self.title_label.pack(pady=(25, 15))

        # ==================================================
        # PANNEAU GAUCHE : BOUTONS DE CHARGEMENT
        # ==================================================
        self.load_button = tk.Button(
            self.control_frame,
            text="Load Data Files",
            command=self.load_files,
            font=FONT_MAIN,
            bg="#34495e",
            fg=TEXT_LIGHT,
            activebackground="#5d6d7e",
            activeforeground=TEXT_LIGHT,
            bd=0, height=1, width=22, cursor="hand2"
        )
        self.load_button.pack(pady=5)

        self.clear_cache_button = tk.Button(
            self.control_frame,
            text="Clear Cache / Reset",
            command=self.clear_cache,
            font=("Segoe UI", 9),
            bg="#7f8c8d",
            fg=TEXT_LIGHT,
            activebackground="#95a5a6",
            activeforeground=TEXT_LIGHT,
            bd=0, height=1, width=22, cursor="hand2"
        )
        self.clear_cache_button.pack(pady=(2, 15))

        # ==================================================
        # PANNEAU GAUCHE : LISTE DES FICHIERS
        # ==================================================
        self.file_label = tk.Label(
            self.control_frame,
            text="Loaded Files",
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            font=FONT_BOLD
        )
        self.file_label.pack(pady=(10, 2))

        self.file_listbox = tk.Listbox(
            self.control_frame,
            width=32,
            height=8,
            selectmode=tk.EXTENDED,
            exportselection=False,
            bg="#34495e",
            fg=TEXT_LIGHT,
            selectbackground="#1abc9c", # Vert émeraude pour la sélection
            selectforeground=TEXT_LIGHT,
            font=("Consolas", 9),
            bd=0, highlightthickness=1, highlightbackground="#455a64"
        )
        self.file_listbox.pack(padx=15, pady=5)
        
        self.ordre_selection = []
        self.file_listbox.bind('<<ListboxSelect>>', self.maj_ordre_selection)

        # ==================================================
        # PANNEAU GAUCHE : CHOIX DU PLOT
        # ==================================================
        self.plot_label = tk.Label(
            self.control_frame,
            text="Analysis Type",
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            font=FONT_BOLD
        )
        self.plot_label.pack(pady=(15, 2))

        self.plot_options = [
            "1 - Grouping Comparison",
            "2 - Dead Time Correction",
            "3 - Efficiency vs Energy",
            "4 - Efficiency vs ToF",
            "5 - Maxwellian Comparison",
            "6 - Least Square Maxwell Fit",
            "7 - Curve Fit Maxwell",
            "8 - Energy Spectrum",
            "9 - Reactor Power Comparison",
            "10 - Reactor Power vs Neutron Rate",
            "11 - Cross Section Grouping Comparision",
            "12 - Cross Section",
        ]

        self.plot_combobox = ttk.Combobox(
            self.control_frame,
            values=self.plot_options,
            state="readonly",
            width=28,
            font=("Segoe UI", 9)
        )
        self.plot_combobox.pack(pady=5)
        self.plot_combobox.current(0)

        # ==================================================
        # PANNEAU GAUCHE : ACTIONS COMPACTES ET DISTINCTES
        # ==================================================
        self.btn_frame = tk.Frame(self.control_frame, bg=BG_DARK)
        self.btn_frame.pack(pady=20)
        
        self.display_t_min = tk.DoubleVar(value=150)
        self.display_t_max = tk.DoubleVar(value=3700)

        self.display_E_min = tk.DoubleVar(value=0.003)
        self.display_E_max = tk.DoubleVar(value=0.2)
        self.display_y_min = tk.DoubleVar(value=PARAMS.get('y_min', 0.0))
        self.display_y_max = tk.DoubleVar(value=PARAMS.get('y_max', 20.0))
        
        limits_frame = tk.LabelFrame(
            self.control_frame,
            text="Display Limits",
            bg=BG_DARK,
            fg="white",
            font=FONT_BOLD
        )
        
        limits_frame.pack(fill="x", padx=10, pady=10)
        
        # ==================================================
        # t_min
        # ==================================================
        
        tk.Label(
            limits_frame,
            text="t_min (µs)",
            bg=BG_DARK,
            fg=TEXT_LIGHT
        ).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        
        entry_tmin = tk.Entry(
            limits_frame,
            textvariable=self.display_t_min,
            width=10
        )
        
        entry_tmin.grid(row=0, column=1, padx=5)
        
        entry_tmin.bind("<Return>", self.update_live_zoom)
        
        tk.Scale(
            limits_frame,
            from_=0,
            to=1000,
            resolution=10,
            orient="horizontal",
            variable=self.display_t_min,
            command=self.update_live_zoom,
            length=120,
            sliderlength=18,
            showvalue=False,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            highlightthickness=0
        ).grid(row=0, column=2)
        
        
        # ==================================================
        # t_max
        # ==================================================
        
        tk.Label(
            limits_frame,
            text="t_max (µs)",
            bg=BG_DARK,
            fg=TEXT_LIGHT
        ).grid(row=1, column=0, padx=5, pady=2, sticky="w")
        
        entry_tmax = tk.Entry(
            limits_frame,
            textvariable=self.display_t_max,
            width=10
        )
        
        entry_tmax.grid(row=1, column=1, padx=5)
        
        entry_tmax.bind("<Return>", self.update_live_zoom)
        
        tk.Scale(
            limits_frame,
            from_=0,
            to=5000,
            resolution=10,
            orient="horizontal",
            variable=self.display_t_max,
            command=self.update_live_zoom,
            length=120,
            sliderlength=18,
            showvalue=False,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            highlightthickness=0
        ).grid(row=1, column=2)
        
        
        # ==================================================
        # E_min
        # ==================================================
        
        tk.Label(
            limits_frame,
            text="E_min (eV)",
            bg=BG_DARK,
            fg=TEXT_LIGHT
        ).grid(row=2, column=0, padx=5, pady=2, sticky="w")
        
        entry_emin = tk.Entry(
            limits_frame,
            textvariable=self.display_E_min,
            width=10
        )
        
        entry_emin.grid(row=2, column=1, padx=5)
        
        entry_emin.bind("<Return>", self.update_live_zoom)
        
        tk.Scale(
            limits_frame,
            from_=0.001,
            to=0.01,
            resolution=0.001,
            orient="horizontal",
            variable=self.display_E_min,
            command=self.update_live_zoom,
            length=120,
            sliderlength=18,
            showvalue=False,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            highlightthickness=0
        ).grid(row=2, column=2)
        
        
        # ==================================================
        # E_max
        # ==================================================
        
        tk.Label(
            limits_frame,
            text="E_max (eV)",
            bg=BG_DARK,
            fg=TEXT_LIGHT
        ).grid(row=3, column=0, padx=5, pady=2, sticky="w")
        
        entry_emax = tk.Entry(
            limits_frame,
            textvariable=self.display_E_max,
            width=10
        )
        
        entry_emax.grid(row=3, column=1, padx=5)
        
        entry_emax.bind("<Return>", self.update_live_zoom)
        
        tk.Scale(
            limits_frame,
            from_=0.01,
            to=1,
            resolution=0.01,
            orient="horizontal",
            variable=self.display_E_max,
            command=self.update_live_zoom,
            length=120,
            sliderlength=18,
            showvalue=False,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            highlightthickness=0
        ).grid(row=3, column=2)

        # ==================================================
        # Y_min
        # ==================================================
        tk.Label(
            limits_frame,
            text="Y min",
            bg=BG_DARK,
            fg=TEXT_LIGHT
        ).grid(row=4, column=0, padx=5, pady=2, sticky="w")

        entry_ymin = tk.Entry(
            limits_frame,
            textvariable=self.display_y_min,
            width=10
        )
        entry_ymin.grid(row=4, column=1, padx=5)
        entry_ymin.bind("<Return>", self.on_change_y_limits)

        tk.Scale(
            limits_frame,
            from_=0.0,
            to=100.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.display_y_min,
            command=self.on_change_y_limits,
            length=120,
            sliderlength=18,
            showvalue=False,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            highlightthickness=0
        ).grid(row=4, column=2)

        # ==================================================
        # Y_max
        # ==================================================
        tk.Label(
            limits_frame,
            text="Y max",
            bg=BG_DARK,
            fg=TEXT_LIGHT
        ).grid(row=5, column=0, padx=5, pady=2, sticky="w")

        entry_ymax = tk.Entry(
            limits_frame,
            textvariable=self.display_y_max,
            width=10
        )
        entry_ymax.grid(row=5, column=1, padx=5)
        entry_ymax.bind("<Return>", self.on_change_y_limits)

        tk.Scale(
            limits_frame,
            from_=0.1,
            to=200.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.display_y_max,
            command=self.on_change_y_limits,
            length=120,
            sliderlength=18,
            showvalue=False,
            bg=BG_DARK,
            fg=TEXT_LIGHT,
            highlightthickness=0
        ).grid(row=5, column=2)

        self.plot_button = tk.Button(
            self.btn_frame,
            text="Plot",
            command=self.execute_plot,
            font=FONT_BOLD,
            bg="#2ecc71", # Vert vif distinct
            fg=TEXT_LIGHT,
            activebackground="#27ae60",
            bd=0, width=10, height=1, cursor="hand2"
        )
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            self.btn_frame,
            text="Clear",
            command=self.clear_plot,
            font=FONT_BOLD,
            bg="#e67e22", # Orange vif distinct
            fg=TEXT_LIGHT,
            activebackground="#d35400",
            bd=0, width=10, height=1, cursor="hand2"
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # ==================================================
        # PANNEAU GAUCHE : BOUTON QUIT SÉPARÉ
        # ==================================================
        self.quit_button = tk.Button(
            self.control_frame,
            text="Quit Application",
            command=self.root.quit,
            font=FONT_MAIN,
            bg="#e74c3c", # Rouge écarlate
            fg=TEXT_LIGHT,
            activebackground="#c0392b",
            bd=0, width=22, height=1, cursor="hand2"
        )
        self.quit_button.pack(side=tk.BOTTOM, pady=30)

        # ==================================================
        # INITIAL TRACÉ VIDE (PLACEHOLDER)
        # ==================================================
        self.clear_plot()

        # ==================================================
        # TAB 2: PHYSICAL PARAMETERS PANEL
        # ==================================================
        self.param_frame = tk.LabelFrame(
            self.tab_config,
            text=" Global Settings & Physical Constants ",
            bg="#f8f9fa",
            font=("Segoe UI", 12, "bold"),
            padx=20,
            pady=20
        )
        self.param_frame.pack(padx=40, pady=40, fill=tk.BOTH, expand=True)

        # --- Section Matériau ---
        tk.Label(self.param_frame, text="Thickness (cm) :", bg="#f8f9fa", font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w", pady=10)
        self.thickness_var = tk.StringVar(value=str(PARAMS["thickness"]))
        self.thickness_entry = tk.Entry(self.param_frame, textvariable=self.thickness_var, width=15, font=("Segoe UI", 11))
        self.thickness_entry.grid(row=0, column=1, pady=10, padx=10)

        tk.Label(self.param_frame, text="Atom Density (cm⁻³) :", bg="#f8f9fa", font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", pady=10)
        self.density_var = tk.StringVar(value=f"{PARAMS['atom_density']:.2e}")
        self.density_entry = tk.Entry(self.param_frame, textvariable=self.density_var, width=15, font=("Segoe UI", 11))
        self.density_entry.grid(row=1, column=1, pady=10, padx=10)

        # --- Section Bornes Énergie (E_min / E_max) ---
        tk.Label(self.param_frame, text="E min (eV) :", bg="#f8f9fa", font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w", pady=10)
        self.emin_var = tk.StringVar(value=str(PARAMS["E_min"]))
        self.emin_entry = tk.Entry(self.param_frame, textvariable=self.emin_var, width=15, font=("Segoe UI", 11))
        self.emin_entry.grid(row=2, column=1, pady=10, padx=10)

        tk.Label(self.param_frame, text="E max (eV) :", bg="#f8f9fa", font=("Segoe UI", 11)).grid(row=3, column=0, sticky="w", pady=10)
        self.emax_var = tk.StringVar(value=str(PARAMS["E_max"]))
        self.emax_entry = tk.Entry(self.param_frame, textvariable=self.emax_var, width=15, font=("Segoe UI", 11))
        self.emax_entry.grid(row=3, column=1, pady=10, padx=10)

        # --- Section Bornes Temps (t_min / t_max) ---
        tk.Label(self.param_frame, text="t min :", bg="#f8f9fa", font=("Segoe UI", 11)).grid(row=4, column=0, sticky="w", pady=10)
        self.tmin_var = tk.StringVar(value=str(PARAMS["t_min"]))
        self.tmin_entry = tk.Entry(self.param_frame, textvariable=self.tmin_var, width=15, font=("Segoe UI", 11))
        self.tmin_entry.grid(row=4, column=1, pady=10, padx=10)

        tk.Label(self.param_frame, text="t max :", bg="#f8f9fa", font=("Segoe UI", 11)).grid(row=5, column=0, sticky="w", pady=10)
        self.tmax_var = tk.StringVar(value=str(PARAMS["t_max"]))
        self.tmax_entry = tk.Entry(self.param_frame, textvariable=self.tmax_var, width=15, font=("Segoe UI", 11))
        self.tmax_entry.grid(row=5, column=1, pady=10, padx=10)

        # --- Bouton de sauvegarde ---
        self.apply_params_button = tk.Button(
            self.param_frame,
            text="Apply and Save Parameters",
            command=self.save_physical_parameters,
            bg="#2ecc71",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=15,
            pady=5, bd=0, cursor="hand2"
        )
        self.apply_params_button.grid(row=6, column=0, columnspan=2, pady=25)

    # ======================================================
    # FONCTIONS & MÉTHODES
    # ======================================================

    def maj_ordre_selection(self, event):
        indices_actuels = list(self.file_listbox.curselection())
        if not indices_actuels:
            return       
        self.ordre_selection = [i for i in self.ordre_selection if i in indices_actuels]
        for i in indices_actuels:
            if i not in self.ordre_selection:
                self.ordre_selection.append(i)

    def load_files(self):
        if self.is_loading:
            return

        fichiers = filedialog.askopenfilenames(
            title="Select neutron data files",
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
            initialdir="data"
        )
        if not fichiers:
            return

        # --- CREATION OF THE MULTI-FILE PROGRESS WINDOW ---
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Loading Progress")
        
        # Dynamic height calculation
        win_width = 580
        win_height = min(650, 40 + (len(fichiers) * 38))

        # 1. Get screen dimensions
        screen_width = progress_win.winfo_screenwidth()
        screen_height = progress_win.winfo_screenheight()

        # 2. Calculate X and Y coordinates to center the window
        x = (screen_width // 2) - (win_width // 2)
        y = (screen_height // 2) - (win_height // 2)

        # 3. Apply size AND position to the window
        progress_win.geometry(f"{win_width}x{win_height}+{x}+{y}")
        progress_win.resizable(False, False)
        progress_win.transient(self.root)
        progress_win.grab_set()

        # Grid configuration for fluid alignment
        progress_win.columnconfigure(0, weight=3)  # Filename column
        progress_win.columnconfigure(1, weight=4)  # Progressbar column
        progress_win.columnconfigure(2, weight=1)  # Percentage string column

        rows_tracker = []
        
        # Generation of all UI rows simultaneously before calculation starts
        for idx, fichier in enumerate(fichiers):
            filename = os.path.basename(fichier)
            # Truncate filename if too long to keep the layout clean
            display_name = filename if len(filename) <= 28 else filename[:25] + "..."
            
            # Left: Filename Label
            lbl_name = tk.Label(progress_win, text=display_name, anchor="w", font=("Segoe UI", 9))
            lbl_name.grid(row=idx, column=0, padx=12, pady=6, sticky="ew")
            
            # Middle: Dedicated File Progressbar
            p_bar = ttk.Progressbar(progress_win, orient="horizontal", mode="determinate", maximum=100)
            p_bar.grid(row=idx, column=1, padx=10, pady=6, sticky="ew")
            p_bar["value"] = 0
            
            # Right: Percentage / Status Label
            lbl_pct = tk.Label(progress_win, text="0%", width=12, anchor="w", font=("Segoe UI", 9, "bold"), fg="#7f8c8d")
            lbl_pct.grid(row=idx, column=2, padx=8, pady=6, sticky="w")
            
            rows_tracker.append({
                "p_bar": p_bar,
                "lbl_pct": lbl_pct,
                "lbl_name": lbl_name,
                "filename": filename,
                "path": fichier
            })

        # Render layout shell
        progress_win.update()

        try:
            self.is_loading = True
            self.root.config(cursor="watch")
            
            for item in rows_tracker:
                # Highlight active row in Blue
                item["lbl_name"].config(fg="#2980b9")
                item["lbl_pct"].config(text="Loading...", fg="#2980b9")
                progress_win.update()
                
                filename = item["filename"]
                fichier = item["path"]
                
                if filename in self.datasets:
                    # Cache hit setup (instantly filled)
                    item["p_bar"]["value"] = 100
                    item["lbl_pct"].config(text="100% (Cached)", fg="#27ae60")
                    item["lbl_name"].config(fg="#27ae60")
                    progress_win.update()
                    continue  
                
                try:
                    # Atomic file calculation
                    self.datasets[filename] = process_neutron_data(fichier)
                    self.file_listbox.insert(tk.END, filename)
                    
                    # Update row layout to Green 100% on success
                    item["p_bar"]["value"] = 100
                    item["lbl_pct"].config(text="100%", fg="#27ae60")
                    item["lbl_name"].config(fg="#27ae60")
                except Exception as e:
                    # Update row layout to Red on failure
                    item["lbl_pct"].config(text="Failed", fg="#e74c3c")
                    item["lbl_name"].config(fg="#e74c3c")
                    messagebox.showerror("Parsing Error", f"Could not parse file:\n{filename}\n\n{e}")
                
                progress_win.update()
                
            # Brief pause at the end so the user can visualize all 100% marks completed
            progress_win.after(500)
            
        finally:
            self.is_loading = False
            self.root.config(cursor="")
            self.status_label.config(text="Ready")
            progress_win.destroy()

    def clear_cache(self):
        if messagebox.askyesno("Clear Cache", "Are you sure you want to unload all files and clear cache?"):
            self.datasets.clear()
            self.file_listbox.delete(0, tk.END)
            self.ordre_selection.clear()
            self.plot_history.clear()
            self.history_combobox.set("No history yet")
            self.history_combobox.configure(values=[])
            self.clear_plot()

    def execute_plot(self):
        # Aucune altération du curseur ou blocage "is_loading" ici car l'affichage est instantané
        choix = self.plot_combobox.get()
        numero_plot = choix.split('-')[0].strip()
        
        if not self.datasets:
            messagebox.showwarning("Warning", "Please load data files first.")
            return
        
        if self.ordre_selection:
            fichiers = [self.file_listbox.get(i) for i in self.ordre_selection]
        else:
            messagebox.showwarning("Selection Error", "Please select at least one file in the list to plot.")   
            return

        try:
            self.clear_plot()
        
            if numero_plot == "1":
                self.current_fig = plot_1(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "2":
                self.current_fig = plot_2(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "3":
                self.current_fig = plot_3(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "4":
                self.current_fig = plot_4(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "5":
                self.current_fig = plot_5(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "6":
                self.current_fig = plot_6(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "7":
                self.current_fig, self.fit_results = plot_7(fichiers, self.datasets, choice_sub=7.2, frame=self.plot_frame)
                
            elif numero_plot == "8":
                if self.fit_results is None:
                    messagebox.showwarning("Warning", "Please execute plot 7 first.")
                    return
                self.current_fig = plot_8(fichiers, self.datasets, self.fit_results, choice_sub=8.2, frame=self.plot_frame)
                
            elif numero_plot == "9":
                self.current_fig = plot_9(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "10":
                self.current_fig = plot_10(fichiers, self.datasets, frame=self.plot_frame)
                
            elif numero_plot == "11":
                choix_ref = messagebox.askyesno("Reference File", "Do you want to compare with a reference file?")
                fichier_ref = ""
                if choix_ref:
                    chemin_complet_ref = filedialog.askopenfilename(
                        title="Select reference cross section file (3 columns)",
                        filetypes=[("Data files", "*.dat *.txt"), ("All files", "*.*")],
                        initialdir="data"
                    )
                    if chemin_complet_ref:
                        fichier_ref = chemin_complet_ref 

                self.current_fig = plot_11(fichiers, self.datasets, thickness=PARAMS["thickness"], atom_density=PARAMS["atom_density"], fichier_ref=fichier_ref, frame=self.plot_frame)
            
            elif numero_plot == "12":
                choix_ref = messagebox.askyesno("Reference File", "Do you want to compare with a reference file?")
                fichier_ref = ""
                if choix_ref:
                    chemin_complet_ref = filedialog.askopenfilename(
                        title="Select reference cross section file (3 columns)",
                        filetypes=[("Data files", "*.dat *.txt"), ("All files", "*.*")],
                        initialdir="data"
                    )
                    if chemin_complet_ref:
                        fichier_ref = chemin_complet_ref 

                self.current_fig = plot_12(fichiers, self.datasets, thickness=PARAMS["thickness"], atom_density=PARAMS["atom_density"], fichier_ref=fichier_ref, frame=self.plot_frame)
            
            self.update_live_zoom()
            
            if not self.is_replaying:
                self.add_to_history(choix, fichiers)

        except Exception as e:
            messagebox.showerror("Plot Error", str(e))
            
    def add_to_history(self, plot_name, files):
        """Ajoute un tracé réussi au menu déroulant de l'historique."""
        affichage_label = f"Plot {plot_name.split('-')[0].strip()} ({len(files)} files) - {', '.join(files[:2])}"
        if len(files) > 2:
            affichage_label += "..."

        self.plot_history.append({
            "nom_complet": plot_name,
            "fichiers": files,
            "label": affichage_label
        })
        
        # Mettre à jour les choix de la Combobox
        liste_labels = [item["label"] for item in self.plot_history]
        self.history_combobox.configure(values=liste_labels)
        self.history_combobox.set(affichage_label)

    def replay_plot_from_history(self, event):
        """Rejoue le graphique sélectionné depuis la combobox d'historique."""
        label_selectionne = self.history_combobox.get()
        
        # Retrouver l'item correspondant dans nos données
        historique_item = None
        for item in self.plot_history:
            if item["label"] == label_selectionne:
                historique_item = item
                break
                
        if not historique_item:
            return
            
        nom_complet = historique_item["nom_complet"]
        fichiers_sauvegardes = historique_item["fichiers"]
        
        self.plot_combobox.set(nom_complet)
        
        self.ordre_selection = []
        self.file_listbox.selection_clear(0, tk.END)
        for f in fichiers_sauvegardes:
            for idx in range(self.file_listbox.size()):
                if self.file_listbox.get(idx) == f:
                    self.ordre_selection.append(idx)
                    self.file_listbox.select_set(idx) 
        
        self.is_replaying = True
        try:
            self.execute_plot()
        finally:
            self.is_replaying = False
            
    def save_physical_parameters(self):
        """Convertit et enregistre les saisies de l'IHM dans le dictionnaire global."""
        try:
            # Conversion systematique en float pour securiser les calculs
            PARAMS["thickness"] = float(self.thickness_var.get())
            PARAMS["E_min"] = float(self.emin_var.get())
            PARAMS["E_max"] = float(self.emax_var.get())
            PARAMS["t_min"] = float(self.tmin_var.get())
            PARAMS["t_max"] = float(self.tmax_var.get())
            
            # float() prend nativement en charge les notations scientifiques (ex: 2.3e22)
            PARAMS["atom_density"] = float(self.density_var.get())
            
            messagebox.showinfo("Success", "Parameters successfully applied to global config.")
            
        except ValueError as e:
            # Securite si l'utilisateur commet une erreur de saisie (ex: une lettre ou un champ vide)
            messagebox.showerror("Parsing Error", f"Please enter valid numerical values.\nDetails: {e}")

    def clear_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(12, 8), dpi=100)
        fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_fig = fig
        
    def action_grouper_fichiers(self):
        """Déclenche l'ouverture de l'explorateur pour le groupement de fichiers (Option 98)"""
        if self.is_loading:
            return

        print("Opening file explorer via menu...")
        fichiers_selectionnes = filedialog.askopenfilenames(
            title="Select the files you want to group together",
            filetypes=[("Data files", "*.dat *.txt"), ("All files", "*.*")]
        )
        
        if fichiers_selectionnes:
            liste_chemins = list(fichiers_selectionnes)
            print(f"Selected {len(liste_chemins)} files to merge.")
            try:
                # Verrouillage et affichage du statut pour le traitement I/O lourd du merge
                self.is_loading = True
                self.root.config(cursor="watch")
                self.status_label.config(text="Merging data files... Please wait.")
                self.root.update()
                
                grouped_file(liste_chemins)
            finally:
                self.is_loading = False
                self.root.config(cursor="")
                self.status_label.config(text="Ready")
        else:
            print("Operation cancelled: No files selected.\n")
            
    def save_current_plot(self):
        """Opens a file dialog to save the currently displayed Matplotlib figure."""
        if not hasattr(self, 'current_fig') or self.current_fig is None:
            messagebox.showwarning("Save Error", "No active plot available to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Plot As",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("PDF Document", "*.pdf"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            try:
                # Saves the figure with high resolution and tight borders
                self.current_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Plot saved successfully!")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save plot:\n{e}")
                
    def export_current_data(self):
        """Scans the active Matplotlib figure, extracts curves and their uncertainties (if any), and exports to CSV."""
        if not hasattr(self, 'current_fig') or self.current_fig is None:
            messagebox.showwarning("Export Error", "No active plot available. Please generate a plot first.")
            return

        axes = self.current_fig.get_axes()
        available_curves = {}

        for ax in axes:
            panel_title = ax.get_title() or ax.get_ylabel() or "Plot"
            x_label = ax.get_xlabel() or "X_Axis"
            y_label = ax.get_ylabel() or "Y_Axis"
            
            # --- TIROIR 1 : Lignes classiques (Pas d'incertitudes) ---
            for line in ax.get_lines():
                label = line.get_label()
                if label and not label.startswith('_'):
                    unique_name = f"Panel: '{panel_title}' ➔ Line: {label}"
                    available_curves[unique_name] = {
                        "x": line.get_xdata(),
                        "y": line.get_ydata(),
                        "yerr": None, # Pas d'incertitude ici
                        "x_header": x_label.replace(" ", "_"),
                        "y_header": y_label.replace(" ", "_")
                    }
            
            # --- TIROIR 2 : Barres d'erreur (Extraction des incertitudes) ---
            for container in ax.containers:
                label = container.get_label()
                if label and not label.startswith('_'):
                    if hasattr(container, 'lines') and len(container.lines) > 0:
                        data_line = container.lines[0]
                        x_data = data_line.get_xdata()
                        y_data = data_line.get_ydata()
                        
                        # Extraction géométrique des barres d'erreur verticales
                        y_err = None
                        if len(container.lines) > 2 and container.lines[2]:
                            try:
                                v_bars = container.lines[2][0] # Collection des barres verticales
                                paths = v_bars.get_paths()
                                if paths and len(paths) == len(x_data):
                                    # Calcul de la demi-hauteur de chaque segment d'erreur (y_max - y_min) / 2
                                    y_err = np.array([(p.vertices[1, 1] - p.vertices[0, 1]) / 2.0 for p in paths])
                            except Exception:
                                y_err = None # Repli sécurisé en cas de géométrie complexe

                        name_suffix = " (with uncertainties)" if y_err is not None else ""
                        unique_name = f"Panel: '{panel_title}' ➔ Data: {label}{name_suffix}"
                        
                        available_curves[unique_name] = {
                            "x": x_data,
                            "y": y_data,
                            "yerr": y_err, # Stockage du tableau d'incertitudes
                            "x_header": x_label.replace(" ", "_"),
                            "y_header": y_label.replace(" ", "_")
                        }

            # --- TIROIR 3 : Nuages de points (Pas d'incertitudes standards) ---
            for collection in ax.collections:
                label = collection.get_label()
                if label and not label.startswith('_'):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        unique_name = f"Panel: '{panel_title}' ➔ Scatter: {label}"
                        available_curves[unique_name] = {
                            "x": offsets[:, 0],
                            "y": offsets[:, 1],
                            "yerr": None,
                            "x_header": x_label.replace(" ", "_"),
                            "y_header": y_label.replace(" ", "_")
                        }

        if not available_curves:
            messagebox.showwarning("Export Error", "No labeled data curves found in the current plot.")
            return

        # 2. Window Layout Selection
        export_win = tk.Toplevel(self.root)
        export_win.title("Select Curve to Export")
        export_win.geometry("520x150")
        export_win.resizable(False, False)
        export_win.transient(self.root)
        export_win.grab_set()
        
        sw, sh = export_win.winfo_screenwidth(), export_win.winfo_screenheight()
        export_win.geometry(f"520x150+{sw//2 - 260}+{sh//2 - 75}")

        tk.Label(export_win, text="Select the specific curve you want to extract:", font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
        
        curve_combobox = ttk.Combobox(export_win, values=list(available_curves.keys()), state="readonly", width=65)
        curve_combobox.pack(pady=5, padx=15)
        curve_combobox.current(0)

        # 3. Dynamic physical CSV saving
        def trigger_save():
            selected_curve_name = curve_combobox.get()
            curve_data = available_curves[selected_curve_name]
            
            file_path = filedialog.asksaveasfilename(
                title="Save Curve Data",
                defaultextension=".csv",
                filetypes=[("CSV Tables", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not file_path:
                return

            try:
                # Verification: Do we have uncertainties to export?
                if curve_data["yerr"] is not None:
                    # Compilation of 3 columns: X, Y, and Yerr
                    matrix = np.column_stack((curve_data["x"], curve_data["y"], curve_data["yerr"]))
                    header_string = f"{curve_data['x_header']},{curve_data['y_header']},{curve_data['y_header']}_Uncertainty"
                else:
                    # Classic fallback to 2 columns: X and Y
                    matrix = np.column_stack((curve_data["x"], curve_data["y"]))
                    header_string = f"{curve_data['x_header']},{curve_data['y_header']}"
                
                np.savetxt(file_path, matrix, delimiter=",", header=header_string, comments="")
                messagebox.showinfo("Success", "Curve data extracted and saved successfully!")
                export_win.destroy()
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export data:\n{e}")

        btn_frame = tk.Frame(export_win)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Export Selection", command=trigger_save, bg="#2ecc71", fg="white", font=("Segoe UI", 9, "bold"), bd=0, padx=10, pady=4).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=export_win.destroy, bg="#7f8c8d", fg="white", font=("Segoe UI", 9), bd=0, padx=10, pady=4).pack(side=tk.LEFT, padx=5)
        
    def validate_ranges(self):

        # ===============================
        # TOF
        # ===============================
    
        tmin = max(0, min(self.display_t_min.get(), 5000))
        tmax = max(0, min(self.display_t_max.get(), 5000))
    
        if tmin >= tmax:
            tmin = tmax - 1
    
        self.display_t_min.set(tmin)
        self.display_t_max.set(tmax)
    
        # ===============================
        # ENERGY
        # ===============================
    
        emin = max(0.001, min(self.display_E_min.get(), 0.01))
        emax = max(0.01, min(self.display_E_max.get(), 2))
    
        if emin >= emax:
            emin = emax * 0.5
    
        self.display_E_min.set(emin)
        self.display_E_max.set(emax)

        # ===============================
        # Y
        # ===============================
        ymin = max(0.0, min(self.display_y_min.get(), 1000.0))
        ymax = max(0.1, min(self.display_y_max.get(), 1000.0))

        if ymin >= ymax:
            ymin = max(0.0, ymax * 0.5)

        self.display_y_min.set(ymin)
        self.display_y_max.set(ymax)

    def on_change_y_limits(self, val=None):
        self.apply_y_limits = True
        self.update_live_zoom(val)

    def update_live_zoom(self, val=None):

        if not hasattr(self, 'current_fig'):
            return

        if self.current_fig is None:
            return

        try:

            # ==========================
            # Validation
            # ==========================

            self.validate_ranges()

            tmin = self.display_t_min.get()
            tmax = self.display_t_max.get()

            emin = self.display_E_min.get()
            emax = self.display_E_max.get()
            ymin = self.display_y_min.get()
            ymax = self.display_y_max.get()

            # ==========================
            # Apply limits
            # ==========================

            for ax in self.current_fig.axes:

                xlabel = ax.get_xlabel().lower()

                if "time" in xlabel or "tof" in xlabel:
                    ax.set_xlim(tmin, tmax)

                elif "energy" in xlabel or "ev" in xlabel:
                    ax.set_xlim(emin, emax)

                if self.apply_y_limits:
                    ax.set_ylim(ymin, ymax)

            self.current_fig.canvas.draw_idle()

        except Exception as e:
            print(e)