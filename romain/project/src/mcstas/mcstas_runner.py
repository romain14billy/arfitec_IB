import itertools
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.mcstas.experiment import Experiment


# =====================================================
# LOG
# =====================================================

def log(title, msg="", width=80):
    print("\n" + "─" * width)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {title}")
    if msg:
        print(msg)
    print("─" * width)


# =====================================================
# PARAMS
# =====================================================

def get_dummy_parameters(config):
    dummy = {}
    for k, v in config["scan_grid"].items():
        dummy[k] = v[0]
    return {
        **dummy,
        **config.get("fixed_parameters", {}),
    }


# =====================================================
# MCSTAS COMPILE
# =====================================================

def compile_mcstas(config):
    dummy = get_dummy_parameters(config)

    cmd = [
        "mcrun",
        "-c",
        "--mpi=auto",
        str(config["instr_file"]),
    ]

    for k, v in dummy.items():
        cmd.append(f"{k}={v}")

    log("COMPILATION MCSTAS", " ".join(cmd))
    subprocess.run(cmd, check=True)
    log("COMPILATION TERMINÉE ✔")


# =====================================================
# MCSTAS RUN
# =====================================================

def run_mcstas(config, output_dir, params):
    cmd = [
        "mcrun",
        "--mpi=auto",
        str(config["instr_file"]),
        "-d",
        str(output_dir),
        "-n",
        str(int(config["ncount"])),
    ]

    for k, v in config.get("fixed_parameters", {}).items():
        cmd.append(f"{k}={v}")

    for k, v in params.items():
        cmd.append(f"{k}={v}")

    log("RUN MCSTAS", " ".join(cmd))
    subprocess.run(cmd, check=True)


# =====================================================
# SCAN ITERATOR
# =====================================================

def iter_scan_points(scan_grid: dict):
    keys = list(scan_grid.keys())
    values = [scan_grid[k] for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# =====================================================
# FOLDER NAME
# =====================================================

def build_folder_name(scan_point: dict):
    return "_".join([f"{k}{v}" for k, v in scan_point.items()])


# =====================================================
# COMPLETION CHECK
# =====================================================

def scan_point_done(base_dir: Path) -> bool:
    """
    True si déjà calculé correctement.
    """
    return (base_dir / "done.flag").exists() or (base_dir / "metadata.json").exists()


# =====================================================
# SINGLE POINT
# =====================================================

def run_scan_point(config, scan_point):
    folder_name = build_folder_name(scan_point)
    base_dir = config["paths"]["raw"] / folder_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 SKIP SI DÉJÀ FAIT
    if not config.get("force", False) and scan_point_done(base_dir):
        log("SKIP ✔ (déjà existant)", str(scan_point))
        return

    log("SCAN", str(scan_point))

    smat_dir = base_dir / "smat"
    mat_dir = base_dir / "mat"

    # -------------------------
    # SMAT
    # -------------------------
    if not smat_dir.exists() or not (smat_dir / "metadata.json").exists():
        run_mcstas(
            config,
            smat_dir,
            {
                **scan_point,
                "use_sample": 0,
            },
        )

    # -------------------------
    # MAT
    # -------------------------
    if not mat_dir.exists() or not (mat_dir / "metadata.json").exists():
        run_mcstas(
            config,
            mat_dir,
            {
                **scan_point,
                "use_sample": 1,
            },
        )

    # -------------------------
    # METADATA
    # -------------------------
    meta = {
        "experiment": config["name"],
        "scan_point": scan_point,
        "ncount": config["ncount"],
        "timestamp": datetime.now().isoformat(),
    }

    with open(base_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    # -------------------------
    # FLAG FINAL
    # -------------------------
    (base_dir / "done.flag").write_text("done")

    log("FIN POINT ✔", str(scan_point))


# =====================================================
# CAMPAIGN
# =====================================================

def run_campaign(config):
    log("DEBUT CAMPAGNE")
    start = time.time()

    for scan_point in iter_scan_points(config["scan_grid"]):
        run_scan_point(config, scan_point)

    log("CAMPAGNE TERMINÉE ✔", f"Durée totale : {time.time() - start:.1f}s")


# =====================================================
# EXPERIMENT BUILDER
# =====================================================

def build_experiments(config):
    experiments = []

    for scan_point in iter_scan_points(config["scan_grid"]):
        name = build_folder_name(scan_point)
        folder = config["paths"]["raw"] / name

        experiments.append(
            Experiment(
                name=name,
                scan_point=scan_point,
                folder=folder
            )
        )

    return experiments


# =====================================================
# CLEAN PROJECT
# =====================================================

def clean_project(clean_root: Path):
    """
    Supprime tout sauf les notebooks.
    """
    if not clean_root.exists():
        return

    for item in clean_root.iterdir():
        if item.suffix == ".ipynb":
            continue

        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)