"""Tableau de bord interactif pour visualiser les résultats d'analyse."""

from dataclasses import fields, is_dataclass

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display


# =====================================================
# STATE
# =====================================================

RESULTS = {}
CURVES = {}

output = widgets.Output()

curve_checkboxes = {}
result_checkboxes = {}

xscale_selector = widgets.Dropdown(options=["linear", "log"], value="linear", description="X")
yscale_selector = widgets.Dropdown(options=["linear", "log"], value="linear", description="Y")
show_errorbars = widgets.Checkbox(value=True, description="Show error bars")

xmin = widgets.FloatText(value=0.0, description="xmin")
xmax = widgets.FloatText(value=1.0, description="xmax")
ymin = widgets.FloatText(value=0.0, description="ymin")
ymax = widgets.FloatText(value=1.0, description="ymax")

AUTO = True


# =====================================================
# CURVES EXTRACTION (NO CHANGE)
# =====================================================

def is_spectrum(obj):
    return hasattr(obj, "x") and hasattr(obj, "y")


def is_dual(obj):
    return hasattr(obj, "smat") and hasattr(obj, "mat")


def walk(obj, prefix="", xu=None, yu=None):

    if is_spectrum(obj):
        return {
            prefix: {
                "x": obj.x,
                "y": obj.y,
                "yerr": getattr(obj, "yerr", None),
                "xu": xu,
                "yu": yu,
            }
        }

    if is_dual(obj):
        return {
            prefix + "_smat": {
                "x": obj.smat.x,
                "y": obj.smat.y,
                "yerr": getattr(obj.smat, "yerr", None),
                "xu": xu,
                "yu": yu,
            },
            prefix + "_mat": {
                "x": obj.mat.x,
                "y": obj.mat.y,
                "yerr": getattr(obj.mat, "yerr", None),
                "xu": xu,
                "yu": yu,
            },
        }

    if is_dataclass(obj):
        out = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            new_xu = getattr(obj, "unit_x", xu)
            new_yu = getattr(obj, "unit_y", yu)

            out.update(
                walk(
                v,
                prefix + "." + f.name if prefix else f.name,
                new_xu,
                new_yu
            )
        )             
        return out

    return {}


def build_curves(results):
    return {k: walk(v, prefix="") for k, v in results.items()}


# =====================================================
# SELECTION
# =====================================================

def get_selection():
    out = []
    for curve_name, res_map in result_checkboxes.items():
        if not curve_checkboxes[curve_name].value:
            continue
        for res_name, cb in res_map.items():
            if cb.value:
                out.append((res_name, curve_name))
    return out


# =====================================================
# SAFE AXIS (CRITICAL FIX FOR LOG)
# =====================================================

def apply_axes(fig):

    xmode = xscale_selector.value
    ymode = yscale_selector.value

    xmin_v = float(xmin.value)
    xmax_v = float(xmax.value)
    ymin_v = float(ymin.value)
    ymax_v = float(ymax.value)

    # -------------------------
    # X axis
    # -------------------------
    if xmode == "log":

        xmin_v = max(xmin_v, 1e-300)
        xmax_v = max(xmax_v, 1e-300)

        if xmin_v > xmax_v:
            xmin_v, xmax_v = xmax_v, xmin_v

        x_range = [
            np.log10(xmin_v),
            np.log10(xmax_v)
        ]

    else:

        x_range = [
            xmin_v,
            xmax_v
        ]

    # -------------------------
    # Y axis
    # -------------------------
    if ymode == "log":

        ymin_v = max(ymin_v, 1e-300)
        ymax_v = max(ymax_v, 1e-300)

        if ymin_v > ymax_v:
            ymin_v, ymax_v = ymax_v, ymin_v

        y_range = [
            np.log10(ymin_v),
            np.log10(ymax_v)
        ]

    else:

        y_range = [
            ymin_v,
            ymax_v
        ]

    fig.update_xaxes(
        type=xmode,
        range=x_range
    )

    fig.update_yaxes(
        type=ymode,
        range=y_range
    )


# =====================================================
# PLOT
# =====================================================

def update_plot(*_):

    global AUTO

    sel = get_selection()

    fig = go.Figure()

    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    titles = set()
    x_units = set()
    y_units = set()

    for res_name, curve_name in sel:

        c = CURVES[res_name][curve_name]
        x_units.add(c.get("xu"))
        y_units.add(c.get("yu"))

        x = np.asarray(c["x"], dtype=float)
        y = np.asarray(c["y"], dtype=float)

        yerr_raw = c.get("yerr", None)
        if yerr_raw is None:
            yerr = np.full_like(y, np.nan)
        else:
            yerr = np.asarray(yerr_raw)
            if yerr.shape != y.shape:
                yerr = np.broadcast_to(yerr, y.shape)
            try:
                yerr = yerr.astype(float)
            except (TypeError, ValueError):
                yerr = np.full_like(y, np.nan)

        n = min(len(x), len(y), len(yerr))
        if n == 0:
            continue

        x, y, yerr = x[:n], y[:n], yerr[:n]

        x_min = min(x_min, np.nanmin(x))
        x_max = max(x_max, np.nanmax(x))
        y_min = min(y_min, np.nanmin(y))
        y_max = max(y_max, np.nanmax(y))

        has_error = np.any(np.isfinite(yerr))
        trace_kwargs = {
            "x": x,
            "y": y,
            "mode": "lines+markers" if has_error else "lines",
            "name": f"[{res_name}] {curve_name}",
        }

        if show_errorbars.value and has_error:
            yerr_safe = np.where(np.isfinite(yerr), yerr, 0.0)
            trace_kwargs["error_y"] = {
                "type": "data",
                "array": yerr_safe,
                "visible": True,
            }

        scatter_cls = go.Scattergl if len(x) > 2000 else go.Scatter
        fig.add_trace(scatter_cls(**trace_kwargs))

        titles.add(res_name)

    # =========================
    # AUTO SCALE (ONLY WHEN ON)
    # =========================

    if AUTO and x_min < np.inf and y_min < np.inf:
        xmin.value = float(x_min)
        xmax.value = float(x_max)
        ymin.value = float(y_min)
        ymax.value = float(y_max)
    
    axis_labels = {
        "s": "Time of flight (s)",
        "eV": "Energy (eV)",
        "barn": "Cross section (barn)",
        "%": "Relative error (%)",
    }

    x_unit = next(iter(x_units)) if len(x_units) == 1 else "mixed units"
    y_unit = next(iter(y_units)) if len(y_units) == 1 else "mixed units"

    fig.update_xaxes(
        title=axis_labels.get(x_unit, f"x ({x_unit})")
    )

    fig.update_yaxes(
        title=axis_labels.get(y_unit, f"y ({y_unit})")
    )

    # =========================
    # APPLY AXES
    # =========================

    apply_axes(fig)

    # =========================
    # LAYOUT (KEEP EVERYTHING)
    # =========================

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=750,
        hovermode="x unified",
        legend_title="Legend",
        title=" + ".join(sorted(titles)) if titles else "Plot"
    )

    with output:
        output.clear_output(wait=True)
        display(fig)


# =====================================================
# UI (KEEP YOUR ACCORDION IDEA)
# =====================================================

def build_ui():

    global curve_checkboxes, result_checkboxes

    curve_checkboxes = {}
    result_checkboxes = {}

    all_curves = set()
    for c in CURVES.values():
        all_curves.update(c.keys())

    categories = {}
    for c in all_curves:
        root = c.split(".")[0]
        categories.setdefault(root, []).append(c)

    blocks = []

    for cat, curves in categories.items():

        curve_blocks = []

        for curve in sorted(curves):

            cb_curve = widgets.Checkbox(value=False, description=curve)
            curve_checkboxes[curve] = cb_curve

            res_map = {}
            res_widgets = []

            for res in CURVES.keys():
                cb = widgets.Checkbox(value=False, description=res)
                res_map[res] = cb
                res_widgets.append(cb)

            result_checkboxes[curve] = res_map

            acc = widgets.Accordion([widgets.VBox(res_widgets)])
            acc.set_title(0, "Results")

            curve_blocks.append(widgets.VBox([cb_curve, acc]))

        acc_cat = widgets.Accordion([widgets.VBox(curve_blocks)])
        acc_cat.set_title(0, "Curves")

        blocks.append(widgets.VBox([widgets.HTML(f"<b>{cat}</b>"), acc_cat]))

    return widgets.VBox(blocks)


# =====================================================
# LAUNCH
# =====================================================

def launch_dashboard(results):

    global RESULTS, CURVES, AUTO

    RESULTS = results
    CURVES = build_curves(results)

    ui = build_ui()

    btn_auto = widgets.Button(description="Auto scale", button_style="success")
    btn_manual = widgets.Button(description="Manual scale", button_style="warning")

    def set_auto(_):
        global AUTO
        AUTO = True
        update_plot()

    def set_manual(_):
        global AUTO
        AUTO = False
        update_plot()

    btn_auto.on_click(set_auto)
    btn_manual.on_click(set_manual)

    xscale_selector.observe(update_plot, "value")
    yscale_selector.observe(update_plot, "value")
    show_errorbars.observe(update_plot, "value")

    for cb_map in result_checkboxes.values():
        for cb in cb_map.values():
            cb.observe(update_plot, "value")

    for cb in curve_checkboxes.values():
        cb.observe(update_plot, "value")

    for w in [xmin, xmax, ymin, ymax]:
        w.observe(update_plot, "value")

    # Force initial UI update after display to avoid repeated plot rebuilds
    ui.layout.min_width = '280px'

    display(
        widgets.HBox([
            widgets.VBox([
                xscale_selector,
                yscale_selector,
                show_errorbars,
                xmin,
                xmax,
                ymin,
                ymax,
                btn_auto,
                btn_manual,
                ui
            ]),
            output
        ])
    )

    update_plot()