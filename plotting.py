"""
Matplotlib figures for the radiator app.

Figures are built with matplotlib.figure.Figure (not pyplot), so there is no global
state to leak between Streamlit reruns.
"""

from __future__ import annotations

import math

from matplotlib import patches
from matplotlib.figure import Figure

from radiator_model import KELVIN_OFFSET, CalcResult, RadiatorInputs

TUBE_COLOR = "#1f4fd8"
PLATE_COLOR = "#d62728"
WATER_COLOR = "#2b8cbe"
AIR_COLOR = "#7a8aa0"


def _mm(x: float) -> str:
    return f"{x * 1000:g} mm"


def geometry_figure(inp: RadiatorInputs) -> Figure:
    """Front view (X-Y) and right-side view (Z-Y) of the core, drawn to scale."""
    Tl, Ph, Th, Tw, Pt, Pw = inp.Tl, inp.Ph, inp.Th, inp.Tw, inp.Pt, inp.Pw
    Nt, Np = inp.Nt, inp.Np

    vertical_spacing = Ph / (Nt + 1) - Th
    plate_spacing = Tl / (Np + 1) - Pt
    tube_y0 = vertical_spacing + Th / 2
    plate_x0 = plate_spacing + Pt / 2

    # Axis limits grow with the core; they equal the original 0.30 / 0.15 / 0.275 m defaults.
    x_max = max(0.30, math.ceil(1.15 * Tl / 0.05) * 0.05)
    z_max = max(0.15, math.ceil(1.2 * max(Tw, Pw) / 0.025) * 0.025)
    y_max = max(0.275, 1.22 * Ph)
    tube_x_side = (z_max - Tw) / 2
    plate_x_side = (z_max - Pw) / 2

    fig = Figure(figsize=(12, 6.4), layout="constrained")
    ax_front, ax_side = fig.subplots(1, 2)

    def rect(ax, xy, w, h, color):
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor=color, facecolor=color,
                                       alpha=0.14, linewidth=0))
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor=color, facecolor="none", linewidth=1.5))

    for i in range(Nt):
        y = tube_y0 + i * (Th + vertical_spacing)
        rect(ax_front, (0.0, y), Tl, Th, TUBE_COLOR)
        rect(ax_side, (tube_x_side, y), Tw, Th, TUBE_COLOR)
    for i in range(Np):
        rect(ax_front, (plate_x0 + i * (Pt + plate_spacing), 0.0), Pt, Ph, PLATE_COLOR)
    rect(ax_side, (plate_x_side, 0.0), Pw, Ph, PLATE_COLOR)

    def dim(ax, x0, x1, y, text):
        ax.annotate("", xy=(x0, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1))
        ax.text((x0 + x1) / 2, y + 0.006 * y_max / 0.275, text, ha="center", va="bottom",
                fontsize=9, color="#333")

    dim(ax_front, 0.0, Tl, Ph * 1.04, f"Tl = {_mm(Tl)}")
    dim(ax_side, plate_x_side, plate_x_side + Pw, Ph * 1.04, f"Pw = {_mm(Pw)}")
    dim(ax_side, tube_x_side, tube_x_side + Tw, Ph * 1.13, f"Tw = {_mm(Tw)}")

    ax_front.set(xlabel="X axis (m)", ylabel="Y axis (m)", xlim=(0, x_max), ylim=(0, y_max),
                 title="Front View of Crossflow Radiator")
    ax_side.set(xlabel="Z axis (m)", ylabel="Y axis (m)", xlim=(0, z_max), ylim=(0, y_max),
                title="Right Side View of Crossflow Radiator")
    for ax in (ax_front, ax_side):
        ax.set_aspect("equal")
        ax.minorticks_on()
        ax.grid(which="both", axis="both", linestyle="--", alpha=0.5)

    fig.legend(
        handles=(patches.Patch(facecolor=TUBE_COLOR, alpha=0.4, edgecolor=TUBE_COLOR, label="Tubes"),
                 patches.Patch(facecolor=PLATE_COLOR, alpha=0.4, edgecolor=PLATE_COLOR, label="Plates")),
        loc="outside lower center", ncol=2, frameon=False,
    )
    return fig


def tube_section_figure(inp: RadiatorInputs) -> Figure:
    """Zoomed cross-section of one tube: outer shape, wall and water passage (mm)."""
    Tw, Th, Tt = inp.Tw * 1000, inp.Th * 1000, inp.Tt * 1000
    fig = Figure(figsize=(11, 2.4), layout="constrained")
    ax = fig.subplots()

    ax.add_patch(patches.FancyBboxPatch(
        (0, 0), Tw, Th, boxstyle=f"round,pad=0,rounding_size={Th / 2}",
        facecolor="#c7d6f5", edgecolor=TUBE_COLOR, linewidth=1.6))
    ax.add_patch(patches.FancyBboxPatch(
        (Tt, Tt), Tw - 2 * Tt, Th - 2 * Tt, boxstyle=f"round,pad=0,rounding_size={(Th - 2 * Tt) / 2}",
        facecolor="#e6f3fb", edgecolor=WATER_COLOR, linewidth=1.0, linestyle="--"))

    pad = max(Tw, 10) * 0.06
    arrow = dict(arrowstyle="<->", color="#444", lw=1)
    ax.annotate("", xy=(0, Th + pad), xytext=(Tw, Th + pad), arrowprops=arrow)
    ax.text(Tw / 2, Th + pad * 1.3, f"Tw = {Tw:g} mm", ha="center", va="bottom", fontsize=9)
    ax.annotate("", xy=(-pad, 0), xytext=(-pad, Th), arrowprops=arrow)
    ax.text(-pad * 1.4, Th / 2, f"Th = {Th:g} mm", ha="right", va="center", fontsize=9)
    ax.annotate(f"Tt = {Tt:g} mm", xy=(Tw * 0.5, Tt / 2), xytext=(Tw * 0.62, -Th * 0.9 - pad),
                ha="center", va="top", fontsize=9, arrowprops=dict(arrowstyle="->", color="#444", lw=1))
    ax.text(Tw / 2, Th / 2, "water passage", ha="center", va="center", fontsize=9, color=WATER_COLOR)

    ax.set_aspect("equal")
    ax.set_xlim(-Tw * 0.2, Tw * 1.05)
    ax.set_ylim(-Th * 1.8 - pad * 1.6, Th + pad * 3.5)
    ax.axis("off")
    ax.set_title("Tube cross-section (not to scale with the core views)", fontsize=11)
    return fig


def temperature_figure(result: CalcResult) -> Figure:
    """Inlet vs outlet temperature for each fluid."""
    inp = result.inputs
    v = result.values
    fig = Figure(figsize=(10, 4), layout="constrained")
    ax_w, ax_a = fig.subplots(1, 2)
    panels = (
        (ax_w, "Water", inp.Twi - KELVIN_OFFSET, v["Tco_C"], WATER_COLOR),
        (ax_a, "Air", inp.Tai - KELVIN_OFFSET, v["Tao_C"], AIR_COLOR),
    )
    for ax, name, t_in, t_out, color in panels:
        bars = ax.bar(["Inlet", "Outlet"], [t_in, t_out], color=[color, color], alpha=0.85, width=0.55)
        ax.bar_label(bars, fmt="%.2f °C", padding=3)
        lo, hi = min(0.0, t_in, t_out), max(t_in, t_out)
        ax.set_ylim(lo * 1.15 if lo < 0 else 0, hi + (hi - lo) * 0.15 + 1e-9)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{name}: ΔT = {t_out - t_in:+.2f} K")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    return fig
