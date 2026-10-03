"""
Streamlit app: tube & plate crossflow radiator calculator (laminar water side).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from plotting import geometry_figure, temperature_figure, tube_section_figure
from radiator_model import (
    INPUT_META,
    KELVIN_OFFSET,
    OUTPUT_META,
    PROPERTY_KEYS,
    RadiatorInputs,
    Row,
    calculate,
    default_properties,
    input_rows,
    validate_inputs,
)

st.set_page_config(page_title="Crossflow Radiator Calculator", page_icon="🌡️", layout="wide")

D = RadiatorInputs()  # defaults = original script values

PROPERTY_FIELDS = {
    "Water": [("Cpw", "Specific heat Cp,w (J/(kg·K))", "%.2f"),
              ("rho_w", "Density ρw (kg/m³)", "%.2f"),
              ("kw", "Thermal conductivity kw (W/(m·K))", "%.4f"),
              ("mu_w", "Dynamic viscosity μw (Pa·s)", "%.4e")],
    "Air": [("Cpa", "Specific heat Cp,a (J/(kg·K))", "%.2f"),
            ("rho_a", "Density ρa (kg/m³)", "%.4f"),
            ("ka", "Thermal conductivity ka (W/(m·K))", "%.5f"),
            ("mu_a", "Dynamic viscosity μa (Pa·s)", "%.4e")],
}


def reset_defaults() -> None:
    for key in [k for k in st.session_state if k.startswith("w_")]:
        del st.session_state[key]


def fnum(label: str, key: str, value: float, step: float, fmt: str, help: str | None = None) -> float:
    return st.number_input(label, min_value=0.0, value=float(value), step=float(step), format=fmt,
                           key=f"w_{key}", help=help)


def fmt_value(v: float | bool) -> str:
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if float(v).is_integer() and abs(v) < 1e6:
        return f"{v:.0f}"
    return f"{v:.6g}"


def rows_to_df(rows: list[Row]) -> pd.DataFrame:
    return pd.DataFrame({
        "Parameter": [r.name for r in rows],
        "Symbol": [r.symbol for r in rows],
        "Value": [fmt_value(r.value) for r in rows],
        "Unit": [r.unit for r in rows],
    })


def show_grouped(rows: list[Row]) -> None:
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.group, []).append(r)
    for group, grows in groups.items():
        st.markdown(f"**{group}**")
        st.dataframe(rows_to_df(grows), hide_index=True)


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Inputs")
    st.button("Reset all inputs to defaults", on_click=reset_defaults, width="stretch")

    with st.expander("Operating conditions", expanded=True):
        Vw = fnum("Water inlet velocity Vw (m/s)", "Vw", D.Vw, 0.005, "%.4f")
        Twi_C = st.number_input("Water inlet temperature Twi (°C)", min_value=-50.0, max_value=300.0,
                                value=D.Twi - KELVIN_OFFSET, step=1.0, format="%.2f", key="w_Twi")
        Va = fnum("Air face velocity Va (m/s)", "Va", D.Va, 0.05, "%.3f")
        Tai_C = st.number_input("Air inlet temperature Tai (°C)", min_value=-50.0, max_value=300.0,
                                value=D.Tai - KELVIN_OFFSET, step=1.0, format="%.2f", key="w_Tai")

    with st.expander("Tube and plate dimensions", expanded=True):
        Tw = fnum("Tube width Tw (mm)", "Tw", D.Tw * 1000, 0.5, "%.3f")
        Th = fnum("Tube height Th (mm)", "Th", D.Th * 1000, 0.1, "%.3f")
        Tt = fnum("Tube wall thickness Tt (mm)", "Tt", D.Tt * 1000, 0.05, "%.3f")
        Pt = fnum("Plate (fin) thickness Pt (mm)", "Pt", D.Pt * 1000, 0.1, "%.3f")
        Pw = fnum("Plate (fin) width Pw (mm)", "Pw", D.Pw * 1000, 0.5, "%.3f")

    with st.expander("Core geometry and materials"):
        Tl = fnum("Tube length Tl (mm)", "Tl", D.Tl * 1000, 5.0, "%.2f")
        Nt = st.number_input("Number of tubes Nt", min_value=1, value=D.Nt, step=1, key="w_Nt")
        Np = st.number_input("Number of plates Np", min_value=1, value=D.Np, step=1, key="w_Np")
        Ph = fnum("Plate (fin) height Ph (mm)", "Ph", D.Ph * 1000, 5.0, "%.2f")
        kt = fnum("Tube conductivity kt (W/(m·K))", "kt", D.kt, 5.0, "%.1f",
                  help="Default 230 W/(m·K): aluminium 1060 H18.")
        kp = fnum("Plate conductivity kp (W/(m·K))", "kp", D.kp, 5.0, "%.1f",
                  help="Default 230 W/(m·K): aluminium 1060 H18.")

    with st.expander("Model constants (advanced)"):
        p_atm = fnum("Air pressure (Pa)", "p_atm", D.p_atm, 500.0, "%.0f",
                     help="Used for air density, ρa = p / (R·Tai).")
        Nuw = fnum("Water-side Nusselt number Nuw", "Nuw", D.Nuw, 0.01, "%.2f",
                   help="4.36 is the fully developed laminar value for constant heat flux.")

    Twi_K, Tai_K = Twi_C + KELVIN_OFFSET, Tai_C + KELVIN_OFFSET
    auto = default_properties(Twi_K, Tai_K, p_atm) if Twi_K > 0 and Tai_K > 0 and p_atm > 0 else None

    overrides: dict[str, float] = {}
    if auto:
        for fluid, fields_ in PROPERTY_FIELDS.items():
            t_label = f"{Twi_C:g}" if fluid == "Water" else f"{Tai_C:g}"
            with st.expander(f"{fluid} properties"):
                manual = st.checkbox("Enter values manually", key=f"w_manual_{fluid}",
                                     help="Off: values come from the built-in table / ideal-gas equation "
                                          "at the inlet temperature. On: edit any value below.")
                if manual:
                    st.caption(f"Fields start at the automatic values for {t_label} °C and reset if the "
                               "inlet temperature changes.")
                    for key, label, fmt in fields_:
                        # temperature in the key resets prefilled values when the temperature changes
                        val = st.number_input(label, min_value=0.0, value=float(auto[key]),
                                              step=float(auto[key]) / 100, format=fmt,
                                              key=f"w_{key}_{t_label}_{p_atm:g}")
                        if abs(val - auto[key]) > 1e-6 * abs(auto[key]):
                            overrides[key] = val
                else:
                    st.caption(f"Automatic values at {t_label} °C:")
                    st.dataframe(
                        pd.DataFrame({"Property": [f[1] for f in fields_],
                                      "Value": [f"{auto[f[0]]:.6g}" for f in fields_]}),
                        hide_index=True)

# --------------------------------------------------------------------- main
st.title("🌡️ Tube & Plate Crossflow Radiator Calculator")
st.caption("Single-case numerical calculation, laminar water flow in flat tubes, air across plate fins. "
           "Change the inputs in the sidebar; results update immediately.")

inp = RadiatorInputs(
    Vw=Vw, Twi=Twi_K, Va=Va, Tai=Tai_K,
    Tw=Tw / 1000, Th=Th / 1000, Tt=Tt / 1000, Pt=Pt / 1000, Pw=Pw / 1000,
    Tl=Tl / 1000, Nt=int(Nt), Np=int(Np), Ph=Ph / 1000, kt=kt, kp=kp,
    p_atm=p_atm, Nuw=Nuw,
)

errors = validate_inputs(inp)
if errors:
    st.error("The inputs cannot be calculated yet:\n\n" + "\n".join(f"- {e}" for e in errors))
    with st.expander("Geometry preview unavailable until the inputs are valid"):
        st.write("Fix the points above and the preview and results will appear.")
    st.stop()

try:
    res = calculate(inp, overrides)
except (ValueError, ZeroDivisionError, OverflowError) as exc:
    st.error(f"Calculation failed: {exc}")
    st.stop()

v = res.values

# headline numbers
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Heat transfer rate Q", f"{v['Q']:.1f} W")
c2.metric("Effectiveness ε", f"{v['epsilon']:.4f}")
c3.metric("NTU", f"{v['NTU']:.4f}")
c4.metric("Water outlet", f"{v['Tco_C']:.2f} °C", f"{v['Tco_C'] - Twi_C:+.2f} K")
c5.metric("Air outlet", f"{v['Tao_C']:.2f} °C", f"{v['Tao_C'] - Tai_C:+.2f} K")

if res.all_checks_passed:
    st.success("All validity checks passed (laminar water flow, Prandtl range, thermal development, "
               "air Reynolds range).")
else:
    failed = [c for c in res.checks if not c.passed]
    st.warning("Results are shown, but the equations are outside their validity range:\n\n" +
               "\n".join(f"- **{c.name}** ({c.criterion}, {c.value}): {c.message}" for c in failed))
for w in res.warnings:
    st.warning(w)

tab_res, tab_geo, tab_temp, tab_chk = st.tabs(["Results", "Geometry plot", "Temperatures", "Checks and notes"])

with tab_res:
    st.subheader("Inputs used")
    show_grouped(input_rows(inp))
    st.subheader("Calculated results")
    show_grouped(res.rows)

    export = pd.DataFrame(
        [("Input", r.group, r.name, r.symbol, r.value, r.unit) for r in input_rows(inp)] +
        [("Output", r.group, r.name, r.symbol, r.value, r.unit) for r in res.rows],
        columns=["Section", "Group", "Parameter", "Symbol", "Value", "Unit"],
    )
    st.download_button("Download inputs and results (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="radiator_results.csv", mime="text/csv")

with tab_geo:
    fig_geo = geometry_figure(inp)
    st.pyplot(fig_geo)
    st.download_button("Download geometry plot (PNG)", fig_png(fig_geo),
                       file_name="radiator_geometry.png", mime="image/png")
    st.pyplot(tube_section_figure(inp))
    st.caption("Schematic drawn from the entered dimensions, not a manufacturing drawing. "
               "Tubes are drawn as rectangles in the core views; the equations treat the tube "
               "cross-section as a flat oval (rectangle with semicircular ends), shown in the detail above.")

with tab_temp:
    st.pyplot(temperature_figure(res))
    st.dataframe(pd.DataFrame({
        "Fluid": ["Water", "Air"],
        "Inlet (°C)": [f"{Twi_C:.2f}", f"{Tai_C:.2f}"],
        "Outlet (°C)": [f"{v['Tco_C']:.2f}", f"{v['Tao_C']:.2f}"],
        "Change (K)": [f"{v['Tco_C'] - Twi_C:+.3f}", f"{v['Tao_C'] - Tai_C:+.3f}"],
        "Heat-capacity rate (W/K)": [f"{v['Cw']:.4g}", f"{v['Ca']:.4g}"],
    }), hide_index=True)

with tab_chk:
    st.subheader("Validity checks")
    st.dataframe(pd.DataFrame({
        "Check": [c.name for c in res.checks],
        "Criterion": [c.criterion for c in res.checks],
        "Value": [c.value for c in res.checks],
        "Status": ["✅ Pass" if c.passed else "⚠️ Fail" for c in res.checks],
    }), hide_index=True)
    st.subheader("Model notes")
    for note in res.notes:
        st.markdown(f"- {note}")
    with st.expander("Key equations"):
        st.latex(r"D_h = \frac{4A_c}{P_{wet}},\qquad Re = \frac{\rho V D_h}{\mu},\qquad Pr = \frac{\mu c_p}{k}")
        st.latex(r"Nu_a = 0.644\,Re_a^{1/2}\,Pr_a^{1/3},\qquad h = \frac{Nu\,k}{D_h}")
        st.latex(r"\eta_f = \frac{\tanh(mL_c)}{mL_c},\quad m=\sqrt{\frac{2h_a}{k_p P_t}},\quad "
                 r"\eta_0 = 1-\frac{A_p}{A_r}(1-\eta_f)")
        st.latex(r"\frac{1}{UA} = \frac{1}{\eta_0 h_a A_r} + \frac{T_t}{k_t\,\bar A_t} + \frac{1}{h_w A_{ti}}")
        st.latex(r"\varepsilon = 1-\exp\!\left[\frac{NTU^{0.22}}{C_r}\left(e^{-C_r NTU^{0.78}}-1\right)\right],"
                 r"\qquad Q = \varepsilon\,C_{min}(T_{wi}-T_{ai})")
