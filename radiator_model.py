"""
Calculation engine for a single-case tube & plate crossflow radiator (laminar water side).

Pure Python (standard library only), no Streamlit or plotting imports, so it can be
tested and reused on its own. All quantities are SI internally: m, m^2, K, W, Pa.s.

The equations are a direct port of the original spreadsheet-style script.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, fields
from typing import Any

AIR_GAS_CONSTANT_J_KG_K = 287.05
STD_ATM_PRESSURE_PA = 101325.0
KELVIN_OFFSET = 273.15

# Property anchors from the original sheet: (temperature K, value).
WATER_TABLE: dict[str, tuple[tuple[float, float], ...]] = {
    "Cpw": ((353.15, 4197.0), (363.15, 4205.0), (373.15, 4216.0)),
    "rho_w": ((353.15, 971.8), (363.15, 965.3), (373.15, 958.4)),
    "kw": ((353.15, 0.670), (363.15, 0.675), (373.15, 0.679)),
    "mu_w": ((353.15, 3.55e-4), (363.15, 3.15e-4), (373.15, 2.82e-4)),
}
AIR_TABLE: dict[str, tuple[tuple[float, float], ...]] = {
    "Cpa": ((298.15, 1006.5), (308.15, 1006.7), (318.15, 1007.0)),
    "ka": ((298.15, 0.0262), (308.15, 0.0269), (318.15, 0.0276)),
    "mu_a": ((298.15, 1.837e-5), (308.15, 1.884e-5), (318.15, 1.931e-5)),
}
WATER_RANGE_K = (WATER_TABLE["Cpw"][0][0], WATER_TABLE["Cpw"][-1][0])
AIR_RANGE_K = (AIR_TABLE["Cpa"][0][0], AIR_TABLE["Cpa"][-1][0])

PROPERTY_KEYS = ("Cpw", "rho_w", "kw", "mu_w", "Cpa", "rho_a", "ka", "mu_a")


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class RadiatorInputs:
    """All user inputs, SI units. Defaults reproduce the original script."""

    # operating conditions
    Vw: float = 0.025        # water inlet velocity, m/s
    Twi: float = 353.15      # water inlet temperature, K
    Va: float = 0.75         # air face velocity, m/s
    Tai: float = 277.15      # air inlet temperature, K
    # tube and plate dimensions
    Tw: float = 0.032        # tube width, m
    Th: float = 0.0035       # tube height, m
    Tt: float = 0.00025      # tube wall thickness, m
    Pt: float = 0.0025       # plate (fin) thickness, m
    Pw: float = 0.042        # plate (fin) width, m
    # core geometry and materials
    Tl: float = 0.25         # tube length, m
    Nt: int = 5              # number of tubes
    Np: int = 5              # number of plates
    Ph: float = 0.25         # plate (fin) height, m
    kt: float = 230.0        # tube conductivity, W/(m K)
    kp: float = 230.0        # plate conductivity, W/(m K)
    # model constants
    p_atm: float = STD_ATM_PRESSURE_PA   # air pressure for density, Pa
    Nuw: float = 4.36                    # water-side Nusselt number (laminar, const. flux)


FLOAT_FIELDS = tuple(f.name for f in fields(RadiatorInputs) if f.name not in ("Nt", "Np"))


# --------------------------------------------------------------- properties
def piecewise_linear(temp_K: float, points: tuple[tuple[float, float], ...]) -> float:
    """Linear interpolation, with linear extrapolation beyond the end points."""
    if temp_K <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
        return y0 + (temp_K - x0) * (y1 - y0) / (x1 - x0)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if temp_K <= x1:
            return y0 + (temp_K - x0) * (y1 - y0) / (x1 - x0)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return y0 + (temp_K - x0) * (y1 - y0) / (x1 - x0)


def default_properties(Twi: float, Tai: float, p_atm: float = STD_ATM_PRESSURE_PA) -> dict[str, float]:
    """Water and air properties from the built-in tables / ideal-gas law."""
    props = {k: piecewise_linear(Twi, pts) for k, pts in WATER_TABLE.items()}
    props.update({k: piecewise_linear(Tai, pts) for k, pts in AIR_TABLE.items()})
    props["rho_a"] = p_atm / (AIR_GAS_CONSTANT_J_KG_K * Tai)
    return {k: props[k] for k in PROPERTY_KEYS}


# ------------------------------------------------------------------ outputs
# key -> (group, parameter name, symbol, unit), in display order
OUTPUT_META: dict[str, tuple[str, str, str, str]] = {
    "Cpw": ("Fluid properties", "Water specific heat", "Cp,w", "J/(kg·K)"),
    "rho_w": ("Fluid properties", "Water density", "ρw", "kg/m³"),
    "kw": ("Fluid properties", "Water thermal conductivity", "kw", "W/(m·K)"),
    "mu_w": ("Fluid properties", "Water dynamic viscosity", "μw", "Pa·s"),
    "Prw": ("Fluid properties", "Water Prandtl number", "Prw", "–"),
    "Cpa": ("Fluid properties", "Air specific heat", "Cp,a", "J/(kg·K)"),
    "rho_a": ("Fluid properties", "Air density", "ρa", "kg/m³"),
    "ka": ("Fluid properties", "Air thermal conductivity", "ka", "W/(m·K)"),
    "mu_a": ("Fluid properties", "Air dynamic viscosity", "μa", "Pa·s"),
    "Pra": ("Fluid properties", "Air Prandtl number", "Pra", "–"),
    "Opt": ("Tube geometry and areas", "Tube outer perimeter", "Opt", "m"),
    "Wpt": ("Tube geometry and areas", "Tube inner (wetted) perimeter", "Wpt", "m"),
    "At": ("Tube geometry and areas", "Tube outer surface area", "At", "m²"),
    "Ati": ("Tube geometry and areas", "Tube inner surface area", "Ati", "m²"),
    "Act": ("Tube geometry and areas", "Tube inner flow cross-section", "Act", "m²"),
    "Acto": ("Tube geometry and areas", "Tube outer cross-section (plate intersection)", "Acto", "m²"),
    "Ato": ("Tube geometry and areas", "Exposed tube outer surface area", "Ato", "m²"),
    "Ap": ("Tube geometry and areas", "Plate (fin) surface area", "Ap", "m²"),
    "Ar": ("Tube geometry and areas", "Total air-side surface area", "Ar", "m²"),
    "St": ("Core spacing and air passage", "Vertical gap between tubes", "St", "m"),
    "Sp": ("Core spacing and air passage", "Horizontal gap between plates", "Sp", "m"),
    "Apa": ("Core spacing and air passage", "Air passage free-flow area", "Apa", "m²"),
    "Awpa": ("Core spacing and air passage", "Air passage wetted perimeter", "Awpa", "m"),
    "Dht": ("Hydraulic diameters and flow regime", "Tube hydraulic diameter", "Dht", "m"),
    "Dha": ("Hydraulic diameters and flow regime", "Air passage hydraulic diameter", "Dha", "m"),
    "Dexp": ("Hydraulic diameters and flow regime", "Thermal-development limit diameter", "Dexp", "m"),
    "thermally_developed": ("Hydraulic diameters and flow regime", "Water flow thermally developed (Dht < Dexp)", "–", "flag"),
    "Rew": ("Hydraulic diameters and flow regime", "Water Reynolds number", "Rew", "–"),
    "Vmax_air": ("Hydraulic diameters and flow regime", "Air velocity in free-flow area", "Vmax", "m/s"),
    "Rea": ("Hydraulic diameters and flow regime", "Air Reynolds number", "Rea", "–"),
    "Nuw": ("Heat transfer coefficients", "Water Nusselt number", "Nuw", "–"),
    "Nua": ("Heat transfer coefficients", "Air Nusselt number", "Nua", "–"),
    "hw": ("Heat transfer coefficients", "Water-side convection coefficient", "hw", "W/(m²·K)"),
    "ha": ("Heat transfer coefficients", "Air-side convection coefficient", "ha", "W/(m²·K)"),
    "m": ("Plate (fin) efficiency", "Fin parameter", "m", "1/m"),
    "Lc": ("Plate (fin) efficiency", "Fin characteristic length", "Lc", "m"),
    "nf": ("Plate (fin) efficiency", "Plate (fin) efficiency", "ηf", "–"),
    "n0": ("Plate (fin) efficiency", "Overall surface efficiency", "η0", "–"),
    "UA": ("Heat exchanger performance", "Overall conductance", "UA", "W/K"),
    "Cw": ("Heat exchanger performance", "Water heat-capacity rate", "Cw", "W/K"),
    "Ca": ("Heat exchanger performance", "Air heat-capacity rate", "Ca", "W/K"),
    "Cmin": ("Heat exchanger performance", "Minimum heat-capacity rate", "Cmin", "W/K"),
    "Cmax": ("Heat exchanger performance", "Maximum heat-capacity rate", "Cmax", "W/K"),
    "Cr": ("Heat exchanger performance", "Heat-capacity rate ratio", "Cr", "–"),
    "NTU": ("Heat exchanger performance", "Number of transfer units", "NTU", "–"),
    "epsilon": ("Heat exchanger performance", "Effectiveness", "ε", "–"),
    "Q": ("Heat exchanger performance", "Heat transfer rate", "Q", "W"),
    "Tco": ("Heat exchanger performance", "Water outlet temperature", "Tco", "K"),
    "Tco_C": ("Heat exchanger performance", "Water outlet temperature", "Tco", "°C"),
    "Tao": ("Heat exchanger performance", "Air outlet temperature", "Tao", "K"),
    "Tao_C": ("Heat exchanger performance", "Air outlet temperature", "Tao", "°C"),
}

# key -> (group, parameter name, symbol, unit, scale from SI to display unit)
INPUT_META: dict[str, tuple[str, str, str, str, float]] = {
    "Vw": ("Operating conditions", "Water inlet velocity", "Vw", "m/s", 1.0),
    "Twi": ("Operating conditions", "Water inlet temperature", "Twi", "K", 1.0),
    "Twi_C": ("Operating conditions", "Water inlet temperature", "Twi", "°C", 1.0),
    "Va": ("Operating conditions", "Air face velocity", "Va", "m/s", 1.0),
    "Tai": ("Operating conditions", "Air inlet temperature", "Tai", "K", 1.0),
    "Tai_C": ("Operating conditions", "Air inlet temperature", "Tai", "°C", 1.0),
    "Tw": ("Tube and plate dimensions", "Tube width", "Tw", "mm", 1000.0),
    "Th": ("Tube and plate dimensions", "Tube height", "Th", "mm", 1000.0),
    "Tt": ("Tube and plate dimensions", "Tube wall thickness", "Tt", "mm", 1000.0),
    "Pt": ("Tube and plate dimensions", "Plate (fin) thickness", "Pt", "mm", 1000.0),
    "Pw": ("Tube and plate dimensions", "Plate (fin) width", "Pw", "mm", 1000.0),
    "Tl": ("Core geometry and materials", "Tube length", "Tl", "mm", 1000.0),
    "Nt": ("Core geometry and materials", "Number of tubes", "Nt", "–", 1.0),
    "Np": ("Core geometry and materials", "Number of plates", "Np", "–", 1.0),
    "Ph": ("Core geometry and materials", "Plate (fin) height", "Ph", "mm", 1000.0),
    "kt": ("Core geometry and materials", "Tube thermal conductivity", "kt", "W/(m·K)", 1.0),
    "kp": ("Core geometry and materials", "Plate thermal conductivity", "kp", "W/(m·K)", 1.0),
    "p_atm": ("Model constants", "Air pressure (for air density)", "p", "Pa", 1.0),
    "Nuw_in": ("Model constants", "Water-side Nusselt number", "Nuw", "–", 1.0),
}


def input_values(inp: RadiatorInputs) -> dict[str, float]:
    """Inputs keyed like INPUT_META, in SI (temperatures in both K and °C)."""
    return {
        "Vw": inp.Vw, "Twi": inp.Twi, "Twi_C": inp.Twi - KELVIN_OFFSET,
        "Va": inp.Va, "Tai": inp.Tai, "Tai_C": inp.Tai - KELVIN_OFFSET,
        "Tw": inp.Tw, "Th": inp.Th, "Tt": inp.Tt, "Pt": inp.Pt, "Pw": inp.Pw,
        "Tl": inp.Tl, "Nt": inp.Nt, "Np": inp.Np, "Ph": inp.Ph, "kt": inp.kt, "kp": inp.kp,
        "p_atm": inp.p_atm, "Nuw_in": inp.Nuw,
    }


@dataclass(frozen=True)
class Row:
    group: str
    name: str
    symbol: str
    value: float | bool
    unit: str


@dataclass(frozen=True)
class Check:
    name: str
    criterion: str
    value: str
    passed: bool
    message: str


@dataclass
class CalcResult:
    values: dict[str, float | bool]
    rows: list[Row]
    checks: list[Check]
    notes: list[str]
    warnings: list[str]
    overridden: tuple[str, ...] = ()
    inputs: RadiatorInputs = field(default_factory=RadiatorInputs)

    @property
    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)


def input_rows(inp: RadiatorInputs) -> list[Row]:
    vals = input_values(inp)
    rows = []
    for key, (group, name, symbol, unit, scale) in INPUT_META.items():
        v = vals[key]
        rows.append(Row(group, name, symbol, v if key in ("Nt", "Np") else v * scale, unit))
    return rows


# --------------------------------------------------------------- validation
def _is_count(x: Any) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool) and x >= 1


def validate_inputs(inp: RadiatorInputs) -> list[str]:
    """Return a list of human-readable problems (empty list means inputs are usable)."""
    errors: list[str] = []
    labels = {k: v[1] for k, v in INPUT_META.items()}
    labels["Nuw"] = labels["Nuw_in"]
    for name in FLOAT_FIELDS:
        value = getattr(inp, name)
        if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            errors.append(f"{labels.get(name, name)} ({name}) must be a positive number.")
    for name in ("Nt", "Np"):
        if not _is_count(getattr(inp, name)):
            errors.append(f"{labels[name]} ({name}) must be a whole number of at least 1.")
    if errors:
        return errors

    if not (inp.Tw > inp.Th > 2 * inp.Tt > 0):
        errors.append("Tube dimensions must satisfy Tw > Th > 2·Tt: width greater than height, "
                      "and the wall thickness less than half the height.")
    St = inp.Ph / (inp.Nt + 1) - inp.Th
    Sp = inp.Tl / (inp.Np + 1) - inp.Pt
    if St <= 0:
        errors.append(f"The tubes do not fit: vertical gap St = Ph/(Nt+1) − Th = {St * 1000:.2f} mm. "
                      "Reduce the number of tubes or tube height, or increase plate height.")
    if Sp <= 0:
        errors.append(f"The plates do not fit: horizontal gap Sp = Tl/(Np+1) − Pt = {Sp * 1000:.2f} mm. "
                      "Reduce the number of plates or plate thickness, or increase tube length.")
    return errors


# -------------------------------------------------------------- calculation
def calculate(
    inp: RadiatorInputs,
    overrides: dict[str, float] | None = None,
    strict: bool = False,
) -> CalcResult:
    """
    Run the full single-case calculation.

    overrides: optional manual fluid properties, any of PROPERTY_KEYS, replacing the
               table / ideal-gas values (Prandtl numbers are always derived).
    strict:    raise ValueError if a validity check fails (the original script's
               behaviour). When False, failed checks are reported in `result.checks`.
    """
    errors = validate_inputs(inp)
    if errors:
        raise ValueError(" ".join(errors))

    overrides = dict(overrides or {})
    unknown = set(overrides) - set(PROPERTY_KEYS)
    if unknown:
        raise ValueError(f"Unknown property override(s): {', '.join(sorted(unknown))}.")
    for k, v in overrides.items():
        if not (isinstance(v, numbers.Real) and math.isfinite(v) and v > 0):
            raise ValueError(f"Property override {k} must be a positive number.")

    Vw, Twi, Va, Tai = inp.Vw, inp.Twi, inp.Va, inp.Tai
    Tw, Th, Tt, Pt, Pw = inp.Tw, inp.Th, inp.Tt, inp.Pt, inp.Pw
    Tl, Nt, Np, Ph, kt, kp = inp.Tl, inp.Nt, inp.Np, inp.Ph, inp.kt, inp.kp

    # Fluid properties (table values, optionally overridden)
    props = default_properties(Twi, Tai, inp.p_atm)
    props.update(overrides)
    Cpw, rho_w, kw, mu_w = props["Cpw"], props["rho_w"], props["kw"], props["mu_w"]
    Cpa, rho_a, ka, mu_a = props["Cpa"], props["rho_a"], props["ka"], props["mu_a"]
    Prw = mu_w * Cpw / kw
    Pra = mu_a * Cpa / ka

    # Tube perimeters and areas
    Opt = math.pi * Th + 2 * (Tw - Th)
    Wpt = math.pi * (Th - 2 * Tt) + 2 * ((Tw - 2 * Tt) - (Th - 2 * Tt))
    At = Nt * Opt * Tl
    Ati = Nt * Wpt * Tl
    Act = (Tw - 2 * Tt - Th) * (Th - 2 * Tt) + math.pi * ((Th - Tt) / 2) ** 2
    Acto = (Tw - Th) * Th + math.pi * ((Th - Tt) / 2) ** 2
    Ato = At - (Np * Nt * 2 * Acto)
    Ap = Np * (2 * ((Pw * Ph) + (Pt * Ph) + (Pw * Pt))) - (Np * Nt * 2 * Acto)
    Ar = Ato + Ap

    # Core spacings, air passage area and wetted perimeter
    St = (Ph / (Nt + 1)) - Th
    Sp = (Tl / (Np + 1)) - Pt
    edge_count = (((Nt + 1) * (Np + 1)) - ((Nt - 1) * (Np - 1))) / 2
    Apa = ((Nt - 1) * (Np - 1) * St * Sp
           + edge_count * (St + Th / 2) * (Sp + Pt / 2))
    Awpa = (2 * (Nt - 1) * (Np - 1) * St
            + 2 * (Nt - 1) * (Np - 1) * Sp
            + 2 * edge_count * (St + Th / 2)
            + 2 * edge_count * (Sp + Pt / 2))

    bad = {n: v for n, v in dict(Act=Act, Acto=Acto, Ato=Ato, Ap=Ap, Ar=Ar, St=St, Sp=Sp,
                                 Apa=Apa, Awpa=Awpa).items() if v <= 0}
    if bad:
        raise ValueError("The geometry gives a non-positive " + ", ".join(bad) +
                         ". Check the tube and plate dimensions and counts.")

    Dht = 4 * Act / Wpt
    Dha = 4 * Apa / Awpa
    Dexp = math.sqrt(Tl * mu_w / (0.05 * rho_w * Vw * Prw))
    thermally_developed = Dht < Dexp
    Rew = rho_w * Vw * Dht / mu_w

    Vmax_air = Va * (Tl * Ph / Apa)
    Rea = rho_a * Vmax_air * Dha / mu_a

    checks = [
        Check("Water flow is laminar", "Rew < 2300", f"Rew = {Rew:.6g}", Rew < 2300,
              "Water-side Nusselt number assumes laminar flow."),
        Check("Water Prandtl number in range", "0.5 < Prw < 100", f"Prw = {Prw:.4g}", 0.5 < Prw < 100,
              "Prandtl number is outside the range the laminar Nusselt number is quoted for."),
        Check("Water flow thermally developed", "Dht < Dexp", f"Dht = {Dht:.4g} m, Dexp = {Dexp:.4g} m",
              bool(thermally_developed),
              "Tube is too short for fully developed thermal flow; Nu = 4.36 under-predicts "
              "the entrance-region heat transfer."),
        Check("Air Reynolds number within correlation range", "Rea < 3.5×10⁵", f"Rea = {Rea:.6g}", Rea < 3.5e5,
              "Air Reynolds number is outside the stated range of the Blasius expression."),
    ]
    if strict:
        failed = [c for c in checks if not c.passed]
        if failed:
            raise ValueError("Validity check failed: " + "; ".join(f"{c.name} ({c.criterion})" for c in failed))

    # Nusselt numbers and convection coefficients
    Nuw = inp.Nuw
    Nua = 0.644 * math.sqrt(Rea) * Pra ** (1 / 3)
    hw = Nuw * kw / Dht
    ha = Nua * ka / Dha

    # Fin efficiency and overall surface efficiency
    m = math.sqrt(2 * ha / (kp * Pt))
    Lc = St / 2 + Pt / 2
    nf = math.tanh(m * Lc) / (m * Lc)
    n0 = 1 - ((Ap / Ar) * (1 - nf))

    # Conductance, heat-capacity rates, crossflow effectiveness
    UA = 1 / (
        1 / (n0 * ha * Ar)
        + Tt / (kt * ((At + Ati) / 2))
        + 1 / (hw * Ati)
    )
    Cw = rho_w * Vw * Act * Nt * Cpw
    Ca = rho_a * Va * Tl * Ph * Cpa
    Cmin = min(Cw, Ca)
    Cmax = max(Cw, Ca)
    Cr = Cmin / Cmax
    NTU = UA / Cmin
    eps_exponent = (NTU ** 0.22 / Cr) * math.expm1(-Cr * NTU ** 0.78)
    eps = -math.expm1(eps_exponent)
    Q = eps * Cmin * (Twi - Tai)
    Tco = Twi - Q / Cw
    Tao = Tai + Q / Ca

    values: dict[str, float | bool] = {
        "Cpw": Cpw, "rho_w": rho_w, "kw": kw, "mu_w": mu_w, "Prw": Prw,
        "Cpa": Cpa, "rho_a": rho_a, "ka": ka, "mu_a": mu_a, "Pra": Pra,
        "Opt": Opt, "Wpt": Wpt, "At": At, "Ati": Ati, "Act": Act, "Acto": Acto,
        "Ato": Ato, "Ap": Ap, "Ar": Ar, "St": St, "Sp": Sp, "Apa": Apa, "Awpa": Awpa,
        "Dht": Dht, "Dha": Dha, "Dexp": Dexp, "thermally_developed": bool(thermally_developed),
        "Rew": Rew, "Vmax_air": Vmax_air, "Rea": Rea, "Nuw": Nuw, "Nua": Nua,
        "hw": hw, "ha": ha, "m": m, "Lc": Lc, "nf": nf, "n0": n0,
        "UA": UA, "Cw": Cw, "Ca": Ca, "Cmin": Cmin, "Cmax": Cmax,
        "Cr": Cr, "NTU": NTU, "epsilon": eps, "Q": Q,
        "Tco": Tco, "Tco_C": Tco - KELVIN_OFFSET, "Tao": Tao, "Tao_C": Tao - KELVIN_OFFSET,
    }

    overridden = tuple(k for k in PROPERTY_KEYS if k in overrides)
    rows = []
    for key, (group, name, symbol, unit) in OUTPUT_META.items():
        if key in overridden:
            name += " (manual)"
        rows.append(Row(group, name, symbol, values[key], unit))

    warnings: list[str] = []
    notes: list[str] = []
    if not WATER_RANGE_K[0] <= Twi <= WATER_RANGE_K[1]:
        warnings.append("Water properties were extrapolated from the supplied 353.15–373.15 K table "
                        "(80–100 °C). Treat results with caution or enter properties manually.")
    if Twi >= 373.15:
        warnings.append("Water inlet is at or above 100 °C. The model treats water as single-phase liquid; "
                        "confirm the system pressure.")
    if not AIR_RANGE_K[0] <= Tai <= AIR_RANGE_K[1]:
        notes.append("Air Cp, conductivity and viscosity were extrapolated from the supplied "
                     "298.15–318.15 K (25–45 °C) anchor values.")
    if Pw < Tw:
        warnings.append("Plate width Pw is smaller than tube width Tw, so the tubes protrude beyond the plates. "
                        "The equations do not account for this.")
    if overridden:
        notes.append("Manually entered properties: " + ", ".join(overridden) + ".")
    notes.extend((
        f"Water Nusselt number is the fixed laminar reference value Nu = {Nuw:g}.",
        "Air Nusselt number uses the Blasius expression Nu = 0.644·Re^0.5·Pr^(1/3); "
        "change it if another correlation suits your model better.",
        "The tube cross-section and air-core geometry formulas produce the schematic shown in the Geometry tab.",
        "Water is treated as single-phase liquid.",
    ))

    return CalcResult(values=values, rows=rows, checks=checks, notes=notes, warnings=warnings,
                      overridden=overridden, inputs=inp)
