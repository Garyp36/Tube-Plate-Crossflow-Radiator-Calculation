"""Regression tests. Reference numbers come from the original single-file script."""

import math
from dataclasses import replace

import pytest

from radiator_model import (
    OUTPUT_META,
    RadiatorInputs,
    calculate,
    default_properties,
    input_rows,
    piecewise_linear,
    validate_inputs,
)

REFERENCE = {
    "Cpa": 1006.0799999999999, "rho_a": 1.2736326337550734, "mu_a": 1.7382999999999996e-05,
    "Prw": 2.223783582089552, "Pra": 0.7071851451678122,
    "Act": 9.229576810088556e-05, "Ar": 0.1864898910493668, "Apa": 0.04005076388888888,
    "Dht": 0.005557912028273283, "Dha": 0.03925262713710116, "Rew": 380.3647119067589,
    "Rea": 3366.0352055296594, "Nua": 33.28814318411491, "hw": 525.5930617720753,
    "ha": 20.9722467254954, "nf": 0.9900666470974894, "n0": 0.9943061112440544,
    "UA": 3.570493218833887, "Cw": 47.055204520941146, "Ca": 60.064515007889256,
    "NTU": 0.07587881627939579, "epsilon": 0.06950358262695576, "Q": 248.55840245416465,
    "Tco": 347.86772772035135, "Tao": 281.2881904510761,
}


def test_defaults_match_original_script():
    res = calculate(RadiatorInputs())
    for key, expected in REFERENCE.items():
        assert res.values[key] == pytest.approx(expected, rel=1e-9), key
    assert res.values["thermally_developed"] is True
    assert res.all_checks_passed


def test_every_output_has_a_row_with_unit_and_symbol():
    res = calculate(RadiatorInputs())
    assert [r.symbol for r in res.rows] and len(res.rows) == len(OUTPUT_META)
    assert all(r.unit and r.symbol and r.name for r in res.rows)


def test_celsius_conversion_rows():
    res = calculate(RadiatorInputs())
    assert res.values["Tco_C"] == pytest.approx(res.values["Tco"] - 273.15)


def test_piecewise_interpolation_and_extrapolation():
    pts = ((0.0, 0.0), (10.0, 10.0), (20.0, 30.0))
    assert piecewise_linear(5, pts) == 5
    assert piecewise_linear(15, pts) == 20
    assert piecewise_linear(-5, pts) == -5
    assert piecewise_linear(30, pts) == 50


def test_property_override_changes_result_and_is_labelled():
    base = calculate(RadiatorInputs())
    props = default_properties(353.15, 277.15)
    res = calculate(RadiatorInputs(), {"mu_w": props["mu_w"] * 1.5})
    assert res.values["mu_w"] == pytest.approx(props["mu_w"] * 1.5)
    assert res.values["Rew"] == pytest.approx(base.values["Rew"] / 1.5)
    assert res.overridden == ("mu_w",)
    assert any("(manual)" in r.name for r in res.rows if r.symbol == "μw")


def test_unknown_or_bad_override_rejected():
    with pytest.raises(ValueError):
        calculate(RadiatorInputs(), {"bogus": 1.0})
    with pytest.raises(ValueError):
        calculate(RadiatorInputs(), {"kw": -1.0})


def test_validation_messages():
    assert validate_inputs(RadiatorInputs()) == []
    assert validate_inputs(replace(RadiatorInputs(), Tt=0.002))            # Th <= 2 Tt
    assert validate_inputs(replace(RadiatorInputs(), Tw=0.003))            # Tw <= Th
    assert validate_inputs(replace(RadiatorInputs(), Nt=80))               # tubes do not fit
    assert validate_inputs(replace(RadiatorInputs(), Np=200))              # plates do not fit
    assert validate_inputs(replace(RadiatorInputs(), Vw=0.0))
    assert validate_inputs(replace(RadiatorInputs(), Nt=0))
    assert validate_inputs(replace(RadiatorInputs(), Va=float("nan")))
    with pytest.raises(ValueError):
        calculate(replace(RadiatorInputs(), Nt=80))


def test_turbulent_water_flags_check_and_strict_raises():
    inp = replace(RadiatorInputs(), Vw=0.5)
    res = calculate(inp)
    assert not res.all_checks_passed
    assert [c.name for c in res.checks if not c.passed][0] == "Water flow is laminar"
    with pytest.raises(ValueError):
        calculate(inp, strict=True)


def test_extrapolation_notes():
    res = calculate(RadiatorInputs())  # Tai = 4 C is below the air table
    assert any("extrapolated" in n for n in res.notes)
    hot = calculate(replace(RadiatorInputs(), Twi=383.15))
    assert any("100 °C" in w for w in hot.warnings)


def test_single_tube_and_plate_do_not_crash():
    res = calculate(replace(RadiatorInputs(), Nt=1, Np=1))
    assert math.isfinite(res.values["Q"]) and res.values["Q"] > 0


def test_input_rows_in_display_units():
    rows = {r.symbol + r.unit: r.value for r in input_rows(RadiatorInputs())}
    assert rows["Twmm"] == pytest.approx(32.0)
    assert rows["Twi°C"] == pytest.approx(80.0)
    assert rows["Nt–"] == 5
