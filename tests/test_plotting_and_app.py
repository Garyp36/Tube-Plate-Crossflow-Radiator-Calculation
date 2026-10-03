from dataclasses import replace
from pathlib import Path

from streamlit.testing.v1 import AppTest

from plotting import geometry_figure, temperature_figure, tube_section_figure
from radiator_model import RadiatorInputs, calculate

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_figures_build_for_default_and_large_cores():
    for inp in (RadiatorInputs(), replace(RadiatorInputs(), Tl=0.6, Ph=0.5, Nt=8, Np=12, Pw=0.2, Tw=0.18)):
        assert len(geometry_figure(inp).axes) == 2
        assert tube_section_figure(inp).axes
        assert len(temperature_figure(calculate(inp)).axes) == 2


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) == 5
    assert at.metric[0].label.startswith("Heat transfer rate")


def test_app_shows_error_for_bad_geometry():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.number_input(key="w_Nt").set_value(80).run()
    assert not at.exception
    assert at.error and "do not fit" in at.error[0].value
    assert len(at.metric) == 0


def test_app_manual_property_override():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.checkbox(key="w_manual_Water").check().run()
    mu = [n for n in at.number_input if n.label.startswith("Dynamic viscosity μw")][0]
    mu.set_value(mu.value * 2).run()
    assert not at.exception
