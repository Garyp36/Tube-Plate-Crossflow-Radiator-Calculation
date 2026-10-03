# Tube & Plate Crossflow Radiator Calculator

A Streamlit app for single-case thermal calculations on a tube-and-plate crossflow radiator
with laminar water flow inside flat tubes and air across plate fins. It is a port of a
spreadsheet-style Python script: same equations, wrapped in an interactive interface.

## What it does

- **Inputs** in the sidebar: operating conditions (water and air velocity and inlet temperature),
  tube and plate dimensions (mm), core geometry and materials, and two model constants.
- **Fluid properties**: calculated from the built-in water table and air equations by default.
  Tick *Enter values manually* under *Water properties* or *Air properties* to override them.
- **Results** in tables with Parameter, Symbol, Value and Unit, grouped by topic, plus a table of
  the inputs used. Download everything as CSV.
- **Geometry plot**: front and right-side views drawn to scale, with a tube cross-section detail.
  Download as PNG.
- **Temperatures** chart (inlet vs outlet for both fluids) and **validity checks** (laminar flow,
  Prandtl range, thermal development, air Reynolds range).

## Run locally

```bash
git clone <your-repo-url>
cd radiator-streamlit-app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. At <https://share.streamlit.io>, choose *New app*, select the repository and set the main
   file to `app.py`.

## Project layout

```
app.py              Streamlit interface
radiator_model.py   Calculation engine (standard library only)
plotting.py         Matplotlib figures
tests/              pytest suite (regression values from the original script, validation, app smoke tests)
requirements.txt    Runtime dependencies
.streamlit/         Theme and server settings
```

Run the tests with `pip install -r requirements-dev.txt` then `pytest`.

## Model notes

- Water properties are linearly interpolated between 80, 90 and 100 °C table values; air Cp,
  conductivity and viscosity between 25, 35 and 45 °C. Outside those ranges the values are
  extrapolated and the app says so. Air density is `p / (R·T)`.
- Water Nusselt number is the fixed laminar value 4.36 (editable under *Model constants*).
  Air Nusselt number is `0.644·Re^0.5·Pr^(1/3)`.
- Crossflow effectiveness uses the unmixed-unmixed correlation
  `ε = 1 − exp[(NTU^0.22 / Cr)·(exp(−Cr·NTU^0.78) − 1)]`.
- Where the original script stopped with an error when a validity check failed, the app still
  shows the results and flags the failed check prominently. `calculate(..., strict=True)` in
  `radiator_model.py` restores the original behaviour.
- The tube area formulas are kept exactly as in the original sheet. `Act` and `Acto` use a
  semicircle radius of `(Th − Tt)/2` rather than `(Th − 2·Tt)/2` and `Th/2`; check this against your
  derivation if the areas matter for your report.
- Water is treated as single-phase liquid.

## Licence

Add a licence file of your choice before publishing.
