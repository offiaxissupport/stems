#!/usr/bin/env python3
"""Schemas for CityLearn's mixed commercial / multi-family neighbourhood (2020 challenge).

Nine buildings per climate zone: by CityLearn's documentation a medium office, a
fast-food restaurant, a standalone retail store, a strip-mall retail store and
five medium multi-family buildings. Each has a battery and (seven of nine) a
hot-water tank; four have PV. There are no thermal dynamics: heating and cooling
loads are met as given, so the indoor temperature is not the controller's to
move and comfort cannot be studied here. What can be studied is whether the
safety layer and the storage control carry over to other kinds of buildings and
to other climates (zone 1: mean 20.9 degC; zone 4: mean 10.3 degC, minimum -18.8).

The dataset is used as cached by CityLearn; nothing is copied. Two changes:

* the schema ships no tariff, so the one used for the Travis houses (the 2022
  challenge file) is attached to every building -- the same prices in every
  experiment of this project;
* the price and demand observations the controllers read are switched on.

The chilled-water storage of these buildings is left idle: the action layout of
``STEMSEnvironment`` has no slot for it.

    .venv/Scripts/python setup_citylearn_mixed.py
"""

from __future__ import annotations

import json
from pathlib import Path

from citylearn.data import DataSet

REPO = Path(__file__).resolve().parent
ZONES = (1, 2, 3, 4)
PRICED_FROM = "citylearn_challenge_2022_phase_all"
OBSERVATIONS_ON = ("electricity_pricing", "electricity_pricing_predicted_1",
                   "electricity_pricing_predicted_2", "dhw_demand", "cooling_demand",
                   "net_electricity_consumption", "electrical_storage_soc", "dhw_storage_soc",
                   "non_shiftable_load", "solar_generation", "carbon_intensity",
                   "indoor_dry_bulb_temperature", "hour", "day_type")


def dataset_dir(name: str) -> Path:
    """Folder of a dataset in CityLearn's cache (downloaded if absent)."""
    schema_path = Path(DataSet().get_schema(name)["root_directory"])
    return schema_path


def build(zone: int) -> Path:
    name = f"citylearn_challenge_2020_climate_zone_{zone}"
    root = dataset_dir(name)
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    pricing = dataset_dir(PRICED_FROM) / "pricing.csv"
    if not pricing.exists():
        raise FileNotFoundError(pricing)
    schema["root_directory"] = str(root)
    for building in schema["buildings"].values():
        building["pricing"] = str(pricing)
    for n in OBSERVATIONS_ON:
        # The 2020 schema does not list the two demand observations; the
        # simulator computes them all the same once they are declared.
        entry = schema["observations"].setdefault(n, {"shared_in_central_agent": False})
        entry["active"] = True
    out = REPO / "citylearn_schemas" / f"cl2020_zone{zone}" / "schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    for z in ZONES:
        print("wrote", build(z).relative_to(REPO))
