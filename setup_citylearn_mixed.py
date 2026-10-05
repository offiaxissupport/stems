#!/usr/bin/env python3

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
        entry = schema["observations"].setdefault(n, {"shared_in_central_agent": False})
        entry["active"] = True
    out = REPO / "citylearn_schemas" / f"cl2020_zone{zone}" / "schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    for z in ZONES:
        print("wrote", build(z).relative_to(REPO))
