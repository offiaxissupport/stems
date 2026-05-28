#!/usr/bin/env python3
"""Create a real 8-building CityLearn Travis County schema for STEMS.

CityLearn v2.6 ships ``tx_travis_county_neighborhood`` with 100 real
ResStock buildings.  The paper evaluates a representative 8-building subset,
whereas the 2023 challenge local/online schemas expose only 3 buildings.  This
script creates a local schema that selects 8 real Travis buildings, activates
the observations used by STEMS, and attaches CityLearn pricing/carbon files so
cost and emission metrics are non-zero.

Usage:
    python -B setup_citylearn_8b.py --validate

By default the selected buildings all expose the same 3-action profile:
``dhw_storage``, ``electrical_storage``, and ``cooling_or_heating_device``.

Then run:
    python train.py --paper-reproduction
    python evaluate.py --paper-reproduction
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


STEMS_OBSERVATIONS: List[str] = [
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "cooling_demand",
    "dhw_demand",
    "occupant_count",
    "indoor_dry_bulb_temperature_cooling_set_point",
]

STEPPABLE_TRAVIS_3_ACTION: List[str] = [
    "resstock-amy2018-2021-release-1-134795",
    "resstock-amy2018-2021-release-1-15942",
    "resstock-amy2018-2021-release-1-344233",
    "resstock-amy2018-2021-release-1-400973",
    "resstock-amy2018-2021-release-1-412815",
    "resstock-amy2018-2021-release-1-427553",
    "resstock-amy2018-2021-release-1-494842",
    "resstock-amy2018-2021-release-1-519975",
]

STEPPABLE_TRAVIS_2_ACTION: List[str] = [
    "resstock-amy2018-2021-release-1-113808",
    "resstock-amy2018-2021-release-1-155572",
    "resstock-amy2018-2021-release-1-171136",
    "resstock-amy2018-2021-release-1-208044",
    "resstock-amy2018-2021-release-1-216680",
    "resstock-amy2018-2021-release-1-224341",
    "resstock-amy2018-2021-release-1-252982",
    "resstock-amy2018-2021-release-1-294190",
]


def _dataset_roots() -> List[Path]:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    return [
        local / "citylearn" / "citylearn" / "Cache" / "v2.6.0b1" / "datasets",
        local / "intelligent-environments-lab" / "citylearn" / "Cache" / "v2.6.0b1" / "datasets",
        Path(r"C:\temp\citylearn_src\data\datasets"),
    ]


def _find_schema(dataset_name: str) -> Path:
    for root in _dataset_roots():
        schema = root / dataset_name / "schema.json"
        if schema.is_file():
            return schema
    roots = "\n  ".join(str(p) for p in _dataset_roots())
    raise FileNotFoundError(
        f"Could not find {dataset_name}/schema.json in CityLearn dataset roots:\n  {roots}\n"
        "Run setup_citylearn.sh or instantiate CityLearnEnv('tx_travis_county_neighborhood') "
        "once to populate the cache."
    )


def _find_file(dataset_name: str, file_name: str) -> Path:
    schema = _find_schema(dataset_name)
    path = schema.parent / file_name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {file_name} in {schema.parent}")
    return path


def _csv_data_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _expected_signal_rows(schema: Dict) -> int:
    start = int(schema.get("simulation_start_time_step") or 0)
    end = int(schema.get("simulation_end_time_step") or 0)
    return end - start + 1


def _validate_signal_length(path: Path, expected_rows: int, label: str) -> None:
    rows = _csv_data_rows(path)
    if rows < expected_rows:
        raise ValueError(
            f"{label} file has {rows} rows but the generated schema needs at least {expected_rows}: {path}\n"
            "Use an 8760-step dataset such as --price-carbon-dataset citylearn_challenge_2022_phase_all."
        )


def build_schema(
    source_schema: Path,
    building_count: int,
    pricing_file: Optional[Path],
    carbon_file: Optional[Path],
    action_profile: str,
) -> Dict:
    data = json.loads(source_schema.read_text())
    buildings = data.get("buildings", {})
    if not isinstance(buildings, dict):
        raise ValueError("Expected CityLearn schema['buildings'] to be a dictionary")
    if len(buildings) < building_count:
        raise ValueError(f"Source schema has only {len(buildings)} buildings")

    selected = set(_select_buildings(buildings, building_count, action_profile))
    data["root_directory"] = str(source_schema.parent)
    data["central_agent"] = False

    for obs_name in STEMS_OBSERVATIONS:
        if obs_name not in data["observations"]:
            raise KeyError(f"Required observation is absent from Travis schema: {obs_name}")
        data["observations"][obs_name]["active"] = True

    for name, cfg in buildings.items():
        include = name in selected
        cfg["include"] = include
        if include:
            if pricing_file is not None:
                cfg["pricing"] = str(pricing_file)
            if carbon_file is not None:
                cfg["carbon_intensity"] = str(carbon_file)

    return data


def _select_buildings(buildings: Dict, building_count: int, action_profile: str) -> List[str]:
    profiles = {
        "3-action": tuple(),
        "2-action": ("dhw_storage",),
    }
    if action_profile == "auto":
        profile_order = ["3-action", "2-action"]
    else:
        profile_order = [action_profile]

    for profile in profile_order:
        inactive_target = profiles[profile]
        candidates = [
            name
            for name, cfg in buildings.items()
            if tuple(sorted(cfg.get("inactive_actions", []))) == inactive_target
        ]
        preferred = STEPPABLE_TRAVIS_3_ACTION if profile == "3-action" else STEPPABLE_TRAVIS_2_ACTION
        candidates = [name for name in preferred if name in candidates] + [
            name for name in candidates if name not in preferred
        ]
        if len(candidates) >= building_count:
            return candidates[:building_count]

    available = {
        name: sum(
            1
            for cfg in buildings.values()
            if tuple(sorted(cfg.get("inactive_actions", []))) == inactive
        )
        for name, inactive in profiles.items()
    }
    raise ValueError(
        f"Could not select {building_count} Travis buildings for action_profile={action_profile}; "
        f"available={available}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create STEMS 8-building Travis CityLearn schema")
    parser.add_argument("--source-dataset", default="tx_travis_county_neighborhood")
    parser.add_argument("--price-carbon-dataset", default="citylearn_challenge_2022_phase_all")
    parser.add_argument("--buildings", type=int, default=8)
    parser.add_argument(
        "--action-profile",
        choices=["auto", "2-action", "3-action"],
        default="3-action",
        help="Select a homogeneous Travis action profile; 3-action matches the original STEMS action layout.",
    )
    parser.add_argument("--output", default="citylearn_schemas/tx_travis_8b/schema.json")
    parser.add_argument("--validate", action="store_true", help="Instantiate CityLearnEnv with the generated schema")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_schema = _find_schema(args.source_dataset)
    pricing_file = _find_file(args.price_carbon_dataset, "pricing.csv")
    carbon_file = _find_file(args.price_carbon_dataset, "carbon_intensity.csv")
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_schema(source_schema, args.buildings, pricing_file, carbon_file, args.action_profile)
    expected_rows = _expected_signal_rows(data)
    _validate_signal_length(pricing_file, expected_rows, "Pricing")
    _validate_signal_length(carbon_file, expected_rows, "Carbon intensity")
    output_path.write_text(json.dumps(data, indent=2))
    selected = [name for name, cfg in data["buildings"].items() if cfg.get("include")]

    print(f"[setup] Wrote {output_path}")
    print(f"[setup] Source: {source_schema}")
    print(f"[setup] Pricing: {pricing_file}")
    print(f"[setup] Carbon: {carbon_file}")
    print(f"[setup] Included buildings ({len(selected)}):")
    for name in selected:
        print(f"  - {name}")

    if args.validate:
        from citylearn.citylearn import CityLearnEnv

        env = CityLearnEnv(schema=str(output_path), central_agent=False)
        obs, _ = env.reset()
        zero_actions = [
            [0.0] * space.shape[0]
            for space in env.action_space
        ]
        env.step(zero_actions)
        print(
            "[setup] Validation OK: "
            f"buildings={len(env.observation_space)}, "
            f"obs_dim={env.observation_space[0].shape[0]}, "
            f"action_dim={env.action_space[0].shape[0]}, "
            f"first_obs_dim={len(obs[0])}"
        )
        print(f"[setup] Action names: {env.action_names[0]}")
        print("[setup] Use with: --schema " + str(output_path))


if __name__ == "__main__":
    main()