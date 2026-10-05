#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent
BASE_SCHEMA = REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json"
OUT_DIR = REPO / "citylearn_schemas" / "tx_travis_8b_ev"

EV_OBSERVATIONS: List[str] = [
    "electric_vehicle_charger_connected_state",
    "connected_electric_vehicle_at_charger_battery_capacity",
    "connected_electric_vehicle_at_charger_departure_time",
    "connected_electric_vehicle_at_charger_required_soc_departure",
    "connected_electric_vehicle_at_charger_soc",
    "electric_vehicle_charger_incoming_state",
    "incoming_electric_vehicle_at_charger_estimated_arrival_time",
]

EV_ACTION = "electric_vehicle_storage"

CSV_COLUMNS = [
    "electric_vehicle_charger_state",
    "electric_vehicle_id",
    "electric_vehicle_departure_time",
    "electric_vehicle_required_soc_departure",
    "electric_vehicle_estimated_arrival_time",
    "electric_vehicle_estimated_soc_arrival",
]

CHARGER_POWERS_KW = [11.0, 7.4, 11.0, 7.4, 11.0, 7.4]
BATTERY_CAPACITIES_KWH = [60.0, 40.0, 75.0, 52.0, 60.0, 40.0]

STATE_PARKED, STATE_INCOMING, STATE_AWAY = 1, 2, 3


def _daily_trip(rng: np.random.Generator, weekend: bool) -> Optional[Tuple[int, int]]:
    stay_home_p = 0.45 if weekend else 0.12
    if rng.random() < stay_home_p:
        return None
    if weekend:
        depart = rng.normal(9.5, 1.5)
        back = rng.normal(15.5, 2.0)
    else:
        depart = rng.normal(7.5, 0.8)
        back = rng.normal(17.5, 1.2)
    depart = int(np.clip(round(depart), 5, 12))
    back = int(np.clip(round(back), depart + 2, 23))
    return depart, back


def generate_charger_schedule(num_steps: int, rng: np.random.Generator,
                              start_weekday: int = 0) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = [
        {c: "" for c in CSV_COLUMNS} for _ in range(num_steps)
    ]
    state = np.full(num_steps, STATE_PARKED, dtype=int)
    depart_at = np.full(num_steps, -1, dtype=int)
    required_soc = np.full(num_steps, np.nan)
    arrival_soc = np.full(num_steps, np.nan)

    num_days = num_steps // 24
    for day in range(num_days):
        base = day * 24
        weekend = ((start_weekday + day) % 7) >= 5
        trip = _daily_trip(rng, weekend)
        if trip is None:
            continue
        depart_h, back_h = trip
        d_step, b_step = base + depart_h, base + back_h
        if b_step >= num_steps:
            b_step = num_steps - 1
        state[d_step + 1:b_step] = STATE_AWAY
        state[b_step] = STATE_INCOMING
        depart_at[base:d_step + 1] = d_step
        required_soc[base:d_step + 1] = rng.uniform(0.70, 0.90)
        hours_away = max(1, b_step - d_step)
        arrival_soc[b_step] = float(np.clip(
            rng.uniform(0.25, 0.55) - 0.01 * (hours_away - 8), 0.10, 0.60))

    next_departure = -1
    next_required = np.nan
    for t in range(num_steps - 1, -1, -1):
        if state[t] == STATE_AWAY:
            next_departure, next_required = -1, np.nan
        elif depart_at[t] >= 0:
            next_departure, next_required = depart_at[t], required_soc[t]
        elif state[t] == STATE_PARKED and next_departure >= 0:
            depart_at[t], required_soc[t] = next_departure, next_required

    ev_id = "Electric_Vehicle_{ev}"
    for t in range(num_steps):
        row = rows[t]
        row["electric_vehicle_charger_state"] = int(state[t])
        if state[t] == STATE_AWAY:
            continue
        row["electric_vehicle_id"] = ev_id
        if state[t] == STATE_PARKED:
            countdown = depart_at[t] - t if depart_at[t] >= 0 else 0
            row["electric_vehicle_departure_time"] = float(max(countdown, 0))
            soc = required_soc[t]
            row["electric_vehicle_required_soc_departure"] = round(
                float(soc if np.isfinite(soc) else 0.8) * 100.0, 1)
        else:
            row["electric_vehicle_estimated_arrival_time"] = 0.0
            soc = arrival_soc[t]
            row["electric_vehicle_estimated_soc_arrival"] = round(
                float(soc if np.isfinite(soc) else 0.4) * 100.0, 1)
    return rows


def write_charger_csv(path: Path, rows: List[Dict[str, object]], ev_name: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out["electric_vehicle_id"]:
                out["electric_vehicle_id"] = ev_name
            writer.writerow(out)


def build_schema(num_ev_buildings: int, seed: int, out_dir: Path) -> Dict:
    if not BASE_SCHEMA.is_file():
        raise RuntimeError(
            f"Base schema {BASE_SCHEMA} not found. Run "
            "`python -B setup_citylearn_8b.py --validate` first.")
    with open(BASE_SCHEMA) as f:
        schema = json.load(f)

    included = [name for name, b in schema["buildings"].items() if b.get("include")]
    if num_ev_buildings > len(included):
        raise ValueError(
            f"Requested {num_ev_buildings} EV buildings but only "
            f"{len(included)} are included in the base schema.")
    out_dir.mkdir(parents=True, exist_ok=True)

    start = int(schema.get("simulation_start_time_step", 0))
    end = int(schema.get("simulation_end_time_step", 8759))
    num_steps = end - start + 1

    for name in EV_OBSERVATIONS:
        schema["observations"][name] = {"active": True}
    schema["actions"][EV_ACTION] = {"active": True}

    ev_defs: Dict[str, Dict] = {}
    rng = np.random.default_rng(seed)

    for i, building in enumerate(included):
        if i >= num_ev_buildings:
            schema["buildings"][building].pop("chargers", None)
            continue

        ev_name = f"Electric_Vehicle_{i + 1}"
        charger_id = f"charger_{i + 1}_1"
        capacity = BATTERY_CAPACITIES_KWH[i % len(BATTERY_CAPACITIES_KWH)]
        power = CHARGER_POWERS_KW[i % len(CHARGER_POWERS_KW)]

        rows = generate_charger_schedule(num_steps, rng)
        csv_path = out_dir / f"{charger_id}.csv"
        write_charger_csv(csv_path, rows, ev_name)

        ev_defs[ev_name] = {
            "include": True,
            "battery": {
                "type": "citylearn.energy_model.Battery",
                "autosize": False,
                "attributes": {
                    "capacity": capacity,
                    "nominal_power": power,
                    "initial_soc": 0.4,
                    "depth_of_discharge": 0.85,
                },
            },
        }
        schema["buildings"][building]["chargers"] = {
            charger_id: {
                "type": "citylearn.electric_vehicle_charger.Charger",
                "charger_simulation": str(csv_path.resolve()),
                "autosize": False,
                "attributes": {
                    "nominal_power": power,
                    "efficiency": 0.95,
                    "charger_type": 0,
                    "max_charging_power": power,
                    "min_charging_power": 1.4,
                    "max_discharging_power": min(7.2, power),
                    "min_discharging_power": 0.0,
                },
            }
        }

    schema["electric_vehicles_def"] = ev_defs
    return schema


def summarise(schema: Dict) -> None:
    ev_buildings = [n for n, b in schema["buildings"].items()
                    if b.get("include") and b.get("chargers")]
    total = [n for n, b in schema["buildings"].items() if b.get("include")]
    print(f"[ev-schema] buildings: {len(total)}  with chargers: {len(ev_buildings)}")
    for name in ev_buildings:
        cid, cfg = next(iter(schema["buildings"][name]["chargers"].items()))
        ev = f"Electric_Vehicle_{cid.split('_')[1]}"
        cap = schema["electric_vehicles_def"][ev]["battery"]["attributes"]["capacity"]
        kw = cfg["attributes"]["max_charging_power"]
        rate = kw * cfg["attributes"]["efficiency"] / cap
        print(f"    {cid:14s} {kw:5.1f} kW  battery {cap:5.1f} kWh  "
              f"rho={rate:.3f}/h  fill={1 / rate:4.1f} h")


def validate(schema_path: Path) -> None:
    import numpy as np
    from stems.environment import STEMSEnvironment

    print("\n[ev-schema] validating on the real simulator ...")
    env = STEMSEnvironment(schema=str(schema_path), seed=0, heat_pump=True)
    obs, _ = env.reset()

    assert env.env_type == "CityLearn", env.env_type
    assert env.absent_observations == [], (
        f"thermal features went missing: {env.absent_observations}")
    assert env.ev_slots >= 1, "no EV slots discovered"
    print(f"    B={env.num_buildings} obs_dim={env.obs_dim} "
          f"action_dim={env.action_dim} ev_slots={env.ev_slots}")
    print(f"    actions: {env.action_names}")

    mask = env.action_presence_mask()
    ev_slot = env.ev_action_indices()[0]
    owners = int(mask[:, ev_slot].sum())
    assert 0 < owners <= env.num_buildings
    print(f"    buildings with a charger: {owners}/{env.num_buildings}")

    for name in ("dhw_storage", "cooling_or_heating_device", "electrical_storage"):
        assert name in env.action_names, f"{name} lost from the merged schema"
    assert env.resolve_control_indices("thermal") is not None
    assert env.resolve_control_indices("ev") == [ev_slot]
    assert len(env.resolve_control_indices("all")) == env.action_dim
    print(f"    isolation modes ok: thermal={env.resolve_control_indices('thermal')} "
          f"ev={env.resolve_control_indices('ev')} "
          f"all={env.resolve_control_indices('all')}")

    layout = env.ev_obs_layout()[0]
    dhw_idx = env.dhw_action_index

    connected_seen = away_seen = 0
    actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    actions[:, ev_slot] = 1.0
    actions[:, dhw_idx] = 1.0
    ev_soc_start = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)
    for _ in range(48):
        obs, _, term, trunc, _ = env.step(actions)
        conn = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
        connected_seen += int(conn.sum())
        away_seen += int((~conn).sum())
        if term or trunc:
            break
    ev_soc_end = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)
    dhw_soc = np.array([o[18] for o in obs], dtype=np.float32)

    assert connected_seen > 0, "no vehicle ever connected"
    assert away_seen > 0, "no vehicle ever away -- schedule is degenerate"
    assert np.any(ev_soc_end > ev_soc_start), "EV charging never moved SOC"
    assert np.any(dhw_soc > 0.5), "DHW tank never charged -- thermal side broken"
    print(f"    48-step rollout: connected {connected_seen}, away {away_seen} "
          f"(building-steps)")
    print(f"    EV soc moved: {ev_soc_start.max():.2f} -> {ev_soc_end.max():.2f}   "
          f"DHW soc max: {dhw_soc.max():.2f}")
    print("[ev-schema] VALID: thermal and EV both live in one environment.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the merged Travis + EV CityLearn schema")
    ap.add_argument("--ev-buildings", type=int, default=6,
                    help="how many of the 8 buildings own a charger (default 6)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--validate", action="store_true",
                    help="load the result in STEMSEnvironment and check it")
    args = ap.parse_args()

    out_dir = Path(args.out)
    schema = build_schema(args.ev_buildings, args.seed, out_dir)
    schema_path = out_dir / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"[ev-schema] wrote {schema_path}")
    summarise(schema)
    print("[ev-schema] EV schedules are SYNTHETIC (ResStock carries no vehicle "
          "data); the thermal side is the unmodified Travis data.")

    if args.validate:
        validate(schema_path)


if __name__ == "__main__":
    main()
