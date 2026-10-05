#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig, ThermalConfig
from stems.deadline import coupled_feasibility
from stems.environment import STEMSEnvironment
from stems.ev import EVChargerSpec, EVObsLayout, EVReadinessBarrier
from stems.thermal import build_thermal_stack

SCHEMA = "citylearn_schemas/tx_travis_8b_ev/schema.json"
_IDX_NET, _IDX_PRICE = 20, 21


def build_ev_barrier(env: STEMSEnvironment) -> Optional[EVReadinessBarrier]:
    layouts = env.ev_obs_layout()
    if not layouts:
        return None
    layout = layouts[0]
    info = env.ev_info()
    power = info["max_charging_power"][:, 0].astype(np.float32)
    eff = info["efficiency"][:, 0].astype(np.float32)
    spec = EVChargerSpec(
        max_charging_power_kw=power,
        efficiency=np.where(power > 0, eff, 1.0),
        action_bound=np.ones(env.num_buildings, dtype=np.float32),
        action_index=env.ev_action_indices()[0],
    )
    return EVReadinessBarrier(
        EVObsLayout(connected_state=layout["connected_state"],
                    departure_time=layout["departure_time"],
                    required_soc_departure=layout["required_soc_departure"],
                    soc=layout["soc"],
                    battery_capacity=layout["battery_capacity"]),
        spec, name="ev")


class FleetPolicy:
    def __init__(self, num_buildings: int, ev_slot: int, dhw_idx: int,
                 layout: Dict[str, int], seed: int = 0) -> None:
        self.B = num_buildings
        self.ev_slot = ev_slot
        self.dhw_idx = dhw_idx
        self.layout = layout
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: List[np.ndarray], action_dim: int) -> np.ndarray:
        a = np.zeros((self.B, action_dim), dtype=np.float32)
        conn = np.array([o[self.layout["connected_state"]] for o in obs]) > 0.5
        soc = np.array([o[self.layout["soc"]] for o in obs], dtype=np.float32)
        req = np.array([o[self.layout["required_soc_departure"]] for o in obs],
                       dtype=np.float32)
        a[:, self.ev_slot] = np.where(conn & (soc < req), 1.0, 0.0)
        return a


def rollout(env: STEMSEnvironment, cap_kw: float, coordination: str,
            steps: int, seed: int) -> Dict[str, float]:
    bi = env.battery_info()
    ev_barrier = build_ev_barrier(env)
    if ev_barrier is None:
        raise RuntimeError("schema exposes no EV chargers")
    tcfg = ThermalConfig(dhw_readiness=False, weather_anticipation=False,
                         cop_aware_power=False)
    _, cop = build_thermal_stack(env, tcfg, enable=False)

    cfg = CBFConfig(P_grid_max=cap_kw, P_building_max=cap_kw)
    shield = CBFShield(cfg, env.num_buildings, soc_rate=bi["soc_rate"],
                       nominal_power=bi["nominal_power"],
                       elec_idx=env.electrical_storage_action_index,
                       safety_cfg=SafetyConfig(),
                       enforce_soc=False,
                       deadline_barriers=[ev_barrier],
                       coordination=coordination,
                       hvac_idx=-1)

    layout = env.ev_obs_layout()[0]
    ev_slot = env.ev_action_indices()[0]
    policy = FleetPolicy(env.num_buildings, ev_slot, env.dhw_action_index,
                         layout, seed)
    owners = np.array([env.building_has_action(b, ev_slot)
                       for b in range(env.num_buildings)], dtype=bool)

    obs, _ = env.reset()
    mask = np.zeros(env.action_dim, dtype=np.float32)
    mask[ev_slot] = 1.0

    infeasible_steps = 0
    total_shortfall = 0.0
    departures = 0
    missed = 0
    missed_soc_gap = 0.0
    at_risk_steps = 0
    connected_steps = 0
    grid_violation_steps = 0
    baseline_violation_steps = 0
    peak = 0.0
    cost = 0.0
    ev_kwh = 0.0
    n = 0

    prev_conn = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
    prev_soc = np.array([o[layout["soc"]] for o in obs], dtype=np.float32)
    prev_req = np.array([o[layout["required_soc_departure"]] for o in obs],
                        dtype=np.float32)

    for _ in range(steps):
        report = coupled_feasibility([ev_barrier], obs, power_cap_kw=cap_kw)
        if not report["feasible"]:
            infeasible_steps += 1
            total_shortfall += float(report["shortfall_kwh"])
        rep = ev_barrier.deadline_report(obs)
        at_risk_steps += int((rep["at_risk"] & owners).sum())
        connected_steps += int((rep["active"] & owners).sum())

        nominal = policy(obs, env.action_dim)
        actions = shield.project(nominal, obs) * mask
        ev_kwh += float((np.maximum(actions[:, ev_slot], 0.0)
                         * ev_barrier._p_charge).sum())

        next_obs, _, term, trunc, _ = env.step(actions)

        net = np.array([o[_IDX_NET] for o in next_obs], dtype=np.float32)
        total_import = float(np.maximum(net, 0.0).sum())
        peak = max(peak, total_import)
        cost += float((np.maximum(net, 0.0)
                       * np.array([o[_IDX_PRICE] for o in obs])).sum())
        if total_import > cap_kw:
            grid_violation_steps += 1

        conn = np.array([o[layout["connected_state"]] for o in next_obs]) > 0.5
        soc = np.array([o[layout["soc"]] for o in next_obs], dtype=np.float32)
        left = prev_conn & ~conn & owners
        for i in np.where(left)[0]:
            departures += 1
            if prev_soc[i] + 1e-3 < prev_req[i]:
                missed += 1
                missed_soc_gap += float(prev_req[i] - prev_soc[i])

        prev_conn, prev_soc = conn, soc
        prev_req = np.array([o[layout["required_soc_departure"]] for o in next_obs],
                            dtype=np.float32)
        obs = next_obs
        n += 1
        if term or trunc:
            break

    return {
        "cap_kw": cap_kw,
        "coordination": coordination,
        "steps": n,
        "infeasible_rate": infeasible_steps / max(n, 1),
        "mean_shortfall_kwh": total_shortfall / max(infeasible_steps, 1),
        "departures": departures,
        "missed_departures": missed,
        "missed_departure_rate": missed / max(departures, 1),
        "mean_missed_soc_gap": missed_soc_gap / max(missed, 1),
        "at_risk_rate": at_risk_steps / max(connected_steps, 1),
        "grid_violation_rate": grid_violation_steps / max(n, 1),
        "peak_kw": peak,
        "cost": cost,
        "ev_electricity_kwh": ev_kwh,
    }


def _fmt(r: Dict[str, float]) -> str:
    return (f"  cap={r['cap_kw']:5.1f}kW {r['coordination']:>11s} | "
            f"infeas={r['infeasible_rate']:.3f} "
            f"missed={r['missed_departures']:3d}/{r['departures']:3d} "
            f"({r['missed_departure_rate']:.3f}) "
            f"at_risk={r['at_risk_rate']:.3f} "
            f"gridviol={r['grid_violation_rate']:.3f} "
            f"peak={r['peak_kw']:5.1f} cost={r['cost']:7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Grid-cap coupling sweep with a deadline-locked EV fleet")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--caps", type=float, nargs="+",
                    default=[80.0, 60.0, 45.0, 35.0, 25.0, 20.0, 15.0])
    ap.add_argument("--schema", type=str, default=SCHEMA)
    ap.add_argument("--out", type=str, default="coupling_study_results.json")
    args = ap.parse_args()

    print(f"[coupling] schema={args.schema}")
    print(f"[coupling] steps={args.steps} seeds={args.seeds}")
    print(f"[coupling] caps={args.caps}\n")
    print("[coupling] NOTE: vehicle schedules are synthetic; buildings are real.\n")

    results: List[Dict[str, float]] = []
    for cap in args.caps:
        for mode in ("independent", "proportional", "edf"):
            per_seed = []
            for s in args.seeds:
                env = STEMSEnvironment(schema=args.schema, seed=s, heat_pump=True)
                per_seed.append(rollout(env, cap, mode, args.steps, s))
            agg = {k: (float(np.mean([r[k] for r in per_seed]))
                       if isinstance(per_seed[0][k], (int, float)) else per_seed[0][k])
                   for k in per_seed[0]}
            agg["cap_kw"], agg["coordination"] = cap, mode
            agg["departures"] = int(round(agg["departures"]))
            agg["missed_departures"] = int(round(agg["missed_departures"]))
            results.append(agg)
            print(_fmt(agg))
        print()

    payload = {"meta": {"schema": args.schema, "steps": args.steps,
                        "seeds": args.seeds, "caps": args.caps,
                        "ev_schedules": "synthetic",
                        "buildings": "real (NREL ResStock Travis County)"},
               "results": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[coupling] wrote {args.out}")


if __name__ == "__main__":
    main()
