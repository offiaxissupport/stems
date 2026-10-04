#!/usr/bin/env python3
"""EV charging under a shared power cap: which rule keeps the deadlines, and at what cap.

Every run replays one season's window of the housing-and-EV schema with a fixed
controller for the house devices (``--base``), a fixed way of asking for charging
(``--policy``) and one fleet shield (``--rule``), under one cap. Nothing is
learned here: the question is what the shield itself guarantees.

House devices (``base``)
    idle       thermostat only; battery and hot-water tank untouched
    rbc        the time-of-use rule: battery charged 10:00-16:00, discharged to cover
               the house's own load 16:00-21:00 -- which is when the cars arrive
    rbc+shed   the same, plus a set-point shift: pre-condition 12:00-16:00, coast
               16:00-21:00 (within the 1.5 degC offset range)

Charging requests (``policy``)
    asap       full power whenever a connected car is below its requirement
    offpeak    the same, but not during the 16:00-21:00 tariff peak
    none       never asks: the shield alone must get the cars charged

Shield (``rule``): see ``stems.fleet.FleetShield``. ``noguard`` is ``independent``
without the latest-start trigger (the request, untouched).

Forecast of the load the fleet must fit around (``forecast``)
    replay     the same scenario recorded without charging: perfect foresight
    causal     persistence, with a calibrated margin (``BaseLoadForecaster``)

Vehicle schedules in this schema are synthetic; the buildings are real.

    .venv/Scripts/python -m experiments.ev_coupling --stage rules --workers 6
    .venv/Scripts/python -m experiments.ev_coupling --stage policy --workers 6
    .venv/Scripts/python -m experiments.ev_coupling --stage flexibility --workers 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EV_SCHEMA = "citylearn_schemas/tx_travis_8b_ev/schema.json"
_IDX_HOUR, _IDX_NET, _IDX_PRICE = 1, 20, 21
PEAK_HOURS = range(17, 22)          # CityLearn hours 17..21 = 16:00-21:00
PREP_HOURS = range(13, 17)          # 12:00-16:00
ALL_RULES = ("noguard", "independent", "static", "proportional", "edf", "llf", "sllf", "lp")
MISS_TOL = 1e-3


def _env(season: str, days: int):
    from experiments.scenario import Scenario
    from stems.environment import STEMSEnvironment

    sc = Scenario(schema=EV_SCHEMA, season=season, days=days)
    return STEMSEnvironment(schema=EV_SCHEMA, seed=0, heat_pump=True,
                            env_kwargs=sc.env_kwargs("eval"), hvac_control="setpoint")


class HouseController:
    """The non-EV devices: returns a (B, action_dim) action with the EV column zero."""

    def __init__(self, env, base: str) -> None:
        from stems.baselines import RuleBasedAgent
        from stems.cbf import CBFShield
        from stems.config import CBFConfig, SafetyConfig

        if base not in ("idle", "rbc", "rbc+shed"):
            raise ValueError(f"unknown base controller {base!r}")
        self.env, self.base = env, base
        self.rule = RuleBasedAgent(env.num_buildings, hvac_control="setpoint",
                                   battery_nominal_power=env.battery_info()["nominal_power"])
        self.shield = CBFShield(CBFConfig(P_grid_max=1e9, P_building_max=1e9), env.num_buildings,
                                battery_model=env.battery_model(),
                                elec_idx=env.electrical_storage_action_index,
                                safety_cfg=SafetyConfig(anticipatory=False, robust_margins=False),
                                enforce_soc=True, hvac_idx=-1)

    def __call__(self, obs: List[np.ndarray]) -> np.ndarray:
        env = self.env
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        if self.base != "idle":
            a[:, :3] = self.rule.select_action(obs)
        if self.base == "rbc+shed":
            hour = int(round(float(obs[0][_IDX_HOUR])))
            heating = env.executed_actions[:, env.hvac_action_index] >= 0.0
            if hour in PREP_HOURS:          # store heat (or coolth) before the peak
                a[:, env.hvac_action_index] = np.where(heating, 1.0, -1.0)
            elif hour in PEAK_HOURS:        # and coast through it
                a[:, env.hvac_action_index] = np.where(heating, -1.0, 1.0)
        return self.shield.project(a, obs)   # keeps the house batteries in their band


def record_base(season: str, days: int, base: str) -> np.ndarray:
    """(T, B) net load of each building with no EV charging: the replay forecast."""
    env = _env(season, days)
    house = HouseController(env, base)
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        out = env.step(house(obs))
        obs, done = out[0], out[2] or out[3]
        rows.append([float(o[_IDX_NET]) for o in obs])
    return np.array(rows)


def run(spec: Dict[str, Any]) -> Dict[str, Any]:
    """One rollout. ``spec``: season, days, base, policy, rule, forecast, cap, base_path."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    out = dict(spec)
    try:
        from stems.fleet import BaseLoadForecaster, FleetShield, fleet_power_bounds

        env = _env(spec["season"], spec["days"])
        model = env.ev_fleet_model()
        layout, e = env.ev_obs_layout()[0], env.ev_action_indices()[0]
        house = HouseController(env, spec["base"])
        replay = np.load(spec["base_path"])
        forecaster = BaseLoadForecaster(
            env.num_buildings, replay=replay if spec["forecast"] == "replay" else None)
        rule = spec["rule"]
        shield = FleetShield(model, layout, e, spec["cap"],
                             "independent" if rule == "noguard" else rule, forecaster,
                             guard_deadlines=rule != "noguard",
                             reserve_hours=int(spec.get("reserve", 0)),
                             lead_margin=bool(spec.get("lead", False)))
        cap = float(spec["cap"])

        obs, _ = env.reset()
        done, n = False, 0
        departures: List[Dict[str, float]] = []
        imports, base_imports, ev_kwh, cost = [], [], 0.0, 0.0
        infeasible_hours, binding_hours, margins, flexibility = 0, 0, [], []
        while not done:
            hour = int(round(float(obs[0][_IDX_HOUR])))
            actions = house(obs)
            col = lambda key: np.array([float(o[layout[key]]) for o in obs])
            wants = (col("connected_state") > 0.5) & (col("soc") < col("required_soc_departure"))
            if spec["policy"] == "none" or (spec["policy"] == "offpeak" and hour in PEAK_HOURS):
                wants = np.zeros_like(wants)
            actions[:, e] = np.where(wants & model.has_ev, 1.0, 0.0)
            actions = shield.project(actions, obs)
            if spec.get("log_flexibility") and n % 6 == 0:
                b = fleet_power_bounds(model, shield.state(obs))
                flexibility.append(b["u_max"] - b["u_min"])
            price = np.array([float(o[_IDX_PRICE]) for o in obs])
            nxt = env.step(actions)
            obs_next, done = nxt[0], nxt[2] or nxt[3]
            shield.observe(obs_next, env.ev_draw_kwh)
            net = np.array([float(o[_IDX_NET]) for o in obs_next])
            imports.append(float(np.maximum(net, 0.0).sum()))
            base_imports.append(float(np.maximum(net - env.ev_draw_kwh, 0.0).sum()))
            cost += float((np.maximum(net, 0.0) * price).sum())
            ev_kwh += float(env.ev_draw_kwh.sum())
            departures += env.ev_departures
            infeasible_hours += int(shield.last.get("feasible") is False)
            binding_hours += int(bool(shield.last.get("binding")))
            margins.append(float(shield.last.get("margin_kw", 0.0)))
            obs = obs_next
            n += 1

        imports, base_imports = np.array(imports), np.array(base_imports)
        over = np.maximum(imports - cap, 0.0)
        avoidable = over[base_imports <= cap]            # the house load alone was under the cap
        short = np.array([max(d["required_soc"] - d["soc"], 0.0) for d in departures])
        short_kwh = np.array([s * d["capacity_kwh"] for s, d in zip(short, departures)])
        gaps = np.array([max(d["required_soc"], 1e-6) for d in departures])
        served = 1.0 - short / gaps                      # share of the requirement delivered
        per_vehicle: Dict[int, List[float]] = {}
        for d, s in zip(departures, served):
            per_vehicle.setdefault(int(d["building"]), []).append(float(s))
        means = np.array([np.mean(v) for v in per_vehicle.values()]) if per_vehicle else np.ones(1)
        out.update(
            status="ok", steps=n, departures=len(departures),
            missed=int((short > MISS_TOL).sum()),
            missed_rate=float((short > MISS_TOL).mean()) if len(short) else 0.0,
            unserved_kwh=float(short_kwh.sum()),
            worst_unserved_share=float((short / gaps).max()) if len(short) else 0.0,
            jain_fairness=float(means.sum() ** 2 / (len(means) * (means ** 2).sum())),
            cap_exceed_rate=float((over > 1e-6).mean()),
            cap_exceed_kwh=float(over.sum()), cap_exceed_max_kw=float(over.max()),
            avoidable_exceed_rate=float((avoidable > 1e-6).sum() / max(n, 1)),
            avoidable_exceed_kwh=float(avoidable.sum()),
            peak_import_kw=float(imports.max()), base_peak_kw=float(base_imports.max()),
            ev_kwh=ev_kwh, cost=cost,
            infeasible_hour_rate=infeasible_hours / max(n, 1),
            binding_hour_rate=binding_hours / max(n, 1),
            mean_margin_kw=float(np.mean(margins)),
            mean_flexibility_kw=float(np.mean(flexibility)) if flexibility else None)
    except Exception as exc:
        out.update(status="error", error=repr(exc), traceback=traceback.format_exc())
    out["seconds"] = round(time.time() - t0, 1)
    return out


STAGES = {
    # which rule keeps the deadlines, with and without foresight
    "rules": dict(bases=["idle"], policies=["asap"], rules=list(ALL_RULES),
                  forecasts=["replay", "causal"]),
    # does it matter how the cars ask?
    "policy": dict(bases=["idle"], policies=["asap", "offpeak", "none"], rules=["llf", "lp"],
                   forecasts=["replay"]),
    # under a causal forecast, what do planning every departure early and holding
    # the later hours to the day-ahead forecast's own margin buy? (The margin only
    # enters the programme: the sorting rules do not look past this hour.)
    "reserve": dict(bases=["idle"], policies=["asap", "none"], rules=["llf", "lp"],
                    forecasts=["causal"], reserves=[0, 1, 2], leads=[False, True]),
    # how much room do the house's battery and thermal mass make for the cars?
    "flexibility": dict(bases=["idle", "rbc", "rbc+shed"], policies=["asap"],
                        rules=["llf", "lp"], forecasts=["replay"]),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="EV charging under a shared cap")
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--seasons", nargs="+", default=["winter", "summer"])
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--caps", type=float, nargs="+",
                    default=[100.0, 60.0, 50.0, 45.0, 40.0, 35.0, 30.0, 27.5, 25.0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/ev_coupling")
    args = ap.parse_args()
    from experiments.runner import code_fingerprint

    fingerprint = code_fingerprint()      # the code as it is when the study starts
    stage = STAGES[args.stage]
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for season, base in itertools.product(args.seasons, stage["bases"]):
        path = out_dir / f"base__{season}{args.days}d__{base}.npy"
        if not path.exists():
            print(f"[ev] recording the load without charging: {season} / {base}", flush=True)
            np.save(path, record_base(season, args.days, base))

    specs = []
    for season, base, policy, rule, forecast, cap, reserve, lead in itertools.product(
            args.seasons, stage["bases"], stage["policies"], stage["rules"],
            stage["forecasts"], args.caps, stage.get("reserves", [0]),
            stage.get("leads", [False])):
        if rule in ("noguard", "independent") and forecast == "causal":
            continue                      # these rules never look at the forecast
        if lead and rule != "lp":
            continue                      # only the programme plans the later hours
        specs.append(dict(stage=args.stage, season=season, days=args.days, base=base,
                          policy=policy, rule=rule, forecast=forecast, cap=cap, reserve=reserve,
                          lead=lead,
                          log_flexibility=rule == "lp" and forecast == "replay",
                          base_path=str(out_dir / f"base__{season}{args.days}d__{base}.npy")))
    print(f"[ev] stage {args.stage}: {len(specs)} runs, workers={args.workers}", flush=True)

    results, t0 = [], time.time()
    with get_context("spawn").Pool(processes=max(1, args.workers), maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(run, specs), 1):
            results.append(r)
            tag = (f"{r['season']:6s} {r['base']:8s} {r['policy']:7s} {r['rule']:12s} "
                   f"{r['forecast']:6s} cap={r['cap']:5.1f}")
            if r["status"] != "ok":
                print(f"[ev] {k}/{len(specs)} ERROR {tag}: {r['error']}", flush=True)
                continue
            print(f"[ev] {k}/{len(specs)} {tag} missed={r['missed']:3d}/{r['departures']:3d} "
                  f"unserved={r['unserved_kwh']:7.1f}kWh over={r['cap_exceed_rate']:.3f} "
                  f"({r['seconds']:.0f}s)", flush=True)
    payload = {"meta": {"stage": args.stage, "seasons": args.seasons, "days": args.days,
                        "caps": args.caps, "schema": EV_SCHEMA, "ev_schedules": "synthetic",
                        "code": fingerprint},
               "results": sorted(results, key=lambda r: (r["season"], r["base"], r["policy"],
                                                         r["rule"], r["forecast"],
                                                         r.get("reserve", 0), r.get("lead", False),
                                                         -r["cap"]))}
    (out_dir / f"{args.stage}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ev] wrote {args.out}/{args.stage}.json in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
