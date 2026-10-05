#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.controllers import ARMS
from experiments.runner import code_fingerprint, run_one
from experiments.scenario import SEASONS, TX_SCHEMA, Scenario


def build_grid(args) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for season in args.seasons:
        for subset in args.subsets:
            scenario = Scenario(schema=args.schema, season=season,
                                subset_seed=None if subset == "ref" else int(subset),
                                n_buildings=args.buildings, days=args.days,
                                hvac_control=args.hvac_control,
                                heat_pump=not args.no_heat_pump,
                                allow_missing_obs=args.allow_missing_obs,
                                grid_cap_kw=args.grid_cap,
                                building_cap_kw=args.building_cap)
            for arm_name in args.arms:
                seeds = args.seeds if ARMS[arm_name].learns else [0]
                for seed in seeds:
                    out = Path(args.out) / scenario.key / f"{arm_name}__seed{seed}.json"
                    specs.append({"scenario": asdict(scenario), "arm": arm_name,
                                  "seed": seed, "episodes": args.episodes,
                                  "learner": {"share_parameters": not args.per_building_actors},
                                  "out": str(out)})
    return specs


def already_done(spec: Dict[str, Any], fingerprint: str) -> bool:
    path = Path(spec["out"])
    path = path if path.is_absolute() else REPO / path
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    meta = record.get("meta", {})
    return (record.get("status") == "ok"
            and (meta.get("code") or {}).get("fingerprint") == fingerprint
            and (meta.get("episodes") == spec["episodes"]
                 or not ARMS[spec["arm"]].learns))


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy vs safety-layer ablation grid")
    ap.add_argument("--seasons", nargs="+", default=["winter", "summer"],
                    choices=SEASONS)
    ap.add_argument("--subsets", nargs="+", default=["ref", "1", "2"],
                    help="'ref' = the schema's own buildings; integers = sampled subsets")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--days", type=int, default=28,
                    help="length of each training and evaluation window")
    ap.add_argument("--buildings", type=int, default=8)
    ap.add_argument("--grid-cap", type=float, default=300.0)
    ap.add_argument("--building-cap", type=float, default=80.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--schema", default=TX_SCHEMA,
                    help="schema to run on; the housing-and-EV schema adds the fleet "
                         "shield to every shielded arm")
    ap.add_argument("--hvac-control", choices=["setpoint", "power"], default="setpoint",
                    help="HVAC action semantics: set-point offset (default) or power fraction")
    ap.add_argument("--no-heat-pump", action="store_true",
                    help="the schema has no heat pump the controller drives "
                         "(datasets without thermal dynamics)")
    ap.add_argument("--allow-missing-obs", action="store_true",
                    help="zero-fill observations the dataset does not have")
    ap.add_argument("--per-building-actors", action="store_true",
                    help="one actor/critic per building (the paper's layout) instead of one "
                         "shared by all buildings")
    ap.add_argument("--out", default="results/ablation_v1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = build_grid(args)
    fingerprint = code_fingerprint()["fingerprint"]
    todo = [s for s in specs if args.force or not already_done(s, fingerprint)]
    print(f"[ablation] {len(specs)} runs in grid, {len(todo)} to do, workers={args.workers}")
    if args.dry_run:
        for s in todo:
            print("   ", s["out"])
        return

    for scenario_json in {json.dumps(s["scenario"], sort_keys=True) for s in todo}:
        Scenario(**json.loads(scenario_json)).schema_path()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    t0, finished = time.time(), 0
    with get_context("spawn").Pool(processes=max(1, args.workers), maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_one, todo):
            finished += 1
            print(f"[ablation] {finished}/{len(todo)} {res['status']:5s} "
                  f"verified={res['verified']} {res['seconds']:8.1f}s  {res['out']}",
                  flush=True)
    print(f"[ablation] finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
