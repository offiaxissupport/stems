#!/usr/bin/env python3
"""Ablation of the constraint-violation tricks on real CityLearn.

The headline STEMS metric is the safety violation rate. On the Travis dataset
this is dominated by the battery SOC bound, which the CBF shield enforces, so
this ablation isolates the *shield's* contribution: the same seeded nominal
policy is replayed through the real environment under different safety
configurations, and we report each configuration's violation rate and cost.

This cleanly separates the four tricks' marginal effects from RL training noise.
(Full policy training / cost-vs-baseline comparison is via ``train.py``.)

Usage:
    .venv/Scripts/python ablation.py --steps 1000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np

from stems.environment import STEMSEnvironment
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig

_IDX_SOC, _IDX_NET, _IDX_PRICE = 19, 20, 21


def _full_safety() -> SafetyConfig:
    return SafetyConfig(feasibility_qp=True, anticipatory=True,
                        robust_margins=True, soc_margin=0.03, power_derate=0.05)


# (label, use_cbf, safety_cfg, miscalibrate). miscalibrate forces the old
# global dSOC=0.1 instead of the real per-building rate.
def _configs() -> List[tuple]:
    return [
        ("no_cbf (raw policy)",         False, None,                                                     False),
        ("cbf, OLD dSOC=0.1 (the bug)", True,  _full_safety(),                                           True),
        ("cbf + real calibration",      True,  SafetyConfig(anticipatory=False, robust_margins=False),  False),
        ("  + robust margins",          True,  SafetyConfig(anticipatory=False, robust_margins=True),   False),
        ("  + anticipatory",            True,  SafetyConfig(anticipatory=True,  robust_margins=False),   False),
        ("FULL (all tricks)",           True,  _full_safety(),                                           False),
    ]


def rollout(env: STEMSEnvironment, use_cbf: bool, safety_cfg: Optional[SafetyConfig],
            miscalibrate: bool, steps: int, seed: int) -> Dict[str, float]:
    bi = env.battery_info()
    soc_rate = np.full(env.num_buildings, 0.1, np.float32) if miscalibrate else bi["soc_rate"]
    cbf = CBFShield(CBFConfig(), env.num_buildings, soc_rate=soc_rate,
                    nominal_power=bi["nominal_power"],
                    elec_idx=env.electrical_storage_action_index,
                    safety_cfg=safety_cfg or SafetyConfig())
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    cbf_cfg = CBFConfig()
    n_soc = n_pow = n_grid = n_any = total = 0
    cost = 0.0
    for _ in range(steps):
        nominal = np.clip(rng.normal(0.3, 0.7, (env.num_buildings, env.action_dim)), -1, 1).astype(np.float32)
        actions = cbf.project(nominal, obs) if use_cbf else nominal
        # Tariff of the hour being simulated: read before stepping (the returned
        # observation already carries the next hour's price).
        price = np.array([o[_IDX_PRICE] for o in obs])
        obs, _, term, trunc, _ = env.step(actions)
        soc = np.array([o[_IDX_SOC] for o in obs])
        net = np.array([o[_IDX_NET] for o in obs])
        sv = (soc < cbf_cfg.SOC_min) | (soc > cbf_cfg.SOC_max)
        pv = np.abs(net) > cbf_cfg.P_building_max
        gv = np.maximum(net, 0).sum() > cbf_cfg.P_grid_max
        n_soc += int(sv.sum()); n_pow += int(pv.sum()); n_grid += int(gv) * env.num_buildings
        n_any += int((sv | pv | gv).sum()); total += env.num_buildings
        cost += float((np.maximum(net, 0) * price).sum())
        if term or trunc:
            break
    return {"violation_rate": n_any / max(total, 1),
            "soc_rate": n_soc / max(total, 1),
            "power_rate": n_pow / max(total, 1),
            "grid_rate": n_grid / max(total, 1),
            "cost": cost}


def main() -> None:
    ap = argparse.ArgumentParser(description="Safety-trick ablation on real CityLearn")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--schema", type=str, default=None)
    ap.add_argument("--out", type=str, default="ablation_results.json")
    args = ap.parse_args()

    print(f"[ablation] real CityLearn | steps={args.steps} | seeds={args.seeds}")
    print(f"[ablation] paper STEMS safety_violation_rate = 0.056 (target to beat)\n")
    results: Dict[str, Dict[str, float]] = {}
    for label, use_cbf, safety_cfg, miscal in _configs():
        per_seed = []
        for s in args.seeds:
            env = STEMSEnvironment(schema=args.schema, seed=s)
            per_seed.append(rollout(env, use_cbf, safety_cfg, miscal, args.steps, s))
        agg = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
        agg["violation_std"] = float(np.std([r["violation_rate"] for r in per_seed]))
        results[label.strip()] = agg
        print(f"{label:30s}  viol={agg['violation_rate']:.4f}+/-{agg['violation_std']:.4f}  "
              f"(soc={agg['soc_rate']:.4f} pow={agg['power_rate']:.4f} grid={agg['grid_rate']:.4f})  "
              f"cost={agg['cost']:.0f}")

    with open(args.out, "w") as f:
        json.dump({"meta": {"env_type": "CityLearn", "steps": args.steps,
                            "seeds": args.seeds, "paper_target": 0.056}, "results": results}, f, indent=2)
    print(f"\n[ablation] wrote {args.out}")


if __name__ == "__main__":
    main()
