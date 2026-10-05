#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np

from stems.config import STEMSConfig
from stems.environment import ACTION_GROUPS, STEMSEnvironment
from stems.thermal import build_thermal_stack
from stems.graph import BuildingGraph
from stems.agent import STEMSAgent
from stems.baselines import RuleBasedAgent
from stems.metrics import MetricsCalculator
from stems.utils import HistoryBuffer, set_seed

_TABLE1 = ["cost", "emission", "avg_daily_peak", "electricity_consumption",
           "ramping_rate", "discomfort_rate", "safety_violation_rate"]
_NORMALISED = _TABLE1[:5]


def run_episode(agent, env: STEMSEnvironment, config: STEMSConfig,
                soc_rate: np.ndarray, max_steps: int,
                count_soc: bool = True, dhw_barrier=None) -> Dict[str, float]:
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    metrics = MetricsCalculator(env.num_buildings, config.cbf, soc_rate=soc_rate,
                                heating_setpoint_idx=env.heating_setpoint_idx,
                                count_soc=count_soc, dhw_barrier=dhw_barrier)
    obs, _ = env.reset()
    hist.update(obs)
    done, steps = False, 0
    while not done:
        actions = agent.select_action(obs, hist.get(), explore=False)
        nxt, _, term, trunc, _ = env.step(actions)
        if hasattr(agent, "observe"):
            agent.observe(nxt)
        metrics.add_step(obs, actions, nxt)
        obs = nxt
        hist.update(obs)
        steps += 1
        done = term or trunc or (max_steps and steps >= max_steps)
    return metrics.compute_all()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate STEMS vs RuleBased (Table I)")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schema", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--heat-pump", action="store_true")
    ap.add_argument("--no-cbf", action="store_true")
    ap.add_argument("--preheat", action="store_true",
                    help="Must match the --preheat setting the checkpoint was trained with.")
    ap.add_argument("--preheat-horizon", type=int, default=None)
    ap.add_argument("--isolate", type=str, default="none",
                    choices=["none"] + list(ACTION_GROUPS),
                    help="Must match the --isolate mode the checkpoint was trained with.")
    ap.add_argument("--out", type=str, default="evaluation.json")
    args = ap.parse_args()

    set_seed(args.seed)
    config = STEMSConfig()
    heat_pump = args.heat_pump or (
        "cooling_or_heating_device" in ACTION_GROUPS.get(args.isolate, []))
    if heat_pump:
        config.heat_pump.enabled = True
    env = STEMSEnvironment(schema=args.schema, seed=args.seed, heat_pump=heat_pump)
    B, battery = env.num_buildings, env.battery_info()
    info = env.get_building_info()
    graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
    control_indices = env.resolve_control_indices(args.isolate)
    count_soc = (control_indices is None
                or env.electrical_storage_action_index in control_indices)

    if args.preheat_horizon is not None:
        config.thermal.preheat_horizon = args.preheat_horizon
    dhw_barrier, cop_model = build_thermal_stack(env, config.thermal, enable=args.preheat)

    agent = STEMSAgent(env.obs_dim, env.action_dim, B, graph, config=config,
                       battery_info=battery, use_cbf=not args.no_cbf,
                       electrical_storage_action_index=env.electrical_storage_action_index,
                       control_indices=control_indices,
                       dhw_barrier=dhw_barrier, cop_model=cop_model,
                       hvac_action_index=env.hvac_action_index)
    agent.load(args.checkpoint)
    rule = RuleBasedAgent(num_buildings=B, hvac_control=env.hvac_control,
                          battery_nominal_power=battery["nominal_power"])

    print(f"[eval] env_type={env.env_type} B={B} obs_dim={env.obs_dim} "
          f"isolate={args.isolate} control_indices={control_indices} "
          f"checkpoint={args.checkpoint}")
    score_barrier, _ = build_thermal_stack(env, config.thermal, enable=args.preheat)
    rb = run_episode(rule, env, config, battery["soc_rate"], args.max_steps,
                     dhw_barrier=score_barrier)
    st = run_episode(agent, env, config, battery["soc_rate"], args.max_steps,
                     count_soc=count_soc, dhw_barrier=score_barrier)

    norm = {m: (st[m] / rb[m] if abs(rb[m]) > 1e-9 else float("nan")) for m in _NORMALISED}
    print("\nTable I (metrics 1-5 normalised to RuleBased=1.0; lower is better)")
    print(f"{'metric':26s} {'RuleBased':>12s} {'STEMS':>12s} {'STEMS/RB':>10s}")
    for m in _TABLE1:
        ntxt = f"{norm[m]:.3f}" if m in norm else "-"
        print(f"{m:26s} {rb[m]:12.4f} {st[m]:12.4f} {ntxt:>10s}")
    print(f"\nSTEMS safety_violation_rate = {st['safety_violation_rate']:.4f} "
          f"(paper STEMS 0.056; RuleBased here {rb['safety_violation_rate']:.4f})")

    with open(args.out, "w") as f:
        json.dump({"meta": {"env_type": env.env_type, "seed": args.seed,
                            "checkpoint": args.checkpoint, "heat_pump": heat_pump,
                            "isolate": args.isolate, "control_indices": control_indices,
                            "preheat": args.preheat,
                            "citylearn_patches": env.citylearn_patches},
                   "rulebased": rb, "stems": st,
                   "stems_normalised": norm}, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
