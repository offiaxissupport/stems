#!/usr/bin/env python3
"""CTDE vs DTDE: does inter-building information help, and when?

Question
--------
E9 (``study_coupling.py``) measured the shared grid constraint and compared
allocation rules *inside the safety layer*. It trained nothing. This study asks
the learning question that E9 made well-posed:

    when a shared constraint binds, does letting the policy see its neighbours
    improve the outcome -- and does that advantage disappear when the constraint
    is slack?

Design
------
The two arms differ by exactly one knob. The spatial encoder is a graph
convolution that adds self-loops before normalising, so setting the adjacency to
zero leaves each building encoding only itself:

    graph on  (CTDE-like)  adjacency = building similarity graph, Eq. 11
    graph off (DTDE)       adjacency = 0  =>  A_hat = I

Everything else is identical: same architecture, same parameter count, same
optimiser, same seeds, same reward, same barriers. Any difference is
attributable to inter-building information flow and nothing else.

The safety shield runs in ``independent`` mode -- no shared-cap allocation -- so
that any coordination must be *learned* rather than supplied by the shield. The
Lagrangian cost critics supply the pressure to avoid violations.

Each arm is trained at several grid caps, from slack to binding. The prediction
under test is that the arms coincide where the cap is slack and separate where it
binds. E9 located that regime at roughly 20-45 kW for this fleet.

Environment: ``tx_travis_8b_ev`` -- 8 real Travis buildings, 6 with chargers.
**Vehicle schedules are synthetic** (see ``setup_citylearn_ev.py``).

Usage
-----
    .venv/Scripts/python study_ctde.py --episodes 8 --steps 1200 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from stems.agent import STEMSAgent
from stems.config import CBFConfig, STEMSConfig, ThermalConfig
from stems.environment import STEMSEnvironment
from stems.ev import EVChargerSpec, EVObsLayout, EVReadinessBarrier
from stems.graph import BuildingGraph
from stems.metrics import MetricsCalculator
from stems.reward import STEMSReward
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed

SCHEMA = "citylearn_schemas/tx_travis_8b_ev/schema.json"
# Charger min/max power ratio: actions below this are floored to the same
# output by CityLearn, so the environment response is flat across [0, DEADBAND].
DEADBAND = 1.4 / 11.0
_IDX_SOC, _IDX_NET = 19, 20


def build_ev_barrier(env: STEMSEnvironment) -> EVReadinessBarrier:
    layout = env.ev_obs_layout()[0]
    info = env.ev_info()
    power = info["max_charging_power"][:, 0].astype(np.float32)
    eff = info["efficiency"][:, 0].astype(np.float32)
    spec = EVChargerSpec(max_charging_power_kw=power,
                         efficiency=np.where(power > 0, eff, 1.0),
                         action_bound=np.ones(env.num_buildings, dtype=np.float32),
                         action_index=env.ev_action_indices()[0])
    return EVReadinessBarrier(
        EVObsLayout(connected_state=layout["connected_state"],
                    departure_time=layout["departure_time"],
                    required_soc_departure=layout["required_soc_departure"],
                    soc=layout["soc"], battery_capacity=layout["battery_capacity"]),
        spec, name="ev")


def departure_stats(records: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Missed departures from a sequence of per-step connection snapshots."""
    departures = missed = 0
    for prev, cur in zip(records[:-1], records[1:]):
        left = prev["connected"] & ~cur["connected"] & prev["owner"]
        for i in np.where(left)[0]:
            departures += 1
            if prev["soc"][i] + 1e-3 < prev["required"][i]:
                missed += 1
    return {"departures": departures, "missed_departures": missed,
            "missed_departure_rate": missed / max(departures, 1)}


def run_arm(cap_kw: float, graph_on: bool, seed: int, episodes: int,
            steps: int, schema: str, use_barrier: bool = True) -> Dict[str, Any]:
    """Train one arm and evaluate it deterministically."""
    set_seed(seed)
    config = STEMSConfig()
    config.cbf = CBFConfig(P_grid_max=cap_kw, P_building_max=cap_kw)
    config.heat_pump.enabled = True
    config.thermal = ThermalConfig(dhw_readiness=False, weather_anticipation=False,
                                   cop_aware_power=False)
    # The deadline barrier emits exactly 0 whenever charging is not yet urgent,
    # so regressing the actor onto the executed action would train it toward
    # that constant. Learn from the policy's own sample instead, treating the
    # shield as part of the environment.
    config.training.actor_target = "raw"

    env = STEMSEnvironment(schema=schema, seed=seed, heat_pump=True)
    B = env.num_buildings
    battery = env.battery_info()
    ev_slot = env.ev_action_indices()[0]
    control_indices = [ev_slot]                      # EVs are the coupled load
    ev_barrier = build_ev_barrier(env)
    layout_for_reward = env.ev_obs_layout()[0]

    info = env.get_building_info()
    graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
    agent = STEMSAgent(env.obs_dim, env.action_dim, B, graph, config=config,
                       battery_info=battery, use_cbf=True,
                       electrical_storage_action_index=env.electrical_storage_action_index,
                       control_indices=control_indices,
                       deadline_barriers=([ev_barrier] if use_barrier else None),
                       hvac_action_index=-1)
    # The single knob. Zeroing the adjacency leaves A_hat = I in the graph
    # convolution, so each building's representation depends only on itself.
    if not graph_on:
        agent.adj = torch.zeros_like(agent.adj)
    # Coordination must be learned, not supplied by the shield.
    agent.cbf.coordination = "independent"
    agent.cbf.enforce_soc = False

    reward_fn = STEMSReward(config.reward, B, cap_kw, cap_kw,
                            heating_setpoint_idx=env.heating_setpoint_idx,
                            ev_layout=layout_for_reward)
    buffer = EpisodeBuffer()
    hist = HistoryBuffer(B, env.obs_dim, config.transformer.window_size)
    layout = env.ev_obs_layout()[0]
    owners = np.array([env.building_has_action(b, ev_slot) for b in range(B)], bool)

    def snapshot(obs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        return {"connected": np.array([o[layout["connected_state"]] for o in obs]) > 0.5,
                "soc": np.array([o[layout["soc"]] for o in obs], dtype=np.float32),
                "required": np.array([o[layout["required_soc_departure"]] for o in obs],
                                     dtype=np.float32),
                "owner": owners}

    train_curve = []
    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, _ = env.reset()
        hist.reset(); hist.update(obs)
        buffer.reset()
        prev_net = [float(o[_IDX_NET]) for o in obs]
        ep_reward, done, n = 0.0, False, 0
        viol = 0
        ep_actions = []

        while not done:
            window = hist.get()
            actions = agent.select_action(obs, window, explore=True)
            ep_actions.append(agent._last_raw_actions[:, ev_slot].copy())
            nxt, _, term, trunc, _ = env.step(actions)
            agent.observe(nxt)
            n += 1
            done = term or trunc or n >= steps

            rewards = reward_fn.compute(obs, actions, nxt, prev_net)
            prev_net = [float(o[_IDX_NET]) for o in nxt]

            net_n = np.array([o[_IDX_NET] for o in nxt], dtype=np.float32)
            total = float(np.maximum(net_n, 0).sum())
            viol += int(total > cap_kw)
            c_soc = np.zeros(B, np.float32)          # battery not controlled
            c_pow = (np.abs(net_n) > cap_kw).astype(np.float32)
            c_grid = np.full(B, float(total > cap_kw), np.float32)
            costs = np.stack([c_soc, c_pow, c_grid], axis=-1)

            hist.update(nxt)
            buffer.add(obs=obs, actions=actions, rewards=rewards, next_obs=nxt,
                       done=done, history=window, next_history=hist.get(),
                       raw_actions=agent._last_raw_actions,
                       safe_actions=agent._last_safe_actions,
                       pre_tanh=agent._last_pre_tanh,
                       behaviour_log_probs=agent._last_log_probs,
                       constraint_costs=costs)
            ep_reward += float(np.mean(rewards))
            obs = nxt

        agent.update(buffer.get_batch())
        # Diagnostic that matters here: the charger floors any action below
        # min_charging_power/max_charging_power (0.127 on this hardware) to the
        # same output, so the response is FLAT across that band. A policy whose
        # actions all fall inside it cannot influence the environment at all,
        # and no architecture comparison run on such a policy means anything.
        acts = np.array(ep_actions, dtype=np.float32)
        above = float((acts > DEADBAND).mean())
        train_curve.append({"episode": ep, "reward": ep_reward,
                            "grid_violation_rate": viol / max(n, 1),
                            "mean_abs_action": float(np.abs(acts).mean()),
                            "frac_above_deadband": above,
                            "seconds": time.time() - t0})
        print(f"      ep {ep:2d}/{episodes}  reward={ep_reward:9.1f}  "
              f"viol={viol / max(n,1):.3f}  |a|={np.abs(acts).mean():.3f}  "
              f"above_deadband={above:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- deterministic evaluation -------------------------------------
    obs, _ = env.reset()
    hist.reset(); hist.update(obs)
    metrics = MetricsCalculator(B, config.cbf, soc_rate=battery["soc_rate"],
                                heating_setpoint_idx=env.heating_setpoint_idx,
                                count_soc=False)
    records = [snapshot(obs)]
    viol = peak = 0
    peak = 0.0
    for _ in range(steps):
        actions = agent.select_action(obs, hist.get(), explore=False)
        nxt, _, term, trunc, _ = env.step(actions)
        agent.observe(nxt)
        metrics.add_step(obs, actions, nxt)
        net_n = np.array([o[_IDX_NET] for o in nxt], dtype=np.float32)
        total = float(np.maximum(net_n, 0).sum())
        peak = max(peak, total)
        viol += int(total > cap_kw)
        hist.update(nxt)
        obs = nxt
        records.append(snapshot(obs))
        if term or trunc:
            break

    m = metrics.compute_all()
    out = {"cap_kw": cap_kw, "graph": "on" if graph_on else "off", "seed": seed,
           "episodes": episodes, "steps": steps,
           "final_train_reward": train_curve[-1]["reward"],
           "eval_grid_violation_rate": viol / max(len(records) - 1, 1),
           "eval_peak_kw": peak, "eval_cost": m["cost"],
           "eval_electricity": m["electricity_consumption"],
           "train_curve": train_curve}
    out.update(departure_stats(records))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CTDE vs DTDE across coupling tightness")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--caps", type=float, nargs="+", default=[80.0, 30.0])
    ap.add_argument("--schema", type=str, default=SCHEMA)
    ap.add_argument("--no-barrier", action="store_true",
                    help="Disable the EV deadline barrier so the policy must learn "
                         "to meet departures itself. With the barrier active the "
                         "task is already solved and the policy has nothing to do.")
    ap.add_argument("--out", type=str, default="ctde_study_results.json")
    args = ap.parse_args()

    print(f"[ctde] schema={args.schema}")
    print(f"[ctde] episodes={args.episodes} steps={args.steps} "
          f"seeds={args.seeds} caps={args.caps}")
    print("[ctde] one knob: graph adjacency on (CTDE-like) vs zeroed (DTDE).")
    print("[ctde] shield coordination=independent, so coordination must be learned.")
    print("[ctde] NOTE: vehicle schedules are synthetic; buildings are real.\n")

    results = []
    for cap in args.caps:
        for graph_on in (True, False):
            for seed in args.seeds:
                tag = f"cap={cap:.0f}kW graph={'on ' if graph_on else 'off'} seed={seed}"
                print(f"[ctde] {tag}")
                r = run_arm(cap, graph_on, seed, args.episodes, args.steps,
                            args.schema, use_barrier=not args.no_barrier)
                results.append(r)
                print(f"      -> missed {r['missed_departures']}/{r['departures']} "
                      f"({r['missed_departure_rate']:.3f})  "
                      f"viol={r['eval_grid_violation_rate']:.3f}  "
                      f"peak={r['eval_peak_kw']:.1f}  cost={r['eval_cost']:.0f}\n",
                      flush=True)

    # -- aggregate -------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'cap':>6} {'graph':>6} {'missed rate':>13} {'viol':>8} {'peak':>8} {'cost':>10}")
    print("-" * 78)
    summary = []
    for cap in args.caps:
        for graph_on in (True, False):
            rows = [r for r in results
                    if r["cap_kw"] == cap and r["graph"] == ("on" if graph_on else "off")]
            agg = {"cap_kw": cap, "graph": "on" if graph_on else "off",
                   "n_seeds": len(rows)}
            for k in ("missed_departure_rate", "eval_grid_violation_rate",
                      "eval_peak_kw", "eval_cost", "final_train_reward"):
                agg[k] = float(np.mean([r[k] for r in rows]))
                agg[k + "_std"] = float(np.std([r[k] for r in rows]))
            summary.append(agg)
            print(f"{cap:6.0f} {agg['graph']:>6} "
                  f"{agg['missed_departure_rate']:.3f}+/-{agg['missed_departure_rate_std']:.3f} "
                  f"{agg['eval_grid_violation_rate']:8.3f} {agg['eval_peak_kw']:8.1f} "
                  f"{agg['eval_cost']:10.0f}")
    print("=" * 78)

    with open(args.out, "w") as f:
        json.dump({"meta": {"schema": args.schema, "episodes": args.episodes,
                            "steps": args.steps, "seeds": args.seeds,
                            "caps": args.caps,
                            "shield_coordination": "independent",
                            "ev_schedules": "synthetic"},
                   "summary": summary, "runs": results}, f, indent=2)
    print(f"[ctde] wrote {args.out}")


if __name__ == "__main__":
    main()
