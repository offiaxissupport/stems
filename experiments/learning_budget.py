#!/usr/bin/env python3
"""How many episodes does the learned policy need? Evaluate as training proceeds.

Trains one ``rl+calibrated`` agent on a scenario's training window and, every
``--every`` episodes, evaluates the deterministic policy on the held-out window.
The rule-based and no-control references are evaluated once on the same window,
so the curve shows where (if anywhere) learning overtakes them.

    .venv/Scripts/python -m experiments.learning_budget --episodes 60 --every 10
    .venv/Scripts/python -m experiments.learning_budget --share-parameters
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.controllers import ARMS, build_controller  # noqa: E402
from experiments.runner import (_window_len, code_fingerprint, evaluate,  # noqa: E402
                                make_config, train)
from experiments.scenario import Scenario  # noqa: E402

KPIS = ("cost", "safety_violation_rate", "discomfort_rate", "peak_import_kw",
        "battery_equivalent_full_cycles", "electricity_consumption")


def main() -> None:
    ap = argparse.ArgumentParser(description="Learning-budget curve")
    ap.add_argument("--season", default="winter")
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="rl+calibrated", choices=[a for a in ARMS if ARMS[a].learns])
    ap.add_argument("--share-parameters", action="store_true")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed

    torch.set_num_threads(2)
    set_seed(args.seed)
    scenario = Scenario(season=args.season, days=args.days)
    schema = scenario.schema_path()
    train_kw, eval_kw = scenario.env_kwargs("train"), scenario.env_kwargs("eval")

    def env(kwargs):
        return STEMSEnvironment(schema=schema, seed=args.seed, heat_pump=True, env_kwargs=kwargs,
                                hvac_control=scenario.hvac_control)

    def config():
        c = make_config(scenario)
        c.actor_critic.share_parameters = args.share_parameters
        if args.lr is not None:
            c.actor_critic.lr = args.lr
        return c

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    record = {"meta": {"scenario": scenario.describe(), "arm": args.arm, "seed": args.seed,
                       "share_parameters": args.share_parameters, "lr": args.lr,
                       "code": code_fingerprint()},
              "references": {}, "curve": []}

    eval_env = env(eval_kw)
    for ref in ("idle+calibrated", "rbc+calibrated"):
        k = evaluate(build_controller(ARMS[ref], eval_env, config()), eval_env, config(),
                     _window_len(eval_kw))["kpis"]
        record["references"][ref] = {key: k[key] for key in KPIS}
        log(f"reference {ref:16s} " + " ".join(f"{key}={k[key]:.4g}" for key in KPIS[:4]))

    train_env, train_cfg = env(train_kw), config()
    agent = build_controller(ARMS[args.arm], train_env, train_cfg)
    done = 0
    while done < args.episodes:
        block = min(args.every, args.episodes - done)
        rows = train(agent, train_env, train_cfg, block, lambda m: None, _window_len(train_kw))
        done += block
        with tempfile.TemporaryDirectory() as tmp:
            agent.save(tmp)
            cfg = config()
            controller = build_controller(ARMS[args.arm], eval_env, cfg)
            controller.load(tmp)
            k = evaluate(controller, eval_env, cfg, _window_len(eval_kw))["kpis"]
        point = {"episodes": done, "train_reward": rows[-1]["reward"],
                 "entropy": rows[-1]["entropy"], **{key: k[key] for key in KPIS}}
        record["curve"].append(point)
        log(f"episodes {done:3d} train_reward={point['train_reward']:8.1f} "
            + " ".join(f"{key}={k[key]:.4g}" for key in KPIS[:5]))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
