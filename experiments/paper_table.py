#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NORMALISED = (("cost", "Cost"), ("emission", "Emission"), ("avg_daily_peak", "Avg. daily peak"),
              ("electricity_consumption", "Consumption"), ("ramping_rate", "Ramping"))
RAW = (("discomfort_rate", "Discomfort rate"), ("safety_violation_rate", "Safety viol. rate"))
PAPER = {"Rule-Based (paper)": (1.000, 1.000, 1.000, 1.000, 1.000, 0.130, 0.351),
         "MPC (paper)": (0.872, 0.914, 0.983, 0.975, 0.981, 0.654, 0.330),
         "Single-Agent SAC (paper)": (0.824, 0.867, 0.856, 0.813, 0.925, 0.485, 0.223),
         "MADDPG (paper)": (0.795, 0.817, 0.854, 0.856, 0.936, 0.357, 0.197),
         "MetaEMS (paper)": (0.836, 0.804, 0.884, 0.835, 0.907, 0.396, 0.231),
         "MARLISA (paper)": (0.847, 0.875, 0.826, 0.838, 0.956, 0.237, 0.214),
         "MADCQ (paper)": (0.814, 0.837, 0.862, 0.823, 0.906, 0.325, 0.155),
         "D-MAPPO (paper)": (0.803, 0.834, 0.841, 0.816, 0.912, 0.152, 0.198),
         "STEMS without CBF (paper)": (0.782, 0.801, 0.823, 0.796, 0.894, 0.145, 0.208),
         "STEMS (paper)": (0.792, 0.824, 0.821, 0.805, 0.883, 0.132, 0.056)}
ARM_ORDER = ["idle", "rbc", "rbc+calibrated", "rl", "rl+basic", "rl+linear", "rl+calibrated",
             "rl-res+calibrated"]
ARM_LABEL = {"idle": "no control", "rbc": "rule-based (ours) = 1",
             "rbc+calibrated": "rule-based + exact barrier", "rl": "learner, no barrier",
             "rl+basic": "learner + uniform-rate barrier",
             "rl+linear": "learner + the paper's barrier (lossless battery)",
             "rl+calibrated": "learner + exact barrier (ours)",
             "rl-res+calibrated": "residual learner + exact barrier"}


def load(root: Path, reference: str) -> Dict[str, List[Dict[str, Any]]]:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*.json"))]
    failed = [r for r in records if r.get("status") != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} runs failed, e.g. {failed[0]['meta']['arm']['name']}: "
                         f"{failed[0].get('error')}")
    prints = {(r["meta"].get("code") or {}).get("fingerprint") for r in records}
    if len(prints) != 1:
        raise SystemExit(f"records from different code: {sorted(map(str, prints))}")
    keys = {r["meta"]["scenario"]["key"] for r in records}
    if len(keys) != 1:
        raise SystemExit(f"more than one scenario in {root}: {sorted(keys)}")
    arms = defaultdict(list)
    for r in records:
        arms[r["meta"]["arm"]["name"]].append(r)
    if reference not in arms:
        raise SystemExit(f"no {reference!r} run to normalise to")
    return arms


def interval(values: List[float]) -> str:
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return f"{v.mean():.3f}"
    from scipy import stats

    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.3f} ± {half:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Results in the layout of the STEMS paper's Table I")
    ap.add_argument("root")
    ap.add_argument("--reference", default="rbc", help="arm every normalised metric is divided by")
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else REPO / args.root
    arms = load(root, args.reference)
    ref = {k: float(np.mean([r["eval"][k] for r in arms[args.reference]])) for k, _ in NORMALISED}
    meta = next(iter(arms.values()))[0]["meta"]
    head = ["Method", "seeds"] + [h for _, h in NORMALISED] + [h for _, h in RAW]
    out = [f"Scenario `{meta['scenario']['key']}`, code `{meta['code']['fingerprint']}`; normalised to "
           f"`{args.reference}`; mean ± 95% interval over seeds.\n",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm in ARM_ORDER:
        runs = arms.get(arm)
        if not runs:
            continue
        cells = [interval([r["eval"][k] / ref[k] for r in runs]) for k, _ in NORMALISED]
        cells += [interval([r["eval"][k] for r in runs]) for k, _ in RAW]
        out.append(f"| {ARM_LABEL.get(arm, arm)} | {len(runs)} | " + " | ".join(cells) + " |")
    for name, row in PAPER.items():
        out.append(f"| *{name}* | 5 | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    out.append("")
    out.append("*Paper rows: Travis County, 5 residential + 2 commercial + 1 mixed-use buildings, its own "
               "rule-based controller and simulator build. Our rows: the scenario above, our rule-based "
               "controller, the corrected simulator. The two blocks are the same protocol and the same "
               "metrics, not the same experiment.*")
    text = "\n".join(out)
    (root / "paper_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
