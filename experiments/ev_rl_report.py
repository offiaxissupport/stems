#!/usr/bin/env python3
"""Table of the controllers-under-a-binding-cap grid (EV schema).

One row per arm and scenario, learning arms averaged over their seeds. With two
scenarios there is nothing to test; the table reports what was measured.

    .venv/Scripts/python -m experiments.ev_rl_report results/ev_rl_v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
COLUMNS = (
    ("cost", "Cost", ".0f"),
    ("cap_exceedance_kwh", "Over cap [kWh]", ".1f"),
    ("grid_violation_rate", "Hours over cap", ".3f"),
    ("peak_import_kw", "Peak [kW]", ".1f"),
    ("ev_missed_departure_rate", "Missed departures", ".3f"),
    ("ev_energy_shortfall_kwh", "EV shortfall [kWh]", ".1f"),
    ("soc_violation_rate", "Battery band", ".3f"),
    ("discomfort_rate", "Discomfort", ".3f"),
    ("battery_equivalent_full_cycles", "Battery cycles", ".1f"),
)
ARM_ORDER = ["idle", "rbc", "idle+calibrated", "rbc+calibrated", "rbc-offpeak+calibrated",
             "rbc-never+calibrated", "rl+calibrated", "rl+calibrated+own",
             "rl+calibrated+floor", "rl-res+calibrated", "rl+calibrated+pen"]
ARM_LABEL = {"idle": "no control (cars never charged)", "rbc": "rule, cars charge on arrival, no shield",
             "idle+calibrated": "no control + shields", "rbc+calibrated": "rule + shields",
             "rl+calibrated": "RL + shields", "rl-res+calibrated": "residual RL + shields",
             "rl+calibrated+pen": "RL + shields + penalty",
             "rbc-offpeak+calibrated": "rule, cars charge off-peak + shields",
             "rbc-never+calibrated": "rule, cars never ask + shields",
             "rl+calibrated+own": "RL + shields, pays for forced charging",
             "rl+calibrated+floor": "RL + shields, floor under the charger request"}


def load(root: Path) -> List[Dict[str, Any]]:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*.json"))]
    failed = [r for r in records if r.get("status") != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} runs failed, e.g. {failed[0]['meta']['arm']['name']} "
                         f"in {failed[0]['meta']['scenario']['key']}: {failed[0].get('error')}")
    prints = {(r["meta"].get("code") or {}).get("fingerprint") for r in records}
    if len(prints) != 1:
        raise SystemExit(f"records from different code: {sorted(map(str, prints))}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the EV controller grid")
    ap.add_argument("root")
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else REPO / args.root
    records = load(root)

    cells = defaultdict(list)
    for r in records:
        cells[(r["meta"]["scenario"]["season"], r["meta"]["arm"]["name"])].append(r)
    seasons = sorted({k[0] for k in cells})
    out: List[str] = []
    meta = records[0]["meta"]
    out.append(f"Schema `{meta['scenario']['schema']}`, cap {meta['scenario']['grid_cap_kw']:g} kW, "
               f"{meta['scenario']['days']}-day windows, code `{meta['code']['fingerprint']}`.\n")
    for season in seasons:
        out.append(f"**{season}**\n")
        out.append("| Arm | seeds | " + " | ".join(c[1] for c in COLUMNS) + " |")
        out.append("|---|---|" + "---|" * len(COLUMNS))
        for arm in ARM_ORDER:
            runs = cells.get((season, arm))
            if not runs:
                continue
            vals = [format(float(np.mean([r["eval"][key] for r in runs])), fmt)
                    for key, _, fmt in COLUMNS]
            out.append(f"| {ARM_LABEL.get(arm, arm)} | {len(runs)} | " + " | ".join(vals) + " |")
        out.append("")
        spread = []
        for arm in ARM_ORDER:
            runs = cells.get((season, arm))
            if runs and len(runs) > 1:
                costs = [r["eval"]["cost"] for r in runs]
                spread.append(f"{ARM_LABEL.get(arm, arm)}: cost per seed "
                              + ", ".join(f"{c:.0f}" for c in costs))
        if spread:
            out.append("*" + "; ".join(spread) + ".*\n")
    text = "\n".join(out)
    (root / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
