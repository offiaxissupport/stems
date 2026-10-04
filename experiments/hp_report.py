#!/usr/bin/env python3
"""Table of the heat-pump-only grid: what a set-point policy saves, and at what comfort.

Every number is on the patched simulator, where the indoor temperature that the
comfort KPI and the comfort reward read is the simulated one (CityLearn >= 2.4
reports the uncontrolled dataset value: see ``stems/environment.py``).

    .venv/Scripts/python -m experiments.hp_report results/heatpump_v1
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
    ("electricity_consumption", "Consumption [kWh]", ".0f"),
    ("discomfort_rate", "Discomfort", ".3f"),
    ("discomfort_degree_hours", "Discomfort [degC h]", ".1f"),
    ("discomfort_rate_worst_building", "Worst house", ".3f"),
    ("peak_import_kw", "Peak [kW]", ".1f"),
    ("avg_daily_peak", "Daily peak [kW]", ".1f"),
    ("hvac_on_transitions_per_building_day", "Switches per day", ".2f"),
)
ARM_ORDER = ["idle", "hp-shift", "rl-hp"]
ARM_LABEL = {"idle": "thermostat at the set point", "hp-shift": "pre-condition and coast (fixed schedule)",
             "rl-hp": "learned set-point policy"}
SEASONS = ["winter", "spring", "summer", "autumn"]


def load(root: Path) -> List[Dict[str, Any]]:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*.json"))]
    failed = [r for r in records if r.get("status") != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} runs failed, e.g. {failed[0]['meta']['arm']['name']} "
                         f"in {failed[0]['meta']['scenario']['key']}: {failed[0].get('error')}")
    prints = {(r["meta"].get("code") or {}).get("fingerprint") for r in records}
    if len(prints) != 1:
        raise SystemExit(f"records from different code: {sorted(map(str, prints))}")
    blind = [r for r in records
             if "endogenous_obs_from_simulated_hour" not in (r["meta"].get("citylearn_patches") or [])]
    if blind:
        raise SystemExit(f"{len(blind)} runs were made without the temperature-observation patch: "
                         "their comfort numbers would be blind to the heat pump")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the heat-pump-only grid")
    ap.add_argument("root")
    args = ap.parse_args()
    root = Path(args.root) if Path(args.root).is_absolute() else REPO / args.root
    records = load(root)
    cells = defaultdict(list)
    for r in records:
        cells[(r["meta"]["scenario"]["season"], r["meta"]["arm"]["name"])].append(r)
    meta = records[0]["meta"]
    out: List[str] = [f"Schema `{meta['scenario']['schema']}`, {meta['scenario']['days']}-day windows, "
                      f"code `{meta['code']['fingerprint']}`; simulator patches "
                      f"{', '.join(meta['citylearn_patches'])}.\n"]
    mean = lambda runs, key: float(np.mean([r["eval"][key] for r in runs]))
    for season in [s for s in SEASONS if any(k[0] == s for k in cells)]:
        out.append(f"**{season}**\n")
        out.append("| Arm | seeds | " + " | ".join(c[1] for c in COLUMNS) + " | Cost vs thermostat |")
        out.append("|---|---|" + "---|" * (len(COLUMNS) + 1))
        ref = cells.get((season, "idle"))
        for arm in ARM_ORDER:
            runs = cells.get((season, arm))
            if not runs:
                continue
            vals = [format(mean(runs, key), fmt) for key, _, fmt in COLUMNS]
            rel = "" if not ref else f"{100 * (mean(runs, 'cost') / mean(ref, 'cost') - 1):+.1f}%"
            out.append(f"| {ARM_LABEL.get(arm, arm)} | {len(runs)} | " + " | ".join(vals) + f" | {rel} |")
        out.append("")
        runs = cells.get((season, "rl-hp"))
        if runs and len(runs) > 1:
            out.append("*learned policy, per seed: cost "
                       + ", ".join(f"{r['eval']['cost']:.0f}" for r in runs) + "; discomfort "
                       + ", ".join(f"{r['eval']['discomfort_rate']:.3f}" for r in runs) + ".*\n")
    text = "\n".join(out)
    (root / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
