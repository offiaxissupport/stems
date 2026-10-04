#!/usr/bin/env python3
"""Aggregate run records into scenario-level estimates, contrasts and tests.

Design
------
A *scenario* is one (season window, building subset, caps) condition. Training
seeds are nested inside scenarios, so they are not independent replicates of the
between-scenario variation: pooling them as if they were inflates n and makes
95% intervals cover far less than 95% (simulated coverage 0.68-0.89 for
within-scenario correlations 0.9-0.25). Every estimate here is therefore built
in two stages:

1. **Within a scenario**, average over seeds. For a contrast between two
   learning arms, the average is over *matched* seeds only (same seed = same
   initial networks and exploration stream, a common-random-numbers pairing); a
   seed present in one arm but not the other is dropped, never substituted. A
   deterministic arm (rule-based control) has one run per scenario and is used
   as is.
2. **Across scenarios**, the scenario-level values are the replicates:
   mean, Student-t 95% interval with ``df = n_scenarios - 1``, and a two-sided
   one-sample t-test of the paired difference against zero.

Seed spread is reported separately (root-mean-square within-scenario SD), as a
description of training noise, not as a source of replication.

Inference is confined to the pre-declared ``PRIMARY_KPIS`` x ``CONTRASTS`` on the
pooled scenarios, with a Holm correction across that family. Every other number
is descriptive and labelled as such.

Runs are excluded only for a failed actuator check (``verified is False``), and
those are listed for investigation; counts of verified / insufficient-evidence /
failed runs are shown per arm. Records produced by different code (fingerprint),
episode counts or window lengths are refused unless ``--allow-mixed`` is given.

Usage
-----
    .venv/Scripts/python -m experiments.aggregate results/ablation_v1
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PRIMARY_KPIS = ["safety_violation_rate", "cost", "discomfort_rate"]

DESCRIPTIVE_KPIS = [
    "soc_violation_rate", "grid_violation_rate", "power_violation_rate", "emission",
    "peak_import_kw", "avg_daily_peak", "ramping_rate", "discomfort_degree_hours",
    "pv_self_consumption", "battery_equivalent_full_cycles",
    "hvac_on_transitions_per_building_day", "barrier_intervention_rate",
    "cost_cv_across_buildings", "electricity_consumption",
]

# (arm, reference, question the difference answers)
CONTRASTS: List[Tuple[str, str, str]] = [
    ("rl+calibrated", "rl", "what the safety layer adds to a learned policy"),
    ("rl+calibrated", "rl+basic", "a calibrated battery model vs the uniform 0.1 rate"),
    ("rl+calibrated", "idle+calibrated", "what learning adds over no control"),
    ("rl+calibrated", "rbc+calibrated", "what learning adds over the time-of-use rule"),
    ("rbc+calibrated", "idle+calibrated", "what the time-of-use rule adds over no control"),
]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def t_critical(n: int) -> float:
    """Two-sided 95% Student-t critical value for n samples."""
    from scipy.stats import t   # required: a normal approximation is wrong at small n

    return float(t.ppf(0.975, n - 1))


def _finite(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def summarize(values: List[Any]) -> Dict[str, Any]:
    """Mean, SD, t interval and two-sided p (H0: mean = 0) of independent replicates."""
    v = np.asarray([x for x in values if _finite(x)], dtype=float)
    out: Dict[str, Any] = {"n": int(v.size), "mean": None, "std": None, "ci95": None, "p": None}
    if v.size == 0:
        return out
    out["mean"] = float(v.mean())
    if v.size == 1:
        return out
    sd = float(v.std(ddof=1))
    out["std"] = sd
    half = t_critical(int(v.size)) * sd / math.sqrt(v.size)
    out["ci95"] = [out["mean"] - half, out["mean"] + half]
    if sd > 0.0:
        from scipy.stats import t

        stat = out["mean"] / (sd / math.sqrt(v.size))
        out["p"] = float(2.0 * t.sf(abs(stat), v.size - 1))
    return out


def holm(pvalues: Dict[Any, Optional[float]], alpha: float = 0.05) -> Dict[Any, Dict[str, Any]]:
    """Holm step-down adjustment. Entries without a p-value are left untested."""
    tested = sorted(((p, k) for k, p in pvalues.items() if p is not None), key=lambda x: x[0])
    m, running, out = len(tested), 0.0, {}
    for rank, (p, k) in enumerate(tested):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = {"p": p, "p_holm": running, "significant": running < alpha}
    for k, p in pvalues.items():
        if p is None:
            out[k] = {"p": None, "p_holm": None, "significant": None}
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def load_records(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(record, dict) and "meta" in record:
            records.append(record)
    return records


def _arm(r: Dict[str, Any]) -> str:
    return r["meta"]["arm"]["name"]


def _learns(r: Dict[str, Any]) -> bool:
    return r["meta"]["arm"].get("policy") == "rl"


def _verified(r: Dict[str, Any]) -> Optional[bool]:
    return (r.get("actuators") or {}).get("verified")


def provenance_conflicts(records: List[Dict[str, Any]]) -> List[str]:
    """Fields that must agree across a grid before its runs can be compared."""
    problems = []
    fields = {
        "code fingerprint": lambda r: (r["meta"].get("code") or {}).get("fingerprint"),
        "window days": lambda r: r["meta"]["scenario"].get("days"),
        "HVAC control": lambda r: r["meta"]["scenario"].get("hvac_control"),
        "caps": lambda r: (r["meta"]["scenario"].get("grid_cap_kw"),
                           r["meta"]["scenario"].get("building_cap_kw")),
    }
    for name, get in fields.items():
        seen = {json.dumps(get(r)) for r in records}
        if len(seen) > 1:
            problems.append(f"{name}: {sorted(seen)}")
    episodes = {r["meta"].get("episodes") for r in records if _learns(r)}
    if len(episodes) > 1:
        problems.append(f"training episodes: {sorted(episodes)}")
    learners = {json.dumps(r["meta"].get("learner"), sort_keys=True) for r in records if _learns(r)}
    if len(learners) > 1:
        problems.append(f"learner settings: {sorted(learners)}")
    return problems


def scenario_table(records: List[Dict[str, Any]], kpi: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """arm -> scenario key -> seed -> value."""
    table: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        v = r["eval"].get(kpi)
        if _finite(v):
            table[_arm(r)][r["meta"]["scenario"]["key"]][int(r["meta"]["seed"])] = float(v)
    return table


def paired_scenario_differences(table, arm: str, ref: str, learns: Dict[str, bool],
                                scenarios: List[str]) -> Dict[str, Dict[str, Any]]:
    """Scenario -> {diff, seeds}: arm minus reference, seeds matched where both learn."""
    out = {}
    for s in scenarios:
        a, b = table.get(arm, {}).get(s), table.get(ref, {}).get(s)
        if not a or not b:
            continue
        if learns.get(arm) and learns.get(ref):
            common = sorted(set(a) & set(b))
            if not common:
                continue
            diff = float(np.mean([a[k] - b[k] for k in common]))
            seeds = len(common)
        else:
            diff = float(np.mean(list(a.values())) - np.mean(list(b.values())))
            seeds = max(len(a), len(b))
        out[s] = {"diff": diff, "seeds": seeds}
    return out


def arm_level(table, arm: str, scenarios: List[str]) -> Dict[str, Any]:
    """Scenario-level mean of one arm, plus the within-scenario seed spread."""
    per = [table[arm][s] for s in scenarios if s in table.get(arm, {})]
    s = summarize([float(np.mean(list(v.values()))) for v in per])
    sds = [float(np.std(list(v.values()), ddof=1)) for v in per if len(v) > 1]
    s["seed_sd"] = float(math.sqrt(np.mean(np.square(sds)))) if sds else None
    return s


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(s: Dict[str, Any], rate: bool = False) -> str:
    if s["mean"] is None:
        return "n/a"
    if s["ci95"] is None:
        return f"{s['mean']:.4g} (n={s['n']})"
    if not s["std"]:
        return f"{s['mean']:.4g} (no variation, n={s['n']})"
    lo, hi = s["ci95"]
    if rate:   # an arm's mean rate cannot leave [0, 1]; the interval is clipped for display
        lo, hi = max(lo, 0.0), min(hi, 1.0)
    return f"{s['mean']:.4g} [{lo:.4g}, {hi:.4g}] (n={s['n']})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse(records: List[Dict[str, Any]], kpis: List[str]) -> Dict[str, Any]:
    learns = {_arm(r): _learns(r) for r in records}
    seasons = sorted({r["meta"]["scenario"]["season"] for r in records})
    scen_season = {r["meta"]["scenario"]["key"]: r["meta"]["scenario"]["season"] for r in records}
    pools = {"pooled": sorted(scen_season)}
    pools.update({season: sorted(k for k, v in scen_season.items() if v == season)
                  for season in seasons})
    arms = sorted(learns)

    result: Dict[str, Any] = {"arms": {}, "contrasts": {}}
    for pool, scenarios in pools.items():
        result["arms"][pool] = {}
        result["contrasts"][pool] = {}
        for kpi in kpis:
            table = scenario_table(records, kpi)
            result["arms"][pool][kpi] = {a: arm_level(table, a, scenarios) for a in arms}
            for arm, ref, _ in CONTRASTS:
                if arm not in learns or ref not in learns:
                    continue
                diffs = paired_scenario_differences(table, arm, ref, learns, scenarios)
                s = summarize([d["diff"] for d in diffs.values()])
                s["seeds_per_scenario"] = [d["seeds"] for d in diffs.values()]
                result["contrasts"][pool].setdefault(f"{arm} - {ref}", {})[kpi] = s

    family = {(c, k): result["contrasts"]["pooled"].get(c, {}).get(k, {}).get("p")
              for c in result["contrasts"]["pooled"] for k in PRIMARY_KPIS if k in kpis}
    result["holm"] = {f"{c} | {k}": v for (c, k), v in holm(family).items()}
    return result


def render(root: Path, records, ok, errors, failed, analysis, kpis) -> str:
    lines = [f"# Ablation summary: {root.as_posix()}", "",
             f"records {len(records)} | ok {len(ok)} | errors {len(errors)} | "
             f"failed actuator check (excluded) {len(failed)}", ""]
    if failed:
        lines += ["## INVESTIGATE: runs whose actuators did not respond", ""]
        lines += [f"- {r['meta']['scenario']['key']} / {_arm(r)} / seed {r['meta']['seed']}"
                  for r in failed]
        lines.append("")
    for r in errors:
        lines.append(f"- error: {r['meta']['scenario']['key']} / {_arm(r)} / "
                     f"seed {r['meta']['seed']}: {r.get('error')}")

    counts = defaultdict(lambda: {"True": 0, "None": 0, "False": 0})
    for r in ok:
        counts[_arm(r)][str(_verified(r))] += 1
    lines += ["## Actuator verification per arm", "", "| arm | verified | insufficient evidence | failed |",
              "|---|---|---|---|"]
    lines += [f"| {a} | {c['True']} | {c['None']} | {c['False']} |" for a, c in sorted(counts.items())]
    lines.append("")

    lines += ["## Primary inference (pooled scenarios, Holm-corrected)", "",
              "Unit of replication: scenario (season x building subset); seeds averaged within it.", "",
              "| contrast | KPI | difference [95% CI] | p | p (Holm) |", "|---|---|---|---|---|"]
    for arm, ref, question in CONTRASTS:
        c = f"{arm} - {ref}"
        for k in PRIMARY_KPIS:
            s = analysis["contrasts"]["pooled"].get(c, {}).get(k)
            if s is None:
                continue
            h = analysis["holm"].get(f"{c} | {k}", {})
            p = "n/a" if h.get("p") is None else f"{h['p']:.3g}"
            ph = "n/a" if h.get("p_holm") is None else f"{h['p_holm']:.3g}" + (" *" if h["significant"] else "")
            lines.append(f"| {c} ({question}) | {k} | {fmt(s)} | {p} | {ph} |")
    lines.append("")

    for pool in analysis["arms"]:
        lines += [f"## Descriptive: {pool}", ""]
        arms = sorted(next(iter(analysis["arms"][pool].values())).keys())
        lines += ["| KPI | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for k in kpis:
            cells = [fmt(analysis["arms"][pool][k][a], rate=k.endswith("_rate")) for a in arms]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        seed = [f"{a}: {analysis['arms'][pool]['cost'][a]['seed_sd']:.3g}"
                for a in arms if "cost" in analysis["arms"][pool]
                and analysis["arms"][pool]["cost"][a].get("seed_sd") is not None]
        if seed:
            lines += ["", "Within-scenario seed SD of cost (training noise, not replication): "
                      + ", ".join(seed)]
        contrasts = analysis["contrasts"][pool]
        if contrasts:
            names = list(contrasts)
            lines += ["", "| KPI | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for k in kpis:
                lines.append(f"| {k} | " + " | ".join(fmt(contrasts[n][k]) for n in names) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate ablation records")
    ap.add_argument("root")
    ap.add_argument("--kpis", nargs="+", default=PRIMARY_KPIS + DESCRIPTIVE_KPIS)
    ap.add_argument("--allow-mixed", action="store_true",
                    help="aggregate records from different code / episodes / windows")
    args = ap.parse_args()
    root = Path(args.root)

    records = load_records(root)
    ok = [r for r in records if r.get("status") == "ok"]
    errors = [r for r in records if r.get("status") != "ok"]
    failed = [r for r in ok if _verified(r) is False]
    valid = [r for r in ok if _verified(r) is not False]

    conflicts = provenance_conflicts(valid)
    if conflicts and not args.allow_mixed:
        raise SystemExit("refusing to aggregate records that are not comparable:\n  "
                         + "\n  ".join(conflicts) + "\n(pass --allow-mixed to override)")

    kpis = [k for k in args.kpis]
    analysis = analyse(valid, kpis)
    text = render(root, records, ok, errors, failed, analysis, kpis)
    print(text)
    summary = {"counts": {"records": len(records), "ok": len(ok), "errors": len(errors),
                          "failed": len(failed)},
               "provenance_conflicts": conflicts, **analysis}
    (root / "summary.md").write_text(text, encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
