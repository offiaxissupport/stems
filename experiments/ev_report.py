#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RULE_ORDER = ["noguard", "independent", "static", "proportional", "edf", "llf", "sllf", "lp"]
RULE_LABEL = {"noguard": "no shield", "independent": "per-vehicle barrier", "static": "static split",
              "proportional": "proportional", "edf": "EDF", "llf": "LLF", "sllf": "sLLF",
              "lp": "LP (joint)"}


def load(path: Path) -> List[Dict[str, Any]]:
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        raise SystemExit(f"{path.name}: {len(failed)} runs failed, e.g. "
                         f"{failed[0]['season']} {failed[0]['rule']} cap {failed[0]['cap']}: "
                         f"{failed[0].get('error')}")
    return results


def pivot(rows, key, value, agg=np.mean):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[key(r)][r["cap"]].append(r[value])
    return {k: {c: float(agg(v)) for c, v in caps.items()} for k, caps in acc.items()}


def table(title: str, data: Dict[Any, Dict[float, float]], order: List[Any], fmt: str,
          label=lambda k: str(k)) -> str:
    caps = sorted({c for v in data.values() for c in v}, reverse=True)
    lines = [f"**{title}**", "", "| | " + " | ".join(f"{c:g} kW" for c in caps) + " |",
             "|---|" + "---|" * len(caps)]
    for k in order:
        if k in data:
            lines.append(f"| {label(k)} | " + " | ".join(
                format(data[k][c], fmt) if c in data[k] else "" for c in caps) + " |")
    return "\n".join(lines) + "\n"


def line_plot(path: Path, title: str, ylabel: str, series: Dict[str, Dict[float, float]],
              order: List[str], labels: Dict[str, str], base_peak: float = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    for m, k in zip(markers, [k for k in order if k in series]):
        caps = sorted(series[k])
        ax.plot(caps, [series[k][c] for c in caps], marker=m, linewidth=1.6, markersize=5,
                label=labels.get(k, k))
    if base_peak is not None:
        ax.axvline(base_peak, color="0.5", linestyle=":", linewidth=1)
        ax.text(base_peak, ax.get_ylim()[1], " house load peak", va="top", fontsize=8, color="0.4")
    ax.set_xlabel("shared import cap [kW]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_xscale("log")
    caps = sorted({c for v in series.values() for c in v})
    ax.set_xticks(caps)
    ax.set_xticklabels([f"{c:g}" for c in caps], fontsize=8)
    ax.minorticks_off()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report the EV coupling study")
    ap.add_argument("--root", default="results/ev_coupling")
    ap.add_argument("--figures", default="docs/figures")
    args = ap.parse_args()
    root, figs = REPO / args.root, REPO / args.figures
    out: List[str] = []

    rules_path = root / "rules.json"
    if rules_path.exists():
        rows = load(rules_path)
        base_peak = float(np.mean([r["base_peak_kw"] for r in rows]))
        n_dep = int(np.mean([r["departures"] for r in rows]))
        out.append(f"## Rules (house devices idle, cars ask for full power; {n_dep} departures "
                   f"per season window; house-load peak {base_peak:.1f} kW)\n")
        for forecast in ("replay", "causal"):
            sub = [r for r in rows if r["forecast"] == forecast
                   or (forecast == "replay" and r["rule"] in ("noguard", "independent"))]
            name = "perfect foresight of the house load" if forecast == "replay" else "causal forecast"
            for value, title, fmt in (
                    ("missed_rate", f"Missed departures, {name}", ".3f"),
                    ("unserved_kwh", f"Energy not delivered at departure [kWh], {name}", ".1f"),
                    ("avoidable_exceed_kwh", f"Energy over the cap caused by charging [kWh], {name}", ".1f")):
                out.append(table(title, pivot(sub, lambda r: r["rule"], value), RULE_ORDER, fmt,
                                 lambda k: RULE_LABEL[k]))
            line_plot(figs / f"ev_unserved_{forecast}.png",
                      f"Energy not delivered at departure ({name})", "kWh per 14-day window",
                      pivot(sub, lambda r: r["rule"], "unserved_kwh"), RULE_ORDER, RULE_LABEL, base_peak)
            line_plot(figs / f"ev_exceed_{forecast}.png",
                      f"Energy imported above the cap because of charging ({name})",
                      "kWh per 14-day window",
                      pivot(sub, lambda r: r["rule"], "avoidable_exceed_kwh"), RULE_ORDER,
                      RULE_LABEL, base_peak)
        out.append(table("Jain fairness of delivered share, perfect foresight",
                         pivot([r for r in rows if r["forecast"] == "replay"],
                               lambda r: r["rule"], "jain_fairness"), RULE_ORDER, ".4f",
                         lambda k: RULE_LABEL[k]))
        out.append(table("Calibrated margin of the causal forecast [kW]",
                         pivot([r for r in rows if r["forecast"] == "causal"],
                               lambda r: r["rule"], "mean_margin_kw"), RULE_ORDER, ".2f",
                         lambda k: RULE_LABEL[k]))
        lp = [r for r in rows if r["rule"] == "lp" and r["forecast"] == "replay"]
        out.append(table("LP shield, perfect foresight: share of hours with an empty safe set",
                         pivot(lp, lambda r: "lp", "infeasible_hour_rate"), ["lp"], ".3f",
                         lambda k: "LP (joint)"))
        out.append(table("LP shield, perfect foresight: mean fleet flexibility u_max - u_min [kW]",
                         pivot([r for r in lp if r.get("mean_flexibility_kw") is not None],
                               lambda r: "lp", "mean_flexibility_kw"), ["lp"], ".1f",
                         lambda k: "LP (joint)"))

    policy_path = root / "policy.json"
    if policy_path.exists():
        rows = load(policy_path)
        out.append("## How the cars ask (perfect foresight)\n")
        keys = [(p, r) for p in ("asap", "offpeak", "none") for r in ("llf", "lp")]
        lab = lambda k: f"{k[0]} + {RULE_LABEL[k[1]]}"
        for value, title, fmt in (("missed_rate", "Missed departures", ".3f"),
                                  ("unserved_kwh", "Energy not delivered [kWh]", ".1f"),
                                  ("ev_kwh", "Energy drawn by the chargers [kWh]", ".0f"),
                                  ("cost", "Neighbourhood cost", ".0f"),
                                  ("avoidable_exceed_kwh",
                                   "Energy over the cap caused by charging [kWh]", ".1f"),
                                  ("peak_import_kw", "Peak import [kW]", ".1f")):
            out.append(table(title, pivot(rows, lambda r: (r["policy"], r["rule"]), value),
                             keys, fmt, lab))
        series = pivot(rows, lambda r: f"{r['policy']}+{r['rule']}", "unserved_kwh")
        order = [f"{p}+{r}" for p, r in keys]
        line_plot(figs / "ev_policy.png",
                  "Energy not delivered at departure, by how the cars ask (perfect foresight)",
                  "kWh per 14-day window", series, order,
                  {f"{p}+{r}": f"{p} + {RULE_LABEL[r]}" for p, r in keys})

    flex_path = root / "flexibility.json"
    if flex_path.exists():
        rows = load(flex_path)
        out.append("## What the house devices free up (cars ask for full power, perfect foresight)\n")
        keys = [(b, r) for r in ("lp", "llf") for b in ("idle", "rbc", "rbc+shed")]
        lab = lambda k: f"{k[0]} + {RULE_LABEL[k[1]]}"
        for value, title, fmt in (("unserved_kwh", "Energy not delivered [kWh]", ".1f"),
                                  ("missed_rate", "Missed departures", ".3f"),
                                  ("base_peak_kw", "House-load peak without charging [kW]", ".1f"),
                                  ("cost", "Neighbourhood cost", ".0f")):
            out.append(table(title, pivot(rows, lambda r: (r["base"], r["rule"]), value),
                             keys, fmt, lab))
        lp = [r for r in rows if r["rule"] == "lp"]
        line_plot(figs / "ev_flexibility.png",
                  "Energy not delivered at departure, LP shield, by house controller",
                  "kWh per 14-day window", pivot(lp, lambda r: r["base"], "unserved_kwh"),
                  ["idle", "rbc", "rbc+shed"],
                  {"idle": "house idle", "rbc": "battery rule", "rbc+shed": "battery rule + set-point shift"})

    reserve_path = root / "reserve.json"
    if reserve_path.exists():
        rows = load(reserve_path)
        out.append("## Causal forecast: planning every departure early, and the later-hours margin\n")
        keys = [(p, r, h, m) for p in ("asap", "none") for r in ("llf", "lp") for m in (False, True)
                for h in (0, 1, 2)]
        lab = lambda k: (f"{k[0]} + {RULE_LABEL[k[1]]}, reserve {k[2]} h"
                         + (", later-hours margin" if k[3] else ""))
        for value, title, fmt in (("missed_rate", "Missed departures", ".3f"),
                                  ("unserved_kwh", "Energy not delivered [kWh]", ".1f"),
                                  ("avoidable_exceed_kwh", "Energy over the cap caused by charging [kWh]", ".1f"),
                                  ("cost", "Neighbourhood cost", ".0f")):
            out.append(table(title, pivot(rows, lambda r: (r["policy"], r["rule"], r.get("reserve", 0),
                                                           bool(r.get("lead", False))),
                                          value), keys, fmt, lab))
        deferred = [r for r in rows if r["policy"] == "none" and r["rule"] == "lp"]
        tag = lambda r: f"{r.get('reserve', 0)}{'m' if r.get('lead') else ''}"
        line_plot(figs / "ev_reserve.png",
                  "Deferred charging under a causal forecast: energy not delivered (LP shield)",
                  "kWh per 14-day window", pivot(deferred, tag, "unserved_kwh"),
                  [f"{h}{m}" for m in ("", "m") for h in (0, 1, 2)],
                  {f"{h}{m}": f"reserve {h} h" + (", later-hours margin" if m else "")
                   for m in ("", "m") for h in (0, 1, 2)})

    text = "\n".join(out)
    (root / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
