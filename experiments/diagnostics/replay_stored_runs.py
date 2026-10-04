"""Does the current code reproduce stored runs of the EV study? (report, 9)

    .venv/Scripts/python experiments/diagnostics/replay_stored_runs.py

The cap-shield changes are meant to leave the shield untouched when no house
storage is attached; this replays stored configurations and compares every
number the study recorded.
"""
import json
import os
import sys
from multiprocessing import get_context

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
MAIN = str(REPO / "results" / "ev_coupling")
PICK = [("rules", dict(season="winter", rule="lp", forecast="causal", cap=40.0)),
        ("rules", dict(season="winter", rule="llf", forecast="replay", cap=35.0)),
        ("rules", dict(season="summer", rule="edf", forecast="causal", cap=25.0)),
        ("policy", dict(season="winter", policy="none", rule="lp", cap=40.0)),
        ("reserve", dict(season="winter", policy="none", rule="lp", cap=35.0, reserve=2)),
        ("flexibility", dict(season="winter", base="rbc+shed", rule="lp", cap=30.0))]
KEYS = ("missed", "unserved_kwh", "cap_exceed_kwh", "avoidable_exceed_kwh", "peak_import_kw", "ev_kwh",
        "cost", "infeasible_hour_rate", "binding_hour_rate", "mean_margin_kw", "jain_fairness")


def one(item):
    from experiments.ev_coupling import run
    stage, want = item
    stored = json.load(open(os.path.join(MAIN, f"{stage}.json")))["results"]
    match = [r for r in stored if all(r.get(k) == v for k, v in want.items())]
    assert len(match) == 1, (stage, want, len(match))
    old = match[0]
    spec = {k: old[k] for k in ("stage", "season", "days", "base", "policy", "rule", "forecast", "cap",
                                "reserve", "log_flexibility")}
    spec["base_path"] = os.path.join(MAIN, os.path.basename(old["base_path"]))
    new = run(spec)
    diffs = {k: (old[k], new.get(k)) for k in KEYS if old[k] != new.get(k)}
    return stage, want, new["status"], diffs


if __name__ == "__main__":
    import stems
    print("code under test:", os.path.dirname(stems.__file__))
    with get_context("spawn").Pool(3) as pool:
        for stage, want, status, diffs in pool.imap_unordered(one, PICK):
            print(f"EQUIV {stage:11s} {want} status={status} ->",
                  "identical on %d fields" % len(KEYS) if not diffs else f"DIFFERS {diffs}")
