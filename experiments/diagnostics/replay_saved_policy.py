"""Evaluate a policy saved by one grid under the shield of the current code.

The same trained policy, the same evaluation window, the same warm-up as
``experiments.runner.run_one``; only the code around the policy differs. Used to
separate what a change to the shield does from what retraining does (report, 6.8).

    .venv/Scripts/python experiments/diagnostics/replay_saved_policy.py results/ev_rl_v1 rl+calibrated winter 0
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from experiments.controllers import ARMS, build_controller          # noqa: E402
from experiments.runner import _window_len, code_fingerprint, evaluate, make_config  # noqa: E402
from experiments.scenario import Scenario                           # noqa: E402
from stems.environment import STEMSEnvironment                      # noqa: E402
from stems.utils import set_seed                                    # noqa: E402

root, arm_name, season, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
stored = None
for path in sorted(root.glob(f"*{season}*/{arm_name}__seed{seed}.json")):
    stored = json.loads(path.read_text(encoding="utf-8"))
    model_dir = path.with_name(f"{path.stem}_model")
if stored is None:
    raise SystemExit(f"no record for {arm_name} {season} seed {seed} under {root}")
meta = stored["meta"]
scenario = Scenario(**{k: meta["scenario"][k] for k in (
    "schema", "season", "subset_seed", "n_buildings", "days", "grid_cap_kw", "building_cap_kw",
    "hvac_control")})
set_seed(seed)
schema = scenario.schema_path()


def env_for(phase):
    return STEMSEnvironment(schema=schema, seed=seed, heat_pump=True,
                            env_kwargs=scenario.env_kwargs(phase), hvac_control=scenario.hvac_control)


eval_env = env_for("eval")
config = make_config(scenario, meta.get("learner") or {})
controller = build_controller(ARMS[arm_name], eval_env, config)
if ARMS[arm_name].learns:
    controller.load(str(model_dir))
if getattr(controller, "fleet_shield", None) is not None:
    evaluate(controller, env_for("train"), config, _window_len(scenario.env_kwargs("train")))
    for inner in (getattr(controller, "base", None), getattr(controller, "base_policy", None)):
        if inner is not None and hasattr(inner, "reset"):
            inner.reset()
new = evaluate(controller, eval_env, config, _window_len(scenario.env_kwargs("eval")))["kpis"]
old = stored["eval"]
keys = ("cost", "cap_exceedance_kwh", "peak_import_kw", "ev_missed_departures",
        "ev_energy_shortfall_kwh", "discomfort_rate", "electricity_consumption")
print(f"REPLAY {arm_name} {season} seed {seed}: stored code {meta['code']['fingerprint']} -> "
      f"current code {code_fingerprint()['fingerprint']}")
for k in keys:
    if k in old and k in new:
        print(f"   {k:28s} stored {old[k]:10.3f}   replayed {new[k]:10.3f}")
