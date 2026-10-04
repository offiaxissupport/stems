"""Do our KPIs equal the simulator's own series on this schema?

The measurement was audited on the Travis houses. Before any number is reported
on another kind of building, the same check is repeated there: cost and
consumption computed by ``MetricsCalculator`` from observations must equal the
values recomputed from CityLearn's own per-building series for the same hours.

    .venv/Scripts/python experiments/diagnostics/kpi_ground_truth.py citylearn_schemas/cl2020_zone1/schema.json winter 7
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from experiments.controllers import ARMS, build_controller          # noqa: E402
from experiments.runner import _window_len, evaluate, make_config   # noqa: E402
from experiments.scenario import Scenario                           # noqa: E402
from stems.environment import STEMSEnvironment                      # noqa: E402

schema, season, days = sys.argv[1], sys.argv[2], int(sys.argv[3])
houses = "tx_travis" in schema
sc = Scenario(schema=schema, season=season, days=days, heat_pump=houses, allow_missing_obs=not houses,
              hvac_control="setpoint" if houses else "power")
kw = sc.env_kwargs("eval")
env = STEMSEnvironment(schema=schema, seed=0, heat_pump=sc.heat_pump, env_kwargs=kw,
                       hvac_control=sc.hvac_control, allow_missing_obs=sc.allow_missing_obs)
config = make_config(sc)
ctrl = build_controller(ARMS["rbc+calibrated"], env, config)
out = evaluate(ctrl, env, config, _window_len(kw))
k, n = out["kpis"], out["steps"]
cost = cons = 0.0
for b in env._env.buildings:
    net = np.asarray(b.net_electricity_consumption, dtype=float)[:n]
    price = np.asarray(b.pricing.electricity_pricing, dtype=float)[:n]
    cost += float((np.maximum(net, 0.0) * price).sum())
    cons += float(np.maximum(net, 0.0).sum())
print(f"{Path(schema).parent.name} {season} {days}d, rule + calibrated barrier, {n} steps, "
      f"{env.num_buildings} buildings ({', '.join(sorted({type(b).__name__ for b in env._env.buildings}))})")
print(f"   cost         ours {k['cost']:12.4f}   simulator's series {cost:12.4f}   difference {k['cost'] - cost:+.2e}")
print(f"   consumption  ours {k['electricity_consumption']:12.4f}   simulator's series {cons:12.4f}   "
      f"difference {k['electricity_consumption'] - cons:+.2e}")
soc = np.array([np.asarray(b.electrical_storage.soc, dtype=float)[:n] for b in env._env.buildings])
outside = float(((soc < config.cbf.SOC_min - 1e-9) | (soc > config.cbf.SOC_max + 1e-9)).mean())
print(f"   battery outside its band: ours {k['soc_violation_rate']:.4f}   simulator's series {outside:.4f}")
