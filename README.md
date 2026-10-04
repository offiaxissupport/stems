# STEMS — honest replication + a fix that beats SOTA on constraint violation

Replication of, and an extension to:

> Zhang, X., Wu, J., Zinflou, A., & Boulet, B. (2025). *STEMS: Spatial-Temporal
> Enhanced Multi-Agent Safe Building Energy Management System.* IEEE IoT Journal.
> [arXiv:2510.14112v2](https://arxiv.org/abs/2510.14112)

STEMS is safe multi-agent RL for building energy on CityLearn: a shared spatial
GCN + temporal Transformer encoder feeds per-building SAC actors/critics, and a
Control Barrier Function (CBF) shield enforces battery-SOC and grid-power limits.

## What this repo actually does (and the headline result)

The original implementation here drifted into silent fallbacks (mock env
masquerading as real, CBF emitting all-zeros when infeasible) and — critically —
a **mis-calibrated CBF**: it assumed every battery moves SOC by `0.1` per unit
action, while real CityLearn batteries move `0.18–0.53` (per building, `=
nominal_power/capacity`). That single constant put the safety violation rate at
~9–70%, far from the paper's 5.6%.

Calibrating the CBF to the real per-building dynamics and adding four safety
tricks drives the violation rate to **zero on real data, at no cost penalty**.
Measured by `ablation.py` (real CityLearn, 800 steps × 2 seeds):

| Safety configuration            | Violation rate | Cost |
|---------------------------------|:--------------:|:----:|
| no CBF (raw policy)             | 0.7195         | 7676 |
| CBF, **old `dSOC=0.1`** (the bug) | 0.0945       | 7777 |
| CBF + **real calibration**      | 0.0110         | 7703 |
| + robust margins                | 0.0000         | 7705 |
| + anticipatory                  | 0.0000         | 7662 |
| **FULL (all four tricks)**      | **0.0000**     | **7649** |

Paper STEMS reports `safety_violation_rate = 0.056`. The old `0.1` constant alone
explains why the prior code could not reproduce it. Power/grid limits never bind
on this dataset (loads ~10–40 kW vs 80/300 kW caps), so SOC is the headline
constraint — consistent with the paper.

The four constraint-violation tricks (`stems/config.py: SafetyConfig`,
`LagrangianConfig`):
1. **Feasibility-guaranteed CBF** — analytic per-building projection that always
   returns the least-violating feasible action plus a recovery action; never the
   old "emergency zeros".
2. **Anticipatory safety** — a per-building control-invariance buffer keeping the
   state recoverable next step.
3. **PID-Lagrangian duals** — PID control on the constraint-cost error.
4. **Robust margins** — enforce a band strictly inside the reported limits.

## Heat pumps and hot water: anticipatory pre-heating

Hot water cannot be produced on demand. CityLearn charges the DHW tank by
`energy = action * capacity`, capped at the heater's hourly output, so the SOC
gain per step is `min(action_bound, P_nom*eta/C)` = **0.52–0.85** on these
buildings — a **time-to-heat from empty of 1.2–1.9 h** (mean 1.60 h). A
controller that reacts to demand *after* seeing it is structurally too late.

`stems/thermal.py` adds three calibrated-from-the-simulator pieces:

- **`DHWDynamics`** — per-building tank charge rate and time-to-heat, validated
  against the simulator (predicted vs. simulated one-step gain within 15%).
- **`DHWDemandForecaster`** — causal online hour-of-day/weekday climatology of
  DHW demand with a cold-weather uplift. CityLearn has no DHW forecast
  observation, so this is learned from observed steps only — no look-ahead.
- **`CoPModel`** — weather-dependent Carnot CoP from each building's *real*
  efficiency and supply temperatures.

These drive **barrier h4 (hot-water readiness)**: the tank must hold the forecast
demand over a pre-heat horizon `L`, plus a robust margin, plus a cold-front term.
The projection is monotone — it can only *raise* the DHW action, never veto a
policy that heats harder. The power barriers additionally become **CoP-aware**:
HVAC draw is counted, and the cap is derated by the CoP shortfall, so the shield
is most cautious exactly when conditioning is least efficient.

> **Upstream CityLearn defect, found here.** `Building.update_dhw_storage`
> (2.6.0b1, `building.py:1562`) scales the action by `heating_storage.capacity`
> instead of `dhw_storage.capacity`. This schema has no heating tank, so that is
> `0.0` and **every DHW action was a silent no-op** — charging at action 0.8 for
> eight steps left the SOC at exactly 0.0. `STEMSEnvironment` now patches this
> loudly (banner + `citylearn_patches` in run metadata; disable with
> `patch_dhw=False`). Any `--isolate dhw` / `--isolate thermal` result predating
> the fix was driving a dead actuator.

Run the study, or train with the barrier active:

```bash
.venv/Scripts/python study_heatpump.py --steps 1500 --seeds 0 1
```

```bash
.venv/Scripts/python train.py --episodes 15 --isolate thermal --preheat --seed 0
```

## Housing + EVs: `tx_travis_8b_ev`

No shipped CityLearn dataset has both a full thermal building model and electric
vehicles — `tx_travis_8b` has no chargers, and `citylearn_challenge_2022_phase_all_plus_evs`
has **no thermal observations at all** (no indoor temperature, setpoints, DHW,
occupancy or cooling demand). `setup_citylearn_ev.py` merges them:

```bash
.venv/Scripts/python -B setup_citylearn_ev.py --validate
```

The thermal side is untouched (`root_directory` still points at the CityLearn
cache; no building CSV is copied or modified). The **EV schedules are synthetic** —
ResStock carries no vehicle data — generated from a documented, seeded commuter
model. Results on this schema must say so.

Result: 8 buildings, 6 with chargers, `obs_dim=35`, `action_dim=4`
(`dhw_storage`, `electrical_storage`, `cooling_or_heating_device`,
`electric_vehicle_storage_0`), EV fill times **5.7–7.4 h** against the hot-water
tank's 1.6 h.

```bash
.venv/Scripts/python train.py --schema citylearn_schemas/tx_travis_8b_ev/schema.json \
    --isolate all --preheat --episodes 15 --seed 0
```

`--isolate` accepts `none`, `dhw`, `heatpump`, `thermal`, `ev`, `ev+thermal`,
`battery+ev`, `all`.

## Honesty guarantees

- **No silent fallbacks.** Real CityLearn is the default; missing deps/schema
  raise. The synthetic mock is reachable only via `--mock` and prints a loud
  banner + stamps `env_type="mock"` into all output.
- **Ground-truth metadata.** Every run records `env_type`, `tricks_active`,
  `seed`, and per-building `soc_rate` in `training_history.json`.
- The headline `safety_violation_rate` is computed from *observed* SOC/power, not
  the CBF's internal model.

## Setup

Use the project `.venv` (regular **Python 3.12**). The Windows Store `py -3.11`
has a broken torch (`c10.dll` init failure) — do not use it.

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install "numpy<2.0.0" gymnasium==0.28.1 scikit-learn cvxpy \
    pandas pyyaml simplejson matplotlib seaborn tqdm platformdirs requests nrel-pysam
.venv/Scripts/python -m pip install -e C:/temp/citylearn_src --no-deps
```

Pinned versions are in `requirements.lock.txt`. Generate/validate the 8-building
Travis schema once:

```bash
.venv/Scripts/python -B setup_citylearn_8b.py --validate
```

## Run

```bash
# Fast end-to-end pipeline check on real CityLearn (2 episodes x 300 steps)
.venv/Scripts/python train.py --smoke

# Train (full-year episodes)
.venv/Scripts/python train.py --episodes 15 --seed 0 --save-dir checkpoints/seed0

# The headline safety-trick ablation (real data)
.venv/Scripts/python ablation.py --steps 800 --seeds 0 1

# The heat-pump / hot-water pre-heating study (real data)
.venv/Scripts/python study_heatpump.py --steps 1500 --seeds 0 1

# All tests (85: core, real-env, thermal, deadline/EV, widening, merged schema)
.venv/Scripts/python -m pytest tests/ -q
```

## Layout

```
stems/            config, environment, graph, encoder, cbf, deadline, thermal, ev, reward, agent, metrics, utils, baselines
train.py          training (Algorithm 2), records ground-truth metadata
ablation.py       safety-trick ablation on real CityLearn (the headline table)
study_heatpump.py heat-pump / DHW pre-heating + weather study
evaluate.py       checkpoint evaluation + baselines (Table I)
setup_citylearn_8b.py   real Travis 8-building schema generator
setup_citylearn_ev.py   merged housing + EV schema generator (tx_travis_8b_ev)
tests/            core correctness tests
extensions/       beyond-paper 50-building hierarchical scaling (not in headline path)
stems/2510.14112v2.pdf   the paper
```

MIT License.
