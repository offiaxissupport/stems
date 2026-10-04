# STEMS — Complete Project Reference

*Safe multi-agent reinforcement learning for housing, heat pumps and electric vehicles on real CityLearn data.*

Replication of and extension to Zhang, Wu, Zinflou & Boulet (2025), *STEMS: Spatial-Temporal Enhanced Multi-Agent
Safe Building Energy Management System*, [arXiv:2510.14112](https://arxiv.org/abs/2510.14112).

Companion long-form document with full derivations: `stems_report.tex` / `stems_report.pdf`.

---

## Table of contents

1. [What this project claims](#1-what-this-project-claims)
2. [Quick start](#2-quick-start)
3. [The environments](#3-the-environments)
4. [The AI architecture](#4-the-ai-architecture)
5. [Safety: one barrier framework, four devices](#5-safety-one-barrier-framework-four-devices)
6. [Reward and metrics](#6-reward-and-metrics)
7. [Device isolation](#7-device-isolation)
8. [Results](#8-results)
9. [Defects found — in the simulator and in our own work](#9-defects-found--in-the-simulator-and-in-our-own-work)
10. [What is real and what is synthetic](#10-what-is-real-and-what-is-synthetic)
11. [Engineering state](#11-engineering-state)
12. [Open work](#12-open-work)

---

## 1. What this project claims

### The thesis

**Multi-agent safe RL for housing and electric vehicles, with constraint violation as the headline metric.**
Heat pumps and hot water are devices inside that system — they provide the flexibility that lets a house charge a
deadline-locked vehicle without breaching a limit.

### The honest version of the results

Three claims, in decreasing order of how much they should impress you:

1. **A defensible engineering claim.** Actuator dynamics must be *read from the plant*, not assumed. A single
   hard-coded constant (`dSOC = 0.1` where real batteries move 0.178–0.533) was the difference between a 70%
   violation rate and zero. The same discipline applied to hot water and EVs produces the same kind of result.

2. **A defensible design claim.** Anticipatory barrier horizons must match *measured* actuation lead time. The
   hot-water deficit collapses 86% exactly at the horizon the physics predicts (L=1 → L=2, against a measured
   1.60 h mean time-to-heat). The EV case validates the same law with nothing to tune, because the deadline is
   declared in the data.

3. **A theory contribution worth developing.** Per-device barriers are feasibility-guaranteed; *coupled* deadline
   constraints under a shared power cap are not. The safe set can be **empty**, and a shield claiming a guarantee
   there is lying. Detection and honest degradation beat "we drove violations to zero" — which any correct
   projection achieves.

### What this project does *not* claim

- **"We beat SOTA."** Driving the measured violation rate to zero is a constraint-satisfaction result, not a
  learning result. The paper's 0.056 was very likely an implementation artifact of the same mis-calibration we
  fixed. Say that plainly rather than claiming a win.
- **"MARL coordination is proven valuable."** The 2026 CityLearn benchmark ([arXiv:2602.19223](https://arxiv.org/abs/2602.19223))
  finds decentralised training *beating* centralised. Our own numbers suggest why: nothing binds. Loads are
  10–40 kW against a 300 kW cap, and under thermal isolation the shield commands only 3.5% of total load. Those
  studies measured coordination in a regime with nothing to coordinate about. **EV fleets are what make it bind** —
  and that experiment has not been run yet.
- **Novel CBF theory.** The deadline barrier is a discrete-time, rate-limited instance of *input-constrained /
  backup* CBF theory ([survey](https://www.sciencedirect.com/science/article/abs/pii/S1367578824000142)). What is
  ours is the constructive horizon rule and the plant-measured rate.

### Why the original code was unusable

Pervasive **silent fallbacks** made it a fake experiment: the environment swapped to a synthetic mock when the
CityLearn import failed; the CBF returned all-zeros when its QP was infeasible; the trainer used raw unsafe
actions when `safe_actions` was missing. The one honest real run sat at ~70% violation.

Everything was rewritten **fail-loud**: the mock is reachable only via `--mock` with a banner, `env_type` is
stamped into every result, and a missing observation or device raises instead of silently resolving to nothing.

---

## 2. Quick start

Use the project `.venv` (**Python 3.12**). The Windows Store `py -3.11` has a broken torch (`c10.dll` init
failure) — do not use it.

```bash
.venv/Scripts/python -B setup_citylearn_8b.py --validate
```

```bash
.venv/Scripts/python -B setup_citylearn_ev.py --validate
```

```bash
.venv/Scripts/python -m pytest tests/ -q
```

```bash
.venv/Scripts/python ablation.py --steps 800 --seeds 0 1
```

```bash
.venv/Scripts/python study_heatpump.py --steps 1500 --seeds 0 1
```

```bash
.venv/Scripts/python train.py --schema citylearn_schemas/tx_travis_8b_ev/schema.json --isolate all --preheat --episodes 15 --seed 0
```

---

## 3. The environments

### 3.1 `tx_travis_8b` — the housing schema

| Parameter | Value |
|---|---|
| Source | NREL ResStock AMY2018–2021, Travis County, TX |
| Buildings | **8** (of 100 candidates), `central_agent = false` |
| Horizon | steps 0–8759, 3600 s/step → **one full year, hourly** |
| Devices/building | bidirectional HeatPump (cooling + heating), ElectricHeater (DHW), Battery, DHW StorageTank, PV |
| Sizing | all `autosize: true` — sized at `reset()` from historical demand + PySAM PV (~17 s reset) |
| Observations | 45 active; STEMS selects **28** by name, **+2** in heat-pump mode |
| Actions | `[dhw_storage, electrical_storage, cooling_or_heating_device]` |

### 3.2 `tx_travis_8b_ev` — housing **and** EVs

No shipped CityLearn dataset has both a thermal building model and vehicles:

| | thermal (heat pump, DHW, comfort) | EVs |
|---|---|---|
| `tx_travis_8b` | ✅ full | ❌ none |
| `citylearn_challenge_2022_phase_all_plus_evs` | ❌ **none at all** | ✅ |

The EV challenge dataset has no indoor temperature, setpoints, DHW, occupancy or cooling demand — it cannot
express the housing side. `setup_citylearn_ev.py` merges them.

| | |
|---|---|
| Buildings | 8, **6 with chargers** (partial penetration — two can only *offer* flexibility) |
| `obs_dim` | **35** = 28 base + 2 heat-pump + 5 EV |
| `action_dim` | **4** = `dhw_storage`, `electrical_storage`, `cooling_or_heating_device`, `electric_vehicle_storage_0` |
| Charger hardware | 11 kW / 7.4 kW, η 0.95, 7.2 kW discharge (**V2G-capable**) |
| Batteries | 40–75 kWh |
| EV fill times | **5.7–7.4 h** vs the hot-water tank's 1.6 h |

No data is copied: `charger_simulation` accepts an **absolute path**, because CityLearn evaluates
`os.path.join(root_directory, charger_simulation)` and `os.path.join` returns an absolute right-hand side
unchanged. `root_directory` still points at the untouched CityLearn cache, so the thermal side is byte-identical
to `tx_travis_8b`.

### 3.3 Heterogeneous-building support

CityLearn buildings need not agree. In the EV challenge dataset native action dims are **1, 2 or 3** and
observation dims **28, 35, 37, 42**, because only some buildings own a charger and charger ids are baked into
observation names.

STEMS needs one shape for every building — the GCN stacks them into a single matrix. The wrapper builds a
**canonical padded layout** plus a per-building map into native indices:

- **Observations** = base features + `5 × ev_slots` (`connected_state`, `departure_time`,
  `required_soc_departure`, `soc`, `battery_capacity`). A building with fewer bays is zero-padded, and a padded
  slot reports `connected_state = 0` — which the EV barrier *already* reads as "no vehicle". An absent charger and
  an empty bay take the same code path instead of needing a special case.
- **Actions** = canonical order, restricted to devices some building owns. Slots a building lacks are **dropped**
  before the vector reaches CityLearn, not sent as zeros: CityLearn sizes its array to the devices that exist.
  `action_presence_mask()` exposes ownership.

Verified on the raw EV dataset: 17 heterogeneous buildings → homogeneous **(38,)** observations, 2 EV slots,
7 of 17 owning a bay. On `tx_travis_8b` every map is the identity, reducing exactly to the previous behaviour.

**Missing base features fail loud.** Constructing STEMS on the EV dataset raises by default;
`allow_missing_obs=True` zero-fills, prints a banner naming every absent feature, and records them in
`absent_observations`.

### 3.4 Device physics (verified in CityLearn source)

**Storage** (battery and DHW tank share the base model):

$$\text{SOC}^{\text{init}} = \max(0,\ \text{SOC}_{t-1}\,C\,(1-\lambda))$$

$$\text{SOC}_t = \frac{1}{C}\begin{cases}\min(\text{SOC}^{\text{init}} + e\sqrt{\eta},\; C) & e \ge 0\\ \max(0,\ \text{SOC}^{\text{init}} + e/\sqrt{\eta}) & e < 0\end{cases}$$

**Heat pump** (Carnot CoP, clipped to [0, 20]):

$$\text{CoP}_{\text{heat}} = \eta\frac{T^{\text{heat}}_{\text{tgt}}+273.15}{T^{\text{heat}}_{\text{tgt}}-T_{\text{out}}},\qquad \text{CoP}_{\text{cool}} = \eta\frac{T^{\text{cool}}_{\text{tgt}}+273.15}{T_{\text{out}}-T^{\text{cool}}_{\text{tgt}}}$$

$$P_{\text{out}} = \min(|a|P_{\text{nom}}, P_{\text{nom}})\cdot\text{CoP},\qquad P_{\text{in}} = P_{\text{out}}/\text{CoP}$$

Heating vs cooling is **not** chosen by the action sign — an exogenous `hvac_mode` schedule gates it. This is why
the comfort term must be season-aware.

**DHW heater:** $P_{\text{out}} = \min(|a|P_{\text{nom}}, P_{\text{nom}})\eta$, η ≈ 0.9–0.99.

CityLearn's built-in reward is **not** used; STEMS computes its own (§6).

---

## 4. The AI architecture

Per timestep, per building:

$$o_i \xrightarrow{\text{normalize}} \underbrace{\text{GCN} \oplus \text{Transformer}}_{\text{shared encoder}} \to r_i \in \mathbb{R}^{64} \xrightarrow{\pi_\theta} a_i^{\text{raw}} \xrightarrow{\text{CBF}} a_i^{\text{safe}} \to \text{CityLearn}$$

**Building similarity graph** (geographic + functional):

$$w_{ij} = \alpha\exp\!\Big(\!-\frac{d_{ij}^2}{2\sigma_d^2}\Big) + \beta\exp\!\Big(\!-\frac{\|f_i-f_j\|^2}{2\sigma_f^2}\Big),\quad w_{ii}=0$$

**Spatial encoder** — 3-layer GCN (Kipf & Welling), $\hat A = A + I$, width 64:

$$H^{(l+1)} = \sigma\big(\hat D^{-1/2}\hat A\hat D^{-1/2} H^{(l)} W^{(l)}\big)$$

**Temporal encoder** — multi-head self-attention over a rolling T = 24 h window (4 heads, 32-d embedding,
sinusoidal positional encodings):

$$e_{\tau,\tau'} = \frac{Q_\tau K_{\tau'}^\top}{\sqrt{d_k}},\qquad z_{i,t} = \text{LayerNorm}\Big(x_{i,t} + \sum_{\tau'}\text{softmax}_{\tau'}(e_{t,\tau'})V_{\tau'}\Big)$$

**Fusion:** $r_i = W_s h_i + W_t z_i + b$.

**Per-building actor and critics** — squashed-Gaussian SAC actor, value critic $V_\phi$, K=3-head cost critic:

$$z\sim\mathcal N(\mu(r_i),\sigma(r_i)^2),\quad a=\tanh z,\quad \log\pi_\theta = \log\mathcal N(z;\mu,\sigma^2) - \sum_d\log(1-\tanh^2 z_d+\epsilon)$$

**Constrained update** (on-policy, one update per full-year episode; GAE γ=0.99, λ=0.95; auto-tuned entropy
temperature α, target entropy −|A|):

$$\hat A_i^{\text{eff}} = \hat A_i^{\text{GAE}} - \sum_k \max(\lambda_k,0)\,\hat A^c_{i,k},\qquad \mathcal L_{\text{actor}} = -\mathbb E[\hat A_i^{\text{eff}}\log\pi_\theta] + \alpha\,\mathbb E[\log\pi_\theta]$$

---

## 5. Safety: one barrier framework, four devices

### 5.1 Classical barriers (battery, power, grid)

$$h_1:\ \text{SOC}\in[0.1,\,0.9]\qquad h_2:\ |e_i|\le 80\ \text{kW}\qquad h_3:\ \textstyle\sum_i\max(e_i,0)\le 300\ \text{kW}$$

$h_1$ is decoupled per building, so it is solved **analytically** — exact, and fast enough for 8760 steps. With
$\rho_i = P_{\text{nom},i}/C_i$ from the *real* battery, robust margin $m$, anticipatory horizon $H$:

$$[\ell_i,u_i] = \Big[\text{SOC}_{\min}+m+\tfrac12\rho_i H,\ \ \text{SOC}_{\max}-m-\tfrac12\rho_i H\Big]$$

$$a^{\text{batt,safe}}_i = \begin{cases}\text{clip}\Big(a^{\text{nom}}_i,\ \frac{\ell_i-\text{SOC}_i}{\rho_i},\ \frac{u_i-\text{SOC}_i}{\rho_i}\Big) & \text{feasible}\\ +1 & \text{SOC}_i<\ell_i\ \text{(recover: charge)}\\ -1 & \text{SOC}_i>u_i\ \text{(recover: discharge)}\end{cases}$$

**Feasibility is guaranteed by construction** — a recovery action always exists, so the shield never falls back to
all-zeros.

**PID-Lagrangian dual ascent** trains the *policy* to be safe. With cost-rate error $\epsilon_k = J_{c_k} - d$
(target d = 0.05):

$$I_k \leftarrow \text{clip}(I_k + k_i\epsilon_k, 0, \lambda_{\max}),\qquad \lambda_k \leftarrow \text{clip}\big(k_p\epsilon_k + I_k + k_d\,\text{relu}(J^{(t)}_{c_k}-J^{(t-1)}_{c_k}),\,0,\,\lambda_{\max}\big)$$

### 5.2 The unification — deadline-constrained storage

Three of the four controllable devices are the **same mathematical object**:

> *A store that must reach a required SOC by a deadline, charging at a finite rate.*

| | DHW tank | Electric vehicle | Battery |
|---|---|---|---|
| Deadline | *implicit* — forecast horizon L | **declared** — `departure_time` | none (a band) |
| Requirement | forecast demand / capacity | **declared** — `required_soc_departure` | SOC_min |
| Capacity | fixed | **varies per arrival** | fixed |
| Rate ρ | `min(bound, P·η/C)` = 0.52–0.85/h | `P_charge·η/C_battery` | `P_nom/C` |
| Lead time | **1.2–1.9 h** | **5.7–7.4 h** | — |
| Slack | always 0 — urgent on sight | **positive** — deferral possible | — |

`stems/deadline.py::DeadlineStorageBarrier` implements the shared mechanism; `DHWReadinessBarrier` and
`EVReadinessBarrier` are subclasses supplying only a `requirement_fn`.

**The horizon rule.** With gap $g = s_{\text{req}} - s$ and rate ρ:

$$\text{steps\_needed} = \lceil g / \rho \rceil, \qquad \text{slack} = \text{steps\_to\_deadline} - \text{steps\_needed}$$

`slack ≤ 0` is a **latest-start trigger**: charge now or miss. DHW passes `steps_to_deadline = 0`, making every
unmet requirement immediately urgent — which reproduces the original hot-water projection exactly. EVs pass real
time-to-departure, which unlocks deferral.

**Projection** (monotone — raises the action, never lowers it, so it can only add readiness and can never veto a
policy that already charges harder):

$$a^{\text{safe}}_i = \max\Big(a^{\text{nom}}_i,\ \min\big(c_i,\ \tfrac{\max(s^{\text{req}}_i - s_i,\,0)}{\rho_i}\,c_i\big)\Big)$$

### 5.3 Hot water: forecasting the requirement

CityLearn exposes `dhw_demand` for the current step only — there is no DHW forecast observation. So the
requirement is an **online hour-of-day climatology** over already-observed steps, per (building, hour, weekend):

$$m_{i,h,w} \leftarrow m_{i,h,w} + \alpha(d_{i,t} - m_{i,h,w})$$

$$\hat D_i(L) = \sum_{k=1}^{L} m_{i,(h+k)\bmod 24,w}\big(1 + g\,\text{relu}(\bar T - \hat T_{t+k})\big)$$

with a cold-weather uplift. Nothing from the future enters: $\hat T_{t+k}$ comes from the 1/2/3-step outdoor
temperature predictions already in the observation vector. During warm-up it falls back to $L \cdot d_{i,t}$,
which never reports less demand than is happening now.

$$h_4:\quad \text{SOC}^{\text{dhw}}_i \geq \text{clip}\Big(\tfrac{\hat D_i(L)}{C_i} + m_{\text{dhw}} + \kappa\,\delta^{\text{cop}}_i,\ 0,\ 0.95\Big)$$

### 5.4 Weather, in two places

**CoP-aware power barriers.** HVAC draws $|a|P_{\text{nom}}$, and the thermal service that buys scales with CoP, so
the same comfort costs more power on a cold hour. The guard counts HVAC draw and derates the cap by the CoP
shortfall against a rated reference:

$$P^{\text{eff}}_{\text{max},i} = P_{\text{max}}\Big(1 - \text{clip}\big(1 - \tfrac{\text{CoP}_i(T_{\text{out}})}{\text{CoP}_i(T_{\text{ref}})},\,0,\,0.5\big)\Big)$$

**Cold-front anticipation.** A falling forecast means more demand *and* less efficiency:

$$\delta^{\text{cop}}_i = \text{clip}\Big(\frac{\text{CoP}_i(T_t) - \min_{k \le H}\text{CoP}_i(\hat T_{t+k})}{\text{CoP}_i(T_t)},\ 0,\ 1\Big)$$

Identically zero under steady or improving weather — it fires only when the forecast justifies it.

### 5.5 Where the guarantee breaks — coupled feasibility

Each barrier is individually feasibility-guaranteed. **That does not survive coupling.** Under a shared cap,
meeting every deadline requires

$$\sum_i \frac{(s^{\text{req}}_i - s_i)\,C_i}{\eta_i} \;\le\; P_{\text{cap}} \cdot \Delta t \cdot T$$

with T the time to the earliest binding deadline. When it fails, **the safe set is empty** — no action sequence
satisfies every deadline and the cap.

This is the crux of the difference between a battery and a fleet. A SOC bound is decoupled and always
recoverable; six deadline-locked vehicles behind one transformer are not.

- `coupled_feasibility()` detects it.
- `prioritise()` degrades by **earliest-deadline-first**, so the victim is a policy choice rather than an artifact
  of array order.
- `CBFShield.feasibility_report()` surfaces it.

Detection and honest degradation is the alternative to a guarantee that does not hold.

---

## 6. Reward and metrics

**Four-part reward**, per building per timestep:

$$r^{\text{econ}}_i = -\mu v_t e_i$$

$$r^{\text{stab}}_i = \alpha_g\Big(1-\tfrac{E_{\text{grid}}}{P_{\text{grid,max}}}\Big)^2 + \alpha_b\Big(1-\tfrac{|e_i|}{P_{\text{bldg,max}}}\Big) - \beta_r\tfrac{|e_i-e_i^{\text{prev}}|}{P_{\text{bldg,max}}}$$

$$r^{\text{comfort}}_i = -\lambda_c\, d(T^{\text{in}}_i)^2,\qquad d(T^{\text{in}}) = \begin{cases}T^{\text{in}}-T^{\text{cool}} & T^{\text{in}}>T^{\text{cool}}\\ T^{\text{heat}}-T^{\text{in}} & T^{\text{in}}<T^{\text{heat}}\\ 0 & \text{deadband}\end{cases}$$

$$r^{\text{renew}}_i = \xi\min\Big(\tfrac{\text{solar}_i}{\max(\text{load}_i,\epsilon)},1\Big)$$

The **dual-setpoint deadband** is a correctness fix over the paper's single cooling setpoint: without it the agent
is penalised for being comfortably warm in winter. Any isolation mode that drives the HVAC actuator auto-enables
it.

**Metrics** (paper Table I), all computed from **observed post-step state** — never the CBF's internal model, so
the safety number cannot be gamed by the shield's own assumptions:

1. Cost — $\sum_t\sum_i \max(e_{i,t},0)v_t$ (imports only)
2. Emission — $\sum_t\sum_i \max(e_{i,t},0)\kappa_t$
3. Avg. daily peak — $\frac1D\sum_d \max_{t\in d}\sum_i\max(e_{i,t},0)$
4. Electricity consumption
5. Ramping rate — $\frac1{T-1}\sum_t\big|\sum_i e_{i,t}-\sum_i e_{i,t-1}\big|$
6. Discomfort rate — fraction of occupied steps outside the band
7. **Safety violation rate** — fraction of (t, building) pairs violating any barrier

Plus DHW: `dhw_readiness_rate`, `dhw_demand_covered_rate`, `dhw_deficit_kwh`, `dhw_soc_mean`.

Metrics 1–5 are normalised to RuleBased = 1.0; 6–7 absolute. Violations are further decomposed into **avoidable**
vs **unavoidable** — a diagnostic beyond the paper's headline metric.

---

## 7. Device isolation

A **name-keyed** registry maps a mode to action indices, resolved by name (exact, then prefix, so per-bay charger
names work):

```
ACTION_GROUPS = {
  dhw:        [dhw_storage]
  heatpump:   [cooling_or_heating_device]
  thermal:    [dhw_storage, cooling_or_heating_device]
  ev:         [electric_vehicle_storage]
  ev+thermal: [dhw_storage, cooling_or_heating_device, electric_vehicle_storage]
  battery+ev: [electrical_storage, electric_vehicle_storage]
  all:        [all four]
}
```

Two effects: **action masking** ($a^{\text{final}} = a^{\text{safe}} \odot \mathbf 1_{\mathcal C}$ — for storage,
$a=0$ is genuinely "do nothing", so freezing is physically meaningful) and **CBF adaptation**
(`enforce_soc = (elec_idx ∈ C)`, so the SOC barrier is skipped rather than constraining an actuator the agent
cannot move).

Asking for a device the schema lacks **raises**. On `tx_travis_8b_ev` all modes resolve:
`thermal=[0,2]`, `heatpump=[2]`, `ev=[3]`, `ev+thermal=[0,2,3]`, `all=[0,1,2,3]`.

---

## 8. Results

### 8.1 Battery calibration — the root cause

The CBF hard-coded `dSOC = 0.1`. Real batteries move $\rho_i = P_{\text{nom},i}/C_i \in [0.178, 0.533]$ — varying
**3× across the 8 buildings**.

**Safety-trick ablation** (real CityLearn, 800 steps × 2 seeds):

| Safety configuration | Violation rate | Cost |
|---|---:|---:|
| No CBF (raw policy) | 0.7195 | 7676 |
| CBF, coarse uniform `dSOC=0.1` (the bug) | 0.0945 | 7777 |
| CBF + real per-building calibration | 0.0110 | 7703 |
| &nbsp;&nbsp;+ robust margins | **0.0000** | 7705 |
| &nbsp;&nbsp;+ anticipatory buffer | **0.0000** | 7662 |
| **FULL (all four tricks)** | **0.0000** | **7649** |
| *Paper STEMS (reported)* | *0.0560* | — |

Calibration precision dominates: the coarse estimate leaves 9.45% violations *even with the shield on* — worse
than the paper. Real calibration alone (0.0110) already beats it. The full configuration is both safest and
cheapest, because power limits never bind on this dataset so the SOC barrier is the only one that matters.

### 8.2 Converged full-year run

`--isolate thermal`, seed 0, 15 × 8760-step episodes. Reward climbs 205 → **671** over episodes 2–15 while the
violation rate stays **0.0000 in every episode** — safety holds *throughout* learning, not just at convergence.

| Metric | RuleBased | STEMS | STEMS/RB |
|---|---:|---:|---:|
| cost | 34008.88 | 15219.87 | **0.448** |
| emission | 21071.71 | 8688.99 | **0.412** |
| avg. daily peak | 36.78 | 16.47 | **0.448** |
| electricity consumption | 142816.61 | 55219.75 | **0.387** |
| ramping rate | 6.368 | 4.804 | **0.754** |
| discomfort rate | 0.0001 | 0.0001 | — |
| **safety violation rate** | **0.6465** | **0.0000** | — |

⚠️ **This run predates the DHW no-op fix (§9.1), so it is a valid *heat-pump-only* result, not a joint
DHW + heat-pump one.** A genuine joint run is open work.

### 8.3 Anticipatory pre-heating study

1500 steps × 2 seeds, `--isolate thermal`, heating season. Nominal policy is a **price-aware reactive**
controller — a strawman already trying to be economical, just not anticipatory — so what the barrier adds is
interpretable as the value of *anticipation specifically*. Every row is graded by the **same** fixed evaluation
barrier, a separate instance from the one being ablated.

| Thermal configuration | Readiness | Covered | Deficit (kWh) | Cost | DHW $/kWh |
|---|---:|---:|---:|---:|---:|
| No pre-heat (reactive) | 0.907 | 0.939 | 1026.5 | 3021.6 | 0.2200 |
| + CoP-aware power guard | 0.907 | 0.939 | 1026.5 | 3021.6 | 0.2200 |
| Pre-heat L=1 | 0.928 | 0.950 | 661.7 | 3025.9 | 0.2300 |
| Pre-heat L=2 | 0.961 | 0.958 | 90.6 | 3033.1 | 0.2411 |
| Pre-heat L=3 | 0.989 | 0.968 | 33.8 | 3039.0 | 0.2936 |
| + weather anticipation (L=2) | 0.982 | 0.964 | 67.3 | 3035.6 | 0.2517 |
| **FULL (L=2, weather + margin)** | **0.991** | **0.970** | **47.4** | 3038.5 | 0.2571 |

**1. The horizon is set by physics.** Deficit collapses **86%** between L=1 and L=2 (661.7 → 90.6 kWh), then
improves only slowly — exactly where the measured 1.60 h mean time-to-heat predicts the discontinuity.

**2. Readiness is _not_ free — this contradicts §8.1.** DHW electricity rises 196.2 → 238.4 kWh and price paid
per kWh 0.2200 → 0.2571, lifting cost 0.6%. The mechanism differs: the battery barrier only ever *restricted* an
action the policy already wanted, whereas h₄ *commands* heating at hours the policy would skip — and since the
requirement follows demand rather than price, those hours are more expensive than a price-follower's. Anticipatory
readiness is a service-quality purchase, not a free lunch.

**3. Weather anticipation beats a longer horizon.** FULL exceeds L=3 on readiness (0.991 vs 0.989) and coverage
while paying **12% less per kWh** (0.2571 vs 0.2936). Extending the horizon raises the requirement at *every*
hour; conditioning on the forecast raises it only when justified.

**4. The CoP-aware guard does nothing here — reported, not hidden.** Rows 1–2 are identical to every digit. Loads
are 10–40 kW against an 80 kW cap, so the power barriers never bind and reserving headroom inside a cap never
approached cannot change any action.

**On the seed spread.** Per-seed std is exactly 0.000 throughout — that is *determinism, not stability*. The
reactive policy is a deterministic function of the observation and the scenario seed is fixed in the schema.

### 8.4 Cold-snap stress

Designing this required **two corrections to my own first attempt**:

1. **A constant temperature offset cannot exercise a cold-front term** — it keys on a *change*, not a level.
   Fixed with a forecast-only ramp (`set_weather_front`).
2. **The cap was far above the load.** Under thermal isolation mean per-building draw is **0.88 kW**; a 12 kW cap
   is no more binding than 80 kW. It had to come down to **2 kW**.

Both perturb the *observation stream*, not CityLearn's physics — this probes whether the anticipation logic
responds to a weather signal; it is not a claim about true energy use in a cold snap.

| Thermal configuration | Readiness | Deficit (kWh) | Cost | DHW $/kWh | Violation |
|---|---:|---:|---:|---:|---:|
| No pre-heat (reactive) | 0.895 | 1163.5 | 3021.6 | 0.2200 | 0.2402 |
| + CoP-aware power guard | 0.895 | 1163.5 | **3008.2** | 0.2200 | 0.2388 |
| Pre-heat L=1 | 0.919 | 734.7 | 3013.1 | 0.2307 | 0.2390 |
| Pre-heat L=2 | 0.961 | 103.2 | 3021.4 | 0.2459 | 0.2390 |
| Pre-heat L=3 | 0.991 | 33.2 | 3028.9 | 0.3127 | 0.2402 |
| + weather anticipation (L=2) | 0.986 | 60.2 | 3024.9 | 0.2550 | 0.2390 |
| **FULL (L=2, weather + margin)** | **0.991** | **43.5** | 3028.3 | 0.2591 | 0.2390 |

**The CoP-aware guard now acts — but cannot repair the constraint.** Cost falls 0.44%, making it the cheapest
configuration, because it trims HVAC draw hardest when CoP is worst. Its effect on violations is negligible, and
that quantifies cleanly: under thermal isolation the shield commands **366.5 of 10597.3 kWh — 3.5% of total
consumption**. The other 96.5% is non-shiftable load, and with the battery frozen no thermal action can bring a
building under a 2 kW cap its baseline already exceeds. The ~24% violation rate is essentially *unavoidable*.
**A shield can only be as effective as the fraction of load it commands** — an argument for re-enabling the
battery, not for tuning the thermal guard harder.

**Weather anticipation is worth more when there is weather to anticipate.** The cold-front term at L=2 lifts
readiness 0.961 → 0.986 and cuts deficit **42%**; FULL matches L=3's readiness while paying **17% less per kWh**
(vs 12% under nominal weather).

### 8.5 Coupling sweep — a deadline-locked EV fleet under a shared cap

The experiment §5.5 was built for. Six vehicles, 55.2 kW aggregate charger capacity, on the merged schema.
Measured first: the **non-vehicle** building load alone peaks at **39.6 kW**; with every vehicle charging on
demand the site peaks at **71.5 kW**, mean 4.2 connected simultaneously. The shared cap is swept 80 → 15 kW.

Three arms, all sharing one deliberately uncoordinated nominal controller (each house charges whatever is
plugged in — what a transformer sees when a street electrifies):

- **`independent`** — each barrier projects alone; correct per device, blind to the connection.
- **`proportional`** — cap enforced, every request scaled equally.
- **`edf`** — cap enforced, allocated earliest-deadline-first.

The last two enforce an *identical total*, so any difference is the priority rule alone. All three necessarily
coincide where the cap is slack — which is exactly why a sweep, not a single operating point, is the right
instrument.

**Service metric:** a *missed departure* is a vehicle disconnecting below its required SOC. Unlike a SOC
excursion it cannot be repaired by any later action.

| Cap (kW) | indep miss | indep viol | prop miss | prop viol | **edf miss** | edf viol | EDF gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 3 | 0.000 | 3 | 0.000 | 4 | 0.000 | −33% |
| 60 | 3 | 0.013 | 1 | 0.000 | 1 | 0.000 | 0% |
| 45 | 3 | 0.065 | 6 | 0.001 | 5 | 0.000 | +17% |
| 35 | 3 | 0.113 | 20 | 0.009 | **11** | 0.011 | **+45%** |
| 30 | 3 | 0.142 | 58 | 0.018 | **34** | 0.023 | **+41%** |
| 25 | 3 | 0.179 | 115 | 0.045 | **66** | 0.047 | **+43%** |
| 20 | 3 | 0.219 | 170 | 0.099 | **115** | 0.095 | **+32%** |
| 15 | 3 | 0.273 | 214 | 0.201 | 179 | 0.177 | +16% |

*(1500 steps, 298 departures, seed 0. Figure: `plots/coupling_sweep.png`.)*

**1. Enforcement converts a breach into a scheduling problem.** `independent` never limits anything: 3 missed
departures at *every* cap, but it exceeds the cap in up to 27% of steps and peaks at 73.4 kW no matter what the
cap says. Enforcing removes nearly all of that — at 35 kW violations fall 0.113 → 0.009 — and converts it into
missed departures instead. That trade *is* the coupled-feasibility condition: when the requirement set cannot be
met, something gives, and the only choice is *what*.

**2. The value of the priority rule is non-monotone — an inverted U.** This is the finding:

- 80 / 60 kW: cap slack, rule worth **nothing**
- 35 → 20 kW: avoids **32–45%** of proportional's missed departures, peaking at **+45%** (11 vs 20) at 35 kW
- 15 kW: falls back to **+16%**

The decline at the tight end matters as much as the peak. Once the cap is far below what the fleet needs, most
deadlines are unreachable under *any* allocation and a priority rule can only reorder failures. **Coordination is
worthless where the constraint doesn't bind, and loses leverage again where the problem is infeasible for
everyone.** A single operating point could have supported almost any conclusion.

**3. Negative — a floor set by uncontrollable load.** Every enforcing arm peaks at exactly 39.6 kW for all caps
≤ 30 kW, and residual violations rise regardless of rule. That is the non-vehicle building load, which the shield
does not command. Below it, no vehicle action can bring the site under the cap. Same bound as §8.4, reached from
a different direction. **Planned:** repeat with battery and thermal actuators also under control.

**4. Negative — the headroom estimate is conservative.** At 80 kW all three arms should be identical (realised
peak 73.4 < 80), yet `edf` records 4 misses against 3. Headroom is computed from the *previously observed* net
import, which already contains that step's charging, so the allocator occasionally fires when the cap is actually
slack. Small, and conservative in the safe direction, but an artifact rather than a property. **Planned:** estimate
next-step non-controllable load separately from controllable draw.

**Scope — read this before citing it.** This characterises the *coupling* and compares **allocation rules inside
the safety layer**. It is **not** a comparison of learned coordination architectures: nothing is trained here, and
the nominal controller is fixed so the shield's contribution can be isolated. It establishes *where* the shared
constraint binds (≈20–45 kW on this fleet), which is a prerequisite for that comparison — a learned coordinator
evaluated at 80 kW would have nothing to coordinate about — but it is not a substitute for it.

---

## 9. Defects found — in the simulator and in our own work

### 9.1 CityLearn: the DHW action was a silent no-op

`Building.update_dhw_storage` (2.6.0b1, `building.py:1562`) computes

$$e = a^{\text{dhw}} \cdot C_{\textit{heating}\_\text{storage}} \cdot \Delta t$$

using the **heating** tank's capacity where it must use the **DHW** tank's. This schema has no heating tank, so
that capacity is `0.0` and **every DHW action resolved to zero energy**. Verified: commanding `a = 0.8` for eight
consecutive steps left `dhw_storage_soc` at exactly 0.0.

Patched loudly per building — only where the defect is present — with a banner and a `citylearn_patches` metadata
stamp, and an opt-out. **This invalidates device-level readings in every pre-fix `dhw` / `thermal` run.**

This is worth upstreaming: every published CityLearn result using `dhw_storage` on a schema without a heating tank
is affected.

### 9.2 My own error: EV departure treated as a clock hour

`ChargerSimulation` documents `electric_vehicle_departure_time` as "number of time steps until the EV departs",
and the shipped `charger_1_1.csv` counts **12, 11, 10 … 0**. My first `EVReadinessBarrier` did clock arithmetic on
it (`hours_until(hour, departure_time)`) — silently wrong deadlines on every bay. Now used verbatim via
`steps_to_departure`. `required_soc_departure` is likewise a **percentage**, divided by 100 on load.

### 9.3 My own error: countdown truncated one step early

CityLearn's CSVs count … 2, 1, **0** while still parked — the departure hour itself is a final charging
opportunity. My generator stopped at 1, silently deleting it.

### 9.4 Metrics artifact: SOC scored on a frozen battery

`soc_violation_rate` pinned at 1.0 under isolation because the metrics calculator scored the *frozen, unmanaged*
battery against the SOC band. `count_soc` is now derived from `control_indices`. Confirmed: `--isolate thermal`
went from 1.0 → 0.0.

**All four are pinned by regression tests.**

---

## 10. What is real and what is synthetic

| Component | Status |
|---|---|
| Building thermal models, loads, weather, pricing, carbon | **Real** — NREL ResStock / CityLearn, unmodified |
| Battery, DHW tank, heat pump, charger parameters | **Real** — read from live device objects, never assumed |
| DHW demand forecast | **Derived** — causal online climatology from observed steps only, no look-ahead |
| **EV arrival / departure / required SOC** | **SYNTHETIC** — ResStock has no vehicle data |
| Cold-snap stress perturbation | **Synthetic, observation-level** — probes the anticipation logic, not true cold-snap energy use |

The EV commuter model: weekday departure ~07:30 / return ~17:30; weekend 09:30 / 15:30; 12% / 45% stay-home;
required SOC U(0.70, 0.90); arrival SOC U(0.25, 0.55) decaying with time away. Seeded and deterministic.
**Every result on `tx_travis_8b_ev` must carry this caveat.**

---

## 11. Engineering state

### Modules

| File | Lines | Role |
|---|---:|---|
| `stems/environment.py` | 1051 | fail-loud wrapper, canonical padded layout, device info, simulator patches |
| `stems/cbf.py` | 393 | analytic CBF shield, deadline barriers, CoP power guard, feasibility report |
| `stems/deadline.py` | 353 | `DeadlineStorageBarrier`, `coupled_feasibility`, `prioritise` |
| `stems/thermal.py` | 387 | DHW dynamics, causal forecaster, CoP model, h₄ |
| `stems/ev.py` | 197 | EV deadline barrier, countdown semantics |
| `stems/metrics.py` | 291 | Table I + DHW readiness, avoidable/unavoidable split |
| `setup_citylearn_ev.py` | 406 | merged housing + EV schema generator |
| `study_heatpump.py` | 369 | pre-heating + weather study, with `--stress` |

Also: `config.py`, `graph.py`, `encoder.py`, `agent.py`, `reward.py`, `utils.py`, `baselines.py`, `train.py`,
`evaluate.py`, `ablation.py`, `extensions/hierarchical.py` (50-building cluster variant).

### Tests — 85/85 passing

| Suite | Count | Covers |
|---|---:|---|
| `test_core.py` | 5 | pipeline correctness |
| `test_real_env_validation.py` | 6 | standing real-env checks — run before any training |
| `test_thermal.py` | 17 | DHW dynamics, forecaster, CoP, h₄, real-env calibration |
| `test_deadline.py` | 22 | generic barrier, EV semantics, coupled feasibility, **legacy-equivalence diff** |
| `test_ev_real.py` | 5 | EV barrier against the raw CityLearn EV dataset |
| `test_env_widening.py` | 16 | heterogeneous padding stays inert; homogeneous path unchanged |
| `test_ev_schema.py` | 14 | merged schema: no thermal feature lost, EV side live, encoding pinned |

Studies: `ablation.py` (§8.1), `study_heatpump.py` (§8.3, `--stress` §8.4), `study_coupling.py` (§8.5).

### Regression protection for heat-pump-only work

Heat-pump-only results are benchmarked in separate doctoral work, so every refactor is verified twice:

- **Op-level** — the pre-refactor projection is kept as a frozen reference function and diffed over 200 randomised
  states × 3 horizons × 2 weather settings.
- **End-to-end** — the full §8.3 study is re-run and diffed against the saved pre-refactor JSON after *every*
  structural change.

Across three rounds (deadline refactor → environment widening → merged schema) the result is unchanged:
**max relative difference 7.359e-09**, always on `avg_daily_peak` (the most summation-order-sensitive metric).

That residual is a **1-ulp float32 effect** from summation order — the legacy code computed
`demand/C + margin + weather`, the base class applies its margin last. Below float32 resolution at these
magnitudes and an order of magnitude under the 1e-6 guard band already in the readiness comparison. The test says
"to float32 precision", **not** "identical", and the tolerance is set there deliberately.

### Runtime

Always `.venv\Scripts\python.exe` (Python 3.12.6). Do **not** use `py -3.11` — the Store app-container sandbox
breaks torch's native DLL loading (`WinError 1114`, `c10.dll`). Pins in `requirements.lock.txt`.

**Stale / unmaintained:** `visualize.py`, `benchmark_sac.py`, `notebook.ipynb` (written against the old API).

---

## 12. Open work

Ordered by what most changes the strength of the claims.

1. **Compare learned coordination across the measured regime.** §8.5 swept the shared cap and found the
   useful operating band (≈20–45 kW on this fleet) and the shape of the effect — non-monotone, peaking at +45%,
   not the monotone relationship originally hypothesised. It compared allocation rules inside the shield; it
   trained nothing. *Planned:* compare centralised-critic against fully decentralised training across that same
   sweep. Evaluating such a comparison at a slack cap would measure nothing, which is why the sweep came first.

2. **Calibrate and run the dead baselines.** `stems/baselines.py` already contains working SAC, DMAPPO, MPC,
   MADDPG, MARLISA, MADCQ and MetaEMS — ~1100 lines that `evaluate.py` never calls; it compares against
   RuleBased only. ⚠️ **`MPCAgent.__init__` has `eta = 0.1`** — the *same* mis-calibration criticised in §8.1.
   Calibrate every baseline from `battery_info()` / `dhw_info()` before publishing a single comparison, or you are
   beating a strawman by exactly the mechanism you criticised.

3. **Re-run the thermal isolation modes post-DHW-fix.** §8.2 and the isolation table stand only as
   heat-pump-only. A genuine joint DHW + heat-pump run is needed.

4. **Real error bars.** Every ± is currently 0.000 because everything is deterministic. Vary stochastic policy
   seeds, weather years, and building subsets; report CIs and paired tests.

5. **Price-aware pre-heating.** §8.3 finding 2: readiness costs 0.6% because h₄ heats as late as the constraint
   allows, ignoring price. Choosing *when within the horizon* should recover most of it — and EVs have the
   positive slack that hot water lacks, so this is where it pays.

6. **Cold climate.** `quebec_neighborhood_*` and `vt_chittenden_county_neighborhood` are heating-dominated.
   Every heat-pump claim is currently untested where heat pumps matter most — and it is where the CoP mechanisms
   stop being inert.

7. **Upstream the DHW bug.** A real defect in a widely-used benchmark that silently disables an actuator. An hour
   of work, a genuine contribution, and a citable line.

8. **Production hygiene.** No `pyproject.toml`, CI, container or license file. The study takes ~40 min for 21k
   steps — the per-step Python forecaster loop is the bottleneck. Property-based invariant tests would beat
   fixed cases.

9. **V2G.** The merged schema is already V2G-capable (7.2 kW discharge) and unused. A vehicle that can discharge
   turns the EV from pure constraint driver into a flexibility provider — which changes the coupling story.

---

*Every number in this document was measured on the real simulator. Where something is synthetic, derived or
invalidated, it is marked as such in place.*
