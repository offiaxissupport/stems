# STEMS: what we are doing, how we got here, and where we stand

*Status on 5 October 2026, just after midnight. Branch `rebuild/audited-pipeline` is pushed to GitHub. A second
branch with tonight's fixes (`fix/learner-leaning`) is local and has experiments running on it; section 7 says
what is finished and what is still running. The technical report with every table is
[REPORT_2026-10.md](REPORT_2026-10.md); this document is the story around it.*

---

## 1. In five lines

1. **Goal:** a controller for a neighbourhood of houses (battery, heat pump, hot-water tank, electric cars)
   that saves money **without breaking limits**: battery limits, comfort, the cars' departure deadlines, and
   one shared limit on how much power the street may draw.
2. **What we found first:** almost nothing in the old results could be trusted. The simulator had three bugs,
   the measurements were mis-timed, and the learning agent had never actually learned.
3. **What we rebuilt:** an audited pipeline where every number is checked against the simulator itself, and
   every run records which code produced it.
4. **What we now know:** an exact safety layer gives zero battery violations at almost no cost; learning beats
   a good rule in summer and loses in winter; for the cars, joint planning matters exactly when charging is
   postponed; and a learning agent placed behind a safety shield stops doing the job itself and lets the
   shield do it.
5. **Where we are tonight:** fixing that last point, and re-running the heat-pump-only benchmark (Manal's) so
   that its comfort numbers are valid.

---

## 2. The objective

### 2.1 Where the project comes from

The starting point is a published method: **STEMS** (Zhang, Wu, Zinflou, Boulet, 2025, IEEE Internet of Things
Journal, arXiv:2510.14112). It is multi-agent reinforcement learning for buildings on the CityLearn simulator:
each building is an agent, a graph network and a transformer share information between them, and a safety
filter (a "control barrier function") keeps the batteries inside their limits. The paper reports a safety
violation rate of 5.6%.

The plan, set on 24 June 2026, had three steps:

1. reproduce the paper honestly on real CityLearn;
2. beat it on the number that matters for safety, the violation rate;
3. add heat pumps and hot water.

On 2 September the thesis was re-centred: **the main subject is safe multi-agent control for housing plus
electric vehicles**. The heat pump is one device inside it. The heat-pump-only model stays important because
**Manal's PhD benchmarks it**, so it must remain reproducible.

### 2.2 The claim we want to be able to defend

"Safety" in most papers means: each device stays inside its own limits. That is easy: a correct per-device
filter always achieves it. The interesting case is when **several devices share one limit**. Six cars behind
one 40 kW connection can each be chargeable in time and still not be chargeable *together*. Then:

- a per-device safety filter is not enough, because the safe set of the group can be empty even when each
  car's own safe set is not;
- something has to plan the group, and say honestly when the plan is impossible;
- and a learning agent has to live with that planner.

That is the contribution we are building toward: **coupled constraints (a shared cap plus deadlines), a shield
that handles them correctly, and an honest account of what learning adds on top**.

### 2.3 Rules we set for ourselves

These were decided early and have not changed:

- **Real simulator, fail loud.** No silent fallback to a fake environment, no safety filter that quietly
  returns zeros. If something cannot be done the code stops and says so.
- **Rigorous, documented code.** Every correction of the simulator announces itself, is recorded in the run,
  can be switched off, and has a test.
- **Honest claims.** Negative results are reported. When we were wrong we say so and keep the correction
  visible (there are several in this document).

---

## 3. The system in plain words

| Piece | What it is | Where |
|---|---|---|
| Simulator | CityLearn 2.6.0b1. Eight real houses from Travis County, Texas (NREL ResStock, 2018 weather). Each has a battery, solar panels, a heat pump for heating and cooling, and an electric hot-water tank. Six have a car charger. | `citylearn_schemas/`, `stems/environment.py` |
| The cars | **Synthetic** commuter schedules (the housing data has no cars): arrive around 17:30, leave around 07:30, each with a required charge at departure. The charger and battery models are CityLearn's real ones. | `setup_citylearn_ev.py` |
| Tariff | 0.22 per kWh, 0.54 between 16:00 and 21:00. Exporting earns nothing. | schema |
| Rule controller | A fixed schedule: charge battery and tank before the evening peak, discharge the battery to cover the house during it. Cars charge when plugged in. | `stems/baselines.py` |
| Learning controller | PPO-Lagrangian with a shared graph + transformer encoder. One policy shared by all houses. | `stems/agent.py` |
| Battery barrier | Keeps each battery between 10% and 90%. It inverts the simulator's own battery equations, so it needs no safety margin. | `stems/battery.py`, `stems/cbf.py` |
| Thermostat | The controller does not command heat-pump power directly. It moves the set point by at most ±1.5 °C and an inner loop tracks it. | `stems/environment.py` |
| Cap shield | Plans all cars together under the shared cap (a linear programme), respects deadlines, and also holds the house batteries and tanks under the same cap. | `stems/fleet.py` |
| Experiments | Scenarios, arms, one JSON record per run with the code hash, aggregation with proper statistics. | `experiments/` |

How a control step works: the controller proposes an action for every house → the battery barrier corrects
the battery part → the cap shield decides the cars' charging and trims storage charging if the cap would be
broken → the simulator runs one hour → costs, comfort and violations are measured.

---

## 4. Timeline

| When | What happened | What we learned |
|---|---|---|
| 24 June | Started. Found the old code was a fake experiment: it silently swapped to a synthetic environment and the safety filter returned zeros when it failed. The barrier assumed every battery moves 10% per step; real ones move 18–53%. | Calibrating the barrier to the real batteries took violations from ~70% to ~0. This became the first result. |
| 30 June | Device isolation: train on the heat pump (and hot water) only, battery frozen. | The basis of the heat-pump-only benchmark. |
| 1–3 September | Hot-water action found dead in CityLearn (the first simulator bug we met). Pre-heating study. Thesis re-centred on housing + EVs. Built the housing-and-EV dataset. First cap sweep. Tried to compare centralised vs decentralised learning. | The comparison was impossible: **the agent was not learning at all**. Also: "a shield strong enough to guarantee the objective removes the learning signal for it" — the first sign of tonight's problem. |
| 14–17 September | Built the clean experiment harness. Found that our code had wrong assumptions about the data (hour runs 1–24, forecasts are 6/12/24 h ahead, the sign of the heat-pump action selects heating or cooling). | Check every convention against the data files, never assume. |
| 3 October | You asked to run the big grid "only when everything is correct, mathematically correct and explainable". We audited first, with four independent reviews. | Three simulator bugs, wrong timing in the measurements, wrong statistics, and the reason the agent never learned. All fixed before running anything. |
| 3–4 October | Grid 1: 80 runs, four seasons, safety layer × controller. | Section 6.1–6.2. |
| 4 October | EV phase: exact car model, the joint shield, a four-stage study (468 runs), the literature search, the report. Then full controllers under a binding cap. | Section 6.3–6.5. The learner "leans on the shield". |
| 4–5 October (now) | Branch pushed. Fixing the leaning. Re-running the heat-pump-only benchmark on the corrected simulator. | Section 5.10–5.11 and 7. |

---

## 5. The problems, and how we reasoned about each

This is the part you asked for: not only what we did, but how we thought.

### 5.1 "Is this even a real experiment?"

**Problem.** Old results looked fine but came from code that hid its failures.

**How we thought.** A result is worth nothing if the code can silently do something else than what the
paper says. So before any science: make failure visible.

**What we did.** Removed every silent fallback. Every run now stores a hash of the code that produced it;
results from different code cannot be mixed; a results table refuses to print if any run in it failed.

**Why the last rule exists.** On 4 October our first EV table showed our method winning clearly. It was
false: the hard winter runs had crashed and the table averaged only the easy season. We caught it, fixed the
crash, and made the report refuse such a table for ever.

### 5.2 The simulator had three bugs, all from one change

**How we found them.** By testing each actuator with a known input. Full cooling in winter made the
simulated house −15 °C while the observation still said 17–23 °C.

| Bug | Effect |
|---|---|
| The indoor-temperature observation is the dataset's *uncontrolled* temperature, not the simulated one | No controller, reward or comfort metric can see what the heat pump does |
| The hot-water storage action is scaled by the wrong tank's capacity (zero here) | The action does nothing |
| Each thermal device's load is counted twice in the first hour of an episode | Crashes, or silently clips storage charging |

All three come from one reordering of the simulator's step in CityLearn 2.4.0 (July 2025). Versions 2.1–2.3.1
(including the 2023 challenge) are fine. We patch them loudly, test them, reproduced them on plain CityLearn
with its own dataset, and wrote the upstream bug report (`docs/citylearn_v2.4_defects.md`, **not filed yet**).

### 5.3 The measurements were mis-timed, and the statistics over-confident

**Problem.** After a step, CityLearn's observation contains the *next* hour's price but *this* hour's
consumption. Our metrics multiplied them. A battery policy timed perfectly to the peak lost 88% of its real
advantage on a test.

**How we thought.** Decide once which hour every observation describes, by measuring it against the
simulator's own series, then use that rule everywhere.

**Result.** Cost, emissions, consumption, peak and discomfort now equal values recomputed from CityLearn's
own data (cost 682.54 = 682.54). Statistics: a *scenario* (season × set of houses) is the unit of replication,
not a seed; a small number of pre-declared comparisons; a correction for multiple testing.

### 5.4 The learner had never learned

**Problem.** Training curves were flat; every arm gave identical results.

**How we thought.** Test the learner on a problem with a known answer before trusting it on the real one.
On a toy task with optimum 0.6 it reached 0.27.

**Causes found.** An exploration bonus about a hundred times too strong pulled every action to zero; the
reference probabilities were computed for a policy that never acted; the end of a time window was treated as
the end of the world; the constraint multiplier never grew.

**Result.** Rebuilt as a standard PPO-Lagrangian. Known-answer task: 0.598. One more decision came from the
literature and from our own September insight: **the shield is treated as part of the environment**, and the
policy is trained on its own action, not on the corrected one.

### 5.5 Comfort

**Problem.** Once the temperature observation was real, comfort was terrible: 66–79% of hours uncomfortable.

**How we thought.** We measured the plant: one unit of heat-pump action moves the simulated temperature
8–12 °C in an hour. No policy can hold a temperature by commanding power directly with that gain. A building
controller does not do that either: it sets a set point and a thermostat does the rest.

**What we did.** The action became a set-point offset (±1.5 °C) tracked by a slow integral loop. Discomfort
fell to 1–5% for every controller.

**What we tried and rejected.** Two "improvements" to the loop made comfort worse and were removed. A fixed
"eco" shift of the set point as a no-learning reference cost 47% more: in this data the heating and cooling
set points are equal, so the loop fights itself. That limitation is still there (see 5.7).

### 5.6 Battery safety

**How we thought.** A barrier needs to know what an action will do. The paper's barrier assumed a linear
battery. The real one loses energy, charges slower when full, and leaks.

**What we did.** Wrote CityLearn's battery equations as a model and inverted it numerically. It matches the
simulator to four decimal places.

**Result (grid 1, eight scenarios).** Zero violations in every scenario, for a cost of +0.3–0.5%. With the
paper's uniform rate: 3.9% violations. With the learning penalty alone and no barrier: 9.1%.

### 5.7 Does learning beat a good rule?

**How we thought.** The fair opponent is not "no control" but a sensible rule. Our first rule was so bad it
cost more than doing nothing (it gave its battery energy away, since export earns nothing). We fixed the rule
first.

**Result.** Summer and spring: learning is 12–41% cheaper. Winter: 10–25% dearer. Pooled over eight
scenarios: not statistically significant.

**Why — we decomposed the saving instead of celebrating it.** Where learning wins, it mostly imports less
energy, by moving the set point to where the house already is; that compensates the thermostat's weakness
from 5.5 more than it is clever trading. In summer it also cycles the battery twice as much (real arbitrage).
In winter it hardly uses the battery and loses.

**An idea from the literature, tested.** Let the agent learn only a bounded *correction* on top of the rule
(a "residual policy"). It halved the winter loss — and halved the summer gain. It bounds both sides.

### 5.8 The cars under a shared cap

**First, the facts we had to establish**, each of which had been wrong at some point:

- the charger cannot draw less than 1.4 kW, the battery takes less power as it fills and loses 0.3% per hour;
- the charge delivered in a car's last hour is never visible in an observation, so departures must be scored
  from the simulator's own record;
- the requirement holds *at departure*: a car exactly at its target one hour early leaves slightly short;
- an allocation is power drawn from the grid, not a command to the charger.

**The question.** Which way of sharing the cap keeps the deadlines? We compared eight: no shield; one barrier
per car; a fixed split of the cap; proportional; earliest deadline first; least laxity first; a smoothed
version from the literature; and our joint linear programme.

**What we found (468 runs, none failed).**

- A barrier per car meets every deadline and breaks the cap by 78–784 kWh. It is not a shield for this problem.
- A fixed split wastes capacity a neighbour is not using.
- **If cars charge as soon as they arrive, the simple sorting rules are as good as the joint programme**
  (within 3%). The programme only adds an exact cap and a fairer spread of unavoidable shortfall.
- **If charging is postponed, only the joint programme works**: at a 40 kW cap the per-car rule leaves 248 kWh
  undelivered, the programme 31. Postponing is 21% cheaper at a 50 kW cap.
- Without perfect foresight, postponing is not free: it needs a time reserve and still misses departures.
- A house battery discharged in the evening halves the cars' shortfall at 40 kW.

**What we got wrong on the way and corrected.** The first table (crashed runs averaged away). Rules that were
handicapped by asking for power a full battery cannot take. Misses "by a hair" caused by the 0.3% loss. A
count of "hours over the cap" that counted a few watts of rounding as violations.

### 5.9 Whole controllers under a binding cap

**How we thought.** The study above tests the shield alone. The real test is a controller with batteries,
tanks, heat pumps and cars together. A one-week test broke the cap by 20 kWh in two ways the study could not
show: the shield assumed the house load would continue while its own controller was about to switch eight
batteries off; and the rule's own battery and tank charging went over the cap with no car plugged in.

**The idea.** Both are the same mistake: the cap was treated as the cars' problem. So:

1. the shield uses the battery and tank actions that are already decided, through exact models of both
   (the tank model matches the simulator to 4e-7 kWh);
2. cars are planned first; storage charging gets what is left — a deadline outranks arbitrage;
3. the forecast starts with history instead of blind.

**Result.** Same week: 20 kWh over the cap → 0. In the full grid: 403–503 kWh without shields, 3–16 with.

### 5.10 The learner leans on the shield — tonight's problem

**What we measured.** The learned policy asked for **0.2–3% of the energy the cars received**. It simply does
not charge the cars. The shield charges them at the last moment that is still feasible — which is the middle
of the night, the cheapest time. So the policy looked 16–35% cheaper than the rule, and missed 9–14 of 67
winter departures where the rule misses 5.

**How we thought about it.** Three separate questions:

1. *Is the shield's last moment actually safe?* No. It planned the coming hours with the forecast "same hour
   yesterday" but only left room for the error of a one-hour forecast (3 kW, where the day-ahead error is
   13 kW in winter).
2. *Does the learner pay for being rescued?* No. And the penalty for a car leaving short was computed from
   the wrong hour, so it did not match reality.
3. *How much of the saving needs learning at all?* We added the obvious rule: "do not charge during the
   expensive hours".

**What we changed.**

- Shield: the later hours of the plan now carry the day-ahead error margin. A reserve of one hour is kept,
  and it now accounts for the 0.3% loss during that hour. These two settings were chosen on the **training**
  weeks only (0 missed of 66 with a controller that never asks; 4 without the margin; 2 without the reserve).
  My first choice used the winter week alone and was wrong; I stopped that run after 15 minutes and redid it.
- Reward: the penalty for a car leaving short now uses what the car really left with.
- Three ways to deal with the learner, running now as separate arms: leave it as it is (control); make it
  **pay** for every kWh the shield has to force; put a **floor** under its request (it may ask for more than
  "half power outside the peak", never less).

**What the non-learning arms already show (final shield, evaluation weeks, 40 kW cap):**

| Cars ask… | Winter cost | Winter missed | Summer cost | Summer missed |
|---|---|---|---|---|
| on arrival | 1768 | 4 of 67 | 1510 | 0 |
| outside the tariff peak | 1497 (−15%) | 5 of 67 | 1212 (−20%) | 0 |
| never (shield alone) | 1464 (−17%) | 7 of 67 | 1136 (−25%) | 0 |

Two things follow. **Most of the "learned" saving on the cars is available to a one-line rule.** And with the
corrected shield, a controller that never asks misses 7 winter departures, where the learned policy behind
the old shield missed 9 and 14: the shield now holds most of the weight, but asking is still safer than not
asking. Four or five misses are unavoidable in that winter window (one night where the energy does not fit
under 40 kW, whatever the controller).

### 5.11 Why the old heat-pump numbers are not valid, and what replaces them

**The old result** (in `RECAP.md`, the one shared with Manal): the learned heat-pump controller cost 45% of
the rule's cost "at the same discomfort, 0.01%".

**Why it cannot be trusted.** The first bug in the table of 5.2: the comfort metric and the comfort reward read
the dataset's temperature, not the simulated one. A controller that turns the heat pump down saves money and the metric
cannot see the house getting cold. Measured tonight (winter, 28 days, heat pump simply switched off):

| Heat pump | Simulator | Cost | Discomfort the metric reports |
|---|---|---|---|
| off | as shipped | 1447 | 1.3% |
| off | corrected | 1446 | **47.7%** |
| thermostat at the set point | corrected | 2128 | 1.5% |

Switching the heat pump off is 32% cheaper and the uncorrected simulator calls it comfortable. That is what
the old benchmark rewarded.

**What "valid" needs.** The corrected simulator, the set-point control from 5.5, the corrected measurement,
and comfort reported next to cost. That benchmark is queued behind the EV runs: thermostat alone; a fixed
pre-heat-and-coast schedule; a learned set-point policy that controls the heat pump and nothing else; four
seasons, two seeds. A second small grid runs the *old* formulation (direct power) on the corrected simulator,
so the old claim can be compared like for like.

---

## 6. What we know now

### 6.1 Safety layer (grid 1: 8 scenarios, 80 runs)

| Controller | Battery violations | Cost | Discomfort |
|---|---|---|---|
| no control | 100% | 1951 | 4.9% |
| rule | 30% | 1677 | 4.9% |
| rule + exact barrier | 0% | 1665 | 4.9% |
| learning, penalty only | 9.1% | 1666 | 2.2% |
| learning + barrier with the paper's uniform rate | 3.9% | 1564 | 2.2% |
| learning + exact barrier | 0% | 1549 | 2.5% |

### 6.2 Learning against the rule

Summer −33% and −41%; spring −12% and −19%; autumn +9% and −18%; winter +10% and +25%. Not significant when
pooled. Residual policy: winter −3% and +12%, summer −16% and −17%.

### 6.3 Cars under a cap (shield alone)

See 5.8. The one-sentence version: **joint planning is needed exactly when charging is deferred.**

### 6.4 Controllers under a 40 kW cap (first grid, before tonight's fix)

| Controller | Winter cost | Winter over cap | Winter missed | Summer cost | Summer missed |
|---|---|---|---|---|---|
| rule, no shields | 1892 | 403 kWh | 1 | 1645 | 1 |
| rule + shields | 1778 | 16 kWh | 5 | 1512 | 0 |
| learning + shields | 1497 | 17 kWh | 9 and 14 | 987 | 0 |
| residual learning + shields | 1751 | 7 kWh | 5 | 1401 | 0 |

### 6.5 Literature

Eighteen ideas were taken from the search, seventeen have code behind them (table in the report, §7). The
Consensus plugin was not connected, so the search used web sources; every cited paper was opened. The list
should be re-checked through Consensus.

---

## 7. Current position

### 7.1 Done and pushed (`rebuild/audited-pipeline`, 4 commits)

The audited pipeline, grid 1, the residual grid, the four-stage EV study, the cap shield, the first
controller grid, the report, 218 tests.

### 7.2 Done tonight, local on `fix/learner-leaning` (2 commits, 227+ tests)

The shield margin for later hours; the reserve that keeps its promise; the corrected departure penalty; the
"pays for forced charging" and "floor" arms; the off-peak and never-ask reference arms; the heat-pump-only
arms; the script that shows why the old heat-pump numbers were invalid.

### 7.3 Running now

| Run | What it answers | State |
|---|---|---|
| Controller grid 2 (18 runs) | Does the learner stop leaning, and what does that cost? | reference arms done (table in 5.10); 12 learning runs in progress |
| Heat-pump-only grid (16 runs) | The valid version of Manal's benchmark | queued |
| Heat-pump-only, old formulation (6 runs) | What the old claim becomes on the corrected simulator | queued |
| EV study, forecast stage again (108 runs) | The effect of the new margin and the corrected reserve at the shield level | queued |

When they finish: results into the report, merge, push.

### 7.4 What is valid and what is not

| Numbers | Status |
|---|---|
| `RECAP.md`, `stems_report.pdf`, every result before 3 October | **Not valid.** Superseded. |
| The heat-pump benchmark shared with Manal (cost 0.448 × rule) | **Not valid** (5.11). Being re-run. |
| Grid 1, residual grid, EV study, controller grid 1 | Valid on the corrected simulator. Limits below. |
| Controller grids under the cap | Two seasons, two seeds: mechanisms are established, sizes are not. |
| Anything with cars | The schedules are synthetic. One fleet. |
| Comfort numbers | Rest on CityLearn's learned temperature model, which reacts unrealistically fast. |

### 7.5 Against Andreas's seven priorities (his email of 3 September)

His message: stop adding functionality, produce a clean and convincing evaluation. His question: *how much
does a calibrated and anticipatory safety layer improve constraint satisfaction in RL-based multi-device
energy management, and what is the cost/comfort trade-off under increasingly constrained conditions?*

| # | What he asked | Status | What exists | What is missing |
|---|---|---|---|---|
| 1 | One clean joint battery / heat-pump / hot-water experiment, all actuators active, results valid | **Done** | Grid 1 on the corrected simulator; every run checks that the actuators respond | In 16 of 48 learned runs one device was used too rarely to confirm. Hot-water "readiness" has no physical meaning in CityLearn, so it is not scored |
| 2 | Several meaningful baselines, e.g. better rules made with Manal | **Not done** | No control, and one time-of-use rule (plus, for the cars, three ways of asking) | No MPC, no other learning method, no rules from Manal |
| 3 | Ablation: rule / RL / RL + basic barrier / RL + calibrated barrier | **Done** | Violations 9.1% → 3.9% → 0%, pre-declared comparisons | The "anticipatory" battery buffer was removed: it was not needed and it distorted cost. Anticipation is now in the car shield, not the battery barrier |
| 4 | Statistics: vary seed, building set, weather period; confidence intervals; controlled forecast errors in training and inference | **Partly** | 4 seasons × 2 building sets × 2 seeds, intervals over scenarios | One weather year. Both building sets are the same *kind* of building (single-family houses). **The forecast-error experiment has not been run** |
| 5 | Sweep the power cap from loose to tight | **Partly** | Nine caps (100 → 25 kW) for the shield without learning | Learned controllers were run at one cap only (40 kW) |
| 6 | A heating-dominated (cold) scenario | **Not done** | — | Texas only. CityLearn ships Vermont and Quebec neighbourhoods |
| 7 | Quantitative EV experiment: deadlines, cost, peaks, devices competing | **Done** at shield level, **thin** with learning | 468 runs; two controller grids | Two seasons, two seeds, synthetic schedules |

Honest reading: points 1, 3 and 7 are answered. Points 2, 4, 5 and 6 are the evaluation work he asked for and
they are mostly still open. And in the last two days we *added* mechanisms (the joint programme, the cap
shield, the tank model, the margins, the floor) — some were needed to make point 7 correct, some came from the
literature search. That is the direction he asked us to leave.

### 7.6 Against the STEMS paper

**We have not beaten the paper in a way a reviewer would accept.** What the paper does (read from its arXiv
page): CityLearn, Travis County, August 2018 – August 2019, eight buildings (five residential, two commercial,
one mixed-use), five seeds, eight baselines (Rule-Based, MPC, Single-Agent SAC, MADDPG, MetaEMS, MARLISA,
MADCQ, D-MAPPO). It reports cost 0.792 and emissions 0.824 of the reference, and safety violations of 5.6%
(from 35.1%).

| | The paper | Us |
|---|---|---|
| Buildings | 8 mixed-use, Travis County | 8 houses, Travis County (ResStock) |
| Period | one full year | 28-day windows, four seasons |
| Learner | SAC actors with GCN + transformer | PPO-Lagrangian with the same encoder |
| Baselines | 8 | no control, one rule |
| Seeds | 5, no intervals | 2, with intervals over scenarios |
| Safety violations | 5.6% | 0% (calibrated barrier), 3.9% (uniform-rate barrier), 9.1% (none) |

What can be said: on our audited setup, a barrier that inverts the simulator's battery model gives zero
violations where a generic one leaves 3.9%. What cannot be said: "better than STEMS". The 5.6% is the paper's
number on the paper's setup; we have not reproduced that setup, its learner, or its baselines.

### 7.7 Different kinds of buildings

Everything so far is on single-family houses. The paper mixes residential, commercial and mixed-use
buildings, and the result has to hold on other kinds too. What CityLearn 2.6.0b1 has, and what our code does
with it today (checked on 5 October):

| Dataset | Buildings | Thermal dynamics | Battery | Loads in our environment today | What it can test |
|---|---|---|---|---|---|
| `citylearn_challenge_2020_climate_zone_1` … `_4` | 9 mixed: office, restaurant, two retail, five multi-family (per the CityLearn documentation; to confirm) | no (loads are always met) | all 9 | yes, with battery and hot-water storage; **no tariff in the schema**, and its chilled-water storage action is not supported yet | the battery-barrier ablation on mixed building types, in four climates (zone 1: 20.9 °C mean; zone 4: 10.3 °C mean, minimum −18.8 °C) |
| `baeda_3dem` | 4 commercial | yes | none | loads, but only hot-water storage is controlled: its cooling action has a different name and is dropped | comfort and set-point control on commercial buildings (cooling only, four months) |
| `vt_chittenden_county_neighborhood`, `quebec_neighborhood_*` | houses, cold climate | yes | — (not downloaded yet) | not tried | the heating-dominated case Andreas asked for |
| `ca_alameda_county_neighborhood` | 100 houses, California | yes | all | not tried | another climate, same kind of building |

So it is feasible, and it is work: a tariff for the mixed set, the cooling actions in the environment's
layout, the simulator checks repeated on each dataset (the three bugs must be verified per building class),
and then the same ablation run on each.

**First step, done on 5 October (branch `eval/building-types`).** The mixed set now runs in the audited
pipeline, in the hot zone and the cold zone. Checked before any number: our cost and consumption equal the
simulator's own series on these buildings (to 1e-4), the battery and hot-water actuators respond, the battery
band matches the simulator's own state-of-charge series. Caps were set so they do not bind (as in grid 1), so
this tests the battery barrier alone. Non-learning arms, four seasons, 28-day windows, 32 runs, none failed:

| Zone, season | No control: cost | Rule: battery outside its band | Rule + exact barrier: violations | Rule + barrier: cost vs no control | Exact barrier alone: cost |
|---|---|---|---|---|---|
| hot, winter | 25,635 | 54.9% | 0 | −12.9% | +0.05% |
| hot, spring | 35,912 | 50.5% | 0 | −7.6% | +0.03% |
| hot, summer | 51,214 | 50.8% | 0 | −4.6% | +0.02% |
| hot, autumn | 25,738 | 52.4% | 0 | −11.3% | +0.05% |
| cold, winter | 24,177 | 54.0% | 0 | −14.3% | +0.05% |
| cold, spring | 22,972 | 52.2% | 0 | −13.3% | +0.05% |
| cold, summer | 36,739 | 51.2% | 0 | −6.6% | +0.03% |
| cold, autumn | 22,874 | 54.1% | 0 | −12.6% | +0.05% |

The exact barrier gives zero violations on offices, restaurants, shops and multi-family buildings, in a hot
and a cold climate, for 0.02–0.05% of cost. These batteries are very different from each other (10 to 100 kW,
33–71% of capacity per hour), which is where a one-size barrier should fail: in a two-episode test the
uniform-rate barrier left 22% violations here, against 3.9% on the houses. The learning arms (RL, RL + uniform
barrier, RL + exact barrier; two zones, winter and summer, two seeds) are queued behind tonight's runs.

Two things this set cannot show: comfort (no thermal dynamics), and a binding cap (the rule's midday charging
takes the peak from 267 to 513 kW in the hot winter, so a realistic cap would bind hard; the shield that
handles a binding cap is so far only wired for schemas with chargers).

---

## 8. Open questions and next steps

0. **Realign with Andreas.** In order of value for his question: the cap sweep with learned controllers
   (point 5); **other kinds of buildings** — the mixed commercial and multi-family set in a hot and a cold
   climate, commercial buildings with thermal dynamics, cold-climate houses (points 4 and 6, and the paper's
   own mix; see 7.7); the forecast-error experiment (point 4); real baselines — an MPC with foresight as the
   upper bound, one or two other learners, rules from Manal (point 2); then a like-for-like table against
   the paper (full year, five seeds). No new mechanisms until these exist.
1. **Finish tonight's runs** and write them up.
2. **Tell Manal** that the old comfort numbers are invalid, and give her the new table when it exists.
3. **Hot water and the weekly Legionella cycle** (her requirement: once a week, 60 °C). It is the same
   mathematical object as a car's departure: storage that must reach a level by a deadline. Design in
   `docs/dhw_legionella_design.md`; needs a heat-pump water-heater model.
4. **A thermostat with a dead zone**, then grid 1 again: it decides how much of the learner's summer gain is
   real arbitrage.
5. **Let the shield discharge batteries** when the house alone would exceed the cap.
6. **More scenarios and seeds** for the controller grids, and a best-possible schedule computed with
   hindsight as a reference.
7. **File the CityLearn bug report.** It is written; it needs your go-ahead because it is public.
8. **Re-check the literature through Consensus** once it is connected.

Decisions that are yours: filing the bug report; what to say to Manal and when; whether the next block of
work is the hot-water deadline (for her) or the stronger EV evidence (for the thesis claim).

---

## 9. Where things are

| What | Where |
|---|---|
| Technical report, all tables | `docs/REPORT_2026-10.md` |
| This overview | `docs/PROJECT_OVERVIEW.md` |
| Simulator bug report (draft) | `docs/citylearn_v2.4_defects.md` |
| Hot-water / Legionella design | `docs/dhw_legionella_design.md` |
| Environment and simulator patches | `stems/environment.py` |
| Battery and tank models | `stems/battery.py` |
| Battery barrier | `stems/cbf.py` |
| Cars: model, rules, joint programme, cap shield | `stems/fleet.py` |
| Learner | `stems/agent.py`, `stems/reward.py` |
| Experiments | `experiments/` (`ablation.py`, `ev_coupling.py`, `aggregate.py`, reports) |
| Scripts behind single numbers | `experiments/diagnostics/` |
| Results | `results/ablation_v1`, `ablation_v2`, `ev_coupling`, `ev_rl_v1` |
| Tests | `tests/` |

Run everything from the project folder with `.venv/Scripts/python`.

---

## 10. Words used here

| Word | Meaning |
|---|---|
| Cap | The limit on the power the whole street may import at once. |
| Shield / barrier | A layer between the controller and the devices that changes an action only when it would break a limit. |
| Deadline | The hour a car leaves; it must have its required charge by then. |
| Laxity | How long a car can still wait before it must charge at full power to make its deadline. |
| Joint programme (LP) | An optimisation that plans all cars over all remaining hours together. |
| Deferred charging | Charging later than on arrival, to avoid expensive hours. |
| Leaning on the shield | The learner not doing a job because the shield will do it for it. |
| Residual policy | A learner that only corrects a rule, within bounds. |
| Scenario | One season with one set of houses. The unit we count results in. |
| Arm | One combination of controller and safety layer in an experiment. |
| Fingerprint | A hash of the code, stored in every run, so results from different code are never mixed. |
