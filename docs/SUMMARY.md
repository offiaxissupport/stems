# STEMS in plain English

*5 October 2026. A short version. The long version is [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md); every table is
in [REPORT_2026-10.md](REPORT_2026-10.md).*

## What the project is

We want a computer to run the energy of a small neighbourhood: each house has a battery, solar panels, a heat
pump and a hot-water tank, and some have an electric car. The goal is to **pay less for electricity without
breaking any limit**: the battery must stay in its safe range, the house must stay comfortable, each car must
be charged before it leaves, and the whole street must stay under one shared power limit.

We compare three ways of doing it: no control, a fixed rule (a simple schedule), and a learning agent. On top
of any of them sits a **safety layer** that corrects an action only when it would break a limit.

## What we did

1. **Checked the simulator before trusting it.** We found three bugs in it. The worst: it told the controller
   the wrong indoor temperature, so nothing could see what the heat pump was doing. We fixed all three and
   wrote the bug report for its authors (not sent yet).
2. **Fixed how we measure.** Costs were being counted with the next hour's price. Our numbers now match the
   simulator's own records exactly.
3. **Rebuilt the learning agent.** The old one had never learned anything; we proved it on a test with a known
   answer, then fixed it.
4. **Made comfort work.** The agent now moves the thermostat's set point a little instead of driving the heat
   pump directly.
5. **Built an exact safety layer for the batteries**, using the simulator's own battery equations.
6. **Tested learning against the rule** in four seasons.
7. **Studied the cars**: eight ways of sharing the power limit between them, at nine different limits.
8. **Ran complete controllers** under a tight limit, found two problems, and fixed them.
9. **Started testing on other kinds of buildings** (offices, shops, restaurants, apartment blocks).

## What we found

**Safety.** With our exact safety layer the batteries never leave their safe range. With a generic safety
layer they do 3.9% of the time, and with none 9.1%. The price of our layer is 0.3–0.5% of the bill. On offices,
shops and apartment blocks, in a hot and a cold climate, it is also zero violations (rule-based controllers so
far; the learning ones are running).

**Does learning beat a simple rule?** Sometimes. In summer the learning agent is 33–41% cheaper. In winter it
is 10–25% more expensive. Over all seasons the difference is not statistically clear.

**The cars.**

- A safety layer that looks at each car alone gets every car charged and breaks the street's limit badly.
- If cars charge as soon as they arrive, simple "most urgent first" rules work as well as planning all cars
  together.
- If charging is postponed to cheaper hours, only planning them together works: at a tight limit the simple
  approach leaves 248 kWh undelivered, the joint plan 31.
- With our shields, energy drawn above the limit falls from 400–500 kWh to 3–16 kWh in two weeks.

**The learning agent was cheating a little.** Behind the safety layer it stopped charging the cars itself
(it asked for less than 3% of their energy) and let the safety layer do it at the last minute. That looked
cheap, and it made 9 and 14 cars out of 67 leave without enough charge in winter. We fixed the safety layer's
planning; now 5 or 6 cars miss, and 4 or 5 are impossible to serve on that winter night whatever you do.

**A one-line rule does as well as the agent on the cars.** "Do not charge during the expensive hours" costs
1497 in winter; the learning agents cost between 1509 and 1638. In summer the agent is still cheaper
(about 1023 against 1212).

**The old heat-pump results were wrong.** Because of the temperature bug, a controller that simply switched
the heating off looked 32% cheaper *and* comfortable (1.3% discomfort reported). The real discomfort was 47.7%.
The corrected benchmark is running now.

## What we cannot say yet

- **We have not beaten the STEMS paper.** We get 0% violations where the paper reports 5.6%, but on different
  buildings, a shorter period and without the paper's eight comparison methods. It is not a fair comparison yet.
- **Too few comparison methods.** We compare against no control and one rule. No other learning method, no
  optimisation-based controller.
- **The learning agent was tested at one power limit only**, in two seasons, with two random seeds.
- **Houses in Texas only** for most results. The car schedules are invented, not measured.
- **No cold-climate houses yet**, and the forecast-error test Andreas asked for has not been run.

## Where we are right now

- The main branch is on GitHub. Tonight's fixes and the work on other building types are saved locally and will
  be pushed when their experiments finish.
- Running tonight: the last learning runs of the fix, the corrected heat-pump benchmark, and the learning
  agents on offices, shops and apartment blocks.
- Next, in the order Andreas asked for: more power limits with the learning agent, more kinds of buildings,
  forecast errors, real comparison methods, a cold climate, and only then a fair table against the paper.
