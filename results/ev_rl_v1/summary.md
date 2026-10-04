Schema `citylearn_schemas/tx_travis_8b_ev/schema.json`, cap 40 kW, 14-day windows, code `52711521326e4e9a`.

**summer**

| Arm | seeds | Cost | Over cap [kWh] | Hours over cap | Peak [kW] | Missed departures | EV shortfall [kWh] | Battery band | Discomfort | Battery cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| rule, cars charge on arrival, no shield | 1 | 1645 | 502.7 | 0.104 | 72.6 | 0.015 | 0.1 | 0.393 | 0.046 | 4.1 |
| rule + shields | 1 | 1512 | 3.4 | 0.015 | 41.5 | 0.000 | 0.0 | 0.000 | 0.046 | 4.1 |
| RL + shields | 2 | 987 | 0.7 | 0.004 | 40.6 | 0.000 | 0.0 | 0.000 | 0.009 | 5.4 |
| residual RL + shields | 2 | 1401 | 4.0 | 0.012 | 41.6 | 0.000 | 0.0 | 0.000 | 0.023 | 8.8 |

*RL + shields: cost per seed 1120, 854; residual RL + shields: cost per seed 1409, 1393.*

**winter**

| Arm | seeds | Cost | Over cap [kWh] | Hours over cap | Peak [kW] | Missed departures | EV shortfall [kWh] | Battery band | Discomfort | Battery cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| rule, cars charge on arrival, no shield | 1 | 1892 | 402.8 | 0.101 | 85.3 | 0.015 | 0.1 | 0.293 | 0.021 | 7.2 |
| rule + shields | 1 | 1778 | 15.8 | 0.033 | 43.1 | 0.075 | 55.5 | 0.000 | 0.021 | 7.0 |
| RL + shields | 2 | 1497 | 17.0 | 0.021 | 45.1 | 0.172 | 92.9 | 0.000 | 0.022 | 4.3 |
| residual RL + shields | 2 | 1751 | 7.4 | 0.018 | 42.1 | 0.075 | 53.3 | 0.000 | 0.015 | 9.1 |

*RL + shields: cost per seed 1503, 1490; residual RL + shields: cost per seed 1727, 1775.*
