Schema `citylearn_schemas/tx_travis_8b_ev/schema.json`, cap 40 kW, 14-day windows, code `205c33df0093a755`.

**summer**

| Arm | seeds | Cost | Over cap [kWh] | Hours over cap | Peak [kW] | Missed departures | EV shortfall [kWh] | Battery band | Discomfort | Battery cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| rule + shields | 1 | 1510 | 3.4 | 0.015 | 41.5 | 0.000 | 0.0 | 0.000 | 0.046 | 4.1 |
| rule, cars charge off-peak + shields | 1 | 1212 | 3.2 | 0.012 | 41.5 | 0.000 | 0.0 | 0.000 | 0.046 | 4.1 |
| rule, cars never ask + shields | 1 | 1136 | 0.0 | 0.000 | 38.5 | 0.000 | 0.0 | 0.000 | 0.046 | 4.1 |
| RL + shields | 2 | 1023 | 0.0 | 0.000 | 39.1 | 0.000 | 0.0 | 0.000 | 0.009 | 5.4 |
| RL + shields, pays for forced charging | 2 | 1004 | 0.9 | 0.003 | 40.0 | 0.000 | 0.0 | 0.000 | 0.010 | 5.8 |
| RL + shields, floor under the charger request | 2 | 1078 | 2.5 | 0.007 | 41.9 | 0.000 | 0.0 | 0.000 | 0.010 | 6.7 |

*RL + shields: cost per seed 1154, 893; RL + shields, pays for forced charging: cost per seed 1060, 948; RL + shields, floor under the charger request: cost per seed 1104, 1052.*

**winter**

| Arm | seeds | Cost | Over cap [kWh] | Hours over cap | Peak [kW] | Missed departures | EV shortfall [kWh] | Battery band | Discomfort | Battery cycles |
|---|---|---|---|---|---|---|---|---|---|---|
| rule + shields | 1 | 1768 | 14.1 | 0.030 | 43.1 | 0.060 | 57.2 | 0.000 | 0.021 | 7.0 |
| rule, cars charge off-peak + shields | 1 | 1497 | 18.3 | 0.030 | 45.1 | 0.075 | 79.1 | 0.000 | 0.021 | 7.0 |
| rule, cars never ask + shields | 1 | 1464 | 13.3 | 0.027 | 43.1 | 0.104 | 79.6 | 0.000 | 0.021 | 7.0 |
| RL + shields | 2 | 1574 | 8.1 | 0.021 | 42.7 | 0.075 | 84.3 | 0.000 | 0.015 | 3.5 |
| RL + shields, pays for forced charging | 2 | 1614 | 8.3 | 0.015 | 42.6 | 0.082 | 76.2 | 0.000 | 0.017 | 5.6 |
| RL + shields, floor under the charger request | 2 | 1577 | 12.3 | 0.019 | 44.8 | 0.075 | 65.0 | 0.000 | 0.020 | 7.1 |

*RL + shields: cost per seed 1509, 1638; RL + shields, pays for forced charging: cost per seed 1610, 1617; RL + shields, floor under the charger request: cost per seed 1599, 1556.*
