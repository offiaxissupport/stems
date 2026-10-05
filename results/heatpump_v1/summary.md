Schema `citylearn_schemas/tx_travis_8b/schema.json`, 28-day windows, code `205c33df0093a755`; simulator patches dhw_storage_capacity, t0_thermal_double_count, endogenous_obs_from_simulated_hour.

**winter**

| Arm | seeds | Cost | Consumption [kWh] | Discomfort | Discomfort [degC h] | Worst house | Peak [kW] | Daily peak [kW] | Switches per day | Cost vs thermostat |
|---|---|---|---|---|---|---|---|---|---|---|
| thermostat at the set point | 1 | 2128 | 7318 | 0.015 | 10.5 | 0.061 | 39.1 | 23.5 | 1.62 | +0.0% |
| pre-condition and coast (fixed schedule) | 1 | 2389 | 8153 | 0.106 | 68.8 | 0.200 | 39.0 | 24.8 | 2.55 | +12.2% |
| learned set-point policy | 2 | 2025 | 6960 | 0.009 | 9.9 | 0.024 | 38.7 | 22.5 | 1.48 | -4.8% |

*learned policy, per seed: cost 2010, 2041; discomfort 0.012, 0.007.*

**spring**

| Arm | seeds | Cost | Consumption [kWh] | Discomfort | Discomfort [degC h] | Worst house | Peak [kW] | Daily peak [kW] | Switches per day | Cost vs thermostat |
|---|---|---|---|---|---|---|---|---|---|---|
| thermostat at the set point | 1 | 1339 | 4384 | 0.011 | 4.1 | 0.023 | 27.3 | 18.8 | 1.43 | +0.0% |
| pre-condition and coast (fixed schedule) | 1 | 1257 | 4305 | 0.045 | 15.2 | 0.080 | 24.0 | 17.3 | 2.31 | -6.1% |
| learned set-point policy | 2 | 1229 | 4084 | 0.005 | 1.9 | 0.013 | 24.7 | 17.0 | 1.43 | -8.2% |

*learned policy, per seed: cost 1219, 1238; discomfort 0.004, 0.005.*

**summer**

| Arm | seeds | Cost | Consumption [kWh] | Discomfort | Discomfort [degC h] | Worst house | Peak [kW] | Daily peak [kW] | Switches per day | Cost vs thermostat |
|---|---|---|---|---|---|---|---|---|---|---|
| thermostat at the set point | 1 | 1789 | 5669 | 0.055 | 19.6 | 0.170 | 35.1 | 26.3 | 1.11 | +0.0% |
| pre-condition and coast (fixed schedule) | 1 | 1550 | 5290 | 0.053 | 34.4 | 0.128 | 33.9 | 21.5 | 1.29 | -13.4% |
| learned set-point policy | 2 | 1519 | 4947 | 0.011 | 3.7 | 0.031 | 30.0 | 21.3 | 1.02 | -15.1% |

*learned policy, per seed: cost 1514, 1524; discomfort 0.011, 0.011.*

**autumn**

| Arm | seeds | Cost | Consumption [kWh] | Discomfort | Discomfort [degC h] | Worst house | Peak [kW] | Daily peak [kW] | Switches per day | Cost vs thermostat |
|---|---|---|---|---|---|---|---|---|---|---|
| thermostat at the set point | 1 | 2051 | 6662 | 0.074 | 67.6 | 0.122 | 33.9 | 22.5 | 2.11 | +0.0% |
| pre-condition and coast (fixed schedule) | 1 | 2184 | 7162 | 0.141 | 123.3 | 0.238 | 38.8 | 23.2 | 2.73 | +6.5% |
| learned set-point policy | 2 | 1855 | 6036 | 0.049 | 42.7 | 0.088 | 29.5 | 20.6 | 2.25 | -9.6% |

*learned policy, per seed: cost 1890, 1820; discomfort 0.048, 0.051.*
