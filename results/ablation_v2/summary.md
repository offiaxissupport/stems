# Ablation summary: results/ablation_v2

records 16 | ok 16 | errors 0 | failed actuator check (excluded) 0

## Actuator verification per arm

| arm | verified | insufficient evidence | failed |
|---|---|---|---|
| idle+calibrated | 0 | 4 | 0 |
| rbc+calibrated | 4 | 0 | 0 |
| rl-res+calibrated | 8 | 0 | 0 |

## Primary inference (pooled scenarios, Holm-corrected)

Unit of replication: scenario (season x building subset); seeds averaged within it.

| contrast | KPI | difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|
| rbc+calibrated - idle+calibrated (what the time-of-use rule adds over no control) | safety_violation_rate | 0 (no variation, n=4) | n/a | n/a |
| rbc+calibrated - idle+calibrated (what the time-of-use rule adds over no control) | cost | -277.2 [-365, -189.3] (n=4) | 0.0021 | 0.00631 * |
| rbc+calibrated - idle+calibrated (what the time-of-use rule adds over no control) | discomfort_rate | 0 (no variation, n=4) | n/a | n/a |
| rl-res+calibrated - rbc+calibrated (what a learned correction adds to the rule) | safety_violation_rate | 0 (no variation, n=4) | n/a | n/a |
| rl-res+calibrated - rbc+calibrated (what a learned correction adds to the rule) | cost | -66.79 [-478.3, 344.7] (n=4) | 0.641 | 1 |
| rl-res+calibrated - rbc+calibrated (what a learned correction adds to the rule) | discomfort_rate | -0.004644 [-0.05799, 0.04871] (n=4) | 0.8 | 1 |

## Descriptive: pooled

| KPI | idle+calibrated | rbc+calibrated | rl-res+calibrated |
|---|---|---|---|
| safety_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) | 0 (no variation, n=4) |
| cost | 2094 [1374, 2815] (n=4) | 1817 [1129, 2505] (n=4) | 1750 [651.2, 2849] (n=4) |
| discomfort_rate | 0.04583 [0.01163, 0.08003] (n=4) | 0.04583 [0.01163, 0.08003] (n=4) | 0.04118 [0, 0.09674] (n=4) |
| soc_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) | 0 (no variation, n=4) |
| grid_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) | 0 (no variation, n=4) |
| power_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) | 0 (no variation, n=4) |
| emission | 1134 [554.2, 1715] (n=4) | 1110 [472, 1748] (n=4) | 1087 [266.6, 1908] (n=4) |
| peak_import_kw | 37.19 [29.09, 45.3] (n=4) | 47.06 [15.41, 78.72] (n=4) | 45.9 [13.25, 78.54] (n=4) |
| avg_daily_peak | 25.68 [23.06, 28.3] (n=4) | 29.92 [15.76, 44.08] (n=4) | 27.45 [13.15, 41.75] (n=4) |
| ramping_rate | 4.999 [4.439, 1] (n=4) | 6.017 [5.144, 1] (n=4) | 5.313 [4.727, 1] (n=4) |
| discomfort_degree_hours | 36.28 [-26.34, 98.89] (n=4) | 36.28 [-26.34, 98.89] (n=4) | 57.34 [-99.37, 214.1] (n=4) |
| pv_self_consumption | 0.5559 [0.4535, 0.6584] (n=4) | 0.6566 [0.5489, 0.7643] (n=4) | 0.6926 [0.5971, 0.7882] (n=4) |
| battery_equivalent_full_cycles | 0.0505 [0.0505, 0.0505] (n=4) | 10.72 [6.276, 15.17] (n=4) | 16.42 [15.01, 17.83] (n=4) |
| hvac_on_transitions_per_building_day | 1.207 [0.7262, 1.688] (n=4) | 1.207 [0.7262, 1.688] (n=4) | 1.14 [0.6809, 1.599] (n=4) |
| barrier_intervention_rate | 0.688 [0.5733, 0.8026] (n=4) | 0.1751 [0.1584, 0.1917] (n=4) | 0.2592 [0.1708, 0.3477] (n=4) |
| cost_cv_across_buildings | 0.424 [0.3693, 0.4788] (n=4) | 0.4458 [0.3923, 0.4993] (n=4) | 0.5077 [0.3767, 0.6387] (n=4) |
| electricity_consumption | 7063 [3921, 1.021e+04] (n=4) | 6947 [3452, 1.044e+04] (n=4) | 6833 [2120, 1.155e+04] (n=4) |
| ev_missed_departure_rate | n/a | n/a | n/a |
| ev_energy_shortfall_kwh | n/a | n/a | n/a |
| cap_exceedance_kwh | 0 (no variation, n=4) | 0 (no variation, n=4) | 0 (no variation, n=4) |

Within-scenario seed SD of cost (training noise, not replication): rl-res+calibrated: 33

| KPI | rbc+calibrated - idle+calibrated | rl-res+calibrated - rbc+calibrated |
|---|---|---|
| safety_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) |
| cost | -277.2 [-365, -189.3] (n=4) | -66.79 [-478.3, 344.7] (n=4) |
| discomfort_rate | 0 (no variation, n=4) | -0.004644 [-0.05799, 0.04871] (n=4) |
| soc_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) |
| grid_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) |
| power_violation_rate | 0 (no variation, n=4) | 0 (no variation, n=4) |
| emission | -24.41 [-86.83, 38.01] (n=4) | -22.82 [-210.3, 164.7] (n=4) |
| peak_import_kw | 9.87 [-14.57, 34.32] (n=4) | -1.167 [-4.34, 2.006] (n=4) |
| avg_daily_peak | 4.238 [-10.58, 19.06] (n=4) | -2.469 [-7.378, 2.439] (n=4) |
| ramping_rate | 1.018 [0.4036, 1.632] (n=4) | -0.7039 [-1.267, -0.1405] (n=4) |
| discomfort_degree_hours | 0 (no variation, n=4) | 21.06 [-73.57, 115.7] (n=4) |
| pv_self_consumption | 0.1006 [0.07796, 0.1233] (n=4) | 0.03607 [0.0146, 0.05754] (n=4) |
| battery_equivalent_full_cycles | 10.67 [6.225, 15.12] (n=4) | 5.696 [1.551, 9.842] (n=4) |
| hvac_on_transitions_per_building_day | 0 (no variation, n=4) | -0.06706 [-0.1172, -0.01693] (n=4) |
| barrier_intervention_rate | -0.5129 [-0.6268, -0.399] (n=4) | 0.08418 [-0.008236, 0.1766] (n=4) |
| cost_cv_across_buildings | 0.02178 [0.01434, 0.02922] (n=4) | 0.06187 [-0.06704, 0.1908] (n=4) |
| electricity_consumption | -116.2 [-510, 277.5] (n=4) | -113.7 [-1350, 1123] (n=4) |
| ev_missed_departure_rate | n/a | n/a |
| ev_energy_shortfall_kwh | n/a | n/a |
| cap_exceedance_kwh | 0 (no variation, n=4) | 0 (no variation, n=4) |

## Descriptive: summer

| KPI | idle+calibrated | rbc+calibrated | rl-res+calibrated |
|---|---|---|---|
| safety_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| cost | 1761 [1306, 2216] (n=2) | 1519 [1487, 1551] (n=2) | 1266 [1169, 1363] (n=2) |
| discomfort_rate | 0.05919 [0.005426, 0.113] (n=2) | 0.05919 [0.005426, 0.113] (n=2) | 0.03082 [0, 0.08338] (n=2) |
| soc_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| grid_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| power_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| emission | 847.8 [654.4, 1041] (n=2) | 790 [703.8, 876.3] (n=2) | 686.6 [549.5, 823.6] (n=2) |
| peak_import_kw | 33.25 [8.478, 58.02] (n=2) | 29.88 [8.62, 51.13] (n=2) | 28.22 [17.54, 38.89] (n=2) |
| avg_daily_peak | 25.9 [20.62, 31.19] (n=2) | 22.23 [13.84, 30.62] (n=2) | 20.05 [18.12, 21.97] (n=2) |
| ramping_rate | 5.266 [2.646, 1] (n=2) | 6.175 [0, 1] (n=2) | 5.25 [0, 1] (n=2) |
| discomfort_degree_hours | 19.83 [17.19, 22.47] (n=2) | 19.83 [17.19, 22.47] (n=2) | 8.84 [-3.946, 21.63] (n=2) |
| pv_self_consumption | 0.5058 [0.152, 0.8595] (n=2) | 0.5999 [0.3486, 0.8513] (n=2) | 0.6433 [0.4709, 0.8158] (n=2) |
| battery_equivalent_full_cycles | 0.0505 [0.05049, 0.05051] (n=2) | 8.362 [0.1245, 16.6] (n=2) | 16.3 [6.594, 26] (n=2) |
| hvac_on_transitions_per_building_day | 1.004 [-0.3313, 2.339] (n=2) | 1.004 [-0.3313, 2.339] (n=2) | 0.9322 [-0.09037, 1.955] (n=2) |
| barrier_intervention_rate | 0.688 [0, 1] (n=2) | 0.1819 [0.1216, 0.2423] (n=2) | 0.2456 [0, 0.9735] (n=2) |
| cost_cv_across_buildings | 0.4081 [0.105, 0.7112] (n=2) | 0.4294 [0.09295, 0.7659] (n=2) | 0.4601 [0.03774, 0.8824] (n=2) |
| electricity_consumption | 5579 [4063, 7095] (n=2) | 5251 [4310, 6192] (n=2) | 4607 [3356, 5859] (n=2) |
| ev_missed_departure_rate | n/a | n/a | n/a |
| ev_energy_shortfall_kwh | n/a | n/a | n/a |
| cap_exceedance_kwh | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |

Within-scenario seed SD of cost (training noise, not replication): rl-res+calibrated: 42

| KPI | rbc+calibrated - idle+calibrated | rl-res+calibrated - rbc+calibrated |
|---|---|---|
| safety_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| cost | -241.7 [-664.5, 181] (n=2) | -253.4 [-318, -188.7] (n=2) |
| discomfort_rate | 0 (no variation, n=2) | -0.02837 [-0.1347, 0.07796] (n=2) |
| soc_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| grid_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| power_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| emission | -57.75 [-164.9, 49.36] (n=2) | -103.5 [-154.3, -52.68] (n=2) |
| peak_import_kw | -3.37 [-6.882, 0.1417] (n=2) | -1.66 [-12.24, 8.921] (n=2) |
| avg_daily_peak | -3.676 [-6.782, -0.5712] (n=2) | -2.178 [-12.49, 8.135] (n=2) |
| ramping_rate | 0.9092 [-3.312, 5.13] (n=2) | -0.9246 [-2.177, 0.328] (n=2) |
| discomfort_degree_hours | 0 (no variation, n=2) | -10.99 [-26.41, 4.438] (n=2) |
| pv_self_consumption | 0.09416 [-0.008209, 0.1965] (n=2) | 0.04342 [-0.03553, 0.1224] (n=2) |
| battery_equivalent_full_cycles | 8.312 [0.07404, 16.55] (n=2) | 7.936 [6.469, 9.403] (n=2) |
| hvac_on_transitions_per_building_day | 0 (no variation, n=2) | -0.07154 [-0.384, 0.2409] (n=2) |
| barrier_intervention_rate | -0.5061 [-1.359, 0.3473] (n=2) | 0.06371 [-0.6038, 0.7312] (n=2) |
| cost_cv_across_buildings | 0.0213 [-0.01202, 0.05462] (n=2) | 0.03067 [-0.05521, 0.1165] (n=2) |
| electricity_consumption | -327.6 [-902.3, 247.1] (n=2) | -643.8 [-954.1, -333.4] (n=2) |
| ev_missed_departure_rate | n/a | n/a |
| ev_energy_shortfall_kwh | n/a | n/a |
| cap_exceedance_kwh | 0 (no variation, n=2) | 0 (no variation, n=2) |

## Descriptive: winter

| KPI | idle+calibrated | rbc+calibrated | rl-res+calibrated |
|---|---|---|---|
| safety_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| cost | 2428 [-1255, 6110] (n=2) | 2115 [-1960, 6190] (n=2) | 2235 [-4066, 8535] (n=2) |
| discomfort_rate | 0.03246 [0, 0.2589] (n=2) | 0.03246 [0, 0.2589] (n=2) | 0.05155 [0, 0.5593] (n=2) |
| soc_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| grid_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| power_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |
| emission | 1421 [-953.3, 3795] (n=2) | 1430 [-992.1, 3852] (n=2) | 1488 [-2056, 5032] (n=2) |
| peak_import_kw | 41.14 [15.82, 66.46] (n=2) | 64.25 [62.03, 66.47] (n=2) | 63.58 [33.56, 93.59] (n=2) |
| avg_daily_peak | 25.46 [0.752, 50.16] (n=2) | 37.61 [34.43, 40.79] (n=2) | 34.85 [-8.558, 78.26] (n=2) |
| ramping_rate | 4.732 [4.3, 1] (n=2) | 5.858 [1.625, 1] (n=2) | 5.375 [4.757, 1] (n=2) |
| discomfort_degree_hours | 52.72 [-483.6, 589] (n=2) | 52.72 [-483.6, 589] (n=2) | 105.8 [-1155, 1367] (n=2) |
| pv_self_consumption | 0.6061 [0.35, 0.8623] (n=2) | 0.7132 [0.6157, 0.8108] (n=2) | 0.742 [0.5014, 0.9825] (n=2) |
| battery_equivalent_full_cycles | 0.0505 [0.05049, 0.05051] (n=2) | 13.08 [8.174, 17.99] (n=2) | 16.54 [6.985, 26.1] (n=2) |
| hvac_on_transitions_per_building_day | 1.411 [-1.231, 4.052] (n=2) | 1.411 [-1.231, 4.052] (n=2) | 1.348 [-0.9244, 3.62] (n=2) |
| barrier_intervention_rate | 0.688 [0, 1] (n=2) | 0.1682 [0.08064, 0.2558] (n=2) | 0.2729 [0, 0.6711] (n=2) |
| cost_cv_across_buildings | 0.44 [0.1046, 0.7754] (n=2) | 0.4623 [0.1909, 0.7336] (n=2) | 0.5553 [-0.2993, 1.41] (n=2) |
| electricity_consumption | 8548 [-6630, 2.373e+04] (n=2) | 8643 [-6800, 2.409e+04] (n=2) | 9059 [-1.382e+04, 3.194e+04] (n=2) |
| ev_missed_departure_rate | n/a | n/a | n/a |
| ev_energy_shortfall_kwh | n/a | n/a | n/a |
| cap_exceedance_kwh | 0 (no variation, n=2) | 0 (no variation, n=2) | 0 (no variation, n=2) |

Within-scenario seed SD of cost (training noise, not replication): rl-res+calibrated: 20.3

| KPI | rbc+calibrated - idle+calibrated | rl-res+calibrated - rbc+calibrated |
|---|---|---|
| safety_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| cost | -312.6 [-704.9, 79.65] (n=2) | 119.8 [-2106, 2345] (n=2) |
| discomfort_rate | 0 (no variation, n=2) | 0.01909 [-0.2622, 0.3003] (n=2) |
| soc_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| grid_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| power_violation_rate | 0 (no variation, n=2) | 0 (no variation, n=2) |
| emission | 8.932 [-38.79, 56.65] (n=2) | 57.84 [-1064, 1180] (n=2) |
| peak_import_kw | 23.11 [0.007803, 46.21] (n=2) | -0.6748 [-28.47, 27.12] (n=2) |
| avg_daily_peak | 12.15 [-15.73, 40.04] (n=2) | -2.761 [-49.35, 43.83] (n=2) |
| ramping_rate | 1.126 [-2.674, 4.927] (n=2) | -0.4832 [-4.098, 3.131] (n=2) |
| discomfort_degree_hours | 0 (no variation, n=2) | 53.11 [-671.2, 777.5] (n=2) |
| pv_self_consumption | 0.1071 [-0.05149, 0.2657] (n=2) | 0.02873 [-0.1142, 0.1717] (n=2) |
| battery_equivalent_full_cycles | 13.03 [8.124, 17.94] (n=2) | 3.456 [-1.189, 8.102] (n=2) |
| hvac_on_transitions_per_building_day | 0 (no variation, n=2) | -0.06259 [-0.4318, 0.3067] (n=2) |
| barrier_intervention_rate | -0.5197 [-1.225, 0.1856] (n=2) | 0.1046 [-0.3812, 0.5905] (n=2) |
| cost_cv_across_buildings | 0.02226 [-0.04181, 0.08633] (n=2) | 0.09307 [-1.033, 1.219] (n=2) |
| electricity_consumption | 95.17 [-169.5, 359.8] (n=2) | 416.4 [-7023, 7856] (n=2) |
| ev_missed_departure_rate | n/a | n/a |
| ev_energy_shortfall_kwh | n/a | n/a |
| cap_exceedance_kwh | 0 (no variation, n=2) | 0 (no variation, n=2) |
