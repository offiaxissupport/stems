"""
Evaluation metrics for STEMS (Table I in the paper).

MetricsCalculator accumulates episode data and computes 7 metrics:
    1. cost                  – total electricity cost
    2. emission              – carbon emissions
    3. avg_daily_peak        – (1/D) sum_d max_t sum_i e_{i,t}
    4. electricity_consumption – total grid draw
    5. ramping_rate          – mean |e_t - e_{t-1}| / (T-1)
    6. discomfort_rate       – proportion of occupied steps with |T_in - T_set| > 2°C (absolute)
    7. safety_violation_rate – proportion of steps violating safety constraints (absolute)

Metrics 1-5 are normalised by baseline values when provided (so baseline = 1.0).
Metrics 6-7 are always absolute.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from stems.config import CBFConfig

# Observation indices
_IDX_PRICE = 21
_IDX_CARBON = 14
_IDX_T_IN = 15
_IDX_T_SET = 27
_IDX_OCCUPANT = 26
_IDX_NET = 20
_IDX_SOC_ELEC = 19

# Comfort threshold (matches RewardConfig)
_T_THRESHOLD = 2.0


class MetricsCalculator:
    """Accumulates episode data and computes Table I metrics.

    Parameters
    ----------
    num_buildings : int
        Number of buildings B.
    cbf_config : CBFConfig
        Safety constraint bounds.
    """

    def __init__(
        self,
        num_buildings: int = 3,
        cbf_config: Optional[CBFConfig] = None,
    ) -> None:
        self.B = num_buildings
        self.cbf = cbf_config or CBFConfig()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear accumulated episode data."""
        self._net_list: List[np.ndarray] = []         # (T,B) over time
        self._price_list: List[np.ndarray] = []       # (T,B)
        self._carbon_list: List[np.ndarray] = []      # (T,B)
        self._t_in_list: List[np.ndarray] = []        # (T,B)
        self._t_set_list: List[np.ndarray] = []       # (T,B)
        self._occupant_list: List[np.ndarray] = []    # (T,B)
        self._soc_list: List[np.ndarray] = []         # (T,B)
        self._action_list: List[np.ndarray] = []      # (T,B,action_dim)

    # ------------------------------------------------------------------
    def add_step(
        self,
        obs_list: List[np.ndarray],
        actions: np.ndarray,
        next_obs_list: List[np.ndarray],
    ) -> None:
        """Accumulate one timestep of data."""
        def extract(obs_list_: List[np.ndarray], idx: int) -> np.ndarray:
            return np.array([obs[idx] for obs in obs_list_], dtype=np.float32)

        self._net_list.append(extract(next_obs_list, _IDX_NET))
        self._price_list.append(extract(next_obs_list, _IDX_PRICE))
        self._carbon_list.append(extract(next_obs_list, _IDX_CARBON))
        self._t_in_list.append(extract(next_obs_list, _IDX_T_IN))
        self._t_set_list.append(extract(next_obs_list, _IDX_T_SET))
        self._occupant_list.append(extract(next_obs_list, _IDX_OCCUPANT))
        self._soc_list.append(extract(next_obs_list, _IDX_SOC_ELEC))
        self._action_list.append(actions.copy())

    # ------------------------------------------------------------------
    def compute_all(
        self, baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Compute all 7 metrics.

        Parameters
        ----------
        baseline_metrics : optional dict of {metric_name: baseline_value}
            If provided, metrics 1-5 are normalised as metric / baseline.

        Returns
        -------
        Dict[str, float]
        """
        if len(self._net_list) == 0:
            return {k: 0.0 for k in [
                "cost", "emission", "avg_daily_peak", "electricity_consumption",
                "ramping_rate", "discomfort_rate", "safety_violation_rate",
            ]}

        net = np.stack(self._net_list, axis=0)         # (T, B)
        price = np.stack(self._price_list, axis=0)     # (T, B)
        carbon = np.stack(self._carbon_list, axis=0)   # (T, B)
        t_in = np.stack(self._t_in_list, axis=0)       # (T, B)
        t_set = np.stack(self._t_set_list, axis=0)     # (T, B)
        occupant = np.stack(self._occupant_list, axis=0)  # (T, B)
        soc = np.stack(self._soc_list, axis=0)         # (T, B)

        T, B = net.shape

        # 1. Total electricity cost
        cost = float((net * price).sum())
        cost = max(cost, 0.0)   # clip to non-negative for reporting

        # 2. Carbon emissions
        emission = float((np.maximum(net, 0.0) * carbon).sum())

        # 3. Average daily peak grid load  (1/D) sum_d max_t sum_i e_{i,t}
        total_net = net.sum(axis=1)   # (T,) aggregated across buildings
        hours_per_day = 24
        num_days = max(1, T // hours_per_day)
        daily_peaks = []
        for d in range(num_days):
            start = d * hours_per_day
            end = min(start + hours_per_day, T)
            peak = float(np.maximum(total_net[start:end], 0.0).max())
            daily_peaks.append(peak)
        avg_daily_peak = float(np.mean(daily_peaks))

        # 4. Total grid electricity consumption
        electricity_consumption = float(np.maximum(net, 0.0).sum())

        # 5. Ramping rate  (1/(T-1)) sum_t |e_t - e_{t-1}|
        if T > 1:
            ramps = np.abs(np.diff(total_net))
            ramping_rate = float(ramps.mean())
        else:
            ramping_rate = 0.0

        # 6. Discomfort rate (absolute) – proportion of occupied steps with |T_in - T_set| > threshold
        occupied_mask = occupant > 0   # (T, B)
        discomfort_mask = np.abs(t_in - t_set) > _T_THRESHOLD   # (T, B)
        total_occupied = float(occupied_mask.sum())
        if total_occupied > 0:
            discomfort_rate = float((occupied_mask & discomfort_mask).sum()) / total_occupied
        else:
            discomfort_rate = 0.0

        # 7. Safety violation rate (absolute) – proportion of steps violating SOC bounds
        soc_violations = (soc < self.cbf.SOC_min) | (soc > self.cbf.SOC_max)   # (T, B)
        safety_violation_rate = float(soc_violations.mean())

        result: Dict[str, float] = {
            "cost": cost,
            "emission": emission,
            "avg_daily_peak": avg_daily_peak,
            "electricity_consumption": electricity_consumption,
            "ramping_rate": ramping_rate,
            "discomfort_rate": discomfort_rate,
            "safety_violation_rate": safety_violation_rate,
        }

        # Normalise metrics 1-5 by baseline
        if baseline_metrics is not None:
            for key in ["cost", "emission", "avg_daily_peak",
                        "electricity_consumption", "ramping_rate"]:
                base = float(baseline_metrics.get(key, 1.0))
                if abs(base) > 1e-10:
                    result[key] = result[key] / base
                else:
                    result[key] = 1.0

        return result
