from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from stems.config import CBFConfig

_IDX_PRICE = 21
_IDX_CARBON = 14
_IDX_T_IN = 15
_IDX_T_SET = 27
_IDX_OCCUPANT = 26
_IDX_NET = 20
_IDX_SOC_ELEC = 19
_IDX_SOC_DHW = 18
_IDX_DHW_DEMAND = 25
_IDX_SOLAR = 17

_HVAC_ON = 0.05
_INTERVENTION_TOL = 1e-3
_EV_TOL = 1e-3

_T_THRESHOLD = 2.0


class MetricsCalculator:
    def __init__(
        self,
        num_buildings: int = 3,
        cbf_config: Optional[CBFConfig] = None,
        soc_rate: Optional[np.ndarray] = None,
        heating_setpoint_idx: Optional[int] = None,
        count_soc: bool = True,
        dhw_barrier: Optional[Any] = None,
        hvac_idx: Optional[int] = None,
        control_indices: Optional[List[int]] = None,
        ev_layout: Optional[Dict[str, int]] = None,
        hours_per_step: float = 1.0,
    ) -> None:
        self.B = num_buildings
        self.cbf = cbf_config or CBFConfig()
        self.heating_setpoint_idx = heating_setpoint_idx
        self.count_soc = count_soc
        self.dhw_barrier = dhw_barrier
        self.soc_rate = (None if soc_rate is None
                         else np.asarray(soc_rate, dtype=np.float32).reshape(-1))
        self.hvac_idx = hvac_idx
        self.control_indices = (list(control_indices)
                                if control_indices is not None else None)
        self.ev_layout = dict(ev_layout) if ev_layout is not None else None
        self.hours_per_step = float(hours_per_step)
        self.reset()

    def reset(self) -> None:
        self._net_list: List[np.ndarray] = []
        self._price_list: List[np.ndarray] = []
        self._carbon_list: List[np.ndarray] = []
        self._t_in_list: List[np.ndarray] = []
        self._t_set_list: List[np.ndarray] = []
        self._t_heat_set_list: List[np.ndarray] = []
        self._occupant_list: List[np.ndarray] = []
        self._soc_list: List[np.ndarray] = []
        self._pre_soc_list: List[np.ndarray] = []
        self._action_list: List[np.ndarray] = []
        self._device_action_list: List[np.ndarray] = []
        self._dhw_ready_list: List[np.ndarray] = []
        self._dhw_deficit_list: List[np.ndarray] = []
        self._dhw_soc_list: List[np.ndarray] = []
        self._dhw_cover_list: List[np.ndarray] = []
        self._dhw_draw_list: List[np.ndarray] = []
        self._solar_list: List[np.ndarray] = []
        self._raw_action_list: List[np.ndarray] = []
        self._ev_departures = 0
        self._ev_missed = 0
        self._ev_shortfall_kwh = 0.0
        self._ev_events: Optional[List[Dict[str, float]]] = None

    def add_step(
        self,
        obs_list: List[np.ndarray],
        actions: np.ndarray,
        next_obs_list: List[np.ndarray],
        raw_actions: Optional[np.ndarray] = None,
        device_actions: Optional[np.ndarray] = None,
    ) -> None:
        def extract(obs_list_: List[np.ndarray], idx: int) -> np.ndarray:
            return np.array([obs[idx] for obs in obs_list_], dtype=np.float32)

        self._net_list.append(extract(next_obs_list, _IDX_NET))
        self._t_in_list.append(extract(next_obs_list, _IDX_T_IN))
        self._soc_list.append(extract(next_obs_list, _IDX_SOC_ELEC))
        self._pre_soc_list.append(extract(obs_list, _IDX_SOC_ELEC))
        self._price_list.append(extract(obs_list, _IDX_PRICE))
        self._carbon_list.append(extract(obs_list, _IDX_CARBON))
        self._solar_list.append(extract(obs_list, _IDX_SOLAR))
        self._t_set_list.append(extract(obs_list, _IDX_T_SET))
        if self.heating_setpoint_idx is not None:
            self._t_heat_set_list.append(extract(obs_list, self.heating_setpoint_idx))
        self._occupant_list.append(extract(obs_list, _IDX_OCCUPANT))
        self._action_list.append(actions.copy())
        self._device_action_list.append(
            np.asarray(actions if device_actions is None else device_actions,
                       dtype=np.float32).copy())
        if raw_actions is not None:
            self._raw_action_list.append(np.asarray(raw_actions, dtype=np.float32).copy())

        if self.ev_layout is not None:
            L = self.ev_layout
            conn_pre = extract(obs_list, L["connected_state"]) > 0.5
            conn_post = extract(next_obs_list, L["connected_state"]) > 0.5
            left = conn_pre & ~conn_post
            if left.any():
                gap = np.maximum(extract(obs_list, L["required_soc_departure"])
                                 - extract(obs_list, L["soc"]), 0.0)
                cap = extract(obs_list, L["battery_capacity"])
                self._ev_departures += int(left.sum())
                self._ev_missed += int((left & (gap > _EV_TOL)).sum())
                self._ev_shortfall_kwh += float((gap * cap)[left].sum())

        if self.dhw_barrier is not None:
            r = self.dhw_barrier.readiness(obs_list)
            self._dhw_ready_list.append(r["ready"].astype(np.float32))
            self._dhw_deficit_list.append(r["deficit_kwh"].astype(np.float32))
            self._dhw_soc_list.append(r["soc"].astype(np.float32))
            demand = extract(next_obs_list, _IDX_DHW_DEMAND)
            stored = r["soc"] * self.dhw_barrier.dyn.capacity
            self._dhw_draw_list.append(demand)
            self._dhw_cover_list.append((stored + 1e-9 >= demand).astype(np.float32))

    def add_ev_departures(self, events: List[Dict[str, float]]) -> None:
        if self._ev_events is None:
            self._ev_events = []
        self._ev_events.extend(dict(e) for e in events)

    def compute_all(
        self, baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        if len(self._net_list) == 0:
            return {k: 0.0 for k in [
                "cost", "emission", "avg_daily_peak", "electricity_consumption",
                "ramping_rate", "discomfort_rate", "safety_violation_rate",
            ]}

        net = np.stack(self._net_list, axis=0)
        price = np.stack(self._price_list, axis=0)
        carbon = np.stack(self._carbon_list, axis=0)
        t_in = np.stack(self._t_in_list, axis=0)
        t_set = np.stack(self._t_set_list, axis=0)
        occupant = np.stack(self._occupant_list, axis=0)
        soc = np.stack(self._soc_list, axis=0)

        T, B = net.shape

        cost = float((np.maximum(net, 0.0) * price).sum())

        emission = float((np.maximum(net, 0.0) * carbon).sum())

        total_net = net.sum(axis=1)
        steps_per_day = max(1, int(round(24 / self.hours_per_step)))
        daily_peaks = [float(np.maximum(total_net[s:s + steps_per_day], 0.0).max())
                       for s in range(0, T, steps_per_day)]
        avg_daily_peak = float(np.mean(daily_peaks))

        electricity_consumption = float(np.maximum(net, 0.0).sum())

        if T > 1:
            ramps = np.abs(np.diff(total_net))
            ramping_rate = float(ramps.mean())
        else:
            ramping_rate = 0.0

        occupied_mask = occupant > 0
        if self._t_heat_set_list:
            t_heat = np.stack(self._t_heat_set_list, axis=0)
            invalid = ~((t_heat > 0.0) & (t_heat <= t_set))
            if invalid.any():
                raise ValueError(
                    f"{int(invalid.sum())} building-steps have a heating set point that is "
                    "missing or above the cooling set point; the comfort band is undefined")
            discomfort_mask = ((t_in - t_set) > _T_THRESHOLD) | ((t_heat - t_in) > _T_THRESHOLD)
        else:
            discomfort_mask = np.abs(t_in - t_set) > _T_THRESHOLD
        total_occupied = float(occupied_mask.sum())
        if total_occupied > 0:
            discomfort_rate = float((occupied_mask & discomfort_mask).sum()) / total_occupied
        else:
            discomfort_rate = 0.0

        pre_soc = np.stack(self._pre_soc_list, axis=0)

        if self.count_soc:
            soc_violations = (soc < self.cbf.SOC_min) | (soc > self.cbf.SOC_max)
        else:
            soc_violations = np.zeros_like(soc, dtype=bool)
        if self.soc_rate is not None:
            delta = self.soc_rate[None, :]
            unavoidable_soc = (
                (pre_soc + delta < self.cbf.SOC_min) |
                (pre_soc - delta > self.cbf.SOC_max)
            )
        else:
            unavoidable_soc = np.zeros_like(soc, dtype=bool)
        avoidable_soc = soc_violations & ~unavoidable_soc

        power_violations = np.abs(net) > self.cbf.P_building_max
        grid_total = np.maximum(net, 0.0).sum(axis=1, keepdims=True)
        grid_violations = np.broadcast_to(
            grid_total > self.cbf.P_grid_max, (T, B)
        )

        any_violation = soc_violations | power_violations | grid_violations
        avoidable_violation = avoidable_soc | power_violations | grid_violations
        unavoidable_violation = any_violation & ~avoidable_violation

        safety_violation_rate = float(any_violation.mean())

        result: Dict[str, float] = {
            "cost": cost,
            "emission": emission,
            "avg_daily_peak": avg_daily_peak,
            "electricity_consumption": electricity_consumption,
            "ramping_rate": ramping_rate,
            "discomfort_rate": discomfort_rate,
            "safety_violation_rate": safety_violation_rate,
            "soc_violation_rate": float(soc_violations.mean()),
            "power_violation_rate": float(power_violations.mean()),
            "grid_violation_rate": float(grid_violations.mean()),
            "avoidable_violation_rate": (float(avoidable_violation.mean())
                                         if self.soc_rate is not None else float("nan")),
            "unavoidable_violation_rate": (float(unavoidable_violation.mean())
                                           if self.soc_rate is not None else float("nan")),
        }

        dt = self.hours_per_step
        grid_series = np.maximum(net, 0.0).sum(axis=1)
        peak_import = float(grid_series.max())
        result["peak_import_kw"] = peak_import
        result["load_factor"] = (float(grid_series.mean()) / peak_import
                                 if peak_import > 1e-9 else float("nan"))
        result["cap_exceedance_kwh"] = float(
            np.maximum(grid_series - self.cbf.P_grid_max, 0.0).sum() * dt)

        solar = np.maximum(np.stack(self._solar_list, axis=0), 0.0)
        export = np.maximum(-net, 0.0)
        pv_used = np.clip(solar - export, 0.0, None)
        gross = np.maximum(net + solar, 0.0)
        total_pv, total_gross = float(solar.sum()), float(gross.sum())
        result["pv_self_consumption"] = (float(pv_used.sum()) / total_pv
                                         if total_pv > 1e-9 else float("nan"))
        result["self_sufficiency"] = (float(pv_used.sum()) / total_gross
                                      if total_gross > 1e-9 else float("nan"))

        if self._t_heat_set_list:
            excess = (np.maximum(t_in - t_set - _T_THRESHOLD, 0.0)
                      + np.maximum(t_heat - _T_THRESHOLD - t_in, 0.0))
        else:
            excess = np.maximum(np.abs(t_in - t_set) - _T_THRESHOLD, 0.0)
        result["discomfort_degree_hours"] = float((excess * occupied_mask).sum() * dt / B)

        throughput = np.abs(soc - pre_soc).sum(axis=0) / 2.0
        result["battery_equivalent_full_cycles"] = float(throughput.mean())

        actions_arr = np.stack(self._action_list, axis=0)
        days = max(T * dt / 24.0, 1e-9)
        device_arr = np.stack(self._device_action_list, axis=0)
        if self.hvac_idx is not None and 0 <= self.hvac_idx < device_arr.shape[-1]:
            on = np.abs(device_arr[:, :, self.hvac_idx]) >= _HVAC_ON
            starts = ((on[1:] & ~on[:-1]).sum(axis=0) if T > 1
                      else np.zeros(B, dtype=np.int64))
            result["hvac_on_transitions_per_building_day"] = float(starts.mean() / days)

        if self._raw_action_list and len(self._raw_action_list) == T:
            raw = np.stack(self._raw_action_list, axis=0)
            cols = (self.control_indices if self.control_indices is not None
                    else list(range(actions_arr.shape[-1])))
            diff = np.abs(actions_arr[:, :, cols] - raw[:, :, cols])
            result["barrier_intervention_rate"] = float(
                (diff.max(axis=-1) > _INTERVENTION_TOL).mean())
            result["barrier_intervention_magnitude"] = float(diff.mean())

        cost_b = (np.maximum(net, 0.0) * price).sum(axis=0)
        mean_cost_b = float(cost_b.mean())
        result["cost_cv_across_buildings"] = (float(cost_b.std()) / mean_cost_b
                                              if mean_cost_b > 1e-9 else float("nan"))
        occ_b = occupied_mask.sum(axis=0)
        disc_b = np.where(occ_b > 0,
                          (occupied_mask & discomfort_mask).sum(axis=0) / np.maximum(occ_b, 1),
                          0.0)
        result["discomfort_rate_worst_building"] = float(disc_b.max())

        if self._ev_events is not None:
            short = np.array([max(e["required_soc"] - e["soc"], 0.0) for e in self._ev_events])
            kwh = np.array([s * e["capacity_kwh"] for s, e in zip(short, self._ev_events)])
            n = len(self._ev_events)
            result["ev_departures"] = float(n)
            result["ev_missed_departures"] = float((short > _EV_TOL).sum())
            result["ev_missed_departure_rate"] = (float((short > _EV_TOL).mean()) if n
                                                  else float("nan"))
            result["ev_energy_shortfall_kwh"] = float(kwh.sum())
        elif self.ev_layout is not None:
            result["ev_departures"] = float(self._ev_departures)
            result["ev_missed_departures"] = float(self._ev_missed)
            result["ev_missed_departure_rate"] = (
                self._ev_missed / self._ev_departures if self._ev_departures
                else float("nan"))
            result["ev_energy_shortfall_kwh"] = float(self._ev_shortfall_kwh)

        if self._dhw_ready_list:
            draw = np.stack(self._dhw_draw_list, axis=0)
            cover = np.stack(self._dhw_cover_list, axis=0)
            drawn = draw > 1e-9
            result["dhw_readiness_rate"] = float(np.stack(self._dhw_ready_list).mean())
            result["dhw_deficit_kwh"] = float(np.stack(self._dhw_deficit_list).sum())
            result["dhw_soc_mean"] = float(np.stack(self._dhw_soc_list).mean())
            result["dhw_demand_covered_rate"] = (
                float(cover[drawn].mean()) if drawn.any() else 1.0)

        if baseline_metrics is not None:
            for key in ["cost", "emission", "avg_daily_peak",
                        "electricity_consumption", "ramping_rate"]:
                base = float(baseline_metrics.get(key, 1.0))
                if abs(base) > 1e-10:
                    result[key] = result[key] / base
                else:
                    result[key] = 1.0

        return result
