from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier

__all__ = ["EVObsLayout", "EVChargerSpec", "EVReadinessBarrier", "steps_to_departure"]


def steps_to_departure(departure_countdown: np.ndarray,
                       connected: np.ndarray) -> np.ndarray:
    d = np.asarray(departure_countdown, dtype=np.float32).reshape(-1)
    conn = np.asarray(connected, dtype=bool).reshape(-1)
    return np.where(conn, np.maximum(d, 0.0), 0.0).astype(np.float32)


@dataclass
class EVObsLayout:
    connected_state: int
    departure_time: int
    required_soc_departure: int
    soc: int
    battery_capacity: int
    incoming_state: Optional[int] = None
    estimated_arrival_time: Optional[int] = None


@dataclass
class EVChargerSpec:
    max_charging_power_kw: np.ndarray
    efficiency: np.ndarray
    action_bound: np.ndarray
    action_index: int


class EVReadinessBarrier(DeadlineStorageBarrier):
    def __init__(self, layout: EVObsLayout, spec: EVChargerSpec,
                 margin: float = 0.0, soc_cap: float = 1.0,
                 rate_derate: float = 0.85, name: str = "ev") -> None:
        self.layout = layout
        self.spec = spec
        f = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)
        self._p_charge = f(spec.max_charging_power_kw)
        self._eta = np.maximum(f(spec.efficiency), 1e-6)
        self._rate_derate = float(rate_derate)
        nominal_capacity = np.full_like(self._p_charge, 50.0)
        super().__init__(
            rate=self._p_charge * self._eta / nominal_capacity,
            action_bound=f(spec.action_bound),
            action_index=spec.action_index,
            capacity=nominal_capacity,
            efficiency=self._eta,
            requirement_fn=self._requirement,
            soc_fn=self._soc,
            margin=margin,
            soc_cap=soc_cap,
            name=name,
        )

    def _col(self, obs_list: List[np.ndarray], idx: int) -> np.ndarray:
        return np.array([float(o[idx]) for o in obs_list], dtype=np.float32)

    def _connected(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return self._col(obs_list, self.layout.connected_state) > 0.5

    def _soc(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return self._col(obs_list, self.layout.soc)

    def current_capacity(self, obs_list: List[np.ndarray]) -> np.ndarray:
        cap = self._col(obs_list, self.layout.battery_capacity)
        return np.where(cap > 1e-3, cap, self.capacity)

    def current_rate(self, obs_list: List[np.ndarray]) -> np.ndarray:
        cap = self.current_capacity(obs_list)
        return np.clip(self._rate_derate * self._p_charge * self._eta
                       / np.maximum(cap, 1e-6), 1e-6, 1.0).astype(np.float32)

    def _requirement(self, obs_list: List[np.ndarray]) -> DeadlineRequirement:
        connected = self._connected(obs_list)
        soc_req = self._col(obs_list, self.layout.required_soc_departure)
        steps = steps_to_departure(
            self._col(obs_list, self.layout.departure_time), connected)
        return DeadlineRequirement(
            soc=np.where(connected, np.maximum(soc_req, 0.0), 0.0),
            steps_to_deadline=steps,
            active=connected,
        )

    def deadline_report(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        u = self.urgency(obs_list)
        at_risk = u["active"] & (u["slack"] < 0.0) & (u["gap"] > 0.0)
        return {"gap": u["gap"], "slack": u["slack"],
                "steps_needed": u["steps_needed"],
                "steps_to_deadline": u["steps_to_deadline"],
                "at_risk": at_risk, "active": u["active"],
                "deficit_kwh": u["deficit_kwh"]}


def unmet_departure_rate(reports: Sequence[Dict[str, np.ndarray]]) -> float:
    num = sum(int(r["at_risk"].sum()) for r in reports)
    den = sum(int(r["active"].sum()) for r in reports)
    return float(num) / den if den else 0.0
