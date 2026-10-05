from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "DeadlineRequirement",
    "DeadlineStorageBarrier",
    "coupled_feasibility",
    "prioritise",
]


@dataclass
class DeadlineRequirement:
    soc: np.ndarray
    steps_to_deadline: np.ndarray
    active: np.ndarray

    def __post_init__(self) -> None:
        self.soc = np.asarray(self.soc, dtype=np.float32).reshape(-1)
        self.steps_to_deadline = np.asarray(
            self.steps_to_deadline, dtype=np.float32).reshape(-1)
        self.active = np.asarray(self.active, dtype=bool).reshape(-1)


RequirementFn = Callable[[List[np.ndarray]], DeadlineRequirement]


class DeadlineStorageBarrier:
    def __init__(
        self,
        rate: np.ndarray,
        action_bound: np.ndarray,
        action_index: int,
        capacity: np.ndarray,
        efficiency: np.ndarray,
        requirement_fn: RequirementFn,
        soc_fn: Callable[[List[np.ndarray]], np.ndarray],
        margin: float = 0.0,
        soc_cap: float = 0.95,
        name: str = "storage",
    ) -> None:
        f = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)
        self.rate = np.maximum(f(rate), 1e-6)
        self.action_bound = f(action_bound)
        self.action_index = int(action_index)
        self.capacity = f(capacity)
        self.efficiency = np.maximum(f(efficiency), 1e-6)
        self.requirement_fn = requirement_fn
        self.soc_fn = soc_fn
        self.margin = float(margin)
        self.soc_cap = float(soc_cap)
        self.name = str(name)


    def current_rate(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return self.rate

    def current_capacity(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return self.capacity

    @property
    def time_to_full_h(self) -> np.ndarray:
        return (1.0 / self.rate).astype(np.float32)

    def action_for_soc_gain(self, gain: np.ndarray,
                            rate: Optional[np.ndarray] = None) -> np.ndarray:
        gain = np.asarray(gain, dtype=np.float32)
        rate = self.rate if rate is None else np.maximum(
            np.asarray(rate, dtype=np.float32), 1e-6)
        return np.clip(gain, 0.0, rate) / rate * self.action_bound

    def required_soc(self, obs_list: List[np.ndarray]) -> np.ndarray:
        req = self.requirement_fn(obs_list)
        soc = np.where(req.active, req.soc + self.margin, 0.0)
        return np.clip(soc, 0.0, self.soc_cap).astype(np.float32)

    def urgency(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        req = self.requirement_fn(obs_list)
        soc = np.asarray(self.soc_fn(obs_list), dtype=np.float32).reshape(-1)
        rate = np.maximum(np.asarray(self.current_rate(obs_list),
                                     dtype=np.float32).reshape(-1), 1e-6)
        capacity = np.asarray(self.current_capacity(obs_list),
                              dtype=np.float32).reshape(-1)
        target = np.clip(np.where(req.active, req.soc + self.margin, 0.0),
                         0.0, self.soc_cap)
        gap = np.where(req.active, np.maximum(target - soc, 0.0), 0.0)
        steps_needed = np.ceil(gap / rate)
        slack = req.steps_to_deadline - steps_needed
        return {"soc": soc, "required_soc": target.astype(np.float32),
                "gap": gap.astype(np.float32),
                "steps_needed": steps_needed.astype(np.float32),
                "steps_to_deadline": req.steps_to_deadline.astype(np.float32),
                "slack": slack.astype(np.float32),
                "active": req.active,
                "rate": rate.astype(np.float32),
                "capacity": capacity.astype(np.float32),
                "deficit_kwh": (gap * capacity).astype(np.float32)}

    def project(self, actions: np.ndarray, obs_list: List[np.ndarray]) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        u = self.urgency(obs_list)
        must_charge = (u["slack"] <= 0.0) & u["active"] & (u["gap"] > 0.0)
        a_min = np.where(must_charge,
                         np.minimum(self.action_for_soc_gain(u["gap"], u["rate"]),
                                    self.action_bound),
                         0.0)
        actions[:, self.action_index] = np.maximum(
            actions[:, self.action_index], a_min)
        return actions

    def readiness(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        u = self.urgency(obs_list)
        ready = (u["soc"] + 1e-6 >= u["required_soc"]) | (~u["active"])
        return {"soc": u["soc"], "required_soc": u["required_soc"],
                "deficit_kwh": u["deficit_kwh"], "ready": ready,
                "slack": u["slack"], "active": u["active"]}

    def energy_still_required_kwh(self, obs_list: List[np.ndarray]) -> np.ndarray:
        u = self.urgency(obs_list)
        return (u["gap"] * u["capacity"] / self.efficiency).astype(np.float32)


def coupled_feasibility(barriers: Sequence[DeadlineStorageBarrier],
                        obs_list: List[np.ndarray],
                        power_cap_kw: float,
                        dt_hours: float = 1.0) -> Dict[str, object]:
    if not barriers:
        return {"feasible": True, "energy_required_kwh": 0.0,
                "energy_available_kwh": float(power_cap_kw * dt_hours),
                "horizon_steps": 0.0, "shortfall_kwh": 0.0, "per_device": {}}

    per_device: Dict[str, float] = {}
    total_required = 0.0
    horizon = np.inf
    for b in barriers:
        u = b.urgency(obs_list)
        e = float(b.energy_still_required_kwh(obs_list).sum())
        per_device[b.name] = e
        total_required += e
        owing = u["active"] & (u["gap"] > 0.0)
        if np.any(owing):
            horizon = min(horizon, float(np.min(u["steps_to_deadline"][owing])))
    if not np.isfinite(horizon):
        horizon = 0.0
    steps = max(horizon, 1.0)
    available = float(power_cap_kw) * float(dt_hours) * steps
    shortfall = max(0.0, total_required - available)
    return {"feasible": shortfall <= 1e-9,
            "energy_required_kwh": total_required,
            "energy_available_kwh": available,
            "horizon_steps": float(steps),
            "shortfall_kwh": shortfall,
            "per_device": per_device}


def prioritise(barriers: Sequence[DeadlineStorageBarrier],
               obs_list: List[np.ndarray],
               power_cap_kw: float,
               dt_hours: float = 1.0) -> Dict[str, np.ndarray]:
    budget = float(power_cap_kw) * float(dt_hours)
    entries = []
    for b in barriers:
        u = b.urgency(obs_list)
        need = b.energy_still_required_kwh(obs_list)
        for i in range(len(need)):
            if u["active"][i] and u["gap"][i] > 0.0:
                deadline = float(u["steps_to_deadline"][i])
                entries.append((deadline, b.name, i, float(need[i])))
    entries.sort(key=lambda e: (e[0], -e[3]))

    allocation: Dict[str, np.ndarray] = {
        b.name: np.zeros(len(b.rate), dtype=np.float32) for b in barriers}
    missed: List[tuple] = []
    for deadline, name, i, need in entries:
        grant = min(need, budget)
        allocation[name][i] = grant
        budget -= grant
        if grant + 1e-9 < need:
            missed.append((name, i, need - grant, deadline))
    return {"allocation_kwh": allocation, "missed": missed,
            "budget_remaining_kwh": max(budget, 0.0)}
