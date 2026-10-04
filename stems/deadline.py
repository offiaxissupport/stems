"""Deadline-constrained storage: one barrier for hot water, EVs, and batteries.

Motivation
----------
Three of the four controllable devices in this system are the *same*
mathematical object seen through different data:

    a storage device that must reach a required state of charge by a
    deadline, while charging at a finite rate.

===============  ===========================  ==============================
Device           Deadline                     Requirement
===============  ===========================  ==============================
DHW tank         implicit: the pre-heat        forecast demand over the
                 horizon L                     horizon, divided by capacity
Electric vehicle **explicit**:                 **explicit**:
                 ``departure_time``            ``required_soc_departure``
Battery          none (a band, not a           SOC_min (handled by the
                 deadline)                     classical CBF in cbf.py)
===============  ===========================  ==============================

``DeadlineStorageBarrier`` implements the shared mechanism once. A device
supplies a ``requirement_fn`` mapping observations to a ``DeadlineRequirement``;
everything else -- the urgency computation, the monotone projection, the
diagnostics -- is device-agnostic.

Relation to the literature
--------------------------
This is a discrete-time, rate-limited instance of *input-constrained* control
barrier functions: safety under actuator limits is certified by predicting
forward under a backup control (here, "charge at maximum") over a finite
horizon. It is not a new theorem; the contribution is the constructive horizon
rule below and the fact that the rate is *measured from the plant* rather than
assumed. (Note: "readiness barrier function" is used for an unrelated
multirotor actuator-authority construction in arXiv:2608.16335; this is a
different object, hence the name ``DeadlineStorageBarrier``.)

The horizon rule
----------------
With gap ``g = s_req - s`` and per-step rate ``rho``, closing the gap takes

    steps_needed = ceil(g / rho)                                        (1)

so the requirement is satisfiable at the deadline only if charging starts by

    slack = steps_to_deadline - steps_needed >= 0.                      (2)

``slack <= 0`` is a *latest-start* trigger: the device must charge now or miss.
For the DHW tank ``steps_to_deadline = 0``, which makes every unmet requirement
immediately urgent -- exactly the behaviour of the original hot-water barrier,
preserved bit-for-bit so that heat-pump-only benchmarks remain reproducible.

Coupled feasibility
-------------------
Individually each barrier is feasibility-guaranteed: the requirement is capped
below a full store, so a satisfying action always exists. That guarantee does
**not** survive coupling. Under a shared power cap, meeting every deadline
requires

    sum_i (s_req_i - s_i) * C_i / eta_i  <=  P_cap * dt * T,            (3)

where T is the time to the earliest binding deadline. When (3) fails the safe
set is *empty* -- no action sequence satisfies every deadline and the cap -- and
a shield that claims to guarantee safety is lying. ``coupled_feasibility``
detects this, and ``prioritise`` degrades gracefully by earliest-deadline-first
rather than failing silently.
"""

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


# ---------------------------------------------------------------------------
# Requirement description
# ---------------------------------------------------------------------------

@dataclass
class DeadlineRequirement:
    """What a storage device must achieve, and by when.

    Attributes
    ----------
    soc : (B,) required state of charge in [0, 1].
    steps_to_deadline : (B,) whole timesteps until the requirement binds. Zero
        means "must hold now" (the hot-water convention); a positive value
        leaves room to defer, which is what makes price-aware scheduling
        possible for EVs.
    active : (B,) bool -- whether a requirement applies at all. False for an
        empty charger bay, or a device the agent does not currently control.
    """
    soc: np.ndarray
    steps_to_deadline: np.ndarray
    active: np.ndarray

    def __post_init__(self) -> None:
        self.soc = np.asarray(self.soc, dtype=np.float32).reshape(-1)
        self.steps_to_deadline = np.asarray(
            self.steps_to_deadline, dtype=np.float32).reshape(-1)
        self.active = np.asarray(self.active, dtype=bool).reshape(-1)


RequirementFn = Callable[[List[np.ndarray]], DeadlineRequirement]


# ---------------------------------------------------------------------------
# The barrier
# ---------------------------------------------------------------------------

class DeadlineStorageBarrier:
    """Monotone, feasibility-guaranteed barrier for deadline-constrained storage.

    Parameters
    ----------
    rate : (B,) achievable SOC gain per timestep at maximum action (``rho``).
        Must be measured from the plant, not assumed -- mis-calibrating this is
        precisely the defect that produced the original 70% violation rate.
    action_bound : (B,) largest admissible action magnitude per building.
    action_index : position of this device's action in the action vector.
    capacity : (B,) store capacity [kWh], for energy accounting.
    efficiency : (B,) charge-path efficiency, for energy accounting.
    requirement_fn : maps observations to a ``DeadlineRequirement``.
    margin : robust margin added to the required SOC.
    soc_cap : requirement ceiling; keeps a satisfying action always available.
    name : label used in diagnostics.
    """

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

    # -- physics --------------------------------------------------------
    # ``rate`` and ``capacity`` are fixed for a plumbed-in store such as a hot
    # water tank, but not for a charger bay: a different vehicle, with a
    # different battery, may be connected at every arrival. Subclasses override
    # these to read the current occupant from the observation stream.

    def current_rate(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """SOC gain per timestep at maximum action, for the store present now."""
        return self.rate

    def current_capacity(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Capacity [kWh] of the store present now."""
        return self.capacity

    @property
    def time_to_full_h(self) -> np.ndarray:
        """Timesteps to fill an empty store at maximum action (``1 / rho``)."""
        return (1.0 / self.rate).astype(np.float32)

    def action_for_soc_gain(self, gain: np.ndarray,
                            rate: Optional[np.ndarray] = None) -> np.ndarray:
        """Smallest action achieving the requested SOC gain (saturating)."""
        gain = np.asarray(gain, dtype=np.float32)
        rate = self.rate if rate is None else np.maximum(
            np.asarray(rate, dtype=np.float32), 1e-6)
        return np.clip(gain, 0.0, rate) / rate * self.action_bound

    # -- requirement ----------------------------------------------------
    def required_soc(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Required SOC including the robust margin, capped below a full store."""
        req = self.requirement_fn(obs_list)
        soc = np.where(req.active, req.soc + self.margin, 0.0)
        return np.clip(soc, 0.0, self.soc_cap).astype(np.float32)

    def urgency(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Per-building gap, steps needed to close it, and scheduling slack.

        ``slack <= 0`` means the device must charge this step or miss its
        deadline (Eq. 2). ``slack > 0`` is deferrable capacity -- the room a
        price-aware scheduler would exploit.
        """
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

    # -- projection -----------------------------------------------------
    def project(self, actions: np.ndarray, obs_list: List[np.ndarray]) -> np.ndarray:
        """Raise the device action to whatever the deadline demands.

        The projection is **monotone**: it never lowers a nominal action, so it
        can only add readiness and can never veto a policy that already charges
        harder. It is also always satisfiable, since the requirement is capped
        below a full store.
        """
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

    # -- diagnostics ----------------------------------------------------
    def readiness(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Whether each store currently meets its requirement, and by how much."""
        u = self.urgency(obs_list)
        ready = (u["soc"] + 1e-6 >= u["required_soc"]) | (~u["active"])
        return {"soc": u["soc"], "required_soc": u["required_soc"],
                "deficit_kwh": u["deficit_kwh"], "ready": ready,
                "slack": u["slack"], "active": u["active"]}

    def energy_still_required_kwh(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Electrical energy still needed to meet the requirement, per building."""
        u = self.urgency(obs_list)
        return (u["gap"] * u["capacity"] / self.efficiency).astype(np.float32)


# ---------------------------------------------------------------------------
# Coupled feasibility -- where the per-device guarantee breaks
# ---------------------------------------------------------------------------

def coupled_feasibility(barriers: Sequence[DeadlineStorageBarrier],
                        obs_list: List[np.ndarray],
                        power_cap_kw: float,
                        dt_hours: float = 1.0) -> Dict[str, object]:
    """Check whether every deadline can still be met under a shared power cap.

    Implements Eq. 3: the total electrical energy still owed must fit inside the
    cap over the time remaining before the earliest deadline binds.

    A per-device barrier is individually feasibility-guaranteed. Coupling
    destroys that: with several deadline-constrained devices behind one
    transformer, the safe set can be *empty*, and no projection can repair it.
    Detecting the condition is the honest alternative to claiming a guarantee
    that does not hold.

    Returns
    -------
    dict with ``feasible``, the energy owed (``energy_required_kwh``), the
    energy available before the earliest deadline (``energy_available_kwh``),
    the binding horizon in steps (``horizon_steps``), and the shortfall.
    """
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
    # A deadline that binds this step still permits one step of charging.
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
    """Earliest-deadline-first allocation of a power budget that cannot cover all.

    When ``coupled_feasibility`` reports an empty safe set, something must slip.
    Rather than let the outcome fall out of projection order -- which would make
    the victim an artifact of array indexing -- this allocates the available
    power by earliest deadline first (EDF), which is the classical optimal
    priority rule for meeting the largest number of hard deadlines on a single
    resource.

    Returns a per-(device, building) power allocation [kW] and the set of
    requirements that are expected to be missed, so the caller can *report* the
    degradation instead of hiding it.
    """
    budget = float(power_cap_kw) * float(dt_hours)
    entries = []
    for b in barriers:
        u = b.urgency(obs_list)
        need = b.energy_still_required_kwh(obs_list)
        for i in range(len(need)):
            if u["active"][i] and u["gap"][i] > 0.0:
                deadline = float(u["steps_to_deadline"][i])
                entries.append((deadline, b.name, i, float(need[i])))
    entries.sort(key=lambda e: (e[0], -e[3]))     # earliest first; larger need breaks ties

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
