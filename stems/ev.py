"""Electric-vehicle charging: the declared-deadline instance of the storage barrier.

Where hot water has to *infer* when it will be wanted, an EV states it. CityLearn
exposes, per charger (``citylearn/data.py``, surfaced in ``building.py``):

======================================================  =========================
Observation                                             Role
======================================================  =========================
``electric_vehicle_charger_<id>_connected_state``       is a car plugged in
``..._at_charger_<id>_departure_time``                  **the deadline**
``..._at_charger_<id>_required_soc_departure``          **the requirement**
``..._at_charger_<id>_soc``                             current state
``..._at_charger_<id>_battery_capacity``                capacity of *this* car
``electric_vehicle_charger_<id>_incoming_state``        a car is on its way
``incoming_..._estimated_arrival_time``                 when it will arrive
======================================================  =========================

**Units, verified against the simulator rather than assumed.**
``departure_time`` is *not* a clock hour: ``ChargerSimulation`` documents it as
"number of time steps expected until the EV departs", and the shipped
``charger_1_1.csv`` counts it down 12, 11, 10 ... 0 across consecutive rows. It
is therefore already the remaining-steps quantity the barrier needs, and doing
clock arithmetic on it would be wrong. ``required_soc_departure`` is stored as a
percentage in the CSV and divided by 100 during loading, so the observation is
already normalised to [0, 1]. Absent values default to -1 (times) and -0.1
(SOCs), which is why every field is gated on the connected flag.

So the EV barrier is the same object as the hot-water barrier with the forecast
replaced by declared data, and one genuinely new property: a **positive slack**.
Hot water is wanted as soon as the forecast says so, so its requirement is always
immediately urgent. A car that departs in nine hours needing three hours of
charge has six hours of deferral to spend, which is what makes price- and
carbon-aware scheduling possible at all -- and it is why the 0.6% cost premium
measured for hot-water pre-heating need not repeat here.

Two things make EVs the load that justifies coordination:

* **The capacity is not fixed.** A different car, with a different battery and a
  different rate, may occupy the bay at every arrival, so ``rho`` must be read
  from the observation stream each step rather than calibrated once.
* **Deadlines couple through the connection.** Independent barriers are each
  feasibility-guaranteed, but a fleet charging behind one transformer can be
  *jointly* infeasible -- see ``stems.deadline.coupled_feasibility``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier

__all__ = ["EVObsLayout", "EVChargerSpec", "EVReadinessBarrier", "steps_to_departure"]


def steps_to_departure(departure_countdown: np.ndarray,
                       connected: np.ndarray) -> np.ndarray:
    """Timesteps remaining until departure, from CityLearn's countdown field.

    The observation is already a countdown (see the module docstring), so this
    only clamps it: negative values are CityLearn's "not applicable" sentinel
    (-1) and appear whenever no vehicle is connected.
    """
    d = np.asarray(departure_countdown, dtype=np.float32).reshape(-1)
    conn = np.asarray(connected, dtype=bool).reshape(-1)
    return np.where(conn, np.maximum(d, 0.0), 0.0).astype(np.float32)


@dataclass
class EVObsLayout:
    """Positions of one charger's observations within the observation vector.

    Indices are resolved by *name* upstream (see
    ``STEMSEnvironment.ev_obs_layout``) so this stays correct if a schema
    reorders its observations.
    """
    connected_state: int
    departure_time: int
    required_soc_departure: int
    soc: int
    battery_capacity: int
    incoming_state: Optional[int] = None
    estimated_arrival_time: Optional[int] = None


@dataclass
class EVChargerSpec:
    """Physical limits of one charger, read from the live ``Charger`` object."""
    max_charging_power_kw: np.ndarray
    efficiency: np.ndarray
    action_bound: np.ndarray
    action_index: int


class EVReadinessBarrier(DeadlineStorageBarrier):
    """Deadline barrier for a charger bay (h5).

    The requirement is declared rather than forecast:

        soc_req_i     = required_soc_departure_i    (active only when connected)
        steps_i       = departure_time_i            (already a countdown)
        rho_i         = P_charge_i * eta_i / C_battery_i  (read per step)

    ``soc_cap`` defaults to 1.0: unlike a hot-water tank, whose requirement is
    capped below full to keep a satisfying action available, a car may
    legitimately be asked to charge to 100%. Feasibility is preserved instead by
    the deadline itself -- if the remaining time cannot cover the gap, the
    barrier reports negative slack rather than pretending the requirement is
    reachable.
    """

    def __init__(self, layout: EVObsLayout, spec: EVChargerSpec,
                 margin: float = 0.0, soc_cap: float = 1.0,
                 rate_derate: float = 0.85, name: str = "ev") -> None:
        self.layout = layout
        self.spec = spec
        f = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)
        self._p_charge = f(spec.max_charging_power_kw)
        self._eta = np.maximum(f(spec.efficiency), 1e-6)
        # The nameplate rate P*eta/C over-predicts the state-of-charge gain the
        # simulator actually delivers -- measured at 0.160 against a predicted
        # 0.174, about 8% optimistic, because the storage model also applies its
        # own efficiency and standby loss. Since the projection is a
        # *latest-start* trigger, an optimistic rate means charging begins too
        # late to finish, and the deadline is missed by a hair every time. The
        # rate is therefore derated: it must be conservative in the same
        # direction the battery calibration is.
        self._rate_derate = float(rate_derate)
        # Nominal capacity/rate for the no-vehicle case; overridden per step.
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

    # -- observation readers -------------------------------------------
    def _col(self, obs_list: List[np.ndarray], idx: int) -> np.ndarray:
        return np.array([float(o[idx]) for o in obs_list], dtype=np.float32)

    def _connected(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return self._col(obs_list, self.layout.connected_state) > 0.5

    def _soc(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return self._col(obs_list, self.layout.soc)

    # -- dynamic physics -----------------------------------------------
    def current_capacity(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Battery capacity of the vehicle currently in the bay [kWh]."""
        cap = self._col(obs_list, self.layout.battery_capacity)
        return np.where(cap > 1e-3, cap, self.capacity)

    def current_rate(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """SOC gain per step for the vehicle currently in the bay."""
        cap = self.current_capacity(obs_list)
        return np.clip(self._rate_derate * self._p_charge * self._eta
                       / np.maximum(cap, 1e-6), 1e-6, 1.0).astype(np.float32)

    # -- requirement ----------------------------------------------------
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

    # -- reporting ------------------------------------------------------
    def deadline_report(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Per-bay view of whether the departure requirement is still reachable.

        ``at_risk`` marks a vehicle whose remaining time is already shorter than
        the charging time it still needs. That is not a projection failure -- no
        action can fix it -- so it must be surfaced rather than absorbed.
        """
        u = self.urgency(obs_list)
        at_risk = u["active"] & (u["slack"] < 0.0) & (u["gap"] > 0.0)
        return {"gap": u["gap"], "slack": u["slack"],
                "steps_needed": u["steps_needed"],
                "steps_to_deadline": u["steps_to_deadline"],
                "at_risk": at_risk, "active": u["active"],
                "deficit_kwh": u["deficit_kwh"]}


def unmet_departure_rate(reports: Sequence[Dict[str, np.ndarray]]) -> float:
    """Fraction of connected vehicle-steps whose departure requirement is at risk.

    The EV analogue of ``dhw_readiness_rate``, and the metric that makes
    constraint violation non-trivial: unlike a battery SOC bound, a departure
    deadline can be missed in ways no shield can retroactively repair.
    """
    num = sum(int(r["at_risk"].sum()) for r in reports)
    den = sum(int(r["active"].sum()) for r in reports)
    return float(num) / den if den else 0.0
