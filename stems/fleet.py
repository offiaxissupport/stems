"""EV charging under one shared power cap: exact plant model, joint feasibility, allocation.

Three pieces.

``EVFleetModel``
    CityLearn's charger and vehicle battery as a one-step map, per building, read
    from the live simulator (``Charger.update_connected_electric_vehicle_soc`` and
    ``Battery.charge``, 2.6.0b1). For a charging action ``a > 0``:

        applied   = clip(a * P_max, P_min, P_max)            [kW at the charger]
        delivered = applied * eta_charger                     [kWh to the battery]
        soc'      = BatteryModel.next_soc(soc, delivered / P_battery)

    so any positive action draws at least ``P_min`` (a dead band), the battery
    accepts less as it fills, and it loses charge while parked. The nameplate
    rate ``P * eta / C`` used before over-predicted the gain by about 8% and was
    patched with a 0.85 derate; this model needs no derate.

``schedule``
    The remaining parking hours of every connected vehicle, the cap, and a
    forecast of the power left for charging form a linear programme. It answers
    exactly (for the model and forecast it is given) the question a per-vehicle
    barrier cannot: *can every deadline still be met together?* When yes, it
    returns the charging vector for this hour closest (in L1) to what the policy
    asked for among those that keep every deadline reachable. When no, the safe
    set is empty and it returns the allocation minimising the weighted energy
    shortfall at departure -- and says so.

``allocate``
    Myopic rules for comparison: scale every request (proportional), earliest
    deadline first, or least laxity first. They look at this hour only.

Approximations of the linear programme, all on the conservative side unless
noted: the battery's power-dependent efficiency is fixed at its minimum; the
1.4 kW dead band is not representable in a linear programme and is applied
afterwards (a tiny allocation is rounded to zero when the vehicle can wait and
up to ``P_min`` when it cannot, which can exceed the cap by at most ``P_min``
per vehicle -- reported as cap exceedance, not hidden).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from stems.battery import BatteryModel, TankModel

__all__ = ["EVFleetModel", "FleetState", "BaseLoadForecaster", "FleetShield", "HouseStorage",
           "schedule",
           "schedule_executable", "allocate", "laxity", "apply_dead_band", "fleet_power_bounds",
           "MYOPIC_RULES",
           "RULES"]

MYOPIC_RULES = ("independent", "static", "proportional", "edf", "llf", "sllf")
RULES = MYOPIC_RULES + ("lp",)


# ---------------------------------------------------------------------------
# Plant model
# ---------------------------------------------------------------------------

class EVFleetModel:
    """One charger bay per building (``has_ev`` False where a building has none)."""

    def __init__(self, has_ev: np.ndarray, p_max: np.ndarray, p_min: np.ndarray,
                 eta_charger: np.ndarray, battery: BatteryModel) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.has_ev = np.asarray(has_ev, dtype=bool).reshape(-1)
        self.p_max, self.p_min, self.eta_c = f(p_max), f(p_min), f(eta_charger)
        self.battery = battery
        self.B = self.has_ev.size
        # Worst-case battery efficiency: the linear programme's (conservative) gain.
        self.sqrt_eta_min = np.sqrt(battery._eta_y.min(axis=1))

    @classmethod
    def from_citylearn(cls, env: Any, slot: int = 0) -> "EVFleetModel":
        """Build from a live ``CityLearnEnv``. Each charger must serve one vehicle
        throughout (true of the schemas used here); anything else raises."""
        buildings = env.buildings
        by_name = {ev.name: ev for ev in env.electric_vehicles}
        flat = np.array([[0.0, 1.0], [1.0, 1.0]])
        has, p_max, p_min, eta = [], [], [], []
        cap, p_nom, loss, eta_curves, pow_curves = [], [], [], [], []
        template = None
        for b in buildings:
            chargers = list(b.electric_vehicle_chargers or [])
            if slot >= len(chargers):
                has.append(False); p_max.append(0.0); p_min.append(0.0); eta.append(1.0)
                cap.append(1.0); p_nom.append(1.0); loss.append(0.0)
                eta_curves.append(None); pow_curves.append(None)
                continue
            c = chargers[slot]
            ids = {str(x) for x in np.asarray(c.charger_simulation.electric_vehicle_id)
                   if str(x) not in ("nan", "", "None")}
            if len(ids) != 1:
                raise ValueError(f"charger {c.charger_id} serves {sorted(ids)}; the fleet "
                                 "model assumes one vehicle per charger")
            if c.charge_efficiency_curve is not None:
                raise ValueError(f"charger {c.charger_id} has a charge efficiency curve; "
                                 "the fleet model assumes a constant charger efficiency")
            bat = by_name[ids.pop()].battery
            has.append(True); p_max.append(c.max_charging_power)
            p_min.append(min(c.min_charging_power, c.max_charging_power)); eta.append(c.efficiency)
            cap.append(bat.capacity); p_nom.append(bat.nominal_power); loss.append(bat.loss_coefficient)
            eta_curves.append(np.asarray(bat.power_efficiency_curve, dtype=np.float64))
            pow_curves.append(np.asarray(bat.capacity_power_curve, dtype=np.float64))
            template = template or (eta_curves[-1], pow_curves[-1])
        if template is None:
            raise ValueError("no building has a charger in this slot")
        # Buildings without a charger get an inert placeholder of the right shape.
        dummy_eta = np.stack([template[0][0], np.ones_like(template[0][1])])
        dummy_pow = np.stack([template[1][0], np.ones_like(template[1][1])])
        eta_curves = [dummy_eta if c is None else c for c in eta_curves]
        pow_curves = [dummy_pow if c is None else c for c in pow_curves]
        battery = BatteryModel(cap, p_nom, loss, eta_curves, pow_curves,
                               float(env.seconds_per_time_step) / 3600.0)
        return cls(has, p_max, p_min, eta, battery)

    # -- one step ---------------------------------------------------------
    def applied_kw(self, action: np.ndarray) -> np.ndarray:
        """Charger power for a charging action in [0, 1] (0 stays 0: the dead band
        starts above it)."""
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        return np.where((a > 0.0) & self.has_ev,
                        np.clip(a * self.p_max, self.p_min, self.p_max), 0.0)

    def _battery_action(self, action: np.ndarray) -> np.ndarray:
        return self.applied_kw(action) * self.eta_c / self.battery.nominal_power

    def next_soc(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Vehicle state of charge after one hour at ``action`` (>= 0)."""
        return self.battery.next_soc(soc, self._battery_action(action))

    def draw_kw(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Grid-side energy the charger takes this hour [kWh]: what the battery
        accepts, divided by the charger efficiency."""
        accepted = self.battery.accepted_kwh(soc, self._battery_action(action))
        return np.where(self.has_ev, accepted / self.eta_c, 0.0)

    def action_for_draw(self, soc: np.ndarray, kw: np.ndarray) -> np.ndarray:
        """Smallest action whose grid draw this hour is at least ``kw``.

        The draw is what the battery accepts, which is below the commanded power
        when the battery limits it (96-99% of nameplate when empty, less as it
        fills). An allocation is a draw; turning it into a command as if the two
        were equal leaves the vehicle a little short every hour -- which showed
        up as one departure in ten missed by a hair with no cap at all.
        """
        soc = np.asarray(soc, dtype=np.float64)
        kw = np.asarray(kw, dtype=np.float64)
        top = self.draw_kw(soc, np.ones(self.B))
        lo, hi = np.zeros(self.B), np.ones(self.B)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            ok = self.draw_kw(soc, mid) >= kw - 1e-9
            lo, hi = np.where(ok, lo, mid), np.where(ok, mid, hi)
        action = np.where(kw >= top - 1e-9, 1.0, hi)
        return np.where((kw > 1e-9) & self.has_ev, action, 0.0).astype(np.float32)

    def useful_kw(self, soc: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Largest charger power a vehicle can put to use this hour [kW]: what its
        battery accepts, and no more than takes it to ``target``.

        A charger asked for more than this draws less than it was allotted, and
        under a shared cap the difference is power nobody else got. Every
        allocation rule is therefore given requests capped at this value.
        """
        soc = np.asarray(soc, dtype=np.float64)
        target = np.minimum(np.asarray(target, dtype=np.float64), 1.0)
        full = np.ones(self.B)
        reaches = self.next_soc(soc, full) > target          # full power overshoots
        lo, hi = np.zeros(self.B), np.ones(self.B)           # least action reaching target
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            ok = self.next_soc(soc, mid) >= target
            lo, hi = np.where(ok, lo, mid), np.where(ok, mid, hi)
        action = np.where(reaches, hi, 1.0)
        need = self.next_soc(soc, np.zeros(self.B)) + 1e-9 < target   # idling falls short
        return np.where(self.has_ev & need, self.draw_kw(soc, action), 0.0)

    def idle_soc(self, soc: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """State of charge after standing still for ``hours`` (standby loss only)."""
        keep = 1.0 - self.battery.loss
        return np.asarray(soc, dtype=np.float64) * keep ** np.maximum(hours, 0)

    def hours_to_charge_by_departure(self, soc: np.ndarray, target: np.ndarray,
                                     slots: np.ndarray, max_steps: int = 24) -> np.ndarray:
        """Fewest hours of full-power charging, placed at the end of the stay, that
        leave the vehicle at ``target`` when it departs.

        The requirement holds at departure, not now: a parked battery loses charge
        every hour, so a vehicle that is exactly at its requirement with an hour to
        go will leave below it. Zero therefore means "can idle until departure and
        still meet it"; ``max_steps + 1`` means it cannot be met.
        """
        soc = np.asarray(soc, dtype=np.float64)
        target = np.minimum(np.asarray(target, dtype=np.float64), 1.0)
        slots = np.asarray(slots, dtype=np.int64)
        need = np.full(self.B, max_steps + 1, dtype=np.int64)
        full = np.ones(self.B)
        for n in range(max_steps + 1):
            pending = need == max_steps + 1
            if not (pending & self.has_ev).any():
                break
            s_ = self.idle_soc(soc, slots - n)
            for _ in range(n):
                s_ = self.next_soc(s_, full)
            need = np.where(pending & (s_ + 1e-9 >= target), n, need)
        return need

    def steps_needed(self, soc: np.ndarray, target: np.ndarray, max_steps: int = 48
                     ) -> np.ndarray:
        """Hours of full-power charging needed to reach ``target`` (``max_steps + 1``
        if it is not reached). Exact for this model, taper and losses included."""
        soc = np.asarray(soc, dtype=np.float64).copy()
        target = np.asarray(target, dtype=np.float64)
        need = np.where(soc + 1e-9 >= target, 0, max_steps + 1).astype(np.int64)
        full = np.ones(self.B)
        for k in range(1, max_steps + 1):
            if not ((need == max_steps + 1) & self.has_ev).any():
                break
            soc = self.next_soc(soc, full)
            need = np.where((need == max_steps + 1) & (soc + 1e-9 >= target), k, need)
        return need


# ---------------------------------------------------------------------------
# Fleet state for one decision
# ---------------------------------------------------------------------------

@dataclass
class FleetState:
    """What the scheduler knows at the start of an hour.

    ``slots`` (B,) is the number of hours a vehicle can still charge in, this one
    included (CityLearn's departure countdown plus one). ``base`` (K, B) is each
    building's net load without EV charging over the coming hours [kW], row 0
    being this hour; it can be negative (PV surplus). ``cap`` is the limit on the
    neighbourhood's import, sum_b max(net_b, 0) -- the quantity the grid KPI
    scores -- so a house's own PV surplus charges its car without using any of
    the shared budget.
    """
    connected: np.ndarray
    soc: np.ndarray
    target: np.ndarray
    slots: np.ndarray
    base: np.ndarray
    cap: float
    price: Optional[np.ndarray] = None
    # Cap the programme plans the *later* hours against (None: the same as
    # ``cap``). A forecast of the hours ahead is wrong by more than the forecast
    # of this hour, so a plan that leans on them needs the wider margin.
    cap_later: Optional[float] = None

    def __post_init__(self) -> None:
        self.connected = np.asarray(self.connected, dtype=bool).reshape(-1)
        self.soc = np.asarray(self.soc, dtype=np.float64).reshape(-1)
        self.target = np.asarray(self.target, dtype=np.float64).reshape(-1)
        self.slots = np.asarray(self.slots, dtype=np.int64).reshape(-1)
        self.base = np.atleast_2d(np.asarray(self.base, dtype=np.float64))
        self.cap = float(self.cap)
        if self.price is not None:
            self.price = np.asarray(self.price, dtype=np.float64).reshape(-1)

    def shared_budget(self, k: int = 0) -> float:
        """Import left under the cap in hour k once every building's base import
        is served [kW]; zero when the base load alone reaches the cap."""
        return max(self.cap - float(np.maximum(self.base[k], 0.0).sum()), 0.0)

    def own_surplus(self, k: int = 0) -> np.ndarray:
        """(B,) PV surplus a building can put into its own car for free [kW]."""
        return np.maximum(-self.base[k], 0.0)


def laxity(model: EVFleetModel, state: FleetState) -> np.ndarray:
    """Hours a vehicle can still wait: ``slots - hours of full charging needed``.

    Zero or less means it must charge at full power from now on (or it is
    already too late). Infinite for a vehicle that is absent, or that can stand
    idle until it leaves and still meet its requirement -- which is not the same
    as being at the requirement now (see ``hours_to_charge_by_departure``).
    """
    need = model.hours_to_charge_by_departure(state.soc, state.target, state.slots)
    active = state.connected & model.has_ev & (need > 0)
    return np.where(active, state.slots - need, np.inf)


# ---------------------------------------------------------------------------
# Myopic allocation rules
# ---------------------------------------------------------------------------

def allocate(model: EVFleetModel, state: FleetState, requested_kw: np.ndarray,
             rule: str) -> np.ndarray:
    """Charger power per building [kW] under this hour's cap, by a myopic rule.

    ``independent`` returns the request unchanged (no enforcement). ``static``
    gives every building a fixed ``cap / B`` of import -- a factored shield that
    needs no communication (ElSayed-Aly et al. 2021) and wastes whatever a
    neighbour leaves unused. The others enforce the same total import and differ
    in who is served first: ``proportional`` scales the part of every request
    that needs shared import alike; ``edf`` serves the earliest departure first;
    ``llf`` the smallest laxity first, ties to the longer remaining charge (the
    less-laxity-and-longer-processing-time order of Xu, Pan and Tong 2016);
    ``sllf`` is smoothed least-laxity-first (Chen et al. 2022): it chooses the
    rates that leave every served vehicle with the same laxity L after this hour,
    L being as large as the budget allows, which maximises the fleet minimum
    laxity and avoids the on/off switching of plain LLF.
    """
    if rule not in MYOPIC_RULES:
        raise ValueError(f"unknown allocation rule {rule!r}; choose from {MYOPIC_RULES}")
    req = np.where(state.connected & model.has_ev, np.maximum(requested_kw, 0.0), 0.0)
    if rule == "independent":
        return req
    if rule == "static":
        share = state.cap / model.B
        return np.minimum(req, np.maximum(share - state.base[0], 0.0))
    free = np.minimum(req, state.own_surplus())          # served by the house's own PV
    need = req - free                                    # competes for the shared budget
    budget = state.shared_budget()
    if need.sum() <= budget + 1e-9:
        return req
    if rule == "proportional":
        return free + need * (budget / need.sum())
    if rule == "sllf":
        return free + _sllf(model, state, need, budget)
    hours_left = model.steps_needed(state.soc, state.target).astype(np.float64)
    key = state.slots.astype(np.float64) if rule == "edf" else laxity(model, state)
    tie = state.slots.astype(np.float64) if rule == "edf" else -hours_left
    order = sorted(np.flatnonzero(need > 0), key=lambda i: (key[i], tie[i], -need[i]))
    grant = np.zeros_like(need)
    for i in order:
        grant[i] = min(need[i], budget)
        budget -= grant[i]
    return free + grant


def _sllf(model: EVFleetModel, state: FleetState, need: np.ndarray, budget: float) -> np.ndarray:
    """Smoothed least-laxity-first split of ``budget`` among ``need`` [kW].

    Chen et al. (2022), Eq. 16-17, with a continuous laxity
    ``l_i = slots_i - e_i / P_i`` (``e_i`` the grid energy still owed):

        r_i = clip(P_i (L - l_i + 1), 0, need_i),   L such that sum_i r_i = budget.

    Charging r_i for an hour moves laxity from l_i to l_i - 1 + r_i / P_i, so every
    vehicle that is neither idle nor at its limit ends the hour at laxity L.
    """
    gap = np.maximum(np.minimum(state.target, 1.0) - state.soc, 0.0)
    owed = gap * model.battery.capacity / (model.eta_c * model.sqrt_eta_min)
    lax = state.slots - owed / np.maximum(model.p_max, 1e-9)
    rates = lambda L: np.clip(model.p_max * (L - lax + 1.0), 0.0, need)
    lo, hi = float(np.min(lax[need > 0])) - 1.0, float(np.max(lax[need > 0])) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rates(mid).sum() < budget else (lo, mid)
    out = rates(hi)
    return out * min(1.0, budget / max(out.sum(), 1e-12))


# ---------------------------------------------------------------------------
# Joint feasibility and minimal intervention (linear programme)
# ---------------------------------------------------------------------------

def schedule(model: EVFleetModel, state: FleetState, requested_kw: Optional[np.ndarray] = None,
             shortfall_weight: Optional[np.ndarray] = None, objective: str = "project",
             now_min: Optional[np.ndarray] = None, now_max: Optional[np.ndarray] = None
             ) -> Dict[str, Any]:
    """Solve the fleet's charging problem over the remaining parking hours.

    Variables: grid-side energy ``x[i, k]`` of vehicle i in hour k (k = 0 is
    now), its state of charge ``s[i, k]`` and its building's import ``y[i, k]``.

        s[i, k+1] = (1 - loss_i) s[i, k] + g_i x[i, k]        g_i = eta_c sqrt(eta_min) / C_i
        eta_c x[i, k] <= P_i (a_m - b_m (1 - loss_i) s[i, k])   each segment m of the battery's
                                                             (concave) power curve
        0 <= x[i, k] <= P_max_i,   x[i, k] = 0 for k >= slots_i
        y[i, k] >= base[k, i] + x[i, k],   y[i, k] >= 0
        sum_i y[i, k] <= cap - sum over other buildings of max(base[k, b], 0)
        s[i, slots_i] + d_i >= target_i,   d_i >= 0             (d_i: shortfall at departure)

    ``objective="project"`` minimises, lexicographically through large weights,
    the weighted energy shortfall, then the largest share of its requirement any
    vehicle misses (so an unavoidable shortfall is spread rather than dumped on one
    car), then the L1 distance of this hour's charging from ``requested_kw``: the
    least change to the request that keeps every deadline reachable. ``objective="cost"`` minimises the weighted shortfall and then the
    cost of the energy that has to be *imported* at ``state.price``.

    ``now_min`` / ``now_max`` (B,) bound this hour's charging per building; they
    are how the charger's dead band is imposed (see ``schedule_executable``).

    Returns ``now_kw`` (this hour's charger power per building), ``feasible``
    (every deadline reachable under this cap and forecast), ``shortfall_soc`` per
    building, ``total_shortfall_kwh`` and ``binding`` (the cap constrains this
    hour).
    """
    from scipy.optimize import linprog

    # A vehicle belongs in the programme if standing idle until it leaves would
    # put it below its requirement (standby loss included), not merely if it is
    # below it now.
    active = np.flatnonzero(state.connected & model.has_ev & (state.slots > 0)
                            & (model.idle_soc(state.soc, state.slots) + 1e-9
                               < np.minimum(state.target, 1.0)))
    now = np.zeros(model.B)
    out = {"now_kw": now, "feasible": True, "shortfall_soc": np.zeros(model.B),
           "total_shortfall_kwh": 0.0, "worst_shortfall_share": 0.0, "binding": False,
           "cap_shadow_price": 0.0, "status": "nothing to charge"}
    if active.size == 0 and objective == "cost":
        return out
    # A request from a vehicle that needs nothing is still honoured in "project"
    # mode (the policy may top up), so it joins the programme with its target met.
    if objective == "project" and requested_kw is not None:
        wants = np.flatnonzero(state.connected & model.has_ev & (state.slots > 0)
                               & (np.asarray(requested_kw) > 1e-9))
        active = np.union1d(active, wants)
    if active.size == 0:
        return out
    if objective not in ("project", "cost", "min_now", "max_now"):
        raise ValueError(f"unknown objective {objective!r}")
    if objective == "cost" and state.price is None:
        raise ValueError("objective='cost' needs state.price")

    K = int(min(state.slots[active].max(), state.base.shape[0]))
    n = active.size
    bat = model.battery
    # Variable layout: x (n*K) | s (n*(K+1)) | y (n*K) | d (n) | z (1) | u (n, project only)
    nx, ns = n * K, n * (K + 1)
    ix = lambda j, k: j * K + k
    is_ = lambda j, k: nx + j * (K + 1) + k
    iy = lambda j, k: nx + ns + j * K + k
    idd = lambda j: nx + ns + nx + j
    iz = nx + ns + nx + n
    iu = lambda j: iz + 1 + j
    nvar = iz + 1 + (n if objective == "project" else 0)

    cost = np.zeros(nvar)
    bounds: List[Any] = [(0.0, None)] * nvar
    A_eq, b_eq, A_ub, b_ub = [], [], [], []

    w = np.ones(model.B) if shortfall_weight is None else np.asarray(shortfall_weight, float)
    big = 1e4
    others = np.setdiff1d(np.arange(model.B), active)
    for j, i in enumerate(active):
        g = model.eta_c[i] * model.sqrt_eta_min[i] / bat.capacity[i]
        keep = 1.0 - bat.loss[i]
        slots_i = int(min(state.slots[i], K))
        row = np.zeros(nvar); row[is_(j, 0)] = 1.0
        A_eq.append(row); b_eq.append(state.soc[i])
        for k in range(K):
            bounds[ix(j, k)] = (0.0, model.p_max[i] if k < slots_i else 0.0)
            row = np.zeros(nvar)
            row[is_(j, k + 1)], row[is_(j, k)], row[ix(j, k)] = 1.0, -keep, -g
            A_eq.append(row); b_eq.append(0.0)
            xs, ys = bat._pow_x[i], bat._pow_y[i]
            for m in range(len(xs) - 1):
                slope = (ys[m + 1] - ys[m]) / (xs[m + 1] - xs[m])
                row = np.zeros(nvar)
                row[ix(j, k)] = model.eta_c[i]
                row[is_(j, k)] = -bat.nominal_power[i] * slope * keep
                A_ub.append(row); b_ub.append(bat.nominal_power[i] * (ys[m] - slope * xs[m]))
            row = np.zeros(nvar); row[ix(j, k)], row[iy(j, k)] = 1.0, -1.0   # y >= base + x
            A_ub.append(row); b_ub.append(-float(state.base[k, i]))
        if now_min is not None or now_max is not None:
            lo = 0.0 if now_min is None else float(now_min[i])
            hi = model.p_max[i] if now_max is None else float(min(now_max[i], model.p_max[i]))
            bounds[ix(j, 0)] = (min(lo, hi), hi)
        for k in range(K + 1):
            bounds[is_(j, k)] = (0.0, 1.0)
        row = np.zeros(nvar); row[is_(j, slots_i)], row[idd(j)] = -1.0, -1.0
        A_ub.append(row); b_ub.append(-min(state.target[i], 1.0))
        cost[idd(j)] = big * w[i] * bat.capacity[i]
        # z >= d_i / gap_i: the largest share of its requirement any vehicle misses.
        gap = max(min(state.target[i], 1.0) - state.soc[i], 1e-3)
        row = np.zeros(nvar); row[idd(j)], row[iz] = 1.0 / gap, -1.0
        A_ub.append(row); b_ub.append(0.0)
        if objective == "project":
            req = 0.0 if requested_kw is None else float(max(requested_kw[i], 0.0))
            for sign in (1.0, -1.0):          # u >= |x0 - req|
                row = np.zeros(nvar); row[ix(j, 0)], row[iu(j)] = sign, -1.0
                A_ub.append(row); b_ub.append(sign * req)
            cost[iu(j)] = 1.0
            # Among equal projections, charge later rather than sooner (keeps the
            # policy's deferral): a vanishing preference that only breaks ties.
            for k in range(K):
                cost[ix(j, k)] += 1e-4 * (K - k) / K
        elif objective == "cost":
            for k in range(K):
                cost[iy(j, k)] += float(state.price[min(k, len(state.price) - 1)])
        else:
            cost[ix(j, 0)] += 1.0 if objective == "min_now" else -1.0
    # Fairness as a tie-break: among allocations with the same (minimal) total
    # shortfall, prefer the one whose worst-off vehicle misses the smallest share
    # of what it needed. Weighted far below a kWh of shortfall, so it never trades
    # delivered energy for evenness.
    cost[iz] = 1e-2 * big * float(bat.capacity[active].mean())
    budgets = []
    for k in range(K):
        row = np.zeros(nvar)
        for j in range(n):
            row[iy(j, k)] = 1.0
        cap_k = state.cap if k == 0 or state.cap_later is None else float(state.cap_later)
        budget = cap_k - float(np.maximum(state.base[k, others], 0.0).sum())
        # The EV buildings' own base import is unavoidable too: the budget can
        # never be below it, or the programme would be infeasible for a reason no
        # charging decision can change.
        floor = float(np.maximum(state.base[k, active], 0.0).sum())
        budgets.append(max(budget, floor))
        A_ub.append(row); b_ub.append(budgets[-1])

    res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq),
                  b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"fleet schedule LP failed: {res.message}")
    x = res.x
    shortfall = np.zeros(model.B)
    for j, i in enumerate(active):
        now[i] = x[ix(j, 0)]
        shortfall[i] = max(x[idd(j)], 0.0)
    shortfall = np.where(shortfall < 1e-6, 0.0, shortfall)
    # Import the EV buildings actually cause this hour (y itself is only bounded
    # below, so it is recomputed from x rather than read from the solution).
    used = float(np.maximum(state.base[0, active] + now[active], 0.0).sum())
    # Shadow price of this hour's cap: what one more kW of import would save, in
    # the programme's objective units (0 when the cap is slack).
    marginals = getattr(getattr(res, "ineqlin", None), "marginals", None)
    cap_row = len(b_ub) - K
    shadow = float(-marginals[cap_row]) if marginals is not None else float("nan")
    return {"now_kw": now, "feasible": bool(shortfall.sum() == 0.0),
            "shortfall_soc": shortfall,
            "total_shortfall_kwh": float((shortfall * bat.capacity).sum()),
            "worst_shortfall_share": float(x[iz]) if shortfall.sum() > 0 else 0.0,
            "binding": bool(used >= budgets[0] - 1e-6), "cap_shadow_price": shadow,
            "status": "ok"}


def schedule_executable(model: EVFleetModel, state: FleetState,
                        requested_kw: Optional[np.ndarray] = None, objective: str = "project",
                        max_rounds: int = 4) -> Dict[str, Any]:
    """``schedule`` with the charger's dead band respected.

    A charger delivers either nothing or at least ``P_min``, which a linear
    programme cannot express. This solves the programme, and while some vehicle
    is given less than ``P_min`` it fixes that vehicle to zero for this hour and
    solves again -- so the others absorb the freed power and the cap still holds.
    If switching a vehicle off costs deliverable energy (the shortfall grows),
    it is fixed to ``P_min`` instead. A handful of small programmes at most.
    """
    sol = schedule(model, state, requested_kw, objective=objective)
    now_min, now_max = np.zeros(model.B), model.p_max.copy()
    for _ in range(max_rounds):
        small = (sol["now_kw"] > 1e-6) & (sol["now_kw"] < model.p_min - 1e-9)
        if not small.any():
            break
        off_max = np.where(small, 0.0, now_max)
        off = schedule(model, state, requested_kw, objective=objective,
                       now_min=np.minimum(now_min, off_max), now_max=off_max)
        if off["total_shortfall_kwh"] <= sol["total_shortfall_kwh"] + 1e-6:
            now_min, now_max, sol = np.minimum(now_min, off_max), off_max, off
            continue
        on_min = np.where(small, model.p_min, now_min)
        try:
            on = schedule(model, state, requested_kw, objective=objective,
                          now_min=on_min, now_max=now_max)
        except RuntimeError:
            # P_min does not fit: a battery too full to take it, or no budget left.
            on = None
        if on is None or on["total_shortfall_kwh"] > off["total_shortfall_kwh"] + 1e-6:
            now_min, now_max, sol = np.minimum(now_min, off_max), off_max, off
        else:
            now_min, sol = on_min, on
    return sol


def apply_dead_band(model: EVFleetModel, state: FleetState, kw: np.ndarray) -> np.ndarray:
    """Make an allocation executable by a charger with a minimum power.

    Any positive action draws at least ``P_min``. An allocation below it is
    dropped when the vehicle can still wait an hour and raised to ``P_min`` when
    it cannot (laxity <= 0), which is the side that protects the deadline.
    """
    kw = np.asarray(kw, dtype=np.float64).copy()
    small = (kw > 1e-6) & (kw < model.p_min)
    if not small.any():
        return kw
    urgent = laxity(model, state) <= 0
    kw[small & urgent] = model.p_min[small & urgent]
    kw[small & ~urgent] = 0.0
    return kw


# ---------------------------------------------------------------------------
# Forecast of the load the fleet has to fit around
# ---------------------------------------------------------------------------

class BaseLoadForecaster:
    """Each building's net load without EV charging, for the coming hours [kW].

    ``replay`` -- a (T, B) array of that load recorded from a run of the same
    scenario without charging. With a non-EV controller that does not react to
    the vehicles this is perfect foresight, and bounds what forecasting can buy.

    Otherwise the forecast is causal. This hour: the non-shiftable load and PV
    are observed, and the controllable remainder (heat pump, hot water, battery)
    is taken to persist from the previous hour. Later hours: the same hour of the
    previous day, or this hour's estimate until a day of history exists.

    ``known_kw`` removes the largest avoidable error of that persistence. The
    shield runs after the house controller has decided, so the stationary
    battery's action for this hour is known and its draw follows from the battery
    model; only the rest of the remainder has to persist. Without it, the hour a
    rule stops discharging eight batteries (21:00 here) is forecast 20 kW too low
    and the cars are given room that is not there.

    ``daily_pattern_days`` handles the scheduled steps no plant model covers (a
    rule that starts heating eight hot-water tanks at 10:00 adds 15 kW in one
    hour). The remainder is forecast as last hour's value plus the change it made
    at this hour of day on the previous days -- the median over up to that many
    days, so one unusual day does not move it. Zero (the default) is plain
    persistence, which is what the shield-level study was run with: its houses
    run on the thermostat alone and have no such steps.

    A causal forecast is wrong by some amount every hour, so the cap is enforced
    against ``cap - margin``, where ``margin`` is the ``quantile`` of the last
    ``window`` one-hour-ahead errors of the neighbourhood import (an online
    conformal margin): the share of hours whose realised import exceeds the
    forecast-plus-margin then tracks ``1 - quantile``.

    ``later_margin`` is the same quantile for the rows the shield *plans* on: the
    later hours are forecast by the same hour of the previous day, and that error
    (import now minus import 24 h ago) is larger than the one-hour-ahead one. A
    shield that defers charging to the last feasible hour is betting on those
    rows, so they carry their own margin.
    """

    def __init__(self, num_buildings: int, replay: Optional[np.ndarray] = None,
                 quantile: float = 0.95, window: int = 168,
                 load_index: int = 16, solar_index: int = 17,
                 daily_pattern_days: int = 0) -> None:
        self.B = int(num_buildings)
        self.daily_pattern_days = int(daily_pattern_days)
        self.replay = None if replay is None else np.asarray(replay, dtype=np.float64)
        self.quantile, self.window = float(quantile), int(window)
        self.load_index, self.solar_index = int(load_index), int(solar_index)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.history: List[np.ndarray] = []          # realised base per hour
        self._prev_exogenous: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None
        self._rest: Optional[np.ndarray] = None       # this hour without the known draw
        self._prev_known = np.zeros(self.B)
        self._now_known = np.zeros(self.B)
        self._remainders: List[np.ndarray] = []      # realised base - observed - known, per hour
        self.errors: List[float] = []
        self.day_errors: List[float] = []            # import now - import 24 h earlier

    def _exogenous(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([float(o[self.load_index]) - float(o[self.solar_index])
                         for o in obs_list], dtype=np.float64)

    def predict(self, obs_list: Sequence[np.ndarray], horizon: int,
                known_kw: Optional[np.ndarray] = None) -> np.ndarray:
        """(horizon, B) forecast, row 0 = the hour about to be simulated.

        ``known_kw``: per building, the draw this hour of devices whose action is
        already decided and whose plant is modelled (the stationary battery).
        """
        if self.replay is not None:
            rows = [self.replay[min(self.t + k, len(self.replay) - 1)] for k in range(horizon)]
            self._last_prediction = np.array(rows[0])
            return np.stack(rows)
        exo = self._exogenous(obs_list)
        known = (np.zeros(self.B) if known_kw is None
                 else np.asarray(known_kw, dtype=np.float64).reshape(-1))
        if self.history and self._prev_exogenous is not None:
            rest = exo + (self.history[-1] - self._prev_exogenous - self._prev_known)
            rest = rest + self._daily_step()
        else:
            rest = exo
        now = rest + known
        rows = [now]
        for k in range(1, horizon):
            back = len(self.history) - 24 + k
            rows.append(self.history[back] if 0 <= back < len(self.history) else rest)
        self._last_prediction = now.copy()
        self._now_exogenous, self._rest, self._now_known = exo, rest, known
        return np.stack(rows)

    def _daily_step(self) -> np.ndarray:
        """Median change of the remainder into this hour of day over the previous
        ``daily_pattern_days`` days (zero until one full day and an hour exist)."""
        n = len(self._remainders)
        steps = [self._remainders[n - 24 * d] - self._remainders[n - 24 * d - 1]
                 for d in range(1, self.daily_pattern_days + 1) if n - 24 * d - 1 >= 0]
        return np.median(steps, axis=0) if steps else np.zeros(self.B)

    def commit(self, known_kw: np.ndarray) -> None:
        """Replace this hour's known draw by the one finally executed (the shield
        may have shed battery charging after the forecast was made)."""
        if self.replay is not None or self._rest is None:
            return
        self._now_known = np.asarray(known_kw, dtype=np.float64).reshape(-1)
        self._last_prediction = self._rest + self._now_known

    def observe(self, realised_base: np.ndarray) -> None:
        """Record the hour just simulated: its net load without EV charging."""
        realised = np.asarray(realised_base, dtype=np.float64).reshape(-1)
        if self._last_prediction is not None:
            self.errors.append(float(np.maximum(realised, 0.0).sum()
                                     - np.maximum(self._last_prediction, 0.0).sum()))
        if self.replay is None and len(self.history) >= 24:
            self.day_errors.append(float(np.maximum(realised, 0.0).sum()
                                         - np.maximum(self.history[-24], 0.0).sum()))
        self.history.append(realised)
        if self.replay is None:
            self._prev_exogenous = getattr(self, "_now_exogenous", None)
            self._prev_known = self._now_known
            if self._prev_exogenous is not None:
                self._remainders.append(realised - self._prev_exogenous - self._prev_known)
        self.t += 1

    @property
    def margin(self) -> float:
        """Calibrated head-room reserve [kW]; zero for a perfect (replay) forecast."""
        if self.replay is not None or len(self.errors) < 24:
            return 0.0
        recent = np.asarray(self.errors[-self.window:])
        return float(max(np.quantile(recent, self.quantile), 0.0))

    @property
    def later_margin(self) -> float:
        """Head-room reserve for the hours after this one [kW]: the quantile of the
        day-ahead forecast's errors, never less than ``margin``. Falls back to
        ``margin`` until a day of such errors exists."""
        now = self.margin
        if self.replay is not None or len(self.day_errors) < 24:
            return now
        recent = np.asarray(self.day_errors[-self.window:])
        return float(max(np.quantile(recent, self.quantile), now))


# ---------------------------------------------------------------------------
# The shield
# ---------------------------------------------------------------------------

@dataclass
class HouseStorage:
    """The stationary battery and the hot-water tank behind the same cap, as the
    cap shield sees them.

    Their actions for this hour are decided before the shield runs, so what they
    draw is computed from their plant models, not forecast. Their charging is
    discretionary -- nothing leaves at a deadline, and in this simulator the
    heater serves every hot-water draw directly whatever the tank holds -- so it
    is what the shield sheds first when the cap binds. ``soc_lo`` / ``soc_hi``
    are the band the battery's state-of-charge barrier enforces: a charge that
    band requires (recovery from below it) is never shed.

    The hot-water demand observation describes the previous hour (measured:
    it equals the simulator's series one step back, exactly), so last hour's
    demand stands in for this hour's where the tank model needs it: to bound a
    discharge, and to bound a charge when the heater is close to its nameplate.
    """

    battery: BatteryModel
    battery_action: int
    battery_soc: int
    soc_lo: Any
    soc_hi: Any
    tank: Optional[TankModel] = None
    tank_action: int = 0
    tank_soc: int = 18
    tank_demand: int = 25

    def read(self, obs_list: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        col = lambda i: np.array([float(o[i]) for o in obs_list], dtype=np.float64)
        return {"battery_soc": col(self.battery_soc), "tank_soc": col(self.tank_soc),
                "demand": col(self.tank_demand)}

    def actions(self, actions: np.ndarray):
        """(battery action, tank action) columns; the tank's is zero without a tank."""
        a = np.asarray(actions, dtype=np.float64)
        tank = a[:, self.tank_action] if self.tank is not None else np.zeros(len(a))
        return a[:, self.battery_action], tank

    def draw_kw(self, levels: Dict[str, np.ndarray], battery_action: np.ndarray,
                tank_action: np.ndarray) -> np.ndarray:
        """Draw of both devices this hour [kW]: + charging, - discharging."""
        kw = self.battery.accepted_kwh(levels["battery_soc"], battery_action) / self.battery.dt
        if self.tank is not None:
            kw = kw + self.tank.drawn_kwh(levels["tank_soc"], tank_action,
                                          levels["demand"]) / self.tank.dt
        return kw

    def floor_action(self, battery_soc: np.ndarray) -> np.ndarray:
        """The battery charge its state-of-charge band requires (zero inside it)."""
        a_lo, _ = self.battery.safe_interval(battery_soc, self.soc_lo, self.soc_hi)
        return np.maximum(np.asarray(a_lo, dtype=np.float64), 0.0)


class FleetShield:
    """Turns requested EV charging into charging that respects the shared cap.

    ``rule``:
        ``independent``  each vehicle on its own: the request, raised to full
                         power once its laxity is spent. No view of the cap.
        ``proportional`` / ``edf`` / ``llf``
                         the same, then this hour's cap enforced by that rule.
        ``lp``           the linear programme of ``schedule``: the closest
                         charging vector to the request from which every deadline
                         remains reachable under the cap, or, when none exists,
                         the one with the least energy shortfall.

    ``guard_deadlines=False`` removes the latest-start trigger from the myopic
    rules, leaving a pure cap allocator.

    ``reserve_hours`` plans every departure that many hours early. A shield that
    defers charging to the last feasible hour has no room left when its forecast
    of the house load turns out wrong; with perfect foresight the reserve is not
    needed, with a causal forecast one hour buys the guarantee back. A departure
    planned ``r`` hours early must still hold when the car actually leaves, so the
    requirement at the planned hour is raised by the standby loss of the hours it
    then stands idle (``target / (1 - loss)^r``). Without that the reserve
    creates the misses it is there to prevent: the car is at its requirement an
    hour early and 0.3% under it at the door.

    ``lead_margin=True`` plans the later hours against ``cap - later_margin``
    (see ``BaseLoadForecaster.later_margin``) instead of the one-hour-ahead
    margin. It is what makes a deferred plan hold when the forecast it rests on
    is the day-ahead one.

    After ``project``, ``last["forced_kw"]`` is, per building, the charging the
    shield added above the request (its deadline rescue) and ``last["cut_kw"]``
    what it took off (the cap).

    ``house`` (``HouseStorage``) puts the stationary batteries and hot-water
    tanks under the same cap. The vehicles are then planned around what those
    devices discharge only, and whatever they wanted to charge is cut back, by
    one common factor, to what is left under the cap once the vehicles are
    served: a deadline outranks arbitrage. Without ``house`` the shield moves
    nothing but the chargers.
    """

    def __init__(self, model: EVFleetModel, layout: Dict[str, int], ev_action_index: int,
                 cap_kw: float, rule: str, forecaster: BaseLoadForecaster,
                 horizon: int = 24, guard_deadlines: bool = True,
                 reserve_hours: int = 0, house: Optional[HouseStorage] = None,
                 lead_margin: bool = False) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
        self.model, self.layout, self.idx = model, dict(layout), int(ev_action_index)
        self.cap, self.rule, self.forecaster = float(cap_kw), rule, forecaster
        self.horizon, self.guard = int(horizon), bool(guard_deadlines)
        self.reserve = int(reserve_hours)
        if house is not None and forecaster.replay is not None:
            raise ValueError("house storage needs the causal forecast: a replayed load "
                             "already contains what it draws")
        self.house = house
        self.lead_margin = bool(lead_margin)
        self.last: Dict[str, Any] = {}

    def state(self, obs_list: Sequence[np.ndarray],
              known_kw: Optional[np.ndarray] = None) -> FleetState:
        col = lambda key: np.array([float(o[self.layout[key]]) for o in obs_list])
        connected = (col("connected_state") > 0.5) & self.model.has_ev
        base = self.forecaster.predict(obs_list, self.horizon, known_kw)
        hours_left = np.where(connected, col("departure_time") + 1, 0).astype(np.int64)
        slots = np.where(connected, np.maximum(hours_left - self.reserve, 1), 0)
        # Hours the car stands idle between the planned completion and its real
        # departure: the planned requirement is raised by their standby loss.
        idle = np.maximum(hours_left - slots, 0)
        keep = 1.0 - self.model.battery.loss
        target = np.where(connected,
                          np.minimum(col("required_soc_departure") / keep ** idle, 1.0), 0.0)
        margin = self.forecaster.margin
        cap_later = (self.cap - max(margin, self.forecaster.later_margin)
                     if self.lead_margin else None)
        return FleetState(connected=connected, soc=np.where(connected, col("soc"), 0.0),
                          target=target, slots=slots, base=base, cap=self.cap - margin,
                          cap_later=cap_later)

    def project(self, actions: np.ndarray, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        levels = discharge = None
        if self.house is not None:
            levels = self.house.read(obs_list)
            a_batt, a_tank = self.house.actions(actions)
            # What the vehicles may count on: the discharges, not the charging.
            discharge = self.house.draw_kw(levels, np.minimum(a_batt, 0.0),
                                           np.minimum(a_tank, 0.0))
        state = self.state(obs_list, discharge)
        asked = np.where(state.connected, np.maximum(actions[:, self.idx], 0.0), 0.0)
        if self.rule == "independent" and not self.guard:
            # No shield at all: the request goes to the charger untouched.
            self.last = {"margin_kw": self.forecaster.margin, "binding": False,
                         "predicted_import_kw": float(np.maximum(
                             state.base[0] + self.model.draw_kw(state.soc, asked), 0.0).sum())}
            actions[:, self.idx] = asked
            return actions
        # Everything below is in grid draw [kW]: what the cap counts.
        requested = self.model.draw_kw(state.soc, asked)
        asked_kw = requested.copy()
        report: Dict[str, Any] = {"margin_kw": self.forecaster.margin}
        if self.rule == "lp":
            sol = schedule_executable(self.model, state, requested, objective="project")
            kw = sol["now_kw"]
            report.update(feasible=sol["feasible"], binding=sol["binding"],
                          shortfall_kwh=sol["total_shortfall_kwh"])
        else:
            # A vehicle out of laxity needs everything it can take; a request is
            # capped at what can be used toward the requirement (see useful_kw).
            useful = self.model.useful_kw(state.soc, state.target)
            if self.guard:
                requested = np.where(laxity(self.model, state) <= 0, useful, requested)
            requested = np.minimum(requested, useful)
            kw = allocate(self.model, state, requested, self.rule)
            report.update(binding=bool(kw.sum() < requested.sum() - 1e-9))
        kw = apply_dead_band(self.model, state, kw)
        report["forced_kw"] = np.maximum(kw - asked_kw, 0.0)
        report["cut_kw"] = np.maximum(asked_kw - kw, 0.0)
        house_kw = state.base[0]
        if self.house is not None:
            others = state.base[0] - discharge + kw      # everything but battery and tank
            actions, storage_kw = self._shed_storage_charging(actions, others, state.cap,
                                                              levels, report)
            self.forecaster.commit(storage_kw)
            house_kw = state.base[0] - discharge + storage_kw
        report["predicted_import_kw"] = float(np.maximum(house_kw + kw, 0.0).sum())
        self.last = report
        actions[:, self.idx] = self.model.action_for_draw(state.soc, kw)
        return actions

    def _shed_storage_charging(self, actions: np.ndarray, others: np.ndarray, cap: float,
                               levels: Dict[str, np.ndarray], report: Dict[str, Any]):
        """Cut battery and tank charging back to what the cap leaves.

        ``others`` is each building's predicted draw without those two devices
        (house load, PV and the vehicle's charging). Battery charging above the
        band's recovery floor and all tank charging are scaled by the largest
        common factor ``s`` in [0, 1] with
        ``sum_b max(others_b + storage_b(s), 0) <= cap``; the import is
        non-decreasing in ``s``, so ``s`` is found by bisection on the plant
        models themselves. Discharging is left alone. If the cap is exceeded with
        no charging at all, nothing here can help and the excess is reported.
        Returns the actions and the storage draw they produce.
        """
        h = self.house
        a_batt, a_tank = h.actions(actions)
        floor = np.minimum(h.floor_action(levels["battery_soc"]), np.maximum(a_batt, 0.0))
        shed_batt, shed_tank = a_batt > floor + 1e-9, a_tank > 1e-9

        def at(s: float):
            batt = np.where(shed_batt, floor + s * (a_batt - floor), a_batt)
            tank = np.where(shed_tank, s * a_tank, a_tank)
            kw = h.draw_kw(levels, batt, tank)
            return float(np.maximum(others + kw, 0.0).sum()), batt, tank, kw

        total, _, _, asked_kw = at(1.0)
        report["storage_shed_kw"] = 0.0
        if total <= cap + 1e-9 or not (shed_batt.any() or shed_tank.any()):
            return actions, asked_kw
        lo, hi = 0.0, 1.0                    # at(hi) is over the cap
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if at(mid)[0] <= cap else (lo, mid)
        _, batt, tank, kw = at(lo)
        report["storage_shed_kw"] = float((asked_kw - kw).sum())
        actions[:, h.battery_action] = batt.astype(np.float32)
        if h.tank is not None:
            actions[:, h.tank_action] = tank.astype(np.float32)
        return actions, kw

    def observe(self, next_obs_list: Sequence[np.ndarray], ev_draw_kwh: np.ndarray,
                net_index: int = 20) -> None:
        """Feed the hour just simulated to the forecaster (its load without EVs)."""
        net = np.array([float(o[net_index]) for o in next_obs_list], dtype=np.float64)
        self.forecaster.observe(net - np.asarray(ev_draw_kwh, dtype=np.float64))


def fleet_power_bounds(model: EVFleetModel, state: FleetState) -> Dict[str, Any]:
    """The fleet's flexibility this hour: the least and the most total charging
    power [kW] from which every deadline is still reachable under the cap.

    ``u_min`` is the joint latest-start requirement -- the coupled counterpart of
    a single vehicle's "must charge now" -- and ``u_max - u_min`` is the room a
    price-aware controller has (the aggregate-flexibility signal of Li et al.
    2021; exact fleet sets: Mukhi et al. 2025). ``feasible`` False means the safe
    set is empty; the bounds are then those of the least-shortfall schedule.
    """
    lo = schedule(model, state, objective="min_now")
    hi = schedule(model, state, objective="max_now")
    return {"u_min": float(lo["now_kw"].sum()), "u_max": float(hi["now_kw"].sum()),
            "feasible": bool(lo["feasible"]),
            "total_shortfall_kwh": float(lo["total_shortfall_kwh"])}
