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


class EVFleetModel:
    def __init__(self, has_ev: np.ndarray, p_max: np.ndarray, p_min: np.ndarray,
                 eta_charger: np.ndarray, battery: BatteryModel) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.has_ev = np.asarray(has_ev, dtype=bool).reshape(-1)
        self.p_max, self.p_min, self.eta_c = f(p_max), f(p_min), f(eta_charger)
        self.battery = battery
        self.B = self.has_ev.size
        self.sqrt_eta_min = np.sqrt(battery._eta_y.min(axis=1))

    @classmethod
    def from_citylearn(cls, env: Any, slot: int = 0) -> "EVFleetModel":
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
        dummy_eta = np.stack([template[0][0], np.ones_like(template[0][1])])
        dummy_pow = np.stack([template[1][0], np.ones_like(template[1][1])])
        eta_curves = [dummy_eta if c is None else c for c in eta_curves]
        pow_curves = [dummy_pow if c is None else c for c in pow_curves]
        battery = BatteryModel(cap, p_nom, loss, eta_curves, pow_curves,
                               float(env.seconds_per_time_step) / 3600.0)
        return cls(has, p_max, p_min, eta, battery)

    def applied_kw(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        return np.where((a > 0.0) & self.has_ev,
                        np.clip(a * self.p_max, self.p_min, self.p_max), 0.0)

    def _battery_action(self, action: np.ndarray) -> np.ndarray:
        return self.applied_kw(action) * self.eta_c / self.battery.nominal_power

    def next_soc(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.battery.next_soc(soc, self._battery_action(action))

    def draw_kw(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        accepted = self.battery.accepted_kwh(soc, self._battery_action(action))
        return np.where(self.has_ev, accepted / self.eta_c, 0.0)

    def action_for_draw(self, soc: np.ndarray, kw: np.ndarray) -> np.ndarray:
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
        soc = np.asarray(soc, dtype=np.float64)
        target = np.minimum(np.asarray(target, dtype=np.float64), 1.0)
        full = np.ones(self.B)
        reaches = self.next_soc(soc, full) > target
        lo, hi = np.zeros(self.B), np.ones(self.B)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            ok = self.next_soc(soc, mid) >= target
            lo, hi = np.where(ok, lo, mid), np.where(ok, mid, hi)
        action = np.where(reaches, hi, 1.0)
        need = self.next_soc(soc, np.zeros(self.B)) + 1e-9 < target
        return np.where(self.has_ev & need, self.draw_kw(soc, action), 0.0)

    def idle_soc(self, soc: np.ndarray, hours: np.ndarray) -> np.ndarray:
        keep = 1.0 - self.battery.loss
        return np.asarray(soc, dtype=np.float64) * keep ** np.maximum(hours, 0)

    def hours_to_charge_by_departure(self, soc: np.ndarray, target: np.ndarray,
                                     slots: np.ndarray, max_steps: int = 24) -> np.ndarray:
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


@dataclass
class FleetState:
    connected: np.ndarray
    soc: np.ndarray
    target: np.ndarray
    slots: np.ndarray
    base: np.ndarray
    cap: float
    price: Optional[np.ndarray] = None
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
        return max(self.cap - float(np.maximum(self.base[k], 0.0).sum()), 0.0)

    def own_surplus(self, k: int = 0) -> np.ndarray:
        return np.maximum(-self.base[k], 0.0)


def laxity(model: EVFleetModel, state: FleetState) -> np.ndarray:
    need = model.hours_to_charge_by_departure(state.soc, state.target, state.slots)
    active = state.connected & model.has_ev & (need > 0)
    return np.where(active, state.slots - need, np.inf)


def allocate(model: EVFleetModel, state: FleetState, requested_kw: np.ndarray,
             rule: str) -> np.ndarray:
    if rule not in MYOPIC_RULES:
        raise ValueError(f"unknown allocation rule {rule!r}; choose from {MYOPIC_RULES}")
    req = np.where(state.connected & model.has_ev, np.maximum(requested_kw, 0.0), 0.0)
    if rule == "independent":
        return req
    if rule == "static":
        share = state.cap / model.B
        return np.minimum(req, np.maximum(share - state.base[0], 0.0))
    free = np.minimum(req, state.own_surplus())
    need = req - free
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


def schedule(model: EVFleetModel, state: FleetState, requested_kw: Optional[np.ndarray] = None,
             shortfall_weight: Optional[np.ndarray] = None, objective: str = "project",
             now_min: Optional[np.ndarray] = None, now_max: Optional[np.ndarray] = None
             ) -> Dict[str, Any]:
    from scipy.optimize import linprog

    active = np.flatnonzero(state.connected & model.has_ev & (state.slots > 0)
                            & (model.idle_soc(state.soc, state.slots) + 1e-9
                               < np.minimum(state.target, 1.0)))
    now = np.zeros(model.B)
    out = {"now_kw": now, "feasible": True, "shortfall_soc": np.zeros(model.B),
           "total_shortfall_kwh": 0.0, "worst_shortfall_share": 0.0, "binding": False,
           "cap_shadow_price": 0.0, "status": "nothing to charge"}
    if active.size == 0 and objective == "cost":
        return out
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
            row = np.zeros(nvar); row[ix(j, k)], row[iy(j, k)] = 1.0, -1.0
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
        gap = max(min(state.target[i], 1.0) - state.soc[i], 1e-3)
        row = np.zeros(nvar); row[idd(j)], row[iz] = 1.0 / gap, -1.0
        A_ub.append(row); b_ub.append(0.0)
        if objective == "project":
            req = 0.0 if requested_kw is None else float(max(requested_kw[i], 0.0))
            for sign in (1.0, -1.0):
                row = np.zeros(nvar); row[ix(j, 0)], row[iu(j)] = sign, -1.0
                A_ub.append(row); b_ub.append(sign * req)
            cost[iu(j)] = 1.0
            for k in range(K):
                cost[ix(j, k)] += 1e-4 * (K - k) / K
        elif objective == "cost":
            for k in range(K):
                cost[iy(j, k)] += float(state.price[min(k, len(state.price) - 1)])
        else:
            cost[ix(j, 0)] += 1.0 if objective == "min_now" else -1.0
    cost[iz] = 1e-2 * big * float(bat.capacity[active].mean())
    budgets = []
    for k in range(K):
        row = np.zeros(nvar)
        for j in range(n):
            row[iy(j, k)] = 1.0
        cap_k = state.cap if k == 0 or state.cap_later is None else float(state.cap_later)
        budget = cap_k - float(np.maximum(state.base[k, others], 0.0).sum())
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
    used = float(np.maximum(state.base[0, active] + now[active], 0.0).sum())
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
            on = None
        if on is None or on["total_shortfall_kwh"] > off["total_shortfall_kwh"] + 1e-6:
            now_min, now_max, sol = np.minimum(now_min, off_max), off_max, off
        else:
            now_min, sol = on_min, on
    return sol


def apply_dead_band(model: EVFleetModel, state: FleetState, kw: np.ndarray) -> np.ndarray:
    kw = np.asarray(kw, dtype=np.float64).copy()
    small = (kw > 1e-6) & (kw < model.p_min)
    if not small.any():
        return kw
    urgent = laxity(model, state) <= 0
    kw[small & urgent] = model.p_min[small & urgent]
    kw[small & ~urgent] = 0.0
    return kw


class BaseLoadForecaster:
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
        self.history: List[np.ndarray] = []
        self._prev_exogenous: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None
        self._rest: Optional[np.ndarray] = None
        self._prev_known = np.zeros(self.B)
        self._now_known = np.zeros(self.B)
        self._remainders: List[np.ndarray] = []
        self.errors: List[float] = []
        self.day_errors: List[float] = []

    def _exogenous(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([float(o[self.load_index]) - float(o[self.solar_index])
                         for o in obs_list], dtype=np.float64)

    def predict(self, obs_list: Sequence[np.ndarray], horizon: int,
                known_kw: Optional[np.ndarray] = None) -> np.ndarray:
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
        n = len(self._remainders)
        steps = [self._remainders[n - 24 * d] - self._remainders[n - 24 * d - 1]
                 for d in range(1, self.daily_pattern_days + 1) if n - 24 * d - 1 >= 0]
        return np.median(steps, axis=0) if steps else np.zeros(self.B)

    def commit(self, known_kw: np.ndarray) -> None:
        if self.replay is not None or self._rest is None:
            return
        self._now_known = np.asarray(known_kw, dtype=np.float64).reshape(-1)
        self._last_prediction = self._rest + self._now_known

    def observe(self, realised_base: np.ndarray) -> None:
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
        if self.replay is not None or len(self.errors) < 24:
            return 0.0
        recent = np.asarray(self.errors[-self.window:])
        return float(max(np.quantile(recent, self.quantile), 0.0))

    @property
    def later_margin(self) -> float:
        now = self.margin
        if self.replay is not None or len(self.day_errors) < 24:
            return now
        recent = np.asarray(self.day_errors[-self.window:])
        return float(max(np.quantile(recent, self.quantile), now))


@dataclass
class HouseStorage:
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
        a = np.asarray(actions, dtype=np.float64)
        tank = a[:, self.tank_action] if self.tank is not None else np.zeros(len(a))
        return a[:, self.battery_action], tank

    def draw_kw(self, levels: Dict[str, np.ndarray], battery_action: np.ndarray,
                tank_action: np.ndarray) -> np.ndarray:
        kw = self.battery.accepted_kwh(levels["battery_soc"], battery_action) / self.battery.dt
        if self.tank is not None:
            kw = kw + self.tank.drawn_kwh(levels["tank_soc"], tank_action,
                                          levels["demand"]) / self.tank.dt
        return kw

    def floor_action(self, battery_soc: np.ndarray) -> np.ndarray:
        a_lo, _ = self.battery.safe_interval(battery_soc, self.soc_lo, self.soc_hi)
        return np.maximum(np.asarray(a_lo, dtype=np.float64), 0.0)


class FleetShield:
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
            discharge = self.house.draw_kw(levels, np.minimum(a_batt, 0.0),
                                           np.minimum(a_tank, 0.0))
        state = self.state(obs_list, discharge)
        asked = np.where(state.connected, np.maximum(actions[:, self.idx], 0.0), 0.0)
        if self.rule == "independent" and not self.guard:
            self.last = {"margin_kw": self.forecaster.margin, "binding": False,
                         "predicted_import_kw": float(np.maximum(
                             state.base[0] + self.model.draw_kw(state.soc, asked), 0.0).sum())}
            actions[:, self.idx] = asked
            return actions
        requested = self.model.draw_kw(state.soc, asked)
        asked_kw = requested.copy()
        report: Dict[str, Any] = {"margin_kw": self.forecaster.margin}
        if self.rule == "lp":
            sol = schedule_executable(self.model, state, requested, objective="project")
            kw = sol["now_kw"]
            report.update(feasible=sol["feasible"], binding=sol["binding"],
                          shortfall_kwh=sol["total_shortfall_kwh"])
        else:
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
            others = state.base[0] - discharge + kw
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
        lo, hi = 0.0, 1.0
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
        net = np.array([float(o[net_index]) for o in next_obs_list], dtype=np.float64)
        self.forecaster.observe(net - np.asarray(ev_draw_kwh, dtype=np.float64))


def fleet_power_bounds(model: EVFleetModel, state: FleetState) -> Dict[str, Any]:
    lo = schedule(model, state, objective="min_now")
    hi = schedule(model, state, objective="max_now")
    return {"u_min": float(lo["now_kw"].sum()), "u_max": float(hi["now_kw"].sum()),
            "feasible": bool(lo["feasible"]),
            "total_shortfall_kwh": float(lo["total_shortfall_kwh"])}
