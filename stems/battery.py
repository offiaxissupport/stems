"""CityLearn's battery, as a one-step model the safety barrier can invert.

The barrier needs to know which actions keep the state of charge inside its
band. A linear ``soc + a * P/C`` model is wrong in ways that matter:

* the charge/discharge efficiency depends on power (0.79-0.87 at low power,
  ~0.95 near 80% power on the Travis batteries) and enters as ``sqrt(eta)`` on
  charge and ``1/sqrt(eta)`` on discharge, so ``P/C`` over-states every charge and
  under-states every discharge;
* the power the battery accepts depends on its state of charge (96-99% of
  nameplate when empty, 22-29% when full);
* it loses 0.16-0.82% of its charge per hour standing still.

With ``P/C`` the barrier parked the state of charge at 0.0991-0.0998 -- just below
a 0.1 limit -- because its recovery charge was smaller than the standby loss.

This module reproduces ``citylearn.energy_model.Battery.charge`` (2.6.0b1) per
building from the parameters of the live simulator, and inverts it by bisection
(the map is monotone in the action). ``tests/test_battery_model.py`` checks it
against the simulator step by step.

Two terms are deliberately left out, both on the safe side:

* CityLearn refuses to discharge below ``1 - depth_of_discharge``, with a limit
  computed from the *previous* step's efficiency -- hidden state a barrier
  reading observations cannot reproduce. The limiter only ever makes a discharge
  smaller, so this model predicts a state of charge at or below the simulator's.
* Capacity degradation (3e-5 to 9e-5 per cycle) slightly lowers how full the
  battery can get, so near 100% this model predicts up to 8e-4 too high.

Measured over a week of random actions on eight batteries: exact to 1e-4
wherever the limiter is inactive and the battery is not full; elsewhere the
prediction errs toward the bound the barrier is protecting.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class BatteryModel:
    """One-step state-of-charge dynamics of B batteries (vectorised over B).

    ``eta_curves`` and ``power_curves`` are the simulator's
    ``power_efficiency_curve`` (normalised power -> efficiency) and
    ``capacity_power_curve`` (state of charge -> fraction of nameplate power
    available): one (2, n) array of breakpoints and values per building.
    """

    def __init__(self, capacity: np.ndarray, nominal_power: np.ndarray, loss: np.ndarray,
                 eta_curves: List[np.ndarray], power_curves: List[np.ndarray],
                 hours_per_step: float = 1.0) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.capacity, self.nominal_power, self.loss = f(capacity), f(nominal_power), f(loss)
        self.B = self.capacity.size
        self.dt = float(hours_per_step)
        self._eta_x, self._eta_y = self._stack(eta_curves)
        self._pow_x, self._pow_y = self._stack(power_curves)

    @staticmethod
    def _stack(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        curves = [np.asarray(c, dtype=np.float64) for c in curves]
        if len({c.shape for c in curves}) != 1:
            raise ValueError("battery curves must have the same number of breakpoints")
        return np.stack([c[0] for c in curves]), np.stack([c[1] for c in curves])

    # ------------------------------------------------------------------
    @classmethod
    def from_citylearn(cls, buildings: List[Any], seconds_per_time_step: float = 3600.0
                       ) -> "BatteryModel":
        es = [b.electrical_storage for b in buildings]
        return cls(capacity=[e.capacity for e in es],
                   nominal_power=[e.nominal_power for e in es],
                   loss=[e.loss_coefficient for e in es],
                   eta_curves=[np.asarray(e.power_efficiency_curve) for e in es],
                   power_curves=[np.asarray(e.capacity_power_curve) for e in es],
                   hours_per_step=seconds_per_time_step / 3600.0)

    @classmethod
    def linear(cls, soc_rate: np.ndarray) -> "BatteryModel":
        """``soc + a * soc_rate`` exactly: lossless, power-independent, no limits.

        The model the original implementation assumed (with ``soc_rate = 0.1``),
        kept for the uncalibrated ablation arm and for the mock environment.
        """
        rate = np.asarray(soc_rate, dtype=np.float64).reshape(-1)
        flat = np.array([[0.0, 1.0], [1.0, 1.0]])
        return cls(capacity=np.ones_like(rate), nominal_power=rate, loss=np.zeros_like(rate),
                   eta_curves=[flat] * rate.size, power_curves=[flat] * rate.size)

    # ------------------------------------------------------------------
    @staticmethod
    def _lookup(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Row-wise piecewise-linear lookup, with CityLearn's choice of segment
        (``idx = max(0, argmax(x <= xs) - 1)``, so a query past the last
        breakpoint extrapolates the first segment, as the simulator does)."""
        le = x[:, None] <= xs
        idx = np.maximum(np.argmax(le, axis=1) - 1, 0)
        r = np.arange(xs.shape[0])
        x0, x1, y0, y1 = xs[r, idx], xs[r, idx + 1], ys[r, idx], ys[r, idx + 1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _step(self, soc: np.ndarray, action: np.ndarray):
        """(next state of charge, energy at the terminals [kWh]: + in, - out)."""
        soc = np.asarray(soc, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        cap, p_nom = self.capacity, self.nominal_power
        e_init = np.maximum(soc * cap * (1.0 - self.loss), 0.0)           # after standby loss
        max_power = p_nom * self._lookup(self._pow_x, self._pow_y, e_init / cap)
        requested = action * p_nom * self.dt                              # kWh at the terminals
        eta = self._lookup(self._eta_x, self._eta_y,
                           np.minimum(np.abs(requested), max_power) / p_nom)
        root = np.sqrt(eta)
        charge = np.minimum(np.minimum(max_power, p_nom),
                            np.minimum(cap - e_init, np.maximum(requested, 0.0)))
        discharge = np.maximum(-max_power, np.minimum(requested, 0.0))
        e_final = np.where(requested >= 0.0,
                           np.minimum(e_init + charge * root, cap),
                           np.maximum(e_init + discharge / root, 0.0))
        terminal = np.where(requested >= 0.0, (e_final - e_init) / root,
                            (e_final - e_init) * root)
        return e_final / cap, terminal

    def next_soc(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        """State of charge after one step of ``action`` in [-1, 1], per building."""
        return self._step(soc, action)[0]

    def accepted_kwh(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Energy at the battery terminals over the step [kWh]: what it takes in
        (positive) or gives out (negative), after its own limits."""
        return self._step(soc, action)[1]

    # ------------------------------------------------------------------
    def safe_interval(self, soc: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      a_max: float = 1.0, iters: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Per building, the actions [a_lo, a_hi] with ``lo <= next_soc <= hi``.

        ``next_soc`` is non-decreasing in the action, so each end is found by
        bisection (24 halvings of [-1, 1]: 1e-7). Where no action reaches the
        band in one step the interval collapses onto the action that gets
        closest -- full charge below it, full discharge above it: recovery, never
        an empty set.
        """
        soc = np.asarray(soc, dtype=np.float64)
        lo, hi = np.broadcast_to(lo, soc.shape), np.broadcast_to(hi, soc.shape)
        top = self.next_soc(soc, np.full(self.B, a_max))
        bottom = self.next_soc(soc, np.full(self.B, -a_max))

        # smallest action with next_soc >= lo: invariant f(left) < lo <= f(right)
        left, right = np.full(self.B, -a_max), np.full(self.B, a_max)
        for _ in range(iters):
            mid = 0.5 * (left + right)
            ok = self.next_soc(soc, mid) >= lo
            left, right = np.where(ok, left, mid), np.where(ok, mid, right)
        a_lo = np.where(bottom >= lo, -a_max, right)

        # largest action with next_soc <= hi: invariant f(left) <= hi < f(right)
        left, right = np.full(self.B, -a_max), np.full(self.B, a_max)
        for _ in range(iters):
            mid = 0.5 * (left + right)
            ok = self.next_soc(soc, mid) <= hi
            left, right = np.where(ok, mid, left), np.where(ok, right, mid)
        a_hi = np.where(top <= hi, a_max, left)

        a_lo = np.where(top < lo, a_max, a_lo)            # cannot reach the band: recover
        a_hi = np.where(top < lo, a_max, a_hi)
        a_lo = np.where(bottom > hi, -a_max, a_lo)
        a_hi = np.where(bottom > hi, -a_max, a_hi)
        crossed = a_lo > a_hi                             # band narrower than one step
        mid = 0.5 * (a_lo + a_hi)
        return (np.where(crossed, mid, a_lo).astype(np.float32),
                np.where(crossed, mid, a_hi).astype(np.float32))


class TankModel:
    """CityLearn's hot-water tank and its electric heater: what a storage action
    adds to (or takes off) the building's electricity this hour.

    Reproduces ``Building.update_dhw_storage`` and ``StorageDevice.charge``
    (2.6.0b1) for an ``ElectricHeater``. With ``e = action * capacity``:

        charge     stored = min(E + min(e, P * eta_h - demand) * r, C) - E,
                   drawn  = stored / r / eta_h
        discharge  out    = min(-e, demand),  E' = max(E - out / r, 0),
                   the heater serves that much less: drawn = -(E - E') * r / eta_h

    where ``E = soc * C * (1 - loss)`` is the tank after its standby loss, ``r``
    the square root of the round-trip efficiency, ``eta_h`` the heater efficiency
    and ``P`` its nameplate power. The heater serves the hour's hot-water draw
    first, so only what is left of its output can go into the tank: this is the
    one place a charge depends on ``demand``. A heat-pump water heater would need
    the outdoor temperature and is not handled: ``from_citylearn`` refuses it.
    """

    def __init__(self, capacity, heater_power, heater_efficiency, storage_efficiency, loss,
                 hours_per_step: float = 1.0) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.capacity, self.heater_power = f(capacity), f(heater_power)
        self.heater_efficiency, self.storage_efficiency = f(heater_efficiency), f(storage_efficiency)
        self.loss = f(loss)
        self.B = self.capacity.size
        self.dt = float(hours_per_step)

    @classmethod
    def from_citylearn(cls, buildings: List[Any], seconds_per_time_step: float = 3600.0
                       ) -> "TankModel":
        from citylearn.energy_model import ElectricHeater

        for b in buildings:
            if not isinstance(b.dhw_device, ElectricHeater):
                raise TypeError(f"{b.name}: TankModel covers an ElectricHeater, "
                                f"not {type(b.dhw_device).__name__}")
        return cls(capacity=[b.dhw_storage.capacity for b in buildings],
                   heater_power=[b.dhw_device.nominal_power for b in buildings],
                   heater_efficiency=[b.dhw_device.efficiency for b in buildings],
                   storage_efficiency=[b.dhw_storage.round_trip_efficiency for b in buildings],
                   loss=[b.dhw_storage.loss_coefficient for b in buildings],
                   hours_per_step=seconds_per_time_step / 3600.0)

    def drawn_kwh(self, soc: np.ndarray, action: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Electricity the action adds this step [kWh], relative to leaving the
        tank alone: positive when charging, negative when the tank serves demand
        the heater would otherwise have served."""
        soc = np.asarray(soc, dtype=np.float64)
        e = np.asarray(action, dtype=np.float64) * self.capacity * self.dt
        demand = np.maximum(np.asarray(demand, dtype=np.float64), 0.0)
        C, r = self.capacity, self.storage_efficiency
        e_init = np.maximum(soc * C * (1.0 - self.loss), 0.0)
        spare = np.maximum(self.heater_power * self.heater_efficiency * self.dt - demand, 0.0)
        heat_in = np.minimum(np.maximum(e, 0.0), spare)
        stored = np.minimum(e_init + heat_in * r, C) - e_init
        charge = np.maximum(stored, 0.0) / r / self.heater_efficiency
        heat_out = np.minimum(np.maximum(-e, 0.0), demand)
        e_final = np.maximum(e_init - heat_out / np.maximum(r, 1e-9), 0.0)
        discharge = (e_init - e_final) * r / self.heater_efficiency
        return np.where(e >= 0.0, charge, -discharge)
