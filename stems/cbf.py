"""Control Barrier Function safety shield (Eq. 16-20, Algorithm 1).

The shield projects each building's nominal action onto the safe set defined by
three barriers: battery SOC bounds (h1, Eq. 16), per-building net power (h2,
Eq. 17) and total grid import (h3, Eq. 18).

Calibration (the central correctness fix). The battery SOC change per unit
action is **per building** and read from the real CityLearn battery model
(``soc_rate = nominal_power / capacity``, ~0.18-0.53 here), not a single
hard-coded constant. The previous code used 0.1 for every building, under-
estimating the true delta 2-5x, which let "safe" actions blow through the SOC
bounds -- the main source of the ~70% violation rate on real data.

The SOC barrier is decoupled per building, so the projection is solved
analytically (vectorised, no QP), which is both exact and fast enough for the
8760 x 8 step budget. The coupled power/grid barriers are handled by an analytic
guard that scales down import-increasing actions; on the Travis dataset they
essentially never bind (loads ~10-40 kW vs 80/300 kW caps).

Constraint-violation tricks implemented here (ablated via ``SafetyConfig``):
  * feasibility_qp  -- always returns the least-violating feasible action and a
                       recovery action when the state is already out of bounds;
                       never the old "emergency zeros".
  * robust_margins  -- enforce a band strictly inside the reported limits.
  * anticipatory    -- a per-building control-invariance buffer (scaled by the
                       battery's max one-step move) keeping the state recoverable.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from stems.battery import BatteryModel
from stems.config import CBFConfig, SafetyConfig
from stems.deadline import (DeadlineStorageBarrier, coupled_feasibility,
                            prioritise)
from stems.thermal import CoPModel, DHWReadinessBarrier

# STEMS observation indices (see OBS_NAMES in environment.py)
_IDX_T_OUT = 2        # outdoor_dry_bulb_temperature
_IDX_SOC_ELEC = 19    # electrical_storage_soc
_IDX_NET = 20         # net_electricity_consumption


class CBFShield:
    """Feasibility-guaranteed CBF safety shield.

    Parameters
    ----------
    config : CBFConfig
        Reported safety bounds (SOC band, power caps).
    num_buildings : int
        Number of buildings B.
    soc_rate : array-like, shape (B,)
        Per-building SOC change per unit battery action (from env.battery_info).
    nominal_power : array-like, shape (B,) | None
        Per-building battery rated power (kW), for the power barrier.
    action_scale : float
        Action magnitude bound (paper: 1.0).
    elec_idx : int
        Index of the battery action within the action vector.
    safety_cfg : SafetyConfig | None
        Trick switches (robust margins, anticipatory buffer).
    """

    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        num_buildings: int = 8,
        soc_rate: Optional[np.ndarray] = None,
        nominal_power: Optional[np.ndarray] = None,
        action_scale: float = 1.0,
        elec_idx: int = 1,
        safety_cfg: Optional[SafetyConfig] = None,
        enforce_soc: bool = True,
        dhw_barrier: Optional[DHWReadinessBarrier] = None,
        cop_model: Optional[CoPModel] = None,
        hvac_idx: int = -1,
        deadline_barriers: Optional[List[DeadlineStorageBarrier]] = None,
        coordination: str = "independent",
        battery_model: Optional[BatteryModel] = None,
    ) -> None:
        self.cfg = config or CBFConfig()
        self.safety = safety_cfg or SafetyConfig()
        self.B = num_buildings
        self.action_scale = float(action_scale)
        self.elec_idx = int(elec_idx)
        # When the battery is not an agent-controlled actuator (e.g. heat-pump-only
        # studies) the SOC barrier is meaningless and is skipped.
        self.enforce_soc = bool(enforce_soc)
        # The battery model the SOC barrier inverts. ``battery_model`` is the
        # simulator's own dynamics (STEMSEnvironment.battery_model()); a bare
        # ``soc_rate`` gives the linear model soc + a * rate. There is no default:
        # an assumed rate is the mis-calibration this project set out to remove.
        if battery_model is None:
            if soc_rate is None:
                if self.enforce_soc:
                    raise ValueError("CBFShield needs battery_model or soc_rate to enforce "
                                     "the state-of-charge band")
                soc_rate = np.zeros(num_buildings, dtype=np.float32)
            battery_model = BatteryModel.linear(np.asarray(soc_rate, dtype=np.float64))
        self.battery = battery_model
        # Largest one-step move at nameplate power (used by the optional
        # anticipatory buffer and reported as the barrier's calibration).
        self.soc_rate = (battery_model.nominal_power * battery_model.dt
                         / battery_model.capacity).astype(np.float32)
        self.nominal_power = (np.asarray(nominal_power, dtype=np.float32).reshape(-1)
                              if nominal_power is not None else None)
        # The grid-total guard below predicts from the last hour's net load. A cap
        # shield with a forecast (``stems.fleet.FleetShield`` with ``house``)
        # replaces it and switches it off; the per-building guard stays.
        self.grid_guard = True
        # Barrier h4 (anticipatory hot-water readiness) and the weather-dependent
        # CoP model used by the power guard. Both optional; when absent the shield
        # behaves exactly as before (battery-only).
        self.dhw_barrier = dhw_barrier
        self.cop_model = cop_model
        self.hvac_idx = int(hvac_idx)
        # Deadline-constrained stores (hot water, EV bays). ``dhw_barrier`` is
        # kept as its own argument for the heat-pump-only studies, whose results
        # are benchmarked elsewhere; it is simply the first entry of this list.
        # How deadline barriers behave when their joint demand exceeds the grid
        # cap:
        #   "independent"  each projects on its own -- correct per device,
        #                  collectively blind to the shared connection.
        #   "proportional" the cap is enforced, every device scaled equally.
        #   "edf"          the cap is enforced, allocated earliest-deadline-first.
        # independent vs the others measures whether enforcing the shared cap
        # matters; proportional vs edf isolates the value of the priority rule
        # alone, holding the enforced total identical. All three coincide
        # wherever the cap is slack, which is what makes a sweep the right
        # instrument.
        if coordination not in ("independent", "proportional", "edf"):
            raise ValueError("coordination must be 'independent', 'proportional' "
                             f"or 'edf', got {coordination!r}")
        self.coordination = coordination
        self.deadline_barriers: List[DeadlineStorageBarrier] = list(
            deadline_barriers or [])
        if dhw_barrier is not None and dhw_barrier not in self.deadline_barriers:
            self.deadline_barriers.insert(0, dhw_barrier)

    # ------------------------------------------------------------------
    # Enforced (robust + anticipatory) bounds
    # ------------------------------------------------------------------

    def enforced_soc_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-building enforced (SOC_lo, SOC_hi) inside the reported band.

        Combines the fixed robust margin (Trick 4) with the per-building
        control-invariance buffer (Trick 2). The buffer is scaled by the
        battery's largest one-step move so aggressive batteries keep more
        headroom and remain recoverable.
        """
        margin = self.safety.soc_margin if self.safety.robust_margins else 0.0
        margin += self.safety.soc_tolerance
        lo = np.full(self.B, self.cfg.SOC_min + margin, dtype=np.float32)
        hi = np.full(self.B, self.cfg.SOC_max - margin, dtype=np.float32)
        if self.safety.anticipatory:
            buf = 0.5 * self.soc_rate * float(self.safety.invariance_horizon)
            lo = lo + buf
            hi = hi - buf
        # Guarantee a non-empty band even for very aggressive batteries.
        mid = 0.5 * (self.cfg.SOC_min + self.cfg.SOC_max)
        lo = np.minimum(lo, mid - 1e-3)
        hi = np.maximum(hi, mid + 1e-3)
        return lo, hi

    def power_cap(self) -> float:
        derate = self.safety.power_derate if self.safety.robust_margins else 0.0
        return float(self.cfg.P_building_max * (1.0 - derate))

    def grid_cap(self) -> float:
        derate = self.safety.power_derate if self.safety.robust_margins else 0.0
        return float(self.cfg.P_grid_max * (1.0 - derate))

    # ------------------------------------------------------------------
    # Projection (Algorithm 1) -- analytic, feasibility-guaranteed
    # ------------------------------------------------------------------

    def project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        """Project nominal actions onto the safe set.

        Returns the minimally-modified safe action where one exists, and the
        recovery action (drive SOC back toward the band) where the state is
        already outside it. Never returns an all-zero "emergency" action.
        """
        actions = np.asarray(actions, dtype=np.float32)
        B, action_dim = actions.shape
        safe = actions.copy()
        if not self.enforce_soc:
            # No battery control -> no SOC projection, but the thermal barriers
            # (hot-water readiness, CoP-aware power) still apply.
            safe = self._apply_deadline_barriers(safe, states)
            return self._apply_hvac_power_guard(safe, states)
        soc = np.array([float(s[_IDX_SOC_ELEC]) for s in states], dtype=np.float32)
        lo, hi = self.enforced_soc_bounds()
        # The actions that keep the next state of charge in [lo, hi] under the
        # battery model; where the band is out of reach the interval collapses
        # onto the recovery action (full charge below it, full discharge above).
        a_lo, a_hi = self.battery.safe_interval(soc, lo, hi, a_max=self.action_scale)
        # Minimal correction: the nearest point of that interval.
        safe[:, self.elec_idx] = np.clip(actions[:, self.elec_idx], a_lo, a_hi)

        # Coupled power / grid guard (rarely binds on this dataset). Charging
        # increases import; reduce the battery action if a building or the grid
        # would exceed its enforced cap.
        if self.nominal_power is not None:
            safe = self._apply_power_guard(safe, states, a_lo)
        safe = self._apply_deadline_barriers(safe, states)
        return self._apply_hvac_power_guard(safe, states)

    def _apply_power_guard(self, safe: np.ndarray, states: List[np.ndarray],
                           a_lo: np.ndarray) -> np.ndarray:
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        nom = self.nominal_power
        pred_import = net + np.maximum(safe[:, self.elec_idx], 0.0) * nom  # charging adds draw
        p_cap = self.power_cap()
        over = pred_import > p_cap
        for i in np.where(over)[0]:
            # Reduce charge so building import <= cap, but never below what the
            # state-of-charge band requires (charging when below it).
            allowed = max(0.0, (p_cap - net[i]) / max(nom[i], 1e-6))
            safe[i, self.elec_idx] = float(np.clip(allowed, min(a_lo[i], safe[i, self.elec_idx]),
                                                   safe[i, self.elec_idx]))
        if not self.grid_guard:
            return safe
        # Grid total import guard: if the sum of positive imports would exceed the
        # grid cap, scale all charging by the largest common factor that fits.
        # (The factor is not cap / total: only the charging shrinks, not the load
        # under it, and a building exporting PV absorbs its own charge for free.
        # The import is non-decreasing in the factor, so bisection finds it.)
        charge = np.maximum(safe[:, self.elec_idx], 0.0) * nom
        g_cap = self.grid_cap()
        imported = lambda s: float(np.maximum(net + s * charge, 0.0).sum())
        if imported(1.0) > g_cap and charge.sum() > 1e-9:
            lo_s, hi_s = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo_s + hi_s)
                lo_s, hi_s = (mid, hi_s) if imported(mid) <= g_cap else (lo_s, mid)
            charging = safe[:, self.elec_idx] > 0
            # never below the charge the state-of-charge band requires
            keep = np.minimum(np.maximum(a_lo, 0.0), safe[:, self.elec_idx])
            scaled = np.maximum(safe[:, self.elec_idx] * lo_s, keep)
            safe[charging, self.elec_idx] = scaled[charging]
        return safe

    # ------------------------------------------------------------------
    # Deadline-constrained stores (hot water, EV bays)
    # ------------------------------------------------------------------

    def _apply_deadline_barriers(self, safe: np.ndarray,
                                 states: List[np.ndarray]) -> np.ndarray:
        """Apply every deadline barrier in turn.

        Each projection is monotone and touches only its own action index, so
        the order of application does not matter and no barrier can undo
        another. What they *can* do jointly is exceed the power cap -- see
        ``feasibility_report``, which is diagnostic rather than corrective
        precisely because no per-device projection can repair a jointly
        infeasible set.
        """
        for barrier in self.deadline_barriers:
            safe = barrier.project(safe, states)
        if self.coordination != "independent":
            safe = self._apply_shared_power_allocation(safe, states)
        return safe

    def _apply_shared_power_allocation(self, safe: np.ndarray,
                                       states: List[np.ndarray]) -> np.ndarray:
        """Enforce the shared import cap across all deadline-constrained devices.

        Independent projection is correct for each device in isolation but has no
        view of the connection they share, so several stores can collectively
        request more instantaneous power than the cap allows. Note that this is a
        *different* condition from ``coupled_feasibility``: that one asks whether
        enough energy can flow before the earliest deadline, which overnight is
        usually true; this one asks whether the power requested *right now* fits,
        which is what actually binds when a street charges at once.

        Two allocation rules are supported. ``proportional`` scales every request
        by the same factor. ``edf`` serves the earliest deadline first, so a
        vehicle leaving in an hour is charged ahead of one leaving at dawn. Both
        enforce the identical total, so any difference between them is
        attributable to the priority rule and nothing else.

        The allocation is an upper bound: a device asking for less than its share
        keeps its own action.
        """
        cap = self.grid_cap()
        net = np.array([float(s_[_IDX_NET]) for s_ in states], dtype=np.float32)
        baseline = float(np.maximum(net, 0.0).sum())
        headroom = max(cap - baseline, 0.0)

        entries = []          # (deadline, barrier, building, requested kW)
        requested_total = 0.0
        for barrier in self.deadline_barriers:
            power = getattr(barrier, "_p_charge", None)
            if power is None:
                continue      # device exposes no electrical rating; skip
            u = barrier.urgency(states)
            a = np.maximum(safe[:, barrier.action_index], 0.0)
            kw = a * power
            for i in range(len(kw)):
                if kw[i] > 1e-9:
                    entries.append((float(u["steps_to_deadline"][i]), barrier,
                                    i, float(kw[i])))
                    requested_total += float(kw[i])

        if requested_total <= headroom + 1e-9 or requested_total <= 1e-9:
            return safe       # the cap is slack: every rule agrees here

        if self.coordination == "proportional":
            scale = headroom / requested_total
            for _, barrier, i, kw in entries:
                idx = barrier.action_index
                safe[i, idx] = safe[i, idx] * scale
            return safe

        # Earliest deadline first; a larger outstanding request breaks ties.
        entries.sort(key=lambda e: (e[0], -e[3]))
        budget = headroom
        for _, barrier, i, kw in entries:
            grant = min(kw, budget)
            budget -= grant
            idx = barrier.action_index
            safe[i, idx] = safe[i, idx] * (grant / kw) if kw > 1e-9 else 0.0
        return safe

    def feasibility_report(self, states: List[np.ndarray]) -> Optional[dict]:
        """Can every deadline still be met under the grid cap? (Eq. 3.)

        Returns ``None`` when no deadline-constrained store is present. Otherwise
        reports the energy owed, the energy available before the earliest
        deadline, and -- when the safe set is empty -- an earliest-deadline-first
        allocation naming which requirements are expected to slip.

        This is the honest counterpart to the shield's per-device guarantee: the
        SOC barrier is feasibility-guaranteed because it is decoupled, but a
        fleet of deadline-constrained stores behind one cap is not, and a shield
        that silently absorbed that would be claiming a guarantee it cannot keep.
        """
        if not self.deadline_barriers:
            return None
        report = coupled_feasibility(self.deadline_barriers, states,
                                     power_cap_kw=self.grid_cap())
        if not report["feasible"]:
            report["priority"] = prioritise(self.deadline_barriers, states,
                                            power_cap_kw=self.grid_cap())
        return report

    # ------------------------------------------------------------------
    # Weather-aware (CoP) HVAC power guard
    # ------------------------------------------------------------------

    def _apply_hvac_power_guard(self, safe: np.ndarray,
                                states: List[np.ndarray]) -> np.ndarray:
        """Fold the HVAC electrical draw into the per-building/grid power barriers.

        The battery-only guard ignored space conditioning entirely. Heat-pump
        control makes that unsafe to assume: the HVAC action draws
        ``|a| * P_nom`` electrical, and the *thermal service* that buys shrinks
        with the CoP, so a cold hour needs a larger action for the same comfort.
        The guard therefore (i) counts the HVAC draw in the predicted import and
        (ii) reserves extra headroom proportional to the CoP shortfall relative
        to the pump's rated-condition CoP, so the cap is approached more
        cautiously exactly when conditioning is least efficient.
        """
        if self.cop_model is None or self.hvac_idx < 0:
            return safe
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        t_out = np.array([float(s[_IDX_T_OUT]) for s in states], dtype=np.float32)
        # CityLearn splits the one HVAC action by sign -- a_heat = max(a, 0),
        # a_cool = |min(a, 0)| -- and hvac_mode is 3 (both allowed) at every hour
        # of this data, so the mode is the action's own sign. Inferring it from
        # the weather priced a cooling command on a cold day at the heating CoP.
        heating = safe[:, self.hvac_idx] > 0.0

        p_nom = np.where(heating, self.cop_model.p_h, self.cop_model.p_c)
        cop_h = self.cop_model.cop(t_out, heating=True)
        cop_c = self.cop_model.cop(t_out, heating=False)
        cop = np.where(heating, cop_h, cop_c)
        # Rated-condition reference CoP (mild weather): the shortfall below it is
        # the fraction of extra headroom we hold back.
        cop_ref = np.maximum(
            np.where(heating,
                     self.cop_model.cop(np.full_like(t_out, 10.0), heating=True),
                     self.cop_model.cop(np.full_like(t_out, 30.0), heating=False)),
            1e-3)
        shortfall = np.clip(1.0 - cop / cop_ref, 0.0, 0.5)

        a_hvac = safe[:, self.hvac_idx]
        draw = np.abs(a_hvac) * p_nom
        p_cap = self.power_cap() * (1.0 - shortfall)
        pred = net + draw
        over = pred > p_cap
        for i in np.where(over)[0]:
            allowed = max(0.0, (p_cap[i] - net[i]) / max(p_nom[i], 1e-6))
            safe[i, self.hvac_idx] = float(np.sign(a_hvac[i]) *
                                           min(abs(a_hvac[i]), allowed))
        # Aggregate grid guard including HVAC draw.
        total = float(np.maximum(net + np.abs(safe[:, self.hvac_idx]) * p_nom, 0.0).sum())
        g_cap = self.grid_cap()
        if total > g_cap and total > 1e-6:
            safe[:, self.hvac_idx] *= g_cap / total
        return safe

    # ------------------------------------------------------------------
    # Model-based violation predicate (for diagnostics / cost signals)
    # ------------------------------------------------------------------

    def predicted_constraint_costs(self, actions: np.ndarray,
                                   states: List[np.ndarray]) -> np.ndarray:
        """Return a (B, 3) binary cost: would each barrier be violated next step?

        k=0 SOC, k=1 per-building power, k=2 total grid power. Uses the same
        linear model as the projection (so it is consistent with what the shield
        can actually enforce). These feed the Lagrangian cost critics.
        """
        actions = np.asarray(actions, dtype=np.float32)
        B = self.B
        soc = np.array([float(s[_IDX_SOC_ELEC]) for s in states], dtype=np.float32)
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        rate = np.maximum(self.soc_rate, 1e-6)
        next_soc = soc + actions[:, self.elec_idx] * rate

        costs = np.zeros((B, 3), dtype=np.float32)
        costs[:, 0] = ((next_soc < self.cfg.SOC_min) | (next_soc > self.cfg.SOC_max)).astype(np.float32)
        if self.nominal_power is not None:
            pred = net + np.maximum(actions[:, self.elec_idx], 0.0) * self.nominal_power
            costs[:, 1] = (np.abs(pred) > self.cfg.P_building_max).astype(np.float32)
            costs[:, 2] = float(np.maximum(pred, 0.0).sum() > self.cfg.P_grid_max)
        else:
            costs[:, 1] = (np.abs(net) > self.cfg.P_building_max).astype(np.float32)
            costs[:, 2] = float(np.maximum(net, 0.0).sum() > self.cfg.P_grid_max)
        return costs


# ---------------------------------------------------------------------------
# Optional extension (off by default, not part of the headline experiment)
# ---------------------------------------------------------------------------

class NeuralSafetyFilter(nn.Module):
    """Differentiable learned safety filter -- EXPERIMENTAL, off by default.

    A network trained offline on (obs, a_nom) -> a_safe pairs from the CBF
    oracle, with MC-Dropout uncertainty triggering a CBF fallback. It is a
    beyond-paper extension and is *not* used by the headline experiment; the
    feasibility-guaranteed analytic CBF above is. Kept here for ablation only.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128,
                 num_ensemble: int = 5, dropout_rate: float = 0.1,
                 uncertainty_threshold: float = 0.05,
                 cbf_shield: Optional[CBFShield] = None) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.E = num_ensemble
        self.uncertainty_threshold = uncertainty_threshold
        self.cbf = cbf_shield
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())
        nn.init.uniform_(self.head[0].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head[0].bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, a_nom: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(torch.cat([obs, a_nom], dim=-1)))

    def loss(self, obs: torch.Tensor, a_nom: torch.Tensor,
             a_safe_qp: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(obs, a_nom), a_safe_qp)
