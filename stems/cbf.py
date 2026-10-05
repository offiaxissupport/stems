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

_IDX_T_OUT = 2
_IDX_SOC_ELEC = 19
_IDX_NET = 20


class CBFShield:
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
        self.enforce_soc = bool(enforce_soc)
        if battery_model is None:
            if soc_rate is None:
                if self.enforce_soc:
                    raise ValueError("CBFShield needs battery_model or soc_rate to enforce "
                                     "the state-of-charge band")
                soc_rate = np.zeros(num_buildings, dtype=np.float32)
            battery_model = BatteryModel.linear(np.asarray(soc_rate, dtype=np.float64))
        self.battery = battery_model
        self.soc_rate = (battery_model.nominal_power * battery_model.dt
                         / battery_model.capacity).astype(np.float32)
        self.nominal_power = (np.asarray(nominal_power, dtype=np.float32).reshape(-1)
                              if nominal_power is not None else None)
        self.grid_guard = True
        self.dhw_barrier = dhw_barrier
        self.cop_model = cop_model
        self.hvac_idx = int(hvac_idx)
        if coordination not in ("independent", "proportional", "edf"):
            raise ValueError("coordination must be 'independent', 'proportional' "
                             f"or 'edf', got {coordination!r}")
        self.coordination = coordination
        self.deadline_barriers: List[DeadlineStorageBarrier] = list(
            deadline_barriers or [])
        if dhw_barrier is not None and dhw_barrier not in self.deadline_barriers:
            self.deadline_barriers.insert(0, dhw_barrier)


    def enforced_soc_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        margin = self.safety.soc_margin if self.safety.robust_margins else 0.0
        margin += self.safety.soc_tolerance
        lo = np.full(self.B, self.cfg.SOC_min + margin, dtype=np.float32)
        hi = np.full(self.B, self.cfg.SOC_max - margin, dtype=np.float32)
        if self.safety.anticipatory:
            buf = 0.5 * self.soc_rate * float(self.safety.invariance_horizon)
            lo = lo + buf
            hi = hi - buf
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


    def project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        B, action_dim = actions.shape
        safe = actions.copy()
        if not self.enforce_soc:
            safe = self._apply_deadline_barriers(safe, states)
            return self._apply_hvac_power_guard(safe, states)
        soc = np.array([float(s[_IDX_SOC_ELEC]) for s in states], dtype=np.float32)
        lo, hi = self.enforced_soc_bounds()
        a_lo, a_hi = self.battery.safe_interval(soc, lo, hi, a_max=self.action_scale)
        safe[:, self.elec_idx] = np.clip(actions[:, self.elec_idx], a_lo, a_hi)

        if self.nominal_power is not None:
            safe = self._apply_power_guard(safe, states, a_lo)
        safe = self._apply_deadline_barriers(safe, states)
        return self._apply_hvac_power_guard(safe, states)

    def _apply_power_guard(self, safe: np.ndarray, states: List[np.ndarray],
                           a_lo: np.ndarray) -> np.ndarray:
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        nom = self.nominal_power
        pred_import = net + np.maximum(safe[:, self.elec_idx], 0.0) * nom
        p_cap = self.power_cap()
        over = pred_import > p_cap
        for i in np.where(over)[0]:
            allowed = max(0.0, (p_cap - net[i]) / max(nom[i], 1e-6))
            safe[i, self.elec_idx] = float(np.clip(allowed, min(a_lo[i], safe[i, self.elec_idx]),
                                                   safe[i, self.elec_idx]))
        if not self.grid_guard:
            return safe
        charge = np.maximum(safe[:, self.elec_idx], 0.0) * nom
        g_cap = self.grid_cap()
        imported = lambda s: float(np.maximum(net + s * charge, 0.0).sum())
        if imported(1.0) > g_cap and charge.sum() > 1e-9:
            lo_s, hi_s = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo_s + hi_s)
                lo_s, hi_s = (mid, hi_s) if imported(mid) <= g_cap else (lo_s, mid)
            charging = safe[:, self.elec_idx] > 0
            keep = np.minimum(np.maximum(a_lo, 0.0), safe[:, self.elec_idx])
            scaled = np.maximum(safe[:, self.elec_idx] * lo_s, keep)
            safe[charging, self.elec_idx] = scaled[charging]
        return safe


    def _apply_deadline_barriers(self, safe: np.ndarray,
                                 states: List[np.ndarray]) -> np.ndarray:
        for barrier in self.deadline_barriers:
            safe = barrier.project(safe, states)
        if self.coordination != "independent":
            safe = self._apply_shared_power_allocation(safe, states)
        return safe

    def _apply_shared_power_allocation(self, safe: np.ndarray,
                                       states: List[np.ndarray]) -> np.ndarray:
        cap = self.grid_cap()
        net = np.array([float(s_[_IDX_NET]) for s_ in states], dtype=np.float32)
        baseline = float(np.maximum(net, 0.0).sum())
        headroom = max(cap - baseline, 0.0)

        entries = []
        requested_total = 0.0
        for barrier in self.deadline_barriers:
            power = getattr(barrier, "_p_charge", None)
            if power is None:
                continue
            u = barrier.urgency(states)
            a = np.maximum(safe[:, barrier.action_index], 0.0)
            kw = a * power
            for i in range(len(kw)):
                if kw[i] > 1e-9:
                    entries.append((float(u["steps_to_deadline"][i]), barrier,
                                    i, float(kw[i])))
                    requested_total += float(kw[i])

        if requested_total <= headroom + 1e-9 or requested_total <= 1e-9:
            return safe

        if self.coordination == "proportional":
            scale = headroom / requested_total
            for _, barrier, i, kw in entries:
                idx = barrier.action_index
                safe[i, idx] = safe[i, idx] * scale
            return safe

        entries.sort(key=lambda e: (e[0], -e[3]))
        budget = headroom
        for _, barrier, i, kw in entries:
            grant = min(kw, budget)
            budget -= grant
            idx = barrier.action_index
            safe[i, idx] = safe[i, idx] * (grant / kw) if kw > 1e-9 else 0.0
        return safe

    def feasibility_report(self, states: List[np.ndarray]) -> Optional[dict]:
        if not self.deadline_barriers:
            return None
        report = coupled_feasibility(self.deadline_barriers, states,
                                     power_cap_kw=self.grid_cap())
        if not report["feasible"]:
            report["priority"] = prioritise(self.deadline_barriers, states,
                                            power_cap_kw=self.grid_cap())
        return report


    def _apply_hvac_power_guard(self, safe: np.ndarray,
                                states: List[np.ndarray]) -> np.ndarray:
        if self.cop_model is None or self.hvac_idx < 0:
            return safe
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        t_out = np.array([float(s[_IDX_T_OUT]) for s in states], dtype=np.float32)
        heating = safe[:, self.hvac_idx] > 0.0

        p_nom = np.where(heating, self.cop_model.p_h, self.cop_model.p_c)
        cop_h = self.cop_model.cop(t_out, heating=True)
        cop_c = self.cop_model.cop(t_out, heating=False)
        cop = np.where(heating, cop_h, cop_c)
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
        total = float(np.maximum(net + np.abs(safe[:, self.hvac_idx]) * p_nom, 0.0).sum())
        g_cap = self.grid_cap()
        if total > g_cap and total > 1e-6:
            safe[:, self.hvac_idx] *= g_cap / total
        return safe


    def predicted_constraint_costs(self, actions: np.ndarray,
                                   states: List[np.ndarray]) -> np.ndarray:
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


class NeuralSafetyFilter(nn.Module):
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
