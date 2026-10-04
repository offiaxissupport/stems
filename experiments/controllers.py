"""Controllers compared in the ablation, built the same way for every arm.

Each arm combines a policy with a safety layer.

Policies
    ``idle``  no storage action and a zero set-point offset: the house runs on its
              thermostat alone. This is the no-control reference.
    ``rbc``   the time-of-use storage rule of ``stems.baselines.RuleBasedAgent``.
    ``rl``    the learned policy (``stems.agent.STEMSAgent``). ``rl-res`` arms learn
              a bounded correction on the ``rbc`` rule (residual policy learning);
              ``+pen`` arms pay for every shield intervention.

Safety layers (the state-of-charge barrier; the grid and building power caps of
the default scenario never bind, so they are not what these arms differ in)
    ``none``        the policy's action is executed as is.
    ``basic``       the barrier as originally implemented: a linear battery model
                    with one uniform rate (0.1 state of charge per unit action) for
                    every building.
    ``calibrated``  the same barrier inverting the simulator's own battery
                    equations per building (``stems.battery.BatteryModel``).

``basic`` and ``calibrated`` differ in the battery model and nothing else: no
margins, buffers, hot-water or efficiency terms are bundled into either, so the
contrast between them measures calibration alone.

The learned policy's Lagrangian cost critics belong to the learning algorithm and
stay active in every learned arm, including ``rl`` without a shield.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

UNCALIBRATED_SOC_RATE = 0.1   # the uniform constant of the original implementation

RBC_ACTIONS = ["dhw_storage", "electrical_storage", "cooling_or_heating_device"]


@dataclass(frozen=True)
class Arm:
    name: str
    policy: str    # "idle" | "rbc" | "rl"
    barrier: str   # "none" | "basic" | "calibrated"
    residual: bool = False     # rl only: the policy corrects the time-of-use rule
    penalty: float = 0.0       # rl only: weight of the shield-intervention penalty

    @property
    def learns(self) -> bool:
        return self.policy == "rl"


ARMS: Dict[str, Arm] = {a.name: a for a in (
    Arm("idle", "idle", "none"),
    Arm("idle+calibrated", "idle", "calibrated"),
    Arm("rbc", "rbc", "none"),
    Arm("rbc+calibrated", "rbc", "calibrated"),
    Arm("rl", "rl", "none"),
    Arm("rl+basic", "rl", "basic"),
    Arm("rl+calibrated", "rl", "calibrated"),
    Arm("rl-res+calibrated", "rl", "calibrated", residual=True),
    Arm("rl+calibrated+pen", "rl", "calibrated", penalty=1.0),
)}


class EVRule:
    """A house rule plus "charge the car whenever it is plugged in and short".

    The uncoordinated behaviour a street of chargers has by default: each house
    asks for full power for its own vehicle, with no view of the shared cap.
    """

    def __init__(self, house, action_dim: int, ev_index: int, layout: Dict[str, int]) -> None:
        self.house, self.action_dim, self.ev_index, self.layout = house, action_dim, ev_index, layout

    def reset(self) -> None:
        if hasattr(self.house, "reset"):
            self.house.reset()

    def notify_executed(self, executed: np.ndarray) -> None:
        if hasattr(self.house, "notify_executed"):
            self.house.notify_executed(executed)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        a = np.zeros((len(obs_list), self.action_dim), dtype=np.float32)
        if self.house is not None:
            a[:, :3] = self.house.select_action(obs_list, history, explore)
        col = lambda key: np.array([float(o[self.layout[key]]) for o in obs_list])
        short = (col("connected_state") > 0.5) & (col("soc") < col("required_soc_departure"))
        a[:, self.ev_index] = np.where(short, 1.0, 0.0)
        return a


class IdlePolicy:
    """Zero action: storage untouched, thermostat at its set point."""

    def __init__(self, num_buildings: int, action_dim: int) -> None:
        self.shape = (num_buildings, action_dim)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


class PlainController:
    """A non-learning controller exposing the same interface as ``STEMSAgent``."""

    def __init__(self, base) -> None:
        self.base = base
        self._last_raw_actions: Optional[np.ndarray] = None
        self._last_safe_actions: Optional[np.ndarray] = None

    def _base_action(self, obs_list, history, explore) -> np.ndarray:
        a = np.asarray(self.base.select_action(obs_list, history, explore), dtype=np.float32)
        return np.clip(a, -1.0, 1.0)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        a = self._base_action(obs_list, history, explore)
        self._last_raw_actions, self._last_safe_actions = a.copy(), a.copy()
        return a

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        return None

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class ShieldedController(PlainController):
    """A non-learning controller whose actions pass through a safety shield."""

    def __init__(self, base, shield, dhw_barrier=None, fleet_shield=None) -> None:
        super().__init__(base)
        self.shield = shield
        self.dhw_barrier = dhw_barrier
        self.fleet_shield = fleet_shield

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        raw = self._base_action(obs_list, history, explore)
        safe = np.clip(self.shield.project(raw, obs_list), -1.0, 1.0).astype(np.float32)
        if self.fleet_shield is not None:
            safe = self.fleet_shield.project(safe, obs_list)
        self._last_raw_actions, self._last_safe_actions = raw.copy(), safe.copy()
        if hasattr(self.base, "notify_executed"):
            self.base.notify_executed(safe)      # anti-windup for a stateful base
        return safe

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        # The hot-water requirement is estimated online; it must see every step.
        forecaster = getattr(self.dhw_barrier, "forecaster", None)
        if forecaster is not None:
            forecaster.update(next_obs_list)
        if self.fleet_shield is not None and ev_draw_kwh is not None:
            self.fleet_shield.observe(next_obs_list, ev_draw_kwh)


def safety_layer(barrier: str, env):
    """(safety config, battery model the barrier inverts) for a barrier name."""
    from stems.battery import BatteryModel
    from stems.config import SafetyConfig

    plain = SafetyConfig(anticipatory=False, robust_margins=False)
    if barrier == "none":
        return plain, None
    if barrier == "basic":
        return plain, BatteryModel.linear(np.full(env.num_buildings, UNCALIBRATED_SOC_RATE))
    if barrier == "calibrated":
        return plain, env.battery_model()
    raise ValueError(f"unknown barrier {barrier!r}")


def build_controller(arm: Arm, env, config):
    """Construct the controller for ``arm``, calibrated from ``env``."""
    from stems.agent import STEMSAgent
    from stems.baselines import RuleBasedAgent
    from stems.cbf import CBFShield
    from stems.graph import BuildingGraph

    B = env.num_buildings
    safety, battery_model = safety_layer(arm.barrier, env)
    config.safety = safety
    battery = env.battery_info()

    ev_indices = [] if env.using_mock else env.ev_action_indices()

    def rule():
        if list(env.action_names)[:3] != RBC_ACTIONS:
            raise RuntimeError(f"RuleBasedAgent assumes actions {RBC_ACTIONS}; "
                               f"this environment has {env.action_names}")
        house = RuleBasedAgent(num_buildings=B, hvac_control=env.hvac_control,
                               battery_nominal_power=battery["nominal_power"])
        if not ev_indices:
            return house
        return EVRule(house, env.action_dim, ev_indices[0], env.ev_obs_layout()[0])

    def fleet(barrier):
        """The cap shield of a shielded arm on a schema with chargers: the
        joint-feasibility programme under the scenario's grid cap, with a causal
        load forecast (a learning controller changes the load it must fit around).

        It owns the shared cap for the vehicles *and* the house's battery and
        hot-water tank, so the
        state-of-charge barrier's own grid guard -- which predicts from the last
        hour's net load, vehicle charging included -- is switched off.
        """
        if not ev_indices or arm.barrier == "none":
            return None
        from stems.cbf import _IDX_SOC_ELEC
        from stems.fleet import BaseLoadForecaster, FleetShield, HouseStorage

        lo, hi = barrier.enforced_soc_bounds()
        names = list(env.action_names)
        house = HouseStorage(env.battery_model(), env.electrical_storage_action_index,
                             _IDX_SOC_ELEC, lo, hi,
                             tank=env.dhw_tank_model() if "dhw_storage" in names else None,
                             tank_action=names.index("dhw_storage") if "dhw_storage" in names else 0)
        barrier.grid_guard = False
        return FleetShield(env.ev_fleet_model(), env.ev_obs_layout()[0], ev_indices[0],
                           config.cbf.P_grid_max, "lp",
                           BaseLoadForecaster(B, daily_pattern_days=7), reserve_hours=1,
                           house=house)

    if arm.policy == "rl":
        info = env.get_building_info()
        graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
        config.training.intervention_penalty = float(arm.penalty)
        agent = STEMSAgent(env.obs_dim, env.action_dim, B, graph, config=config,
                           battery_info=battery, use_cbf=arm.barrier != "none",
                           electrical_storage_action_index=env.electrical_storage_action_index,
                           control_indices=None, hvac_action_index=env.hvac_action_index,
                           battery_model=battery_model or env.battery_model(),
                           base_policy=rule() if arm.residual else None)
        agent.fleet_shield = fleet(agent.cbf)
        return agent

    if arm.policy == "idle":
        base = IdlePolicy(B, env.action_dim)
    elif arm.policy == "rbc":
        base = rule()
    else:
        raise ValueError(f"unknown policy {arm.policy!r}")
    if arm.barrier == "none":
        return PlainController(base)
    shield = CBFShield(config.cbf, B, battery_model=battery_model,
                       nominal_power=battery["nominal_power"],
                       action_scale=config.training.action_scale,
                       elec_idx=env.electrical_storage_action_index,
                       safety_cfg=safety, enforce_soc=True, hvac_idx=env.hvac_action_index)
    return ShieldedController(base, shield, fleet_shield=fleet(shield))
