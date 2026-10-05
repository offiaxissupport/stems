from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from stems.baselines import RuleBasedAgent

import numpy as np

UNCALIBRATED_SOC_RATE = 0.1

RESERVE_HOURS = 1
LEAD_MARGIN = True

RBC_ACTIONS = ["dhw_storage", "electrical_storage", "cooling_or_heating_device"]


@dataclass(frozen=True)
class Arm:
    name: str
    policy: str
    barrier: str
    residual: bool = False
    penalty: float = 0.0
    ev_request: str = "asap"
    forced_penalty: float = 0.0
    ev_floor: float = 0.0
    control: Optional[Tuple[str, ...]] = None

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
    Arm("rl+linear", "rl", "linear"),
    Arm("rl+calibrated", "rl", "calibrated"),
    Arm("rl-res+calibrated", "rl", "calibrated", residual=True),
    Arm("rl+calibrated+pen", "rl", "calibrated", penalty=1.0),
    Arm("rbc-offpeak+calibrated", "rbc", "calibrated", ev_request="offpeak"),
    Arm("rbc-never+calibrated", "rbc", "calibrated", ev_request="never"),
    Arm("rl+calibrated+own", "rl", "calibrated", forced_penalty=0.22),
    Arm("rl+calibrated+floor", "rl", "calibrated", ev_floor=0.5),
    Arm("hp-shift", "hp-shift", "none"),
    Arm("rl-hp", "rl", "none", control=("cooling_or_heating_device",)),
)}


class RuleColumns:
    def __init__(self, rule, names) -> None:
        self.rule = rule
        self.columns = [RBC_ACTIONS.index(n) for n in names]

    def reset(self) -> None:
        self.rule.reset()

    def notify_executed(self, executed: np.ndarray) -> None:
        return None

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        return self.rule.select_action(obs_list, history, explore)[:, self.columns]


class EVRule:
    def __init__(self, house, action_dim: int, ev_index: int, layout: Dict[str, int],
                 ev_request: str = "asap") -> None:
        if ev_request not in ("asap", "offpeak", "never"):
            raise ValueError(f"unknown ev_request {ev_request!r}")
        self.house, self.action_dim, self.ev_index, self.layout = house, action_dim, ev_index, layout
        self.ev_request = ev_request

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
        hour = int(round(float(obs_list[0][1])))
        if self.ev_request == "never" or (self.ev_request == "offpeak"
                                          and hour in RuleBasedAgent.PEAK_HOURS):
            short = np.zeros_like(short)
        a[:, self.ev_index] = np.where(short, 1.0, 0.0)
        return a


class ChargerFloor:
    def __init__(self, action_dim: int, ev_index: int, layout: Dict[str, int],
                 fraction: float) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.rule = EVRule(None, action_dim, ev_index, layout, ev_request="offpeak")
        self.ev_index, self.fraction = ev_index, float(fraction)

    def __call__(self, obs_list) -> np.ndarray:
        floor = np.full((len(obs_list), self.rule.action_dim), -1.0, dtype=np.float32)
        floor[:, self.ev_index] = self.fraction * self.rule.select_action(obs_list)[:, self.ev_index]
        return floor


class SetpointShiftPolicy:
    PREP_HOURS = range(13, 17)

    def __init__(self, env) -> None:
        if env.hvac_control != "setpoint":
            raise RuntimeError("SetpointShiftPolicy needs hvac_control='setpoint'")
        self.env = env

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        env = self.env
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        hour = int(round(float(obs_list[0][1])))
        heating = env.executed_actions[:, env.hvac_action_index] >= 0.0
        if hour in self.PREP_HOURS:
            a[:, env.hvac_action_index] = np.where(heating, 1.0, -1.0)
        elif hour in RuleBasedAgent.PEAK_HOURS:
            a[:, env.hvac_action_index] = np.where(heating, -1.0, 1.0)
        return a


class IdlePolicy:
    def __init__(self, num_buildings: int, action_dim: int) -> None:
        self.shape = (num_buildings, action_dim)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


class PlainController:
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
            self.base.notify_executed(safe)
        return safe

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        forecaster = getattr(self.dhw_barrier, "forecaster", None)
        if forecaster is not None:
            forecaster.update(next_obs_list)
        if self.fleet_shield is not None and ev_draw_kwh is not None:
            self.fleet_shield.observe(next_obs_list, ev_draw_kwh)


def safety_layer(barrier: str, env):
    from stems.battery import BatteryModel
    from stems.config import SafetyConfig

    plain = SafetyConfig(anticipatory=False, robust_margins=False)
    if barrier == "none":
        return plain, None
    if barrier == "basic":
        return plain, BatteryModel.linear(np.full(env.num_buildings, UNCALIBRATED_SOC_RATE))
    if barrier == "linear":
        exact = env.battery_model()
        return plain, BatteryModel.linear(exact.nominal_power * exact.dt / exact.capacity)
    if barrier == "calibrated":
        return plain, env.battery_model()
    raise ValueError(f"unknown barrier {barrier!r}")


def build_controller(arm: Arm, env, config):
    from stems.agent import STEMSAgent
    from stems.cbf import CBFShield
    from stems.graph import BuildingGraph

    B = env.num_buildings
    safety, battery_model = safety_layer(arm.barrier, env)
    config.safety = safety
    battery = env.battery_info()

    ev_indices = [] if env.using_mock else env.ev_action_indices()

    def rule():
        names = [n for n in env.action_names if n in RBC_ACTIONS]
        if "electrical_storage" not in names or names != [n for n in RBC_ACTIONS if n in names]:
            raise RuntimeError(f"RuleBasedAgent needs a battery and the action order of "
                               f"{RBC_ACTIONS}; this environment has {env.action_names}")
        house = RuleBasedAgent(num_buildings=B, hvac_control=env.hvac_control,
                               battery_nominal_power=battery["nominal_power"],
                               has_hvac=env.hvac_action_index >= 0)
        if names != RBC_ACTIONS:
            if ev_indices:
                raise RuntimeError("a schema with chargers needs all three house devices")
            return RuleColumns(house, names)
        if not ev_indices:
            return house
        return EVRule(house, env.action_dim, ev_indices[0], env.ev_obs_layout()[0],
                      ev_request=arm.ev_request)

    def fleet(barrier):
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
                           BaseLoadForecaster(B, daily_pattern_days=7),
                           reserve_hours=RESERVE_HOURS, house=house, lead_margin=LEAD_MARGIN)

    if arm.policy == "rl":
        info = env.get_building_info()
        graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
        config.training.intervention_penalty = float(arm.penalty)
        config.training.forced_charge_penalty = float(arm.forced_penalty)
        names = list(env.action_names)
        control = None if arm.control is None else [names.index(n) for n in arm.control]
        agent = STEMSAgent(env.obs_dim, env.action_dim, B, graph, config=config,
                           battery_info=battery, use_cbf=arm.barrier != "none",
                           electrical_storage_action_index=env.electrical_storage_action_index,
                           control_indices=control, hvac_action_index=env.hvac_action_index,
                           battery_model=battery_model or env.battery_model(),
                           base_policy=rule() if arm.residual else None)
        agent.fleet_shield = fleet(agent.cbf)
        if arm.ev_floor and ev_indices:
            agent.request_floor = ChargerFloor(env.action_dim, ev_indices[0],
                                               env.ev_obs_layout()[0], arm.ev_floor)
        return agent

    if arm.policy == "idle":
        base = IdlePolicy(B, env.action_dim)
    elif arm.policy == "rbc":
        base = rule()
    elif arm.policy == "hp-shift":
        base = SetpointShiftPolicy(env)
    else:
        raise ValueError(f"unknown policy {arm.policy!r}")
    if arm.barrier == "none":
        controller = PlainController(base)
        if arm.policy == "hp-shift":
            controller.control_indices = [env.hvac_action_index]
        return controller
    shield = CBFShield(config.cbf, B, battery_model=battery_model,
                       nominal_power=battery["nominal_power"],
                       action_scale=config.training.action_scale,
                       elec_idx=env.electrical_storage_action_index,
                       safety_cfg=safety, enforce_soc=True, hvac_idx=env.hvac_action_index)
    return ShieldedController(base, shield, fleet_shield=fleet(shield))
