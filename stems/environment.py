"""
CityLearn environment wrapper for STEMS.

Tries to import the real CityLearn environment.  If unavailable (e.g. in a
CI/CD environment without the CityLearn package), a realistic mock is used so
that the rest of the codebase can be exercised without modification.

Mock specification (matching CityLearn 2023 Phase-2 schema):
    Buildings  : 3
    obs_dim    : 28  (see OBS_NAMES below)
    action_dim : 3   (dhw_storage, electrical_storage, cooling_device)
    Episode    : 720 timesteps (≈ 1 month at hourly resolution)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# CityLearn import with fallback
# --------------------------------------------------------------------------
_CITYLEARN_AVAILABLE = False
try:
    from citylearn.citylearn import CityLearnEnv  # type: ignore
    _CITYLEARN_AVAILABLE = True
except ImportError:
    CityLearnEnv = None  # type: ignore

# --------------------------------------------------------------------------
# Observable feature names (28-dim) – CityLearn 2023 Phase 2 schema
# --------------------------------------------------------------------------
OBS_NAMES: List[str] = [
    "day_type",                                   # 0
    "hour",                                       # 1
    "outdoor_dry_bulb_temperature",               # 2
    "outdoor_dry_bulb_temperature_predicted_1",   # 3
    "outdoor_dry_bulb_temperature_predicted_2",   # 4
    "outdoor_dry_bulb_temperature_predicted_3",   # 5  (unused slot kept for shape)
    "diffuse_solar_irradiance",                   # 6
    "diffuse_solar_irradiance_predicted_1",       # 7
    "diffuse_solar_irradiance_predicted_2",       # 8
    "diffuse_solar_irradiance_predicted_3",       # 9
    "direct_solar_irradiance",                    # 10
    "direct_solar_irradiance_predicted_1",        # 11
    "direct_solar_irradiance_predicted_2",        # 12
    "direct_solar_irradiance_predicted_3",        # 13
    "carbon_intensity",                           # 14
    "indoor_dry_bulb_temperature",                # 15
    "non_shiftable_load",                         # 16
    "solar_generation",                           # 17
    "dhw_storage_soc",                            # 18
    "electrical_storage_soc",                     # 19
    "net_electricity_consumption",                # 20
    "electricity_pricing",                        # 21
    "electricity_pricing_predicted_1",            # 22
    "electricity_pricing_predicted_2",            # 23
    "cooling_demand",                             # 24
    "dhw_demand",                                 # 25
    "occupant_count",                             # 26
    "indoor_dry_bulb_temperature_cooling_set_point",  # 27
]

OBS_DIM = len(OBS_NAMES)   # 28
ACTION_DIM = 3              # dhw_storage, electrical_storage, cooling_device


# --------------------------------------------------------------------------
# Realistic mock environment
# --------------------------------------------------------------------------

class _MockBuilding:
    """Simulates one building with plausible physics for training."""

    def __init__(self, rng: np.random.Generator, building_id: int) -> None:
        self.rng = rng
        self.id = building_id
        self._soc_dhw = 0.5
        self._soc_elec = 0.5
        self._t_indoor = 22.0
        self._t = 0          # timestep within episode

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._soc_dhw = 0.5 + self.rng.uniform(-0.1, 0.1)
        self._soc_elec = 0.5 + self.rng.uniform(-0.1, 0.1)
        self._t_indoor = 22.0 + self.rng.uniform(-1.0, 1.0)
        self._t = 0

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance one timestep with the given 3-dim action, return 28-dim obs."""
        self._t += 1
        hour = (self._t % 24)
        day_type = 1 + int(self._t / 24) % 7

        # Outdoor weather
        t_out = 15.0 + 10.0 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 1)
        solar = max(0.0, 500.0 * np.sin(np.pi * (hour - 6) / 12)) + self.rng.normal(0, 20)
        carbon = 0.3 + 0.1 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 0.02)
        price = 0.12 + 0.08 * (1.0 if 16 <= hour <= 21 else 0.0) + self.rng.normal(0, 0.005)
        price = float(np.clip(price, 0.05, 0.30))

        # Building loads
        load = 1.5 + 0.5 * np.sin(2 * np.pi * hour / 24) + self.rng.normal(0, 0.2)
        load = float(np.clip(load, 0.1, 5.0))
        cooling = max(0.0, 0.5 * (t_out - 18.0) + self.rng.normal(0, 0.1))
        dhw = max(0.0, 0.3 + 0.1 * self.rng.normal())
        occupant = float(self.rng.integers(0, 5))
        solar_gen = max(0.0, solar * 0.003 + self.rng.normal(0, 0.05))

        # Storage dynamics
        dhw_action = float(np.clip(action[0], -1, 1))
        elec_action = float(np.clip(action[1], -1, 1))
        cool_action = float(np.clip(action[2], -1, 1))

        self._soc_dhw = float(np.clip(self._soc_dhw + dhw_action * 0.05, 0.05, 0.95))
        self._soc_elec = float(np.clip(self._soc_elec + elec_action * 0.1, 0.05, 0.95))

        # Thermal dynamics: cooling action lowers indoor temp
        self._t_indoor += 0.1 * (t_out - self._t_indoor) - 0.5 * max(0, cool_action) + self.rng.normal(0, 0.1)
        self._t_indoor = float(np.clip(self._t_indoor, 15.0, 35.0))

        # Net electricity consumption
        net = load + cooling - solar_gen + 0.05 * abs(dhw_action) + 0.1 * abs(elec_action)
        net = float(np.clip(net, -2.0, 15.0))

        # Predicted values (simple persistence forecast + noise)
        t_pred = [t_out + self.rng.normal(0, 0.5) for _ in range(3)]
        solar_d_pred = [max(0, solar * 0.8 + self.rng.normal(0, 30)) for _ in range(3)]
        solar_i_pred = [max(0, solar * 0.8 + self.rng.normal(0, 30)) for _ in range(3)]
        price_pred = [float(np.clip(price + self.rng.normal(0, 0.01), 0.05, 0.30)) for _ in range(2)]

        power_outage = float(self.rng.random() < 0.002)
        t_set = 22.0 + self.rng.normal(0, 0.5)

        obs = np.array([
            float(day_type),
            float(hour),
            t_out,
            t_pred[0], t_pred[1], t_pred[2],
            solar * 0.6,   # diffuse fraction
            solar_d_pred[0], solar_d_pred[1], solar_d_pred[2],
            solar * 0.4,   # direct fraction
            solar_i_pred[0], solar_i_pred[1], solar_i_pred[2],
            carbon,
            self._t_indoor,
            load,
            solar_gen,
            self._soc_dhw,
            self._soc_elec,
            net,
            price,
            price_pred[0], price_pred[1],
            cooling,
            dhw,
            occupant,
            t_set,
        ], dtype=np.float32)

        return obs


class _MockCityLearnEnv:
    """Minimal realistic mock of CityLearnEnv for the STEMS pipeline."""

    NUM_BUILDINGS = 3
    EPISODE_LEN = 720   # timesteps per episode

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._buildings = [
            _MockBuilding(np.random.default_rng(seed + i), i)
            for i in range(self.NUM_BUILDINGS)
        ]
        self._timestep = 0
        self._prev_net: List[float] = [0.0] * self.NUM_BUILDINGS

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        self._timestep = 0
        for b in self._buildings:
            b.reset()
        obs = [b.step(np.zeros(3)) for b in self._buildings]
        self._prev_net = [float(o[20]) for o in obs]
        return obs, {}

    # ------------------------------------------------------------------
    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        self._timestep += 1
        obs = [b.step(a) for b, a in zip(self._buildings, actions)]
        rewards = [float(-o[20] * o[21]) for o in obs]   # -net_consumption * price
        done = self._timestep >= self.EPISODE_LEN
        self._prev_net = [float(o[20]) for o in obs]
        return obs, rewards, done, False, {}

    # ------------------------------------------------------------------
    @property
    def observation_space(self):
        class _Space:
            shape = (OBS_DIM,)
        return [_Space()] * self.NUM_BUILDINGS

    @property
    def action_space(self):
        class _Space:
            shape = (ACTION_DIM,)
            low = np.full(ACTION_DIM, -1.0)
            high = np.full(ACTION_DIM, 1.0)
        return [_Space()] * self.NUM_BUILDINGS

    # Mock building metadata
    @property
    def buildings(self):
        class _Building:
            pass
        result = []
        for i in range(self.NUM_BUILDINGS):
            b = _Building()
            b.name = f"Building_{i+1}"
            result.append(b)
        return result


# --------------------------------------------------------------------------
# STEMSEnvironment
# --------------------------------------------------------------------------

class STEMSEnvironment:
    """Wraps CityLearn (or mock) for the STEMS training pipeline.

    Parameters
    ----------
    schema : str
        CityLearn dataset schema name.  Ignored when using the mock.
    seed : int
        Random seed for the mock environment.
    """

    SCHEMA = "citylearn_challenge_2023_phase_2_local_evaluation"

    # Fallback local schema paths (CityLearn source checkout)
    _LOCAL_DATA_PATHS = [
        r"C:\temp\citylearn_src\data\datasets",
    ]

    def __init__(self, schema: Optional[str] = None, seed: int = 0) -> None:
        self._seed = seed
        self._schema = schema or self.SCHEMA
        self._comm_dropout: float = 0.0
        self._temp_offset: float = 0.0   # outdoor temperature offset for extreme weather

        if _CITYLEARN_AVAILABLE:
            try:
                self._env = CityLearnEnv(schema=self._schema, central_agent=False)
                self._mock = False
            except Exception as exc1:
                # Try local schema path fallback
                self._env = None
                self._mock = True
                import os
                for base in self._LOCAL_DATA_PATHS:
                    local = os.path.join(base, self._schema, "schema.json")
                    if os.path.isfile(local):
                        try:
                            self._env = CityLearnEnv(schema=local, central_agent=False)
                            self._mock = False
                            break
                        except Exception:
                            pass
                if self._mock:
                    self._env = _MockCityLearnEnv(seed=seed)
        else:
            self._env = _MockCityLearnEnv(seed=seed)
            self._mock = True

        # Cache dimensions
        self._num_buildings: int = len(self._env.observation_space)
        self._action_dim: int = self._env.action_space[0].shape[0]
        self._obs_dim: int = OBS_DIM  # always expose the 28-dim subset downstream

        if self._mock:
            self._obs_indices: Optional[List[Optional[int]]] = None
        else:
            # Build mapping from OBS_NAMES → index in CityLearn's raw observation
            # vector (which may have more than 28 dimensions, e.g. 52).
            # Use a dict for O(1) lookups instead of repeated linear searches.
            raw_obs_names: List[str] = self._env.observation_names[0]
            name_to_idx = {name: i for i, name in enumerate(raw_obs_names)}
            self._obs_indices = [name_to_idx.get(name) for name in OBS_NAMES]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_buildings(self) -> int:
        return self._num_buildings

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def using_mock(self) -> bool:
        return self._mock

    # ------------------------------------------------------------------
    # Comm disruption
    # ------------------------------------------------------------------

    def set_comm_disruption(self, dropout_prob: float) -> None:
        """Set probability that a building's observation is zeroed (comm dropout)."""
        self._comm_dropout = float(np.clip(dropout_prob, 0.0, 1.0))

    def set_temp_offset(self, offset: float) -> None:
        """Set outdoor temperature offset in °C (positive=heatwave, negative=coldwave)."""
        self._temp_offset = float(offset)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        result = self._env.reset()
        if isinstance(result, tuple):
            obs_list, info = result
        else:
            obs_list, info = result, {}

        obs_list = [np.asarray(o, dtype=np.float32) for o in obs_list]
        obs_list = self._extract_obs(obs_list)
        return obs_list, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """Step the environment with a (num_buildings, action_dim) action array."""
        actions = np.clip(actions, -1.0, 1.0)
        action_list = self._remap_actions(actions)

        result = self._env.step(action_list)
        if len(result) == 5:
            obs_list, rewards, terminated, truncated, info = result
        else:
            obs_list, rewards, done, info = result
            terminated, truncated = done, False

        obs_list = [np.asarray(o, dtype=np.float32) for o in obs_list]
        obs_list = self._extract_obs(obs_list)
        rewards = [float(r) for r in rewards]

        # Apply communication dropout: zero out selected buildings' observations
        if self._comm_dropout > 0.0:
            for i in range(self._num_buildings):
                if np.random.random() < self._comm_dropout:
                    obs_list[i] = np.zeros_like(obs_list[i])

        return obs_list, rewards, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_obs(self, raw_obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """Extract the 28-dim OBS_NAMES subset from raw CityLearn observations.

        When using the mock environment the observations are already 28-dim, so
        they are returned unchanged.  For the real CityLearn environment the raw
        vector may contain more observations (e.g. 52); we select only the ones
        listed in OBS_NAMES, in order, filling missing entries with 0.

        If a temperature offset is set, outdoor temperature observations
        (indices 2-5) are perturbed to simulate extreme weather.
        """
        if self._mock:
            result = raw_obs_list
        else:
            result = []
            for raw_obs in raw_obs_list:
                obs = np.zeros(OBS_DIM, dtype=np.float32)
                for j, idx in enumerate(self._obs_indices):  # type: ignore[union-attr]
                    if idx is not None and idx < len(raw_obs):
                        obs[j] = raw_obs[idx]
                result.append(obs)

        # Apply temperature offset for extreme weather simulation
        if self._temp_offset != 0.0:
            for obs in result:
                for idx in (2, 3, 4, 5):  # outdoor temp + 3 predictions
                    obs[idx] += self._temp_offset

        return result

    def _remap_actions(self, actions: np.ndarray) -> List[np.ndarray]:
        """Build per-building action list, remapping cooling_device for CityLearn.

        For the real CityLearn environment the ``cooling_device`` action (index 2)
        must be in [0, 1] (0 = no cooling, 1 = full cooling), but the agent
        produces values in [-1, 1].  We apply the affine map
        ``(a + 1) / 2`` to bring it into the required range.

        The mock environment already accepts [-1, 1] for all dimensions, so no
        remapping is performed there.
        """
        if self._mock:
            return [actions[i] for i in range(self._num_buildings)]
        action_list: List[np.ndarray] = []
        for i in range(self._num_buildings):
            a = actions[i].copy()
            # cooling_device is action index 2; remap [-1, 1] -> [0, 1]
            if len(a) > 2:
                a[2] = np.clip((a[2] + 1.0) / 2.0, 0.0, 1.0)
            action_list.append(a)
        return action_list

    # ------------------------------------------------------------------
    # Building metadata (positions + features for graph construction)
    # ------------------------------------------------------------------

    def get_building_info(self) -> Dict[str, Any]:
        """Return dict with 'positions' and 'features' for BuildingGraph."""
        B = self._num_buildings
        rng = np.random.default_rng(self._seed)

        # Positions: evenly spread around a unit circle for better graph diversity
        angles = np.linspace(0.0, 2.0 * np.pi, B, endpoint=False)
        positions = np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        ).astype(np.float32)  # (B, 2)

        # Functional features: seeded random, reproducible across resets
        features = (
            np.eye(B, dtype=np.float32)
            + 0.1 * rng.standard_normal((B, B)).astype(np.float32)
        )

        return {"positions": positions, "features": features}
