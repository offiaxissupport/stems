"""Scenarios: which buildings, which weather period and which constraint caps.

A scenario fixes the environment side of an experiment, so every controller in
the ablation is evaluated under exactly the same conditions. Three sources of
variation are exposed:

* **Building subset.** ``subset_seed=None`` is the schema's own eight buildings;
  an integer draws a different set of ``n_buildings`` from all candidates in the
  schema (100 for Travis). A subset is written out as a schema copy in which only
  the ``include`` flags change. The Travis schema's ``root_directory`` is
  absolute, so every data file still resolves from the untouched CityLearn cache
  and nothing is duplicated.
* **Weather period.** The Travis dataset carries a single weather year, so
  variation means different periods within it. Each season trains on one block of
  days and evaluates on the block immediately after it, so evaluation is always
  on weather the policy has not seen.

  Windows are realised as CityLearn *episodes* (``episode_time_steps``) inside a
  full-year simulation, never by shortening the simulation. CityLearn autosizes
  every device -- heat pumps, hot-water tank and heater, battery and PV -- from the
  data inside the simulation period, so a shortened simulation builds a different
  house: a 7-day summer window sizes building 134795's heating heat pump at 0 kW
  instead of 8.68 kW, and CityLearn's own demand check then fails at the first
  step. Keeping the simulation on the full year gives every window, arm and seed
  identical, full-year-sized hardware.
* **Constraint caps.** Grid and per-building power limits, for sweeping from
  non-binding to binding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TX_SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
SUBSET_DIR = REPO / "citylearn_schemas" / "_subsets"

# Day of year (0 = 1 January, non-leap) on which each season's training block starts.
SEASON_FIRST_DAY = {"winter": 0, "spring": 90, "summer": 181, "autumn": 273}


def day_window(first_day: int, days: int) -> Tuple[int, int]:
    """Inclusive (start, end) hourly time steps for ``days`` days from ``first_day``."""
    start = first_day * 24
    return start, start + days * 24 - 1


def season_windows(season: str, days: int = 28) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """(training window, evaluation window): consecutive and non-overlapping."""
    if season not in SEASON_FIRST_DAY:
        raise ValueError(f"unknown season {season!r}; choose from {sorted(SEASON_FIRST_DAY)}")
    first = SEASON_FIRST_DAY[season]
    return day_window(first, days), day_window(first + days, days)


def _resolve(schema: str) -> Path:
    p = Path(schema)
    return p if p.is_absolute() else REPO / p


def schema_buildings(schema: str = TX_SCHEMA) -> Tuple[List[str], List[str]]:
    """(all candidate building keys, keys marked ``include``), in schema order."""
    buildings = json.loads(_resolve(schema).read_text(encoding="utf-8"))["buildings"]
    return list(buildings), [k for k, v in buildings.items() if v.get("include", True)]


# Per-building fields every reference building shares and every sampled building
# inherits. ``setup_citylearn_8b.py`` attached the tariff and carbon files to the
# eight included buildings only, so the other candidates carry ``None`` -- which
# CityLearn reads as free, carbon-free electricity: zero cost, zero emissions and
# no economic term in the reward, silently.
REFERENCE_SHARED_FIELDS = ("pricing", "carbon_intensity")


def device_signature(building: Dict[str, Any]) -> Tuple:
    """Which devices a building has and which actions/observations it disables."""
    present = tuple(sorted(k for k, v in building.items()
                           if isinstance(v, dict) and "type" in v and v["type"] is not None))
    return (present, tuple(sorted(building.get("inactive_actions") or [])),
            tuple(sorted(building.get("inactive_observations") or [])))


def _reference(data: Dict[str, Any], schema: str) -> Tuple[Tuple, Dict[str, Any]]:
    """The device signature and shared fields common to the schema's included buildings."""
    included = [v for v in data["buildings"].values() if v.get("include", True)]
    signatures = {device_signature(v) for v in included}
    if len(signatures) != 1:
        raise RuntimeError(f"{schema}: included buildings have {len(signatures)} different "
                           "device sets, so there is no single reference to sample against")
    shared = {}
    for field in REFERENCE_SHARED_FIELDS:
        values = {json.dumps(v.get(field)) for v in included}
        if len(values) != 1:
            raise RuntimeError(f"{schema}: included buildings disagree on {field!r}")
        shared[field] = included[0].get(field)
    return signatures.pop(), shared


def comparable_candidates(schema: str = TX_SCHEMA) -> List[str]:
    """Candidates with the same devices as the reference buildings, in schema order.

    Subsets vary the houses, not the experiment: a house without a hot-water
    tank has no DHW actuator, and drawing it would change what is being
    controlled and scored (it showed up as hot-water readiness halving).
    """
    data = json.loads(_resolve(schema).read_text(encoding="utf-8"))
    signature, _ = _reference(data, schema)
    return [k for k, v in data["buildings"].items() if device_signature(v) == signature]


def sample_buildings(n: int, subset_seed: int, schema: str = TX_SCHEMA) -> List[str]:
    """Draw ``n`` buildings reproducibly from the comparable candidates."""
    candidates = comparable_candidates(schema)
    if n > len(candidates):
        raise ValueError(f"asked for {n} buildings but {schema} has only "
                         f"{len(candidates)} with the reference device set")
    rng = np.random.default_rng(subset_seed)
    chosen = sorted(rng.choice(len(candidates), size=n, replace=False).tolist())
    return [candidates[i] for i in chosen]


def materialize_subset_schema(schema: str, buildings: List[str], tag: str) -> str:
    """Write a copy of ``schema`` that includes exactly ``buildings``; return its path.

    Only two things change: the ``include`` flags, and the newly included
    buildings receive the reference tariff and carbon files
    (``REFERENCE_SHARED_FIELDS``).
    """
    src = _resolve(schema)
    data = json.loads(src.read_text(encoding="utf-8"))
    unknown = sorted(set(buildings) - set(data["buildings"]))
    if unknown:
        raise ValueError(f"buildings not in schema {schema}: {unknown}")
    root = data.get("root_directory")
    if not root or not Path(root).is_absolute():
        raise RuntimeError(
            f"schema {schema} has root_directory={root!r}; a copy stored elsewhere "
            "would not find its data files")
    signature, shared = _reference(data, schema)
    mismatched = sorted(k for k in buildings
                        if device_signature(data["buildings"][k]) != signature)
    if mismatched:
        raise ValueError(f"buildings without the reference device set: {mismatched}")
    for key, value in data["buildings"].items():
        value["include"] = key in buildings
        if value["include"]:
            value.update(shared)
    out = SUBSET_DIR / f"{src.parent.name}__{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.write_text(text, encoding="utf-8")
    return out.relative_to(REPO).as_posix()


@dataclass
class Scenario:
    """Everything about the environment side of one experimental condition."""

    schema: str = TX_SCHEMA
    season: str = "winter"
    subset_seed: Optional[int] = None
    n_buildings: int = 8
    days: int = 28
    grid_cap_kw: float = 300.0
    building_cap_kw: float = 80.0
    # "setpoint": the HVAC action is a set-point offset tracked by the
    # environment's thermostat; "power": the action is the power fraction
    # (the paper's formulation). See stems/environment.py.
    hvac_control: str = "setpoint"
    # Whether the buildings have a heat pump the controller drives (heating
    # observations, set points). False for datasets without thermal dynamics,
    # where the indoor temperature is not the controller's to move.
    heat_pump: bool = True
    # Zero-fill observations the dataset does not have (recorded in the run as
    # ``absent_observations``) instead of refusing the schema.
    allow_missing_obs: bool = False

    @property
    def buildings(self) -> Optional[List[str]]:
        """Explicit subset, or ``None`` for the schema's own included buildings."""
        if self.subset_seed is None:
            return None
        return sample_buildings(self.n_buildings, self.subset_seed, self.schema)

    def schema_path(self) -> str:
        if self.subset_seed is None:
            return self.schema
        return materialize_subset_schema(
            self.schema, self.buildings, f"subset{self.subset_seed}_n{self.n_buildings}")

    def env_kwargs(self, phase: str) -> Dict[str, Any]:
        """CityLearnEnv overrides selecting the ``train`` or ``eval`` window."""
        if phase not in ("train", "eval"):
            raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
        train, evaluation = season_windows(self.season, self.days)
        start, end = train if phase == "train" else evaluation
        # An episode inside the full-year simulation, not a shortened simulation:
        # see the module docstring for why the distinction matters.
        return {"episode_time_steps": [(start, end)]}

    @property
    def key(self) -> str:
        subset = "ref" if self.subset_seed is None else f"subset{self.subset_seed}"
        key = (f"{_resolve(self.schema).parent.name}__{self.season}{self.days}d"
               f"__{subset}n{self.n_buildings}"
               f"__cap{self.grid_cap_kw:g}-{self.building_cap_kw:g}")
        return key if self.hvac_control == "setpoint" else f"{key}__{self.hvac_control}"

    def describe(self) -> Dict[str, Any]:
        train, evaluation = season_windows(self.season, self.days)
        d = asdict(self)
        d.update(key=self.key,
                 buildings=self.buildings or schema_buildings(self.schema)[1],
                 train_window=list(train), eval_window=list(evaluation))
        return d
