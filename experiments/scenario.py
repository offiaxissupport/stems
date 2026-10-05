from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TX_SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
SUBSET_DIR = REPO / "citylearn_schemas" / "_subsets"

SEASON_FIRST_DAY = {"winter": 0, "spring": 90, "summer": 181, "autumn": 273}


def day_window(first_day: int, days: int) -> Tuple[int, int]:
    start = first_day * 24
    return start, start + days * 24 - 1


YEAR = "year"
YEAR_STEPS = 8760
SEASONS = sorted(SEASON_FIRST_DAY) + [YEAR]


def season_windows(season: str, days: int = 28) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if season == YEAR:
        return (0, YEAR_STEPS - 1), (0, YEAR_STEPS - 1)
    if season not in SEASON_FIRST_DAY:
        raise ValueError(f"unknown season {season!r}; choose from {sorted(SEASON_FIRST_DAY)}")
    first = SEASON_FIRST_DAY[season]
    return day_window(first, days), day_window(first + days, days)


def _resolve(schema: str) -> Path:
    p = Path(schema)
    return p if p.is_absolute() else REPO / p


def schema_buildings(schema: str = TX_SCHEMA) -> Tuple[List[str], List[str]]:
    buildings = json.loads(_resolve(schema).read_text(encoding="utf-8"))["buildings"]
    return list(buildings), [k for k, v in buildings.items() if v.get("include", True)]


REFERENCE_SHARED_FIELDS = ("pricing", "carbon_intensity")


def device_signature(building: Dict[str, Any]) -> Tuple:
    present = tuple(sorted(k for k, v in building.items()
                           if isinstance(v, dict) and "type" in v and v["type"] is not None))
    return (present, tuple(sorted(building.get("inactive_actions") or [])),
            tuple(sorted(building.get("inactive_observations") or [])))


def _reference(data: Dict[str, Any], schema: str) -> Tuple[Tuple, Dict[str, Any]]:
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
    data = json.loads(_resolve(schema).read_text(encoding="utf-8"))
    signature, _ = _reference(data, schema)
    return [k for k, v in data["buildings"].items() if device_signature(v) == signature]


def sample_buildings(n: int, subset_seed: int, schema: str = TX_SCHEMA) -> List[str]:
    candidates = comparable_candidates(schema)
    if n > len(candidates):
        raise ValueError(f"asked for {n} buildings but {schema} has only "
                         f"{len(candidates)} with the reference device set")
    rng = np.random.default_rng(subset_seed)
    chosen = sorted(rng.choice(len(candidates), size=n, replace=False).tolist())
    return [candidates[i] for i in chosen]


def materialize_subset_schema(schema: str, buildings: List[str], tag: str) -> str:
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
    schema: str = TX_SCHEMA
    season: str = "winter"
    subset_seed: Optional[int] = None
    n_buildings: int = 8
    days: int = 28
    grid_cap_kw: float = 300.0
    building_cap_kw: float = 80.0
    hvac_control: str = "setpoint"
    heat_pump: bool = True
    allow_missing_obs: bool = False

    @property
    def buildings(self) -> Optional[List[str]]:
        if self.subset_seed is None:
            return None
        return sample_buildings(self.n_buildings, self.subset_seed, self.schema)

    def schema_path(self) -> str:
        if self.subset_seed is None:
            return self.schema
        return materialize_subset_schema(
            self.schema, self.buildings, f"subset{self.subset_seed}_n{self.n_buildings}")

    def env_kwargs(self, phase: str) -> Dict[str, Any]:
        if phase not in ("train", "eval"):
            raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
        train, evaluation = season_windows(self.season, self.days)
        start, end = train if phase == "train" else evaluation
        return {"episode_time_steps": [(start, end)]}

    @property
    def key(self) -> str:
        subset = "ref" if self.subset_seed is None else f"subset{self.subset_seed}"
        span = "year-insample" if self.season == YEAR else f"{self.season}{self.days}d"
        key = (f"{_resolve(self.schema).parent.name}__{span}"
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
