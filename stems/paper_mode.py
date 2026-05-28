"""Shared helpers for strict paper-comparable protocol checks."""

from __future__ import annotations

from typing import List

from stems.environment import STEMSEnvironment


def validate_strict_paper_mode(env: STEMSEnvironment, context: str) -> None:
    """Validate that an environment matches paper-comparable assumptions.

    Parameters
    ----------
    env : STEMSEnvironment
        Environment instance to validate.
    context : str
        Human-readable context label (for example: "training" or "evaluation").
    """
    errors: List[str] = []
    if env.using_mock:
        errors.append("Strict paper mode requires real CityLearn (mock environment detected).")
    if env.num_buildings != 8:
        errors.append(f"Strict paper mode requires 8 buildings, got {env.num_buildings}.")
    if env.obs_dim != 28:
        errors.append(f"Strict paper mode expects obs_dim=28, got {env.obs_dim}.")
    if env.action_dim != 3:
        errors.append(f"Strict paper mode expects the 3-action Travis setup, got action_dim={env.action_dim}.")
    if "electrical_storage" not in env.action_names:
        errors.append("Strict paper mode requires an electrical_storage action for CBF safety.")

    if errors:
        joined = "\n - ".join(errors)
        raise RuntimeError(
            "Strict paper mode validation failed"
            + (f" ({context})" if context else "")
            + ":\n - "
            + joined
            + "\nRun `python -B setup_citylearn_8b.py --validate` to create the real 8-building Travis schema."
        )
