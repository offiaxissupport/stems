"""Tests for the experiment harness: scenarios, arms, actuator evidence, aggregation.

Run:  .venv/Scripts/python -m pytest tests/test_experiments.py -q

These run without real CityLearn: controllers are built on the explicit mock
environment and aggregation runs on synthetic records. The end-to-end path is
exercised separately by a smoke grid
(``python -m experiments.ablation --days 3 --episodes 1``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments import aggregate
from experiments.controllers import (ARMS, UNCALIBRATED_SOC_RATE, PlainController,
                                     ShieldedController, build_controller)
from experiments.runner import ActuatorEvidence
from experiments.scenario import (REFERENCE_SHARED_FIELDS, REPO, TX_SCHEMA, Scenario,
                                  comparable_candidates, day_window,
                                  materialize_subset_schema, sample_buildings,
                                  schema_buildings, season_windows)

# ===========================================================================
# Scenarios
# ===========================================================================


def test_day_window_is_inclusive_and_hourly():
    assert day_window(0, 1) == (0, 23)
    assert day_window(90, 28) == (2160, 2831)


@pytest.mark.parametrize("season", ["winter", "spring", "summer", "autumn"])
def test_evaluation_window_follows_training_window_without_overlap(season):
    (t0, t1), (e0, e1) = season_windows(season, 28)
    assert e0 == t1 + 1, "evaluation must start right after training ends"
    assert t1 - t0 + 1 == e1 - e0 + 1 == 28 * 24
    assert 0 <= t0 and e1 <= 8759


def test_unknown_season_fails_loud():
    with pytest.raises(ValueError):
        season_windows("monsoon")


def test_building_subsets_are_reproducible_and_distinct():
    candidates = comparable_candidates(TX_SCHEMA)
    a, b = sample_buildings(8, 1), sample_buildings(8, 1)
    c = sample_buildings(8, 2)
    assert a == b
    assert a != c
    assert len(set(a)) == 8 and set(a) <= set(candidates)


def test_subsets_only_draw_houses_with_the_reference_devices():
    original = json.loads((REPO / TX_SCHEMA).read_text(encoding="utf-8"))["buildings"]
    candidates = comparable_candidates(TX_SCHEMA)
    reference = schema_buildings(TX_SCHEMA)[1]
    assert set(reference) <= set(candidates) < set(original)
    tankless = [k for k, v in original.items() if "dhw_storage" in (v.get("inactive_actions") or [])]
    assert tankless and not set(tankless) & set(candidates)
    with pytest.raises(ValueError, match="reference device set"):
        materialize_subset_schema(TX_SCHEMA, tankless[:1], "pytest_tankless")


def test_subset_schema_changes_only_include_flags_and_shared_tariff():
    chosen = sample_buildings(8, 7)
    tag = "pytest_subset7_n8"
    out_path = REPO / "citylearn_schemas" / "_subsets" / f"tx_travis_8b__{tag}.json"
    existed = out_path.exists()
    try:
        rel = materialize_subset_schema(TX_SCHEMA, chosen, tag)
        copy = json.loads((REPO / rel).read_text(encoding="utf-8"))
        original = json.loads((REPO / TX_SCHEMA).read_text(encoding="utf-8"))
        assert copy["root_directory"] == original["root_directory"]
        assert Path(copy["root_directory"]).is_absolute()
        assert [k for k, v in copy["buildings"].items() if v["include"]] == chosen
        ref = original["buildings"][schema_buildings(TX_SCHEMA)[1][0]]
        for key in original["buildings"]:
            a = dict(original["buildings"][key]); a.pop("include", None)
            b = dict(copy["buildings"][key]); b.pop("include", None)
            if key in chosen:
                for field in REFERENCE_SHARED_FIELDS:
                    assert ref[field] and b.pop(field) == ref[field], (key, field)
                    a.pop(field, None)
            assert a == b, f"building {key} changed beyond include flag and tariff"
        rest_a = {k: v for k, v in original.items() if k != "buildings"}
        rest_b = {k: v for k, v in copy.items() if k != "buildings"}
        assert rest_a == rest_b
    finally:
        if not existed and out_path.exists():
            out_path.unlink()


def test_subset_schema_rejects_unknown_buildings():
    with pytest.raises(ValueError):
        materialize_subset_schema(TX_SCHEMA, ["not-a-building"], "pytest_bad")


def test_reference_scenario_uses_the_schema_itself():
    s = Scenario()
    assert s.buildings is None
    assert s.schema_path() == TX_SCHEMA
    assert s.describe()["buildings"] == schema_buildings(TX_SCHEMA)[1]


def test_scenario_keys_distinguish_every_varied_field():
    base = Scenario()
    variants = [Scenario(season="summer"), Scenario(subset_seed=1), Scenario(days=7),
                Scenario(grid_cap_kw=30.0), Scenario(building_cap_kw=5.0),
                Scenario(n_buildings=6, subset_seed=1)]
    keys = {base.key} | {v.key for v in variants}
    assert len(keys) == 1 + len(variants)


def test_env_kwargs_match_the_season_windows():
    s = Scenario(season="summer", days=7)
    (t0, t1), (e0, e1) = season_windows("summer", 7)
    # Episodes inside the full-year simulation -- never a shortened simulation,
    # which would re-size every device.
    assert s.env_kwargs("train") == {"episode_time_steps": [(t0, t1)]}
    assert s.env_kwargs("eval") == {"episode_time_steps": [(e0, e1)]}
    assert "simulation_start_time_step" not in s.env_kwargs("train")
    with pytest.raises(ValueError):
        s.env_kwargs("test")


def test_environment_refuses_to_shorten_the_simulation():
    """Windows must be episodes: a shortened simulation re-sizes every device."""
    from stems.environment import STEMSEnvironment

    with pytest.raises(ValueError, match="episode_time_steps"):
        STEMSEnvironment(force_mock=True, env_kwargs={"simulation_start_time_step": 0,
                                                      "simulation_end_time_step": 167})


# ===========================================================================
# Arms and controllers
# ===========================================================================


def test_the_ablation_arms():
    assert {n: (a.policy, a.barrier) for n, a in ARMS.items()} == {
        "idle": ("idle", "none"),
        "idle+calibrated": ("idle", "calibrated"),
        "rbc": ("rbc", "none"),
        "rbc+calibrated": ("rbc", "calibrated"),
        "rl": ("rl", "none"),
        "rl+basic": ("rl", "basic"),
        "rl+calibrated": ("rl", "calibrated"),
        "rl-res+calibrated": ("rl", "calibrated"),
        "rl+calibrated+pen": ("rl", "calibrated"),
        "rbc-offpeak+calibrated": ("rbc", "calibrated"),
        "rbc-never+calibrated": ("rbc", "calibrated"),
        "rl+calibrated+own": ("rl", "calibrated"),
        "rl+calibrated+floor": ("rl", "calibrated"),
        "hp-shift": ("hp-shift", "none"),
        "rl-hp": ("rl", "none"),
    }
    assert [n for n, a in ARMS.items() if a.learns] == [
        "rl", "rl+basic", "rl+calibrated", "rl-res+calibrated", "rl+calibrated+pen",
        "rl+calibrated+own", "rl+calibrated+floor", "rl-hp"]
    assert ARMS["rl-res+calibrated"].residual and ARMS["rl+calibrated+pen"].penalty > 0
    assert ARMS["rl+calibrated+own"].forced_penalty > 0 and ARMS["rl+calibrated"].forced_penalty == 0
    assert ARMS["rl-hp"].control == ("cooling_or_heating_device",)


@pytest.fixture(scope="module")
def mock_env():
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    try:
        env.battery_info()
        env.dhw_info()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"mock environment cannot supply device parameters: {exc!r}")
    env.reset()
    return env


def _config():
    from stems.config import STEMSConfig

    config = STEMSConfig()
    config.heat_pump.enabled = True
    return config


def test_rbc_without_barrier_executes_its_own_action(mock_env):
    ctrl = build_controller(ARMS["rbc"], mock_env, _config())
    assert isinstance(ctrl, PlainController) and not isinstance(ctrl, ShieldedController)
    obs, _ = mock_env.reset()
    a = ctrl.select_action(obs, None, explore=False)
    assert a.shape == (mock_env.num_buildings, 3)
    assert np.array_equal(ctrl._last_raw_actions, ctrl._last_safe_actions)


def test_idle_policy_does_nothing(mock_env):
    ctrl = build_controller(ARMS["idle"], mock_env, _config())
    obs, _ = mock_env.reset()
    assert not ctrl.select_action(obs, None, explore=False).any()


def test_calibrated_barrier_inverts_the_environments_battery_model(mock_env):
    ctrl = build_controller(ARMS["rbc+calibrated"], mock_env, _config())
    assert isinstance(ctrl, ShieldedController)
    np.testing.assert_allclose(ctrl.shield.soc_rate, mock_env.battery_info()["soc_rate"])
    soc = np.full(mock_env.num_buildings, 0.5)
    act = np.full(mock_env.num_buildings, 0.7)
    np.testing.assert_allclose(ctrl.shield.battery.next_soc(soc, act),
                               mock_env.battery_model().next_soc(soc, act))


def test_rl_without_barrier_has_the_shield_switched_off(mock_env):
    agent = build_controller(ARMS["rl"], mock_env, _config())
    assert agent.use_cbf is False


def test_basic_barrier_assumes_the_uniform_rate(mock_env):
    agent = build_controller(ARMS["rl+basic"], mock_env, _config())
    assert agent.use_cbf is True
    np.testing.assert_allclose(agent.cbf.soc_rate, UNCALIBRATED_SOC_RATE)


def test_basic_and_calibrated_differ_only_in_the_battery_model(mock_env):
    """No margin, buffer, hot-water or efficiency term is bundled into either arm."""
    basic = build_controller(ARMS["rl+basic"], mock_env, _config())
    calibrated = build_controller(ARMS["rl+calibrated"], mock_env, _config())
    rbc = build_controller(ARMS["rbc+calibrated"], mock_env, _config())
    assert basic.cbf.safety == calibrated.cbf.safety == rbc.shield.safety
    for shield in (basic.cbf, calibrated.cbf, rbc.shield):
        assert not shield.safety.anticipatory and not shield.safety.robust_margins
        assert shield.dhw_barrier is None and shield.cop_model is None
    np.testing.assert_allclose(calibrated.cbf.soc_rate, rbc.shield.soc_rate)


def test_untrained_residual_policy_is_the_rule(mock_env):
    """Residual policy learning starts from the base controller: a fresh actor's
    mean is ~0, so the deterministic action is the rule's, through the same shield."""
    from stems.utils import HistoryBuffer

    cfg = _config()
    residual = build_controller(ARMS["rl-res+calibrated"], mock_env, cfg)
    rule = build_controller(ARMS["rbc+calibrated"], mock_env, _config())
    assert residual.base_policy is not None
    obs, _ = mock_env.reset()
    hist = HistoryBuffer(mock_env.num_buildings, mock_env.obs_dim, cfg.transformer.window_size)
    hist.update(obs)
    a_res = residual.select_action(obs, hist.get(), explore=False)
    a_rule = rule.select_action(obs, None, explore=False)
    np.testing.assert_allclose(a_res, a_rule, atol=0.02)


def test_residual_exploration_starts_small(mock_env):
    cfg = _config()
    residual = build_controller(ARMS["rl-res+calibrated"], mock_env, cfg)
    scratch = build_controller(ARMS["rl+calibrated"], mock_env, _config())
    import torch

    r = torch.zeros(1, cfg.fusion.output_dim)
    assert float(residual.actors[0](r)[1].exp().mean()) == pytest.approx(
        math.exp(cfg.training.residual_log_std), rel=1e-5)
    assert float(scratch.actors[0](r)[1].exp().mean()) > 0.6


def test_penalty_arm_sets_the_intervention_weight(mock_env):
    cfg = _config()
    build_controller(ARMS["rl+calibrated+pen"], mock_env, cfg)
    assert cfg.training.intervention_penalty == ARMS["rl+calibrated+pen"].penalty > 0
    cfg2 = _config()
    build_controller(ARMS["rl+calibrated"], mock_env, cfg2)
    assert cfg2.training.intervention_penalty == 0.0


def test_own_arm_sets_the_forced_charging_weight(mock_env):
    cfg = _config()
    build_controller(ARMS["rl+calibrated+own"], mock_env, cfg)
    assert cfg.training.forced_charge_penalty == ARMS["rl+calibrated+own"].forced_penalty
    cfg2 = _config()
    build_controller(ARMS["rl+calibrated"], mock_env, cfg2)
    assert cfg2.training.forced_charge_penalty == 0.0


def test_heat_pump_only_policy_drives_the_heat_pump_and_nothing_else(mock_env):
    from stems.utils import HistoryBuffer

    cfg = _config()
    agent = build_controller(ARMS["rl-hp"], mock_env, cfg)
    hvac = mock_env.hvac_action_index
    assert agent.control_indices == [hvac] and not agent.use_cbf
    obs, _ = mock_env.reset()
    hist = HistoryBuffer(mock_env.num_buildings, mock_env.obs_dim, cfg.transformer.window_size)
    hist.update(obs)
    a = agent.select_action(obs, hist.get(), explore=True)
    others = [i for i in range(mock_env.action_dim) if i != hvac]
    assert np.all(a[:, others] == 0.0)


def test_charger_floor_holds_the_request_up_and_touches_nothing_else():
    from experiments.controllers import ChargerFloor

    layout = {"connected_state": 0, "soc": 4, "required_soc_departure": 2, "departure_time": 3}

    def obs(hour, connected=1.0):
        o = np.zeros(30)
        o[0], o[1], o[2], o[3] = connected, hour, 0.8, 5.0
        return [o]
    floor = ChargerFloor(action_dim=4, ev_index=3, layout=layout, fraction=0.5)
    assert floor(obs(23)).tolist() == [[-1.0, -1.0, -1.0, 0.5]]     # off-peak, car short
    assert floor(obs(18)).tolist() == [[-1.0, -1.0, -1.0, 0.0]]     # tariff peak: no floor
    assert floor(obs(23, connected=0.0))[0, 3] == 0.0               # no car
    asked = np.array([[-0.7, 0.2, 0.9, -1.0]], dtype=np.float32)    # a policy that refuses to charge
    assert np.maximum(asked, floor(obs(23)))[0] == pytest.approx([-0.7, 0.2, 0.9, 0.5])
    keen = np.array([[0.0, 0.0, 0.0, 0.9]], dtype=np.float32)
    assert np.maximum(keen, floor(obs(23)))[0, 3] == pytest.approx(0.9)   # asking for more is the policy's
    with pytest.raises(ValueError):
        ChargerFloor(4, 3, layout, fraction=0.0)


def test_rule_on_a_schema_without_a_heat_pump_keeps_only_the_devices_it_has():
    from experiments.controllers import RBC_ACTIONS, RuleColumns
    from stems.baselines import RuleBasedAgent

    rule = RuleBasedAgent(num_buildings=2, hvac_control="power",
                          battery_nominal_power=np.array([5.0, 5.0]), has_hvac=False)
    obs = [np.zeros(28) for _ in range(2)]
    for o in obs:
        o[1], o[16] = 12.0, 3.0                     # a charging hour, 3 kW of load
    full = rule.select_action(obs)
    assert full.shape == (2, 3) and np.all(full[:, 2] == 0.0)      # no thermostat without a heat pump
    two = RuleColumns(rule, ["dhw_storage", "electrical_storage"]).select_action(obs)
    assert two.shape == (2, 2)
    assert two[0] == pytest.approx([RuleBasedAgent.DHW_CHARGE_ACTION, RuleBasedAgent.CHARGE_ACTION])
    assert RBC_ACTIONS == ["dhw_storage", "electrical_storage", "cooling_or_heating_device"]


def test_scenario_records_what_the_schema_lacks():
    plain = Scenario()
    assert plain.heat_pump and not plain.allow_missing_obs
    mixed = Scenario(schema="citylearn_schemas/cl2020_zone1/schema.json", heat_pump=False,
                     allow_missing_obs=True, hvac_control="power")
    d = asdict(mixed)
    assert d["heat_pump"] is False and d["allow_missing_obs"] is True
    assert mixed.key.startswith("cl2020_zone1__") and mixed.key.endswith("__power")


def test_car_request_modes_of_the_rule():
    from experiments.controllers import EVRule

    layout = {"connected_state": 0, "soc": 1, "required_soc_departure": 2, "departure_time": 3}

    def obs(hour):
        o = np.zeros(30)
        o[0], o[2], o[3] = 1.0, 0.8, 5.0          # connected, needs 0.8
        o[1] = hour                                # CityLearn hour, and the car's state of charge
        return [o]
    ask = lambda mode, hour: float(EVRule(None, 2, 1, {**layout, "soc": 4}, ev_request=mode)
                                   .select_action(obs(hour))[0, 1])
    assert ask("asap", 18) == 1.0 and ask("asap", 23) == 1.0
    assert ask("offpeak", 18) == 0.0 and ask("offpeak", 23) == 1.0      # 17..21 is the tariff peak
    assert ask("never", 18) == 0.0 and ask("never", 23) == 0.0
    with pytest.raises(ValueError):
        EVRule(None, 2, 1, layout, ev_request="sometimes")


# ===========================================================================
# Actuator evidence
# ===========================================================================


class _FakeEnv:
    electrical_storage_action_index = 1
    dhw_action_index = 0
    hvac_action_index = -1
    using_mock = True


def _rollout(evidence, responsive_battery: bool, responsive_dhw: bool = True,
             steps: int = 60, B: int = 2, charge_every: int = 2):
    soc = np.full(B, 0.5, dtype=np.float32)
    tank = np.full(B, 0.5, dtype=np.float32)
    for t in range(steps):
        cmd = 0.8 if t % charge_every == 0 else 0.0
        actions = np.zeros((B, 3), dtype=np.float32)
        actions[:, 0] = cmd
        actions[:, 1] = cmd
        pre = [np.zeros(30, dtype=np.float32) for _ in range(B)]
        post = [np.zeros(30, dtype=np.float32) for _ in range(B)]
        for i in range(B):
            pre[i][19], pre[i][18] = soc[i], tank[i]
        if responsive_battery and cmd > 0:
            soc = np.minimum(soc + 0.05, 0.9)
        else:
            soc = np.maximum(soc - 0.01, 0.1)
        if responsive_dhw and cmd > 0:
            tank = np.minimum(tank + 0.05, 1.0)
        else:
            tank = np.maximum(tank - 0.01, 0.0)
        for i in range(B):
            post[i][19], post[i][18] = soc[i], tank[i]
        evidence.add(pre, actions, post)


def test_responsive_storage_is_verified():
    ev = ActuatorEvidence(_FakeEnv())
    _rollout(ev, responsive_battery=True)
    s = ev.summary()
    assert s["battery"]["responds"] is True and s["dhw"]["responds"] is True
    assert s["hvac"] is None
    assert s["verified"] is True


def test_inert_actuator_fails_verification():
    """The failure mode of the hot-water defect: commands sent, nothing moves."""
    ev = ActuatorEvidence(_FakeEnv())
    _rollout(ev, responsive_battery=True, responsive_dhw=False)
    s = ev.summary()
    assert s["dhw"]["responds"] is False
    assert s["verified"] is False


def test_too_few_commands_is_insufficient_not_a_pass():
    ev = ActuatorEvidence(_FakeEnv())
    _rollout(ev, responsive_battery=True, steps=30, charge_every=100)   # charges once
    s = ev.summary()
    assert s["battery"]["responds"] is None
    assert s["verified"] is None


# ===========================================================================
# Aggregation
# ===========================================================================


def test_summarize_confidence_interval_and_p_value():
    from scipy.stats import ttest_1samp

    vals = [1.0, 2.0, 3.0, 6.0]
    s = aggregate.summarize(vals)
    sd = float(np.std(vals, ddof=1))
    half = aggregate.t_critical(4) * sd / math.sqrt(4)
    assert s["n"] == 4 and s["mean"] == pytest.approx(3.0)
    assert s["ci95"] == pytest.approx([3.0 - half, 3.0 + half])
    assert s["p"] == pytest.approx(ttest_1samp(vals, 0.0).pvalue)


def test_summarize_ignores_missing_values_and_needs_two_for_an_interval():
    s = aggregate.summarize([None, float("nan"), 2.0])
    assert s["n"] == 1 and s["mean"] == 2.0 and s["ci95"] is None and s["p"] is None
    assert aggregate.summarize([])["mean"] is None


def test_holm_step_down():
    out = aggregate.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert out["a"]["p_holm"] == pytest.approx(0.03)      # 3 x 0.01
    assert out["c"]["p_holm"] == pytest.approx(0.06)      # 2 x 0.03
    assert out["b"]["p_holm"] == pytest.approx(0.06)      # max(1 x 0.04, previous)
    assert out["a"]["significant"] and not out["b"]["significant"]
    assert out["d"]["p_holm"] is None


def _record(root: Path, scenario: str, arm: str, seed: int, cost: float, status: str = "ok",
            verified=True, fingerprint="abc", episodes=20, season="winter") -> None:
    policy = "rl" if arm.startswith("rl") else "rbc"
    rec = {"meta": {"scenario": {"schema": TX_SCHEMA, "season": season, "days": 28,
                                 "grid_cap_kw": 300.0, "building_cap_kw": 80.0,
                                 "key": scenario},
                    "arm": {"name": arm, "policy": policy}, "seed": seed,
                    "episodes": episodes if policy == "rl" else 0,
                    "code": {"fingerprint": fingerprint}},
           "status": status}
    if status == "ok":
        rec.update(eval={"cost": cost}, actuators={"verified": verified})
    else:
        rec["error"] = "boom"
    path = root / scenario / f"{arm}__seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec), encoding="utf-8")


def _run_aggregate(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["aggregate", str(tmp_path), "--kpis", "cost", *extra])
    aggregate.main()
    return json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_seeds_are_averaged_within_a_scenario_before_replication(tmp_path, monkeypatch):
    for scen, ref, rl in (("A", 10.0, (8.0, 9.0)), ("B", 20.0, (17.0, 19.0))):
        _record(tmp_path, scen, "rbc+calibrated", 0, ref)
        for seed, v in enumerate(rl):
            _record(tmp_path, scen, "rl+calibrated", seed, v)
    _record(tmp_path, "A", "rl+calibrated", 2, 1000.0, verified=False)   # excluded, listed
    _record(tmp_path, "B", "rl+calibrated", 3, 0.0, status="error")      # counted
    summary = _run_aggregate(tmp_path, monkeypatch)

    assert summary["counts"] == {"records": 8, "ok": 7, "errors": 1, "failed": 1}
    arm = summary["arms"]["pooled"]["cost"]["rl+calibrated"]
    assert arm["n"] == 2 and arm["mean"] == pytest.approx((8.5 + 18.0) / 2)
    c = summary["contrasts"]["pooled"]["rl+calibrated - rbc+calibrated"]["cost"]
    assert c["n"] == 2, "two scenarios are two replicates, not four seeds"
    assert c["mean"] == pytest.approx(np.mean([8.5 - 10.0, 18.0 - 20.0]))
    assert "INVESTIGATE" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_learning_contrast_uses_only_matched_seeds(tmp_path, monkeypatch):
    _record(tmp_path, "A", "rl+calibrated", 0, 5.0)
    _record(tmp_path, "A", "rl+calibrated", 1, 100.0)    # no seed-1 partner in `rl`
    _record(tmp_path, "A", "rl", 0, 7.0)
    summary = _run_aggregate(tmp_path, monkeypatch)
    c = summary["contrasts"]["pooled"]["rl+calibrated - rl"]["cost"]
    assert c["mean"] == pytest.approx(-2.0) and c["seeds_per_scenario"] == [1]


def test_records_from_different_code_are_refused(tmp_path, monkeypatch):
    _record(tmp_path, "A", "rbc", 0, 10.0, fingerprint="old")
    _record(tmp_path, "A", "rl", 0, 8.0, fingerprint="new")
    with pytest.raises(SystemExit, match="code fingerprint"):
        _run_aggregate(tmp_path, monkeypatch)
    assert _run_aggregate(tmp_path, monkeypatch, "--allow-mixed")["provenance_conflicts"]


def test_charge_commands_on_a_full_store_are_not_evidence():
    """An unshielded policy pinned at full charge must not be marked as an inert actuator."""
    ev = ActuatorEvidence(_FakeEnv())
    B = 2
    for t in range(60):
        cmd = 0.8 if t % 2 == 0 else 0.0
        actions = np.zeros((B, 3), dtype=np.float32)
        actions[:, 0] = actions[:, 1] = cmd
        pre = [np.zeros(30, dtype=np.float32) for _ in range(B)]
        post = [np.zeros(30, dtype=np.float32) for _ in range(B)]
        for i in range(B):
            pre[i][19] = post[i][19] = 1.0          # battery full: cannot move
            pre[i][18], post[i][18] = 0.5, (0.55 if cmd else 0.49)
        ev.add(pre, actions, post)
    s = ev.summary()
    assert s["battery"]["responds"] is None and s["battery"]["n_charge"] == 0
    assert s["verified"] is not False
