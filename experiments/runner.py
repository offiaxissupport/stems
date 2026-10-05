from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]

_IDX_SOC_DHW, _IDX_SOC, _IDX_NET = 18, 19, 20


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def git_state() -> str:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


FINGERPRINTED = ("stems/*.py", "experiments/*.py")


def code_fingerprint() -> Dict[str, Any]:
    import hashlib

    h = hashlib.sha256()
    files = sorted({p for pattern in FINGERPRINTED for p in REPO.glob(pattern)})
    for path in files:
        h.update(path.relative_to(REPO).as_posix().encode())
        h.update(path.read_bytes())
    try:
        import citylearn
        cl_version = getattr(citylearn, "__version__", "unknown")
    except Exception:
        cl_version = "unavailable"
    return {"git": git_state(), "fingerprint": h.hexdigest()[:16], "files": len(files),
            "citylearn": cl_version}


def make_config(scenario, learner: Optional[Dict[str, Any]] = None):
    from stems.config import CBFConfig, STEMSConfig

    config = STEMSConfig()
    config.cbf = CBFConfig(P_grid_max=scenario.grid_cap_kw,
                           P_building_max=scenario.building_cap_kw)
    config.heat_pump.enabled = bool(getattr(scenario, "heat_pump", True))
    config.actor_critic.share_parameters = bool((learner or {}).get("share_parameters", False))
    return config


class ActuatorEvidence:
    CHARGE, IDLE, MIN_N, MIN_CONTRAST, MIN_CORR, MIN_HVAC_STD = 0.2, 0.05, 20, 0.01, 0.3, 0.02
    HEADROOM = 0.95

    def __init__(self, env) -> None:
        self.env = env
        self.idx = {"battery": env.electrical_storage_action_index,
                    "dhw": env.dhw_action_index,
                    "hvac": env.hvac_action_index}
        self.actions: List[np.ndarray] = []
        self.pre: List[np.ndarray] = []
        self.post: List[np.ndarray] = []

    def add(self, obs, actions, next_obs) -> None:
        self.actions.append(np.asarray(actions, dtype=np.float32).copy())
        self.pre.append(np.stack([np.asarray(o, dtype=np.float32) for o in obs]))
        self.post.append(np.stack([np.asarray(o, dtype=np.float32) for o in next_obs]))

    def _storage(self, name: str, soc_idx: int) -> Optional[Dict[str, Any]]:
        j = self.idx[name]
        if j is None or j < 0 or not self.actions:
            return None
        a = np.stack(self.actions)[:, :, j].ravel()
        pre = np.stack(self.pre)[:, :, soc_idx].ravel()
        d = np.stack(self.post)[:, :, soc_idx].ravel() - pre
        charge, rest = d[(a > self.CHARGE) & (pre < self.HEADROOM)], d[a <= self.IDLE]
        out: Dict[str, Any] = {"n_charge": int(charge.size), "n_rest": int(rest.size)}
        if charge.size < self.MIN_N or rest.size < self.MIN_N:
            out["responds"] = None
            return out
        contrast = float(charge.mean() - rest.mean())
        out.update(contrast=contrast, responds=bool(contrast > self.MIN_CONTRAST))
        return out

    def _hvac(self) -> Optional[Dict[str, Any]]:
        j = self.idx["hvac"]
        if j is None or j < 0 or not self.actions or getattr(self.env, "using_mock", False):
            return None
        a = np.stack(self.actions)[:, :, j]
        try:
            heat = [np.asarray(b.heating_electricity_consumption, dtype=float)
                    for b in self.env._env.buildings]
            cool = [np.asarray(b.cooling_electricity_consumption, dtype=float)
                    for b in self.env._env.buildings]
        except Exception as exc:
            return {"responds": None, "reason": f"consumption unavailable: {exc!r}"}
        out: Dict[str, Any] = {
            "heating": self._mode_correlation(np.maximum(a, 0.0), heat),
            "cooling": self._mode_correlation(np.maximum(-a, 0.0), cool)}
        verdicts = [m["responds"] for m in out.values()]
        out["responds"] = (False if False in verdicts
                           else True if True in verdicts else None)
        return out

    def _mode_correlation(self, cmd: np.ndarray, cons: List[np.ndarray]) -> Dict[str, Any]:
        T, B = cmd.shape
        if float(cmd.std()) < self.MIN_HVAC_STD:
            return {"responds": None, "reason": "command barely varied"}
        best: Optional[float] = None
        for lag in (0, 1):
            xs, ys = [], []
            for b in range(B):
                if cons[b].size >= lag + T:
                    xs.append(cmd[:, b])
                    ys.append(cons[b][lag:lag + T])
            if not xs:
                continue
            x, y = np.concatenate(xs), np.concatenate(ys)
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            best = r if best is None else max(best, r)
        if best is None:
            return {"responds": None, "reason": "no aligned consumption series"}
        return {"correlation": best, "responds": bool(best > self.MIN_CORR)}

    def summary(self) -> Dict[str, Any]:
        checks = {"battery": self._storage("battery", _IDX_SOC),
                  "dhw": self._storage("dhw", _IDX_SOC_DHW),
                  "hvac": self._hvac()}
        verdicts = [c["responds"] for c in checks.values() if c is not None]
        if any(v is False for v in verdicts):
            verified: Optional[bool] = False
        elif verdicts and all(v is True for v in verdicts):
            verified = True
        else:
            verified = None
        return {"verified": verified, **checks}


def train(agent, env, config, episodes: int, log: Callable[[str], None],
          max_steps: int) -> List[Dict[str, Any]]:
    from stems.reward import STEMSReward
    from stems.utils import EpisodeBuffer, HistoryBuffer

    B, cbf = env.num_buildings, config.cbf
    ev_layouts = [] if env.using_mock else env.ev_obs_layout()
    reward_fn = STEMSReward(config.reward, B, cbf.P_grid_max, cbf.P_building_max,
                            heating_setpoint_idx=env.heating_setpoint_idx,
                            ev_layout=ev_layouts[0] if ev_layouts else None)
    buffer = EpisodeBuffer()
    hist = HistoryBuffer(B, env.obs_dim, config.transformer.window_size)
    curve: List[Dict[str, Any]] = []
    fleet_shield = getattr(agent, "fleet_shield", None)
    battery_controlled = agent.elec_idx in agent.control_indices

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs, _ = env.reset()
        hist.reset()
        hist.update(obs)
        buffer.reset()
        if hasattr(getattr(agent, "base_policy", None), "reset"):
            agent.base_policy.reset()
        prev_net = [float(o[_IDX_NET]) for o in obs]
        ep_reward, n, violating, done = 0.0, 0, 0, False

        while not done:
            window = hist.get()
            actions = agent.select_action(obs, window, explore=True)
            nxt, _, term, trunc, _ = env.step(actions)
            agent.observe(nxt, env.ev_draw_kwh)
            n += 1
            done = bool(term or trunc) or n >= max_steps

            rewards = reward_fn.compute(obs, actions, nxt, prev_net,
                                        ev_departures=env.ev_departures if ev_layouts else None)
            prev_net = [float(o[_IDX_NET]) for o in nxt]
            if config.training.forced_charge_penalty and fleet_shield is not None:
                forced = fleet_shield.last.get("forced_kw")
                if forced is not None:
                    rewards = [r - config.training.forced_charge_penalty * float(f)
                               for r, f in zip(rewards, forced)]
            if config.training.intervention_penalty:
                moved = ((agent._last_nominal_actions - agent._last_safe_actions) ** 2).sum(axis=1)
                rewards = [r - config.training.intervention_penalty * float(m)
                           for r, m in zip(rewards, moved)]

            soc = np.array([o[_IDX_SOC] for o in nxt], dtype=np.float32)
            net = np.array([o[_IDX_NET] for o in nxt], dtype=np.float32)
            c_soc = ((soc < cbf.SOC_min) | (soc > cbf.SOC_max)).astype(np.float32)
            if not battery_controlled:
                c_soc[:] = 0.0
            c_pow = (np.abs(net) > cbf.P_building_max).astype(np.float32)
            grid = float(np.maximum(net, 0.0).sum() > cbf.P_grid_max)
            costs = np.stack([c_soc, c_pow, np.full(B, grid, dtype=np.float32)], axis=-1)
            violating += int(costs.max() > 0)

            hist.update(nxt)
            buffer.add(obs=obs, actions=actions, rewards=rewards, next_obs=nxt,
                       done=done, history=window, next_history=hist.get(),
                       raw_actions=agent._last_raw_actions,
                       safe_actions=agent._last_safe_actions,
                       pre_tanh=agent._last_pre_tanh,
                       behaviour_log_probs=agent._last_log_probs,
                       constraint_costs=costs)
            ep_reward += float(np.mean(rewards))
            obs = nxt

        stats = agent.update(buffer.get_batch())
        row = {"episode": ep, "steps": n, "reward": ep_reward,
               "violating_step_rate": violating / max(n, 1),
               "gradient_steps": int(stats.get("gradient_steps", 0)),
               "lambdas": stats.get("lambdas"), "entropy": stats.get("entropy"),
               "approx_kl": stats.get("approx_kl"), "clip_frac": stats.get("clip_frac"),
               "value_loss": stats.get("value"), "reward_scale": stats.get("reward_scale"),
               "seconds": round(time.time() - t0, 1)}
        curve.append(row)
        log(f"ep {ep}/{episodes} reward={ep_reward:.1f} "
            f"violating_steps={row['violating_step_rate']:.3f} "
            f"grad_steps={row['gradient_steps']} ({row['seconds']}s)")
    return curve


def evaluate(controller, env, config, max_steps: int) -> Dict[str, Any]:
    from stems.metrics import MetricsCalculator
    from stems.utils import HistoryBuffer

    B = env.num_buildings
    controlled = getattr(controller, "control_indices", None)
    count_soc = controlled is None or env.electrical_storage_action_index in controlled
    metrics = MetricsCalculator(B, config.cbf, soc_rate=env.battery_info()["soc_rate"],
                                heating_setpoint_idx=env.heating_setpoint_idx,
                                count_soc=count_soc, hvac_idx=env.hvac_action_index)
    evidence = ActuatorEvidence(env)
    hist = HistoryBuffer(B, env.obs_dim, config.transformer.window_size)
    has_ev = not env.using_mock and bool(env.ev_action_indices())

    obs, _ = env.reset()
    hist.update(obs)
    n, done = 0, False
    while not done:
        actions = controller.select_action(obs, hist.get(), explore=False)
        raw = getattr(controller, "_last_nominal_actions",
                      getattr(controller, "_last_raw_actions", None))
        nxt, _, term, trunc, _ = env.step(actions)
        if hasattr(controller, "observe"):
            controller.observe(nxt, env.ev_draw_kwh)
        executed = env.executed_actions
        metrics.add_step(obs, actions, nxt, raw_actions=raw, device_actions=executed)
        if has_ev:
            metrics.add_ev_departures(env.ev_departures)
        evidence.add(obs, executed, nxt)
        hist.update(nxt)
        obs = nxt
        n += 1
        done = bool(term or trunc) or n >= max_steps
    return {"kpis": metrics.compute_all(), "actuators": evidence.summary(), "steps": n}


def _window_len(kwargs: Dict[str, Any]) -> int:
    (start, end), = kwargs["episode_time_steps"]
    return int(end - start + 1)


def run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)

    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed
    from experiments.controllers import ARMS, build_controller
    from experiments.scenario import Scenario

    out = Path(spec["out"])
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")
    log_path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    t_start = time.time()
    scenario = Scenario(**spec["scenario"])
    arm = ARMS[spec["arm"]]
    seed = int(spec["seed"])
    episodes = int(spec.get("episodes", 0))
    learner = dict(spec.get("learner") or {})
    record: Dict[str, Any] = {"meta": {
        "scenario": scenario.describe(), "arm": asdict(arm), "seed": seed,
        "episodes": episodes if arm.learns else 0, "learner": learner if arm.learns else {},
        "code": code_fingerprint(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}}

    try:
        set_seed(seed)
        schema = scenario.schema_path()
        train_kw, eval_kw = scenario.env_kwargs("train"), scenario.env_kwargs("eval")
        model_dir: Optional[Path] = None

        def make_env(kwargs):
            return STEMSEnvironment(schema=schema, seed=seed, heat_pump=scenario.heat_pump,
                                    env_kwargs=kwargs, hvac_control=scenario.hvac_control,
                                    allow_missing_obs=scenario.allow_missing_obs)

        def fit_config(cfg, env):
            if env.hvac_action_index < 0:
                cfg.reward.lambda_indoor = 0.0
            return cfg

        if arm.learns:
            log(f"train {scenario.key} arm={arm.name} seed={seed} episodes={episodes}")
            train_env = make_env(train_kw)
            train_config = fit_config(make_config(scenario, learner), train_env)
            agent = build_controller(arm, train_env, train_config)
            record["train"] = train(agent, train_env, train_config, episodes, log,
                                    _window_len(train_kw))
            model_dir = out.parent / f"{out.stem}_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(model_dir))
            record["meta"]["train_calibration"] = {
                "soc_rate": train_env.battery_info()["soc_rate"],
                "dhw_capacity_kwh": train_env.dhw_info()["capacity"]}
            del agent, train_env

        log("evaluate")
        eval_env = make_env(eval_kw)
        config = fit_config(make_config(scenario, learner), eval_env)
        controller = build_controller(arm, eval_env, config)
        if model_dir is not None:
            controller.load(str(model_dir))
        if getattr(controller, "fleet_shield", None) is not None:
            log("warm-up: load forecast on the training window")
            warm_env = make_env(train_kw)
            warm = evaluate(controller, warm_env, config, _window_len(train_kw))
            record["meta"]["forecast_warmup_steps"] = warm["steps"]
            for inner in (getattr(controller, "base", None),
                          getattr(controller, "base_policy", None)):
                if inner is not None and hasattr(inner, "reset"):
                    inner.reset()
            del warm_env
        result = evaluate(controller, eval_env, config, _window_len(eval_kw))

        record.update(status="ok", eval=result["kpis"], actuators=result["actuators"],
                      eval_steps=result["steps"])
        record["meta"].update(
            eval_calibration={"soc_rate": eval_env.battery_info()["soc_rate"],
                              "dhw_capacity_kwh": eval_env.dhw_info()["capacity"]},
            citylearn_patches=eval_env.citylearn_patches,
            absent_observations=([] if eval_env.using_mock
                                 else list(eval_env.absent_observations)),
            buildings_simulated=([b.name for b in eval_env._env.buildings]
                                 if not eval_env.using_mock else None),
            config={k: asdict(getattr(config, k))
                    for k in ("cbf", "safety", "thermal", "training", "lagrangian", "reward",
                              "actor_critic")})
        k = result["kpis"]
        log(f"done verified={result['actuators']['verified']} "
            f"violation={k.get('safety_violation_rate', float('nan')):.4f} "
            f"cost={k.get('cost', float('nan')):.1f}")
    except Exception as exc:
        record.update(status="error", error=repr(exc), traceback=traceback.format_exc())
        log(record["traceback"])

    record["seconds"] = round(time.time() - t_start, 1)
    out.write_text(json.dumps(record, indent=2, default=_jsonable), encoding="utf-8")
    return {"out": str(out), "status": record["status"], "seconds": record["seconds"],
            "verified": (record.get("actuators") or {}).get("verified")}
