from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stems.config import STEMSConfig
from stems.encoder import STEncoder
from stems.cbf import CBFShield
from stems.graph import BuildingGraph
from stems.utils import RunningNormalizer


class Actor(nn.Module):
    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)

    def set_initial_log_std(self, value: float) -> None:
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(value))

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(r)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def distribution(self, r: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = self.forward(r)
        return torch.distributions.Normal(mean, log_std.exp())

    @staticmethod
    def pre_tanh_of(a: torch.Tensor) -> torch.Tensor:
        return torch.atanh(a.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r).squeeze(-1)


class CostCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_constraints: int = 3) -> None:
        super().__init__()
        self.num_constraints = num_constraints
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_constraints),
        )
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class STEMSAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int,
        building_graph: BuildingGraph,
        config: Optional[STEMSConfig] = None,
        battery_info: Optional[Dict[str, np.ndarray]] = None,
        use_cbf: bool = True,
        device: str = "cpu",
        electrical_storage_action_index: int = 1,
        control_indices: Optional[List[int]] = None,
        dhw_barrier: Optional[object] = None,
        cop_model: Optional[object] = None,
        hvac_action_index: int = -1,
        deadline_barriers: Optional[List[object]] = None,
        coordination: str = "independent",
        battery_model: Optional[object] = None,
        base_policy: Optional[object] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.base_policy = base_policy
        self.action_dim = action_dim
        self.B = num_buildings
        self.graph = building_graph
        self.cfg = config or STEMSConfig()
        self.use_cbf = use_cbf
        self.device = torch.device(device)
        self.elec_idx = int(electrical_storage_action_index)
        self.control_indices = (list(range(action_dim)) if control_indices is None
                                else list(control_indices))

        self.adj = self.graph.compute_edge_weights().to(self.device)

        self.encoder = STEncoder(
            obs_dim=obs_dim,
            spatial_dim=self.cfg.gcn.hidden_dim,
            temporal_dim=self.cfg.transformer.embed_dim,
            output_dim=self.cfg.fusion.output_dim,
            gcn_num_layers=self.cfg.gcn.num_layers,
            num_heads=self.cfg.transformer.num_heads,
            window_size=self.cfg.transformer.window_size,
        ).to(self.device)
        repr_dim = self.cfg.fusion.output_dim
        hidden = self.cfg.actor_critic.hidden_dim
        K = self.cfg.lagrangian.num_constraints

        n = 1 if self.cfg.actor_critic.share_parameters else self.B
        self.actors = nn.ModuleList([Actor(repr_dim, hidden, action_dim)
                                     for _ in range(n)]).to(self.device)
        self.critics = nn.ModuleList([Critic(repr_dim, hidden)
                                      for _ in range(n)]).to(self.device)
        self.cost_critics = nn.ModuleList([CostCritic(repr_dim, hidden, K)
                                           for _ in range(n)]).to(self.device)

        lr = self.cfg.actor_critic.lr
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actors.parameters())
            + list(self.critics.parameters()) + list(self.cost_critics.parameters()), lr=lr)
        self.return_normalizer = RunningNormalizer(1).to(self.device)

        lag = self.cfg.lagrangian
        self._lambdas = torch.full((K,), lag.lambda_init, device=self.device)
        self._cost_integral = torch.full((K,), lag.lambda_init, device=self.device)
        self._prev_cost = torch.zeros(K, device=self.device)
        self._cost_limit = torch.full((K,), lag.cost_limit, device=self.device)

        soc_rate = (np.asarray(battery_info["soc_rate"], dtype=np.float32)
                    if battery_info is not None else None)
        nominal_power = (np.asarray(battery_info["nominal_power"], dtype=np.float32)
                         if battery_info is not None else None)
        self.cbf = CBFShield(
            config=self.cfg.cbf, num_buildings=self.B, soc_rate=soc_rate,
            nominal_power=nominal_power, action_scale=self.cfg.training.action_scale,
            elec_idx=self.elec_idx, safety_cfg=self.cfg.safety,
            enforce_soc=(self.elec_idx in self.control_indices),
            dhw_barrier=dhw_barrier, cop_model=cop_model,
            hvac_idx=hvac_action_index,
            deadline_barriers=deadline_barriers, coordination=coordination,
            battery_model=battery_model,
        )
        self._dhw_forecaster = getattr(dhw_barrier, "forecaster", None)

        self.obs_normalizer = RunningNormalizer(obs_dim).to(self.device)
        self._update_step = 0
        self._last_raw_actions = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_safe_actions = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_pre_tanh = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_log_probs = np.zeros(self.B, dtype=np.float32)
        self._ctrl = torch.tensor(self.control_indices, dtype=torch.long, device=self.device)
        self._last_nominal_actions = np.zeros((self.B, action_dim), dtype=np.float32)
        self.fleet_shield = None
        self.request_floor = None
        if self.base_policy is not None:
            for actor in self.actors:
                actor.set_initial_log_std(self.cfg.training.residual_log_std)

    def select_action(self, obs_list: List[np.ndarray], history: np.ndarray,
                      explore: bool = True) -> np.ndarray:
        self.encoder.eval()
        for actor in self.actors:
            actor.eval()
        with torch.no_grad():
            x = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=self.device)
            h = torch.tensor(history, dtype=torch.float32, device=self.device)
            x_norm = self.obs_normalizer(x)
            h_norm = self.obs_normalizer(h.view(-1, self.obs_dim)).view(h.shape)
            repr_mat = self.encoder(x_norm, self.adj, h_norm)
            raw = np.empty((self.B, self.action_dim), dtype=np.float32)
            pre = np.empty((self.B, self.action_dim), dtype=np.float32)
            logp = np.zeros(self.B, dtype=np.float32)
            for i in range(self.B):
                dist = self.actors[i % len(self.actors)].distribution(repr_mat[i].unsqueeze(0))
                z = dist.sample() if explore else dist.mean
                logp[i] = float(dist.log_prob(z)[..., self._ctrl].sum())
                pre[i] = z.squeeze(0).cpu().numpy()
                raw[i] = torch.tanh(z).squeeze(0).cpu().numpy()

        self._last_raw_actions = raw.copy()
        self._last_pre_tanh = pre
        self._last_log_probs = logp
        if self.base_policy is not None:
            base = np.asarray(self.base_policy.select_action(obs_list), dtype=np.float32)
            nominal = np.clip(base + self.cfg.training.residual_scale * raw, -1.0, 1.0)
        else:
            nominal = raw
        if self.request_floor is not None:
            nominal = np.maximum(nominal, np.asarray(self.request_floor(obs_list), dtype=np.float32))
        self._last_nominal_actions = nominal.copy()
        safe = self.cbf.project(nominal, obs_list) if self.use_cbf else nominal.copy()
        safe = np.clip(safe, -1.0, 1.0).astype(np.float32)
        if self.fleet_shield is not None and self.use_cbf:
            safe = self.fleet_shield.project(safe, obs_list)
        if len(self.control_indices) < self.action_dim:
            mask = np.zeros(self.action_dim, dtype=np.float32)
            mask[self.control_indices] = 1.0
            safe *= mask
        self._last_safe_actions = safe.copy()
        if self.base_policy is not None and hasattr(self.base_policy, "notify_executed"):
            self.base_policy.notify_executed(safe)
        return safe

    def observe(self, next_obs_list: List[np.ndarray],
                ev_draw_kwh: Optional[np.ndarray] = None) -> None:
        if self._dhw_forecaster is not None:
            self._dhw_forecaster.update(next_obs_list)
        if self.fleet_shield is not None and ev_draw_kwh is not None:
            self.fleet_shield.observe(next_obs_list, ev_draw_kwh)

    @staticmethod
    def _compute_gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
                     episode_end: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[0])
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            last = delta + gamma * lam * (1.0 - episode_end[t]) * last
            adv[t] = last
        return adv

    def _update_lambdas(self, mean_costs: torch.Tensor) -> None:
        lag = self.cfg.lagrangian
        with torch.no_grad():
            err = mean_costs - self._cost_limit
            if lag.use_pid:
                self._cost_integral = torch.clamp(
                    self._cost_integral + lag.pid_ki * err, 0.0, lag.lambda_max)
                deriv = torch.relu(mean_costs - self._prev_cost)
                self._lambdas = torch.clamp(
                    lag.pid_kp * err + self._cost_integral + lag.pid_kd * deriv,
                    0.0, lag.lambda_max)
                self._prev_cost = mean_costs.clone()
            else:
                self._cost_integral = torch.clamp(
                    self._cost_integral + lag.lambda_lr * err, 0.0, lag.lambda_max)
                self._lambdas = self._cost_integral.clone()

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        cfgt = self.cfg.training
        gamma, lam = self.cfg.actor_critic.gamma, float(cfgt.gae_lambda)
        N, B = len(batch["obs"]), self.B
        if cfgt.actor_target not in ("raw", "safe"):
            raise ValueError(f"unknown actor_target {cfgt.actor_target!r}")
        raw_target = cfgt.actor_target == "raw"

        required = ["history", "next_history", "safe_actions"]
        if raw_target:
            required += ["pre_tanh", "behaviour_log_probs"]
        for key in required:
            if batch.get(key) is None:
                raise KeyError(f"batch is missing required '{key}'. The trajectory "
                               "collector must always store it (see experiments/runner.py).")

        t = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)
        obs_nb, next_nb = t(batch["obs"]), t(batch["next_obs"])
        if int(self.obs_normalizer.count) == 0:
            with torch.no_grad():
                self.obs_normalizer.update(obs_nb.view(-1, self.obs_dim))
            return {"policy": 0.0, "value": 0.0, "cost_value": 0.0, "entropy": 0.0,
                    "clip_frac": 0.0, "approx_kl": 0.0, "gradient_steps": 0,
                    "stopped_early": False, "reward_scale": 1.0, "warmup": True,
                    "lambdas": self._lambdas.detach().cpu().tolist()}
        hist_nb, next_hist_nb = t(batch["history"]), t(batch["next_history"])
        rewards = t(batch["rewards"])
        episode_end = t(batch["dones"])
        costs = (t(batch["constraint_costs"]) if batch.get("constraint_costs") is not None
                 else None)

        def encode(idx: torch.Tensor, use_next: bool = False) -> torch.Tensor:
            src_o, src_h = (next_nb, next_hist_nb) if use_next else (obs_nb, hist_nb)
            o = self.obs_normalizer(src_o[idx])
            h = self.obs_normalizer(src_h[idx].reshape(-1, self.obs_dim)).view(src_h[idx].shape)
            return self.encoder.batch_forward(o, self.adj, h)

        all_idx = torch.arange(N, device=self.device)
        for m in (self.encoder, *self.actors, *self.critics, *self.cost_critics):
            m.eval()

        with torch.no_grad():
            if cfgt.scale_rewards:
                ret = torch.zeros_like(rewards)
                running = torch.zeros(B, device=self.device)
                for k in reversed(range(N)):
                    running = rewards[k] + gamma * (1.0 - episode_end[k]) * running
                    ret[k] = running
                self.return_normalizer.update(ret.reshape(-1, 1))
                scale = float(self.return_normalizer.var.sqrt().clamp_min(1e-6))
            else:
                scale = 1.0
            r_scaled = rewards / scale

            repr_all, repr_next = encode(all_idx), encode(all_idx, use_next=True)
            values = torch.stack([self.critics[b % len(self.critics)](repr_all[:, b]) for b in range(B)], 1)
            next_values = torch.stack([self.critics[b % len(self.critics)](repr_next[:, b]) for b in range(B)], 1)
            adv = self._compute_gae(r_scaled, values, next_values, episode_end, gamma, lam)
            returns = adv + values
            adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)

            if costs is not None:
                cv = torch.stack([self.cost_critics[b % len(self.cost_critics)](repr_all[:, b]) for b in range(B)], 1)
                cnv = torch.stack([self.cost_critics[b % len(self.cost_critics)](repr_next[:, b]) for b in range(B)], 1)
                cadv = self._compute_gae(costs, cv, cnv, episode_end, gamma, lam)
                cost_returns = cadv + cv
                cadv = cadv - cadv.mean(0)
                lam_k = torch.clamp(self._lambdas, min=0.0)
                eff_adv = (adv - (cadv * lam_k).sum(-1)) / (1.0 + lam_k.sum())
            else:
                cost_returns, eff_adv = None, adv

            if raw_target:
                z_all = t(batch["pre_tanh"])
                old_logp = t(batch["behaviour_log_probs"])
            else:
                z_all = Actor.pre_tanh_of(t(batch["safe_actions"]))
                old_logp = torch.stack(
                    [self.actors[b % len(self.actors)].distribution(repr_all[:, b]).log_prob(z_all[:, b])
                     [..., self._ctrl].sum(-1) for b in range(B)], 1)

        mb = max(2, min(int(cfgt.minibatch_size), N))
        clip, ent_coef, v_coef = (float(cfgt.ppo_clip), float(cfgt.entropy_coef),
                                  float(cfgt.value_coef))
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        sums = {"policy": 0.0, "value": 0.0, "cost_value": 0.0, "entropy": 0.0,
                "clip_frac": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for m in (self.encoder, *self.actors, *self.critics, *self.cost_critics):
            m.train()

        target_kl = cfgt.target_kl
        stopped_early = False
        for _ in range(max(1, int(cfgt.update_epochs))):
            if stopped_early:
                break
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                if idx.numel() < 2:
                    continue
                repr_mb = encode(idx)
                zero = torch.zeros((), device=self.device)
                policy_loss, value_loss, cost_value_loss, entropy = zero, zero, zero, zero
                kl = clipped = 0.0
                for b in range(B):
                    r_b = repr_mb[:, b]
                    dist = self.actors[b % len(self.actors)].distribution(r_b)
                    logp = dist.log_prob(z_all[idx, b])[..., self._ctrl].sum(-1)
                    log_ratio = logp - old_logp[idx, b]
                    ratio = log_ratio.clamp(-20.0, 20.0).exp()
                    a_b = eff_adv[idx, b]
                    policy_loss = policy_loss - torch.min(
                        ratio * a_b, ratio.clamp(1.0 - clip, 1.0 + clip) * a_b).mean()
                    entropy = entropy + dist.entropy()[..., self._ctrl].sum(-1).mean()
                    value_loss = value_loss + F.mse_loss(self.critics[b % len(self.critics)](r_b), returns[idx, b])
                    if cost_returns is not None:
                        cost_value_loss = cost_value_loss + F.mse_loss(
                            self.cost_critics[b % len(self.cost_critics)](r_b), cost_returns[idx, b])
                    with torch.no_grad():
                        kl += float(((ratio - 1.0) - log_ratio).mean())
                        clipped += float(((ratio - 1.0).abs() > clip).float().mean())
                loss = (policy_loss + v_coef * (value_loss + cost_value_loss)
                        - ent_coef * entropy) / B
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, float(cfgt.max_grad_norm))
                self.optimizer.step()

                sums["policy"] += policy_loss.item() / B
                sums["value"] += value_loss.item() / B
                sums["cost_value"] += cost_value_loss.item() / B
                sums["entropy"] += entropy.item() / B
                sums["approx_kl"] += kl / B
                sums["clip_frac"] += clipped / B
                n_updates += 1
                if target_kl and kl / B > 1.5 * target_kl:
                    stopped_early = True
                    break

        with torch.no_grad():
            self.obs_normalizer.update(obs_nb.view(-1, self.obs_dim))
        if costs is not None:
            self._update_lambdas(costs.mean(dim=(0, 1)))
        self._update_step += 1
        k = max(n_updates, 1)
        return {**{key: v / k for key, v in sums.items()},
                "gradient_steps": n_updates, "stopped_early": stopped_early,
                "reward_scale": scale,
                "lambdas": self._lambdas.detach().cpu().tolist()}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, "encoder.pt"))
        torch.save(self.actors.state_dict(), os.path.join(path, "actors.pt"))
        torch.save(self.critics.state_dict(), os.path.join(path, "critics.pt"))
        torch.save(self.cost_critics.state_dict(), os.path.join(path, "cost_critics.pt"))
        torch.save({"lambdas": self._lambdas, "cost_integral": self._cost_integral,
                    "prev_cost": self._prev_cost}, os.path.join(path, "lagrangian.pt"))
        torch.save(self.obs_normalizer.state_dict(), os.path.join(path, "obs_normalizer.pt"))
        torch.save(self.return_normalizer.state_dict(), os.path.join(path, "return_normalizer.pt"))

    def load(self, path: str) -> None:
        map_loc = self.device

        def _load(module: nn.Module, fname: str, label: str) -> None:
            try:
                module.load_state_dict(torch.load(os.path.join(path, fname), map_location=map_loc))
            except RuntimeError:
                raise RuntimeError(
                    f"Checkpoint {path!r} is incompatible with this agent "
                    f"(B={self.B}, obs_dim={self.obs_dim}, action_dim={self.action_dim}); "
                    f"failed loading {label}. Retrain with the current schema/config."
                ) from None

        _load(self.encoder, "encoder.pt", "encoder")
        _load(self.actors, "actors.pt", "actors")
        _load(self.critics, "critics.pt", "critics")
        if os.path.exists(os.path.join(path, "cost_critics.pt")):
            _load(self.cost_critics, "cost_critics.pt", "cost critics")
        lag_path = os.path.join(path, "lagrangian.pt")
        if os.path.exists(lag_path):
            d = torch.load(lag_path, map_location=map_loc)
            with torch.no_grad():
                self._lambdas = d["lambdas"].to(self.device)
                self._cost_integral = d["cost_integral"].to(self.device)
                self._prev_cost = d["prev_cost"].to(self.device)
        if os.path.exists(os.path.join(path, "obs_normalizer.pt")):
            self.obs_normalizer.load_state_dict(
                torch.load(os.path.join(path, "obs_normalizer.pt"), map_location=map_loc))
        if os.path.exists(os.path.join(path, "return_normalizer.pt")):
            self.return_normalizer.load_state_dict(
                torch.load(os.path.join(path, "return_normalizer.pt"), map_location=map_loc))
