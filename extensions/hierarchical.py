from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stems.cbf import CBFShield
from stems.config import CBFConfig, LagrangianConfig
from stems.environment import OBS_DIM, ACTION_DIM, _MockBuilding

_SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import SpectralClustering
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

_IDX_SOC_ELEC = 19
_IDX_NET      = 20

LOCAL_DIM    = 32
CLUSTER_DIM  = 64
HIDDEN_DIM   = 128
NUM_HEADS    = 4


class ClusterAssignment:
    def __init__(
        self,
        num_buildings: int,
        adj: Optional[np.ndarray] = None,
        cluster_size: int = 5,
    ) -> None:
        self.B = num_buildings
        self.K = max(1, math.ceil(num_buildings / cluster_size))
        self.labels: np.ndarray = self._assign(adj)

    def _assign(self, adj: Optional[np.ndarray]) -> np.ndarray:
        if adj is not None and _SKLEARN_AVAILABLE and self.K > 1:
            try:
                sc = SpectralClustering(
                    n_clusters=self.K,
                    affinity="precomputed",
                    random_state=0,
                    assign_labels="kmeans",
                )
                return sc.fit_predict(adj.astype(float)).astype(int)
            except Exception:
                pass
        labels = np.zeros(self.B, dtype=int)
        buildings_per_cluster = max(1, self.B // self.K)
        for i in range(self.B):
            labels[i] = min(i // buildings_per_cluster, self.K - 1)
        return labels

    def buildings_in(self, k: int) -> List[int]:
        return [i for i in range(self.B) if self.labels[i] == k]


class EventTrigger:
    def __init__(self, num_buildings: int, threshold: float = 0.5) -> None:
        self.B = num_buildings
        self.threshold = threshold
        self._last_obs: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_obs = None

    def __call__(self, obs_list: List[np.ndarray]) -> np.ndarray:
        obs = np.array(obs_list, dtype=np.float32)
        if self._last_obs is None:
            fired = np.ones(self.B, dtype=bool)
        else:
            delta = np.linalg.norm(obs - self._last_obs, axis=1)
            fired = delta > self.threshold
        self._last_obs = obs.copy()
        return fired


class LocalEncoder(nn.Module):
    def __init__(self, obs_dim: int, local_dim: int = LOCAL_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, local_dim),
        )
        self.norm = nn.LayerNorm(local_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class ClusterCoordinator(nn.Module):
    def __init__(
        self,
        local_dim: int = LOCAL_DIM,
        cluster_dim: int = CLUSTER_DIM,
        num_heads: int = NUM_HEADS,
    ) -> None:
        super().__init__()
        self.local_dim   = local_dim
        self.cluster_dim = cluster_dim
        self.input_proj = nn.Linear(local_dim, cluster_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=cluster_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cluster_dim)
        self.ff   = nn.Sequential(
            nn.Linear(cluster_dim, cluster_dim * 2), nn.ReLU(),
            nn.Linear(cluster_dim * 2, cluster_dim),
        )
        self.norm2 = nn.LayerNorm(cluster_dim)

    def forward(
        self,
        local_reprs: torch.Tensor,
        fired: torch.Tensor,
        cluster_assignment: ClusterAssignment,
        cached: Optional[torch.Tensor],
    ) -> torch.Tensor:
        K = cluster_assignment.K
        device = local_reprs.device
        dtype  = local_reprs.dtype

        summaries = torch.zeros(K, self.local_dim, device=device, dtype=dtype)
        updated = torch.zeros(K, dtype=torch.bool, device=device)

        for k in range(K):
            members = cluster_assignment.buildings_in(k)
            active  = [i for i in members if fired[i].item()]
            if active:
                idx = torch.tensor(active, dtype=torch.long, device=device)
                summaries[k] = local_reprs[idx].mean(dim=0)
                updated[k] = True
            elif cached is not None:
                pass

        projected = self.input_proj(summaries)

        if cached is not None:
            mask = updated.unsqueeze(1).float()
            projected = mask * projected + (1 - mask) * cached

        x = projected.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.squeeze(0)


class LocalPolicy(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(
        self,
        local_dim: int = LOCAL_DIM,
        cluster_dim: int = CLUSTER_DIM,
        hidden_dim: int = HIDDEN_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        in_dim = local_dim + cluster_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(feat)
        mean    = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(feat)
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        return y, log_prob.sum(dim=-1)

    def log_prob_of(
        self, feat: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self.forward(feat)
        a_clamped = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        x = torch.atanh(a_clamped)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(x) - torch.log(1 - actions.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)


class _QNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat, action], dim=-1)).squeeze(-1)


class _CostCriticNet(nn.Module):
    def __init__(
        self, in_dim: int, num_constraints: int = 3, hidden_dim: int = HIDDEN_DIM
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_constraints),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat, action], dim=-1))


class LargeGridEnv:
    def __init__(
        self,
        num_buildings: int = 50,
        seed: int = 0,
        episode_len: int = 8760,
    ) -> None:
        self.num_buildings = num_buildings
        self.obs_dim       = OBS_DIM
        self.action_dim    = ACTION_DIM
        self._episode_len  = episode_len
        self._seed         = seed
        self.using_mock    = True

        self._buildings = [
            _MockBuilding(np.random.default_rng(seed + i), i % 3)
            for i in range(num_buildings)
        ]
        self._t = 0

    def reset(self) -> Tuple[List[np.ndarray], Dict]:
        self._t = 0
        for b in self._buildings:
            b.reset()
        obs = [b.step(np.zeros(self.action_dim)) for b in self._buildings]
        return obs, {}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        self._t += 1
        obs     = [b.step(actions[i]) for i, b in enumerate(self._buildings)]
        rewards = [float(-o[_IDX_NET] * o[21]) for o in obs]
        done    = self._t >= self._episode_len
        return obs, rewards, done, False, {}

    def get_building_info(self) -> Dict[str, Any]:
        B = self.num_buildings
        positions = [
            [float(i % 10) * 100.0, float(i // 10) * 100.0] for i in range(B)
        ]
        features = [[1.0, 1.0, 1.0, 1.0] for _ in range(B)]
        return {"positions": positions, "features": features}


class HierarchicalSTEMSAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_buildings: int,
        adj: Optional[np.ndarray] = None,
        cluster_size: int = 5,
        event_threshold: float = 0.5,
        local_dim: int = LOCAL_DIM,
        cluster_dim: int = CLUSTER_DIM,
        hidden_dim: int = HIDDEN_DIM,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_ent: float = 0.2,
        cbf_config: Optional[CBFConfig] = None,
        lagrangian_cfg: Optional[LagrangianConfig] = None,
        use_cbf: bool = True,
        device: str = "cpu",
        electrical_storage_action_index: Optional[int] = None,
    ) -> None:
        self.B            = num_buildings
        self.obs_dim      = obs_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.tau          = tau
        self.use_cbf      = use_cbf
        self.device       = torch.device(device)
        if electrical_storage_action_index is None:
            electrical_storage_action_index = 1 if action_dim > 2 else 0
        self.electrical_storage_action_index = int(electrical_storage_action_index)

        if cbf_config is not None:
            cbf_cfg = cbf_config
        else:
            _base = CBFConfig()
            cbf_cfg = CBFConfig(
                SOC_min=_base.SOC_min,
                SOC_max=_base.SOC_max,
                P_building_max=_base.P_building_max,
                P_grid_max=_base.P_grid_max * num_buildings / 3.0,
                gamma_cbf=_base.gamma_cbf,
            )
        lag_cfg       = lagrangian_cfg or LagrangianConfig()
        self._lag_cfg = lag_cfg
        self._cbf_cfg = cbf_cfg

        self.cluster = ClusterAssignment(num_buildings, adj=adj, cluster_size=cluster_size)
        K = self.cluster.K

        self.event_trigger = EventTrigger(num_buildings, threshold=event_threshold)

        self.encoder     = LocalEncoder(obs_dim, local_dim).to(self.device)
        self.coordinator = ClusterCoordinator(local_dim, cluster_dim, NUM_HEADS).to(self.device)

        self._feat_dim = local_dim + cluster_dim
        feat_dim = self._feat_dim

        self.actors = nn.ModuleList([
            LocalPolicy(local_dim, cluster_dim, hidden_dim, action_dim)
            for _ in range(num_buildings)
        ]).to(self.device)

        self.q1_nets = nn.ModuleList([
            _QNet(feat_dim + action_dim, hidden_dim) for _ in range(num_buildings)
        ]).to(self.device)
        self.q2_nets = nn.ModuleList([
            _QNet(feat_dim + action_dim, hidden_dim) for _ in range(num_buildings)
        ]).to(self.device)
        self.q1_targets = copy.deepcopy(self.q1_nets).to(self.device)
        self.q2_targets = copy.deepcopy(self.q2_nets).to(self.device)

        num_constraints = lag_cfg.num_constraints
        self.cost_critics = nn.ModuleList([
            _CostCriticNet(feat_dim + action_dim, num_constraints, hidden_dim)
            for _ in range(num_buildings)
        ]).to(self.device)

        self.log_lambdas = nn.Parameter(
            torch.full((num_constraints,), math.log(lag_cfg.lambda_init), device=self.device)
        )

        self.target_entropy = -float(action_dim)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_ent), device=self.device)
        )

        shared_params = (
            list(self.encoder.parameters())
            + list(self.coordinator.parameters())
        )
        self.shared_opt = optim.Adam(shared_params, lr=lr)
        self.actor_opts = [
            optim.Adam(list(self.actors[i].parameters()), lr=lr)
            for i in range(num_buildings)
        ]
        self.q_opts = [
            optim.Adam(
                list(self.q1_nets[i].parameters())
                + list(self.q2_nets[i].parameters()),
                lr=lr,
            )
            for i in range(num_buildings)
        ]
        self.cost_opts = [
            optim.Adam(self.cost_critics[i].parameters(), lr=lr)
            for i in range(num_buildings)
        ]
        self.lambda_opt = optim.Adam([self.log_lambdas], lr=lag_cfg.lambda_lr)
        self.alpha_opt  = optim.Adam([self.log_alpha],   lr=lr)

        if use_cbf:
            self.cbf = CBFShield(
                cbf_cfg,
                num_buildings,
                electrical_storage_action_index=self.electrical_storage_action_index,
            )
        else:
            self.cbf = None

        self._cached_cluster_latents: Optional[torch.Tensor] = None


    @torch.no_grad()
    def _encode_all(
        self,
        obs_list: List[np.ndarray],
        fired: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_t = torch.tensor(
            np.array(obs_list, dtype=np.float32), dtype=torch.float32, device=self.device
        )

        local_reprs = self.encoder(obs_t)

        fired_arr = np.asarray(
            fired if fired is not None else np.ones(self.B, dtype=np.uint8), dtype=np.uint8
        )
        fired_t = torch.tensor(fired_arr, dtype=torch.bool, device=self.device)

        cluster_latents = self.coordinator(
            local_reprs, fired_t, self.cluster, self._cached_cluster_latents
        )

        self._cached_cluster_latents = cluster_latents.detach()
        return local_reprs, cluster_latents

    def _build_feat(
        self,
        local_reprs: torch.Tensor,
        cluster_latents: torch.Tensor,
    ) -> torch.Tensor:
        cluster_idx = torch.tensor(self.cluster.labels, dtype=torch.long, device=self.device)
        assigned    = cluster_latents[cluster_idx]
        return torch.cat([local_reprs, assigned], dim=-1)


    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = True,
    ) -> np.ndarray:
        fired = self.event_trigger(obs_list)
        local_reprs, cluster_latents = self._encode_all(obs_list, fired)
        feats = self._build_feat(local_reprs, cluster_latents)

        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i in range(self.B):
            feat_i = feats[i].unsqueeze(0)
            if explore:
                with torch.no_grad():
                    a, _ = self.actors[i].sample(feat_i)
            else:
                with torch.no_grad():
                    mean, _ = self.actors[i](feat_i)
                    a = torch.tanh(mean)
            actions[i] = np.array(a.squeeze(0).detach().tolist(), dtype=np.float32)

        if self.cbf is not None:
            actions = self.cbf.project(actions, obs_list)

        return actions


    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        N = len(batch["obs"])
        if N == 0:
            return {}

        device = self.device
        lambdas = torch.clamp(self.log_lambdas.exp(), 0.0, self._lag_cfg.lambda_max)
        alpha   = self.log_alpha.exp().item()

        losses: Dict[str, float] = {
            "actor_loss": 0.0, "q_loss": 0.0,
            "cost_loss": 0.0,  "lambda_loss": 0.0,
        }

        obs_t      = torch.tensor(
            np.array([[batch["obs"][n][b] for b in range(self.B)] for n in range(N)]),
            dtype=torch.float32, device=device,
        )
        next_obs_t = torch.tensor(
            np.array([[batch["next_obs"][n][b] for b in range(self.B)] for n in range(N)]),
            dtype=torch.float32, device=device,
        )
        actions_t  = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
        rewards_t  = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
        dones_t    = torch.tensor(batch["dones"],   dtype=torch.float32, device=device)

        obs_flat      = obs_t.view(N * self.B, self.obs_dim)
        next_obs_flat = next_obs_t.view(N * self.B, self.obs_dim)

        with torch.no_grad():
            local_reprs_flat      = self.encoder(obs_flat)
            local_reprs_next_flat = self.encoder(next_obs_flat)

        local_reprs      = local_reprs_flat.view(N, self.B, -1)
        local_reprs_next = local_reprs_next_flat.view(N, self.B, -1)

        fired_all = torch.ones(self.B, dtype=torch.bool, device=device)

        coord_latents      = []
        coord_latents_next = []
        with torch.no_grad():
            for n in range(N):
                cl = self.coordinator(
                    local_reprs[n], fired_all, self.cluster, None
                )
                coord_latents.append(cl)
                cl_next = self.coordinator(
                    local_reprs_next[n], fired_all, self.cluster, None
                )
                coord_latents_next.append(cl_next)

        cluster_idx = torch.tensor(
            self.cluster.labels, dtype=torch.long, device=device
        )

        feat_dim = self._feat_dim
        feats_all      = torch.zeros(N, self.B, feat_dim, device=device)
        feats_next_all = torch.zeros(N, self.B, feat_dim, device=device)
        for n in range(N):
            assigned      = coord_latents[n][cluster_idx]
            assigned_next = coord_latents_next[n][cluster_idx]
            feats_all[n]      = torch.cat([local_reprs[n], assigned],      dim=-1)
            feats_next_all[n] = torch.cat([local_reprs_next[n], assigned_next], dim=-1)

        feats_det      = feats_all.detach()
        feats_next_det = feats_next_all.detach()

        for b in range(self.B):
            feat_b      = feats_det[:, b, :]
            feat_next_b = feats_next_det[:, b, :]
            act_b       = actions_t[:, b, :]
            rew_b       = rewards_t[:, b]

            with torch.no_grad():
                next_a_b, next_lp_b = self.actors[b].sample(feat_next_b)
                q1_next = self.q1_targets[b](feat_next_b, next_a_b)
                q2_next = self.q2_targets[b](feat_next_b, next_a_b)
                q_next  = torch.min(q1_next, q2_next) - alpha * next_lp_b
                q_tgt   = rew_b + self.gamma * (1 - dones_t) * q_next

            q1_pred = self.q1_nets[b](feat_b, act_b)
            q2_pred = self.q2_nets[b](feat_b, act_b)
            q_loss  = F.mse_loss(q1_pred, q_tgt) + F.mse_loss(q2_pred, q_tgt)

            self.q_opts[b].zero_grad()
            q_loss.backward()
            self.q_opts[b].step()
            losses["q_loss"] += q_loss.item()

            with torch.no_grad():
                cost_next = self.cost_critics[b](feat_next_b, next_a_b)
                soc_b  = torch.tensor(
                    [batch["obs"][n][b][_IDX_SOC_ELEC] for n in range(N)],
                    dtype=torch.float32, device=device,
                )
                net_b  = torch.tensor(
                    [batch["obs"][n][b][_IDX_NET] for n in range(N)],
                    dtype=torch.float32, device=device,
                )
                delta_soc = act_b[:, 1] * 0.1
                new_soc   = soc_b + delta_soc
                c_soc  = ((new_soc < self._cbf_cfg.SOC_min).float()
                          + (new_soc > self._cbf_cfg.SOC_max).float()).clamp(0, 1)
                c_build = (torch.abs(net_b) > self._cbf_cfg.P_building_max).float()
                c_grid  = (net_b.clamp(min=0) > self._cbf_cfg.P_grid_max / self.B).float()
                cost_labels = torch.stack([c_soc, c_build, c_grid], dim=-1)
                cost_tgt    = cost_labels + self.gamma * (1 - dones_t.unsqueeze(1)) * cost_next

            cost_pred = self.cost_critics[b](feat_b, act_b)
            cost_loss = F.mse_loss(cost_pred, cost_tgt)
            self.cost_opts[b].zero_grad()
            cost_loss.backward()
            self.cost_opts[b].step()
            losses["cost_loss"] += cost_loss.item()

        local_reprs_a_flat = self.encoder(obs_flat)
        local_reprs_a      = local_reprs_a_flat.view(N, self.B, -1)

        fired_all_a = torch.ones(self.B, dtype=torch.bool, device=device)
        coord_latents_a = []
        for n in range(N):
            cl = self.coordinator(local_reprs_a[n], fired_all_a, self.cluster, None)
            coord_latents_a.append(cl)

        feats_actor = torch.zeros(N, self.B, self._feat_dim, device=device)
        for n in range(N):
            assigned = coord_latents_a[n][cluster_idx]
            feats_actor[n] = torch.cat([local_reprs_a[n], assigned], dim=-1)

        total_actor_loss = torch.tensor(0.0, device=device)
        for b in range(self.B):
            feat_b_a = feats_actor[:, b, :]
            a_new, lp_new = self.actors[b].sample(feat_b_a)
            with torch.no_grad():
                q1_val = self.q1_nets[b](feats_det[:, b, :], a_new.detach())
                q2_val = self.q2_nets[b](feats_det[:, b, :], a_new.detach())
                cost_val = self.cost_critics[b](feats_det[:, b, :], a_new.detach())
                safety_penalty = (lambdas.unsqueeze(0) * cost_val).sum(dim=-1)
            q1_new = self.q1_nets[b](feats_det[:, b, :], a_new)
            q2_new = self.q2_nets[b](feats_det[:, b, :], a_new)
            q_min  = torch.min(q1_new, q2_new)
            actor_loss_b = (alpha * lp_new - q_min + safety_penalty).mean()
            total_actor_loss = total_actor_loss + actor_loss_b
            losses["actor_loss"] += actor_loss_b.item()

        self.shared_opt.zero_grad()
        for opt in self.actor_opts:
            opt.zero_grad()
        total_actor_loss.backward()
        self.shared_opt.step()
        for opt in self.actor_opts:
            opt.step()

        for b in range(self.B):
            for p, tp in zip(self.q1_nets[b].parameters(), self.q1_targets[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.q2_nets[b].parameters(), self.q2_targets[b].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        lp_sum = torch.tensor(0.0, device=device)
        for b in range(self.B):
            with torch.no_grad():
                _, lp_b = self.actors[b].sample(feats_det[:, b, :])
            lp_sum = lp_sum + lp_b.mean()
        alpha_loss = -(self.log_alpha * (lp_sum / self.B + self.target_entropy).detach())
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        lambda_loss = torch.tensor(0.0, device=device)
        for b in range(self.B):
            feat_b = feats_det[:, b, :]
            act_b  = actions_t[:, b, :]
            with torch.no_grad():
                c_pred = self.cost_critics[b](feat_b, act_b)
            constraint_violation = c_pred.mean(dim=0) - self._lag_cfg.cost_limit
            lambda_loss = lambda_loss - (self.log_lambdas * constraint_violation.detach()).sum()

        self.lambda_opt.zero_grad()
        lambda_loss.backward()
        self.lambda_opt.step()
        log_max = math.log(self._lag_cfg.lambda_max)
        self.log_lambdas.data.clamp_(min=-10.0, max=log_max)
        losses["lambda_loss"] = lambda_loss.item()

        losses["actor_loss"] /= self.B
        losses["q_loss"]     /= self.B
        losses["cost_loss"]  /= self.B
        return losses


    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(),     os.path.join(path, "hier_encoder.pt"))
        torch.save(self.coordinator.state_dict(), os.path.join(path, "hier_coordinator.pt"))
        torch.save(self.actors.state_dict(),      os.path.join(path, "hier_actors.pt"))
        torch.save(self.q1_nets.state_dict(),     os.path.join(path, "hier_q1.pt"))
        torch.save(self.q2_nets.state_dict(),     os.path.join(path, "hier_q2.pt"))
        torch.save(self.cost_critics.state_dict(),os.path.join(path, "hier_cost.pt"))
        torch.save(self.log_lambdas.data,         os.path.join(path, "hier_lambdas.pt"))
        torch.save(self.log_alpha.data,           os.path.join(path, "hier_log_alpha.pt"))
        np.save(os.path.join(path, "cluster_labels.npy"), self.cluster.labels)

    def load(self, path: str) -> None:
        map_loc = self.device
        self.encoder.load_state_dict(
            torch.load(os.path.join(path, "hier_encoder.pt"), map_location=map_loc))
        self.coordinator.load_state_dict(
            torch.load(os.path.join(path, "hier_coordinator.pt"), map_location=map_loc))
        self.actors.load_state_dict(
            torch.load(os.path.join(path, "hier_actors.pt"), map_location=map_loc))
        self.q1_nets.load_state_dict(
            torch.load(os.path.join(path, "hier_q1.pt"), map_location=map_loc))
        self.q2_nets.load_state_dict(
            torch.load(os.path.join(path, "hier_q2.pt"), map_location=map_loc))
        self.cost_critics.load_state_dict(
            torch.load(os.path.join(path, "hier_cost.pt"), map_location=map_loc))
        self.log_lambdas.data.copy_(
            torch.load(os.path.join(path, "hier_lambdas.pt"), map_location=map_loc))
        self.log_alpha.data.copy_(
            torch.load(os.path.join(path, "hier_log_alpha.pt"), map_location=map_loc))
