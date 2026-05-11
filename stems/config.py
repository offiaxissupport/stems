"""
STEMS hyperparameter configuration (Section III-A4 of the paper).

All hyperparameters are collected in a single dataclass so they can be
overridden in a consistent way throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """Parameters for adaptive building similarity graph (Eq 10-11)."""
    alpha: float = 0.5         # weight for geographic proximity
    beta: float = 0.5          # weight for functional similarity
    sigma_d: float = 1.0       # bandwidth for distance kernel
    sigma_f: float = 1.0       # bandwidth for feature kernel


@dataclass
class GCNConfig:
    """Graph Convolutional Network architecture parameters (Eq 12)."""
    num_layers: int = 3
    hidden_dim: int = 64


@dataclass
class TransformerConfig:
    """Temporal Transformer parameters (Eq 13-14)."""
    num_heads: int = 4
    embed_dim: int = 32
    window_size: int = 24      # T = 24 hours of history


@dataclass
class FusionConfig:
    """Spatial-Temporal fusion layer parameters (Eq 15)."""
    output_dim: int = 64


@dataclass
class ActorCriticConfig:
    """Actor/Critic network parameters (Eq 22-26)."""
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99


@dataclass
class RewardConfig:
    """Reward function parameters (Eq 3-9)."""
    mu: float = 1.0                    # economic weight
    alpha_grid: float = 0.5            # grid stability weight
    alpha_build: float = 0.3           # building stability weight
    beta_ramp: float = 0.2             # ramping penalty weight
    lambda_indoor: float = 0.4         # comfort weight
    xi: float = 0.6                    # renewable utilisation weight
    T_ref: float = 22.0                # reference indoor temperature (°C)
    T_comfort_threshold: float = 2.0   # comfort band half-width (°C)


@dataclass
class CBFConfig:
    """Control Barrier Function safety shield parameters (Eq 16-20)."""
    SOC_min: float = 0.1               # minimum battery state-of-charge
    SOC_max: float = 0.9               # maximum battery state-of-charge
    P_grid_max: float = 300.0          # maximum total grid power (kW)
    P_building_max: float = 80.0       # maximum per-building grid draw (kW)
    gamma_cbf: float = 0.1             # CBF decay rate


@dataclass
class TrainingConfig:
    """Training loop parameters (Algorithm 2)."""
    episodes: int = 50
    batch_size: int = 512
    buffer_capacity: int = 100_000
    exploration_noise: float = 0.1
    action_scale: float = 1.0          # paper uses full [-1,1] action space; no scaling


@dataclass
class LagrangianConfig:
    """Multi-constraint Lagrangian multiplier parameters.

    Three safety constraints (matching CBF Eq 16-18):
        k=0  SOC bounds             (h1)
        k=1  per-building power     (h2)
        k=2  total grid power       (h3)

    Each constraint has an independent λ_k updated by gradient ascent on
    the Lagrangian dual.  The cost_limit is the maximum fraction of timesteps
    that may violate each constraint (the 'd' in J_c_k ≤ d).
    """
    num_constraints: int = 3           # fixed: one per CBF constraint
    cost_limit: float = 0.05           # max allowed violation rate (5 % of steps)
    lambda_lr: float = 0.005           # Lagrangian multiplier learning rate (stable dual)
    lambda_init: float = 0.1           # initial λ value — non-zero so constraints respected early
    lambda_max: float = 1.0            # cap on λ; 0.05 was too small to penalize violations


@dataclass
class STEMSConfig:
    """Top-level configuration aggregating all sub-configs."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    gcn: GCNConfig = field(default_factory=GCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cbf: CBFConfig = field(default_factory=CBFConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lagrangian: LagrangianConfig = field(default_factory=LagrangianConfig)
