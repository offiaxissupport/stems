"""STEMS hyperparameter configuration (paper Section III-A4).

Each sub-system has its own dataclass; ``STEMSConfig`` aggregates them. Field
names map to the paper's symbols where possible and carry an equation reference.

Beyond the paper, this configuration also exposes the four constraint-violation
"tricks" we ablate (see ``SafetyConfig`` and ``LagrangianConfig.use_pid``) and a
``HeatPumpConfig`` for bidirectional heat-pump operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """Adaptive building similarity graph (Eq. 10-11)."""
    alpha: float = 0.5       # weight on geographic proximity
    beta: float = 0.5        # weight on functional similarity
    sigma_d: float = 1.0     # distance-kernel bandwidth
    sigma_f: float = 1.0     # feature-kernel bandwidth


@dataclass
class GCNConfig:
    """Spatial Graph Convolutional Network (Eq. 12)."""
    num_layers: int = 3
    hidden_dim: int = 64


@dataclass
class TransformerConfig:
    """Temporal Transformer (Eq. 13-14)."""
    num_heads: int = 4
    embed_dim: int = 32
    window_size: int = 24    # T = 24h history window


@dataclass
class FusionConfig:
    """Spatial-temporal fusion (Eq. 15)."""
    output_dim: int = 64


@dataclass
class ActorCriticConfig:
    """Actor / Critic networks (Eq. 22-26)."""
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    # False: one actor/critic per building (paper). True: one set shared by all.
    share_parameters: bool = False


@dataclass
class RewardConfig:
    """Four-part reward (Eq. 3-9)."""
    mu: float = 1.0                  # economic weight
    alpha_grid: float = 0.5          # grid-stability weight
    alpha_build: float = 0.3         # building-stability weight
    beta_ramp: float = 0.2           # ramping penalty weight
    lambda_indoor: float = 0.4       # comfort weight
    # Renewable utilisation (Eq. 9) depends only on exogenous solar and load, so
    # it cannot change any decision; off by default (paper value 0.6).
    xi: float = 0.0
    T_ref: float = 22.0              # fallback reference indoor temperature (degC)
    T_comfort_threshold: float = 2.0 # comfort band half-width (degC)

    # Electric-vehicle service. Without this term the reward contains no
    # incentive to charge at all -- charging only ever costs money through
    # ``mu`` -- so the optimal policy is to never charge and the learning
    # problem is degenerate. The penalty is the normalised energy a vehicle is
    # short of its requirement at departure, following the standard
    # soft-deadline formulation, plus a small shaping term on the running
    # shortfall so the signal is not delivered only once per trip.
    # Weight derived, not guessed. Filling a 0.5 state-of-charge gap on a 60 kWh
    # battery draws ~30 kWh, which at ~0.25 $/kWh costs ~7.5 through the economic
    # term. A departure penalty of ev_service * shortfall must exceed that or the
    # cost-minimising policy is simply never to charge -- which is exactly what
    # was measured at ev_service = 2.0 (100% of departures missed). Requiring
    # ev_service * 0.5 > 7.5 gives ev_service > 15; 25 leaves margin for cheaper
    # hours and larger batteries.
    ev_service: float = 25.0         # weight on unmet energy at departure
    ev_shaping: float = 0.5          # weight on the per-step remaining shortfall


@dataclass
class CBFConfig:
    """Control Barrier Function safety bounds (Eq. 16-18).

    These are the *reported* hard limits. The CBF can additionally enforce
    tighter *robust* limits via ``SafetyConfig`` margins (Trick 4) so that the
    enforced safe set sits strictly inside the reported one.

    Note on the Travis-County dataset: per-building net load is ~10-40 kW and
    total grid imports peak ~40 kW, so the 80/300 kW power limits essentially
    never bind -- the battery SOC bound is the dominant (and, in the paper,
    the headline) constraint. The power limits are kept at the paper values for
    fidelity and remain meaningful for heat-pump / cold-snap stress tests.
    """
    SOC_min: float = 0.1             # minimum battery state-of-charge
    SOC_max: float = 0.9             # maximum battery state-of-charge
    P_grid_max: float = 300.0        # max total grid import (kW)
    P_building_max: float = 80.0     # max per-building net draw (kW)
    gamma_cbf: float = 1.0           # CBF relaxation rate in [0,1]: how fast a
                                     # barrier may approach its bound each step


@dataclass
class SafetyConfig:
    """Switches and parameters for the constraint-violation tricks.

    Trick 1 (feasibility-guaranteed CBF) and the per-building real battery
    dynamics are always on -- they are correctness fixes, not optional. The
    remaining tricks are flag-gated for ablation.
    """
    # Trick 1: feasibility-guaranteed projection (always returns the least-
    # violating action + recovery toward the safe set; never emergency-zeros).
    feasibility_qp: bool = True
    # Trick 2: anticipatory / control-invariance margin -- keep SOC far enough
    # from the bound that a safe action still exists next step.
    anticipatory: bool = True
    invariance_horizon: int = 1      # steps of recoverability to guarantee
    # Trick 4: robust margins -- shrink the enforced SOC band and derate the
    # power caps relative to the reported CBFConfig limits.
    robust_margins: bool = True
    soc_margin: float = 0.03         # enforced SOC band = [SOC_min+m, SOC_max-m]
    # Always applied, whatever the switches above: the state of charge is
    # observed in float32 and inverted by bisection, so the barrier aims this
    # far inside the band rather than exactly at its edge.
    soc_tolerance: float = 1e-3
    power_derate: float = 0.05       # enforced power cap = (1-derate) * reported


@dataclass
class TrainingConfig:
    """Training loop (Algorithm 2)."""
    episodes: int = 50
    batch_size: int = 512
    buffer_capacity: int = 100_000
    exploration_noise: float = 0.1
    action_scale: float = 1.0        # paper uses the full [-1,1] action range

    # --- PPO-Lagrangian update (Schulman et al. 2017; Ray et al. 2019) -----
    # Each episode's trajectory is replayed for ``update_epochs`` passes in
    # minibatches of ``minibatch_size`` timesteps (x B buildings), with a clipped
    # importance ratio against the behaviour policy's log-probability recorded
    # at sampling time. A 4-week window (671 transitions) gives
    # ceil(671/64) x 10 = 110 gradient steps per episode.
    update_epochs: int = 10          # passes over each episode's trajectory
    minibatch_size: int = 64         # timesteps per gradient step
    ppo_clip: float = 0.2            # trust region on the importance ratio
    gae_lambda: float = 0.95         # GAE bias/variance trade-off
    value_coef: float = 0.5          # weight of the (reward and cost) value losses
    max_grad_norm: float = 0.5       # global gradient-norm clip
    target_kl: float = 0.02          # stop the epochs early past 1.5 x this KL
    # Fixed entropy bonus on the Gaussian policy. The previous SAC-style
    # auto-tuned temperature started at 1.0 and, with Adam at 3e-4 over ~160
    # updates, could not move (alpha stayed 0.95-1.05): a bonus of weight ~1
    # against unit-scale advantages, which pulled every action toward 0. On a
    # known-answer bandit it reached 0.27 for a target of 0.6; at 0.01 it reached
    # 0.61. 0.01 is the standard PPO value.
    entropy_coef: float = 0.01
    # Rewards are divided by a running SD of the discounted return, so value
    # targets are O(1) whatever the reward units (Engstrom et al. 2020).
    scale_rewards: bool = True

    # Which action the actor's log-probability is evaluated at:
    #   "raw"  -- the policy's own sample, with the shield treated as part of the
    #             environment. The importance ratio is between two densities of
    #             the same random variable, so the estimator is consistent.
    #   "safe" -- the post-shield executed action (paper Eq. 24). The shield maps
    #             many samples onto one point (e.g. a deadline barrier emits
    #             exactly 0 whenever charging is not urgent), so the executed
    #             action has no density under the policy and the "ratio" is not
    #             an importance weight: training becomes advantage-weighted
    #             regression onto the shield's output. Kept only to reproduce
    #             the paper's formulation.
    actor_target: str = "raw"

    # Residual policy (Silver et al. 2018). When the agent is given a base
    # controller, the policy's action a in (-1, 1) is a correction on it:
    # nominal = clip(base(obs) + residual_scale * a). A fresh actor outputs a
    # mean near zero, so the untrained deterministic policy IS the base
    # controller and learning starts from its performance instead of from
    # nothing. ``residual_log_std`` is the initial exploration scale of such an
    # actor: sigma = exp(-1.2) = 0.3, small enough that early episodes stay close
    # to the base controller.
    residual_scale: float = 0.5
    residual_log_std: float = -1.2

    # Penalty on shield interventions (Krasowski et al. 2023; Markgraf et al.
    # 2025): w * ||nominal - executed||^2 is subtracted from the reward of the
    # building whose action the shield changed, so a policy cannot rely on the
    # shield for free. 0 switches it off.
    intervention_penalty: float = 0.0

    # Reward lost per kWh of vehicle charging the cap shield had to *force* (its
    # deadline rescue: charging above what the policy asked for). Without it a
    # policy behind the shield pays nothing for never charging a car -- the
    # shield charges it at the last feasible moment, which is also the cheapest
    # -- and that is what the plain policy was measured to do (0.2-3% of the
    # energy requested). Cuts for the cap are not penalised: asking early for
    # more than fits does not put a deadline at risk. 0 switches it off.
    forced_charge_penalty: float = 0.0


@dataclass
class LagrangianConfig:
    """Multi-constraint Lagrangian safety (one lambda_k per CBF constraint).

    Constraints k: 0 = SOC bounds (h1), 1 = per-building power (h2),
    2 = total grid power (h3). Each lambda_k is driven so that the predicted
    violation rate J_c_k stays below ``cost_limit``.

    With ``use_pid`` (Trick 3) the dual update is a PID controller on the
    constraint-cost error e_k = J_c_k - cost_limit, which is markedly more
    stable than plain integral-only dual ascent.
    """
    num_constraints: int = 3
    cost_limit: float = 0.05         # max allowed per-constraint violation rate
    lambda_init: float = 0.1         # non-zero so constraints are respected early
    lambda_max: float = 10.0         # anti-runaway cap

    # Plain dual ascent (used when use_pid=False)
    lambda_lr: float = 0.005

    # Trick 3: PID-Lagrangian gains (used when use_pid=True)
    use_pid: bool = True
    # Gains act on the per-episode violation rate (a number in [0, 1]), not an
    # episode cost sum as in Stooke et al., so they are correspondingly larger.
    # The reward advantage is standardised and the cost advantage only centred
    # (Ray et al. 2019), so the cost dominates once lambda * SD(A_c) > 1. On a
    # known-answer problem (reward pushes an action up, a constraint forbids it,
    # limit 5%): the previous gains (0.05/0.005) never exceeded lambda 0.24;
    # 1.0/0.3 took the violation rate 0.52 -> 0.03 in 30 episodes with lambda
    # settling at ~2.3; 0.5/0.5 overshot to 3.3.
    pid_kp: float = 1.0              # proportional: react to current violation
    pid_ki: float = 0.3              # integral: plain dual ascent
    pid_kd: float = 0.0              # derivative: off (episode costs are noisy)


@dataclass
class HeatPumpConfig:
    """Bidirectional heat-pump parameters (Phase 3).

    In real CityLearn ``action[2]`` (cooling_or_heating_device) is already a
    bidirectional HeatPump, so no extra action is needed. These values are used
    for the CoP-aware CBF power factor and for season-aware comfort.
    """
    enabled: bool = False
    cop_heating_mild: float = 3.5    # CoP at mild outdoor temperature
    cop_heating_cold: float = 2.0    # CoP in a deep cold snap (conservative)
    cop_cooling: float = 4.0
    cold_snap_temp: float = 0.0      # degC below which CoP_cold is used
    rated_power_kw: float = 5.0
    heating_setpoint: float = 20.0   # comfort reference when heating
    cooling_setpoint: float = 22.0   # comfort reference when cooling


@dataclass
class ThermalConfig:
    """Anticipatory DHW pre-heating and weather-aware thermal safety (Phase 4).

    Motivation. The DHW tank cannot be filled instantly: CityLearn charges it by
    ``energy = action * capacity`` but caps that by the heater's one-hour output,
    so the achievable SOC gain per step is
    ``charge_rate = min(action_bound, P_nom * eta / C)`` -- measured 0.56-0.85
    on the Travis buildings, i.e. a **time-to-heat of 1.2-1.8 hours from empty**.
    Water therefore has to be heated *before* it is needed, which is what the
    readiness barrier below enforces.

    Weather enters twice. (i) The space-conditioning heat pump's CoP falls as the
    outdoor temperature moves away from its supply temperature, so the same
    thermal output costs more electrical power on a cold hour -- folded into the
    power barriers via ``cop_aware_power``. (ii) A drop in the forecast outdoor
    temperature raises upcoming thermal demand while lowering CoP, so the
    readiness requirement is inflated ahead of a cold front
    (``weather_anticipation``).
    """
    # --- Barrier h4: DHW readiness -------------------------------------
    dhw_readiness: bool = True
    preheat_horizon: int = 2         # L: hours of forecast demand the tank must cover
    dhw_margin: float = 0.05         # extra SOC held above the forecast requirement
    dhw_soc_cap: float = 0.95        # never demand more than this (tank headroom)

    # --- Demand forecaster ---------------------------------------------
    forecast_alpha: float = 0.2      # EWMA rate for the hour-of-day climatology
    forecast_warmup: int = 24        # steps before the climatology is trusted
    forecast_temp_gain: float = 0.02 # per-degC uplift of demand as T_out falls

    # --- Weather anticipation ------------------------------------------
    weather_anticipation: bool = True
    weather_gain: float = 0.15       # max extra SOC required ahead of a cold front
    weather_horizon: int = 3         # forecast steps available in the observation

    # --- CoP-aware power barrier ---------------------------------------
    cop_aware_power: bool = True


@dataclass
class STEMSConfig:
    """Top-level configuration aggregating every sub-config."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    gcn: GCNConfig = field(default_factory=GCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cbf: CBFConfig = field(default_factory=CBFConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lagrangian: LagrangianConfig = field(default_factory=LagrangianConfig)
    heat_pump: HeatPumpConfig = field(default_factory=HeatPumpConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)

    def tricks_active(self) -> List[str]:
        """Return the list of active constraint-violation tricks (for metadata)."""
        active = ["feasibility_qp", "real_battery_dynamics"]
        if self.safety.anticipatory:
            active.append("anticipatory")
        if self.safety.robust_margins:
            active.append("robust_margins")
        if self.lagrangian.use_pid:
            active.append("pid_lagrangian")
        if self.thermal.dhw_readiness:
            active.append(f"dhw_readiness(L={self.thermal.preheat_horizon})")
        if self.thermal.weather_anticipation:
            active.append("weather_anticipation")
        if self.thermal.cop_aware_power:
            active.append("cop_aware_power")
        return active
