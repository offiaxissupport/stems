# STEMS: Spatial-Temporal Enhanced Multi-Agent Safe Building Energy Management System

Faithful replication of:

> Zhang, X., Wu, J., Zinflou, A., & Boulet, B. (2025). **STEMS: Spatial-Temporal Enhanced Multi-Agent Safe Building Energy Management System.** *IEEE Internet of Things Journal*. [arXiv:2510.14112v2](https://arxiv.org/abs/2510.14112)

---

## Overview

STEMS is a safe multi-agent reinforcement learning framework for building energy management that combines:

- **Spatial GCN** – captures inter-building coordination via adaptive similarity graph (Eq 10-12)
- **Temporal Transformer** – learns diurnal energy patterns over a T=24h window (Eq 13-14)
- **ST Fusion** – merges spatial and temporal representations (Eq 15)
- **CBF Safety Shield** – guarantees battery SOC and power constraints via QP projection (Eq 16-20, Algorithm 1)
- **4-Part Reward** – economic + stability + comfort + renewable components (Eq 3-9)

---

## Repository Structure

```
stems/
├── stems/                  # Python package
│   ├── __init__.py
│   ├── config.py           # All hyperparameters (Section III-A4)
│   ├── environment.py      # CityLearn wrapper + realistic mock
│   ├── graph.py            # Building similarity graph (Eq 10-11)
│   ├── encoder.py          # SpatialGCN + TemporalTransformer + STEncoder (Eq 12-15)
│   ├── cbf.py              # CBF Safety Shield (Eq 16-20, Algorithm 1)
│   ├── reward.py           # 4-part reward function (Eq 3-9)
│   ├── agent.py            # Actor, Critic, STEMSAgent (Eq 22-26, Algorithm 2)
│   ├── baselines.py        # RuleBased, SingleAgentSAC, DMAPPOAgent
│   ├── metrics.py          # MetricsCalculator (Table I metrics)
│   └── utils.py            # ReplayBuffer, HistoryBuffer, helpers
├── train.py                # Main training script (Algorithm 2)
├── evaluate.py             # Evaluation + Table I/II
├── visualize.py            # All paper figures (Figs 2-7)
├── ablation.py             # Ablation study (Table IV)
├── notebook.ipynb          # Complete end-to-end Jupyter notebook
├── requirements.txt        # Python dependencies
└── setup_citylearn.sh      # CityLearn installation script
```

---

## Setup

### Option A: CityLearn from Source (Recommended)

```bash
pip install -r requirements.txt
bash setup_citylearn.sh
```

### Option B: Core Dependencies Only (uses mock environment)

```bash
pip install -r requirements.txt
```

The mock environment provides 3 buildings with 28-dimensional observations and realistic building physics. All code runs without CityLearn installed.

---

## Quick Start

### Train STEMS (15 episodes, ~10 min on CPU)

```bash
python train.py --episodes 15 --save-dir checkpoints/
```

### Evaluate All Agents and Print Table I

```bash
python evaluate.py --checkpoint checkpoints/
```

### Generate All Paper Figures

```bash
python visualize.py --checkpoint checkpoints/ --output-dir plots/
```

### Run Ablation Study (Table IV)

```bash
python ablation.py --episodes 5
```

### Interactive Notebook

```bash
jupyter notebook notebook.ipynb
```

---

## Expected Results (Table I)

Metrics are normalised against the rule-based baseline (lower is better for columns 1-5):

| Agent | Cost | Emission | DayPeak | Consumption | Ramping | Discomfort% | SafetyVio% |
|-------|------|----------|---------|-------------|---------|-------------|------------|
| **STEMS** | **0.82** | **0.79** | **0.85** | **0.88** | **0.80** | **3.4** | **0.1** |
| RuleBased | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 8.9 | 5.2 |
| SingleSAC | 0.93 | 0.91 | 0.94 | 0.95 | 0.92 | 6.3 | 3.8 |
| DMAPPO | 0.89 | 0.87 | 0.91 | 0.92 | 0.88 | 4.9 | 2.1 |

> **Note:** These are reference values from the paper. Results with the mock environment and 3 buildings may differ from the paper's 8-building CityLearn results. Run `setup_citylearn.sh` and use `--episodes 15` for closest reproduction.

---

## Architecture Details

### Hyperparameters (Section III-A4)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Graph | α, β | 0.5, 0.5 |
| Graph | σ_d, σ_f | 1.0, 1.0 |
| GCN | layers, hidden | 3, 64 |
| Transformer | heads, embed_dim, window | 4, 32, 24 |
| Actor/Critic | hidden, lr | 128, 3×10⁻⁴ |
| Training | episodes, batch | 15, 512 |
| CBF | SOC_min, SOC_max | 0.1, 0.9 |

### Key Equations

**Graph weights (Eq 11):**
$$w_{ij} = \alpha \exp\!\left(-\frac{d_{ij}^2}{2\sigma_d^2}\right) + \beta \exp\!\left(-\frac{\|f_i - f_j\|^2}{2\sigma_f^2}\right)$$

**ST Fusion (Eq 15):**
$$r_i = W_s h_i + W_t z_i + b$$

**CBF constraint (Eq 16):**
$$\dot{h}(s, a) + \gamma_{\text{cbf}} h(s) \geq 0$$

---

## Notes

- **3 vs 8 buildings:** The mock environment uses 3 buildings for fast local development. The paper evaluates on 8 buildings from CityLearn 2023 Phase 2. Install CityLearn via `setup_citylearn.sh` for the full 8-building scenario.
- **torch_geometric:** Not required. `FallbackGCNConv` provides an equivalent pure-PyTorch GCN implementation.
- **cvxpy:** Optional. Falls back to analytical SOC clipping if unavailable.

---

## License

MIT License. See paper for additional citation requirements.

