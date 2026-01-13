# Improving Self-Play for No-Press Diplomacy
## A Hybrid Approach Combining Human Imitation, Reinforcement Learning, and Population-Based Training

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Diplomacy_board.svg/800px-Diplomacy_board.svg.png" alt="Diplomacy Board" width="400"/>
</p>

<p align="center">
  <strong>Intelligent Systems Project</strong><br>
  Universitat Politècnica de Catalunya (UPC Barcelona)<br>
  Fall Semester 2025/26
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#research-questions">Research Questions</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#repository-structure">Repository Structure</a> •
  <a href="#results">Results</a> •
  <a href="#references">References</a>
</p>

---

## Authors

| Name | Role | Contact |
|------|------|---------|
| **Giacomo Colosio** | Lead Developer, RL Implementation | giacomo.colosio@estudiantat.upc.edu |
| **Maciej Tasarz** | Data Integration, Analysis | maciej.tasarz@estudiantat.upc.edu |
| **Jakub Seliga** | Behavioral Cloning, Evaluation | jakub.seliga@estudiantat.upc.edu |
| **Luka Ivcevic** | Population-Based Training | luka.ivcevic@estudiantat.upc.edu |

**Supervisor:** Prof. Ulises Cortes, Department of Computer Science, UPC Barcelona

---

## Abstract

Multi-agent reinforcement learning in complex strategic environments remains a fundamental challenge in artificial intelligence. **Diplomacy**, a seven-player game of negotiation and strategy, represents one of the most demanding testbeds for AI research due to its combinatorial action space (~10²⁰ possible actions per turn), need for long-term planning, and multi-agent dynamics with simultaneous moves.

This project investigates **hybrid training approaches** for No-Press Diplomacy (the non-communication variant), combining:

1. **Behavioral Cloning (BC)** from 33,279 human expert games
2. **Self-Play Reinforcement Learning** with PPO
3. **Human-Regularized RL (DiL-πKL)** to prevent strategy collapse
4. **Population-Based Training (PBT)** for robust generalization

Our experiments demonstrate that pure self-play leads to strategy collapse (89.6% draw rate), while combining human regularization with population diversity achieves **39.9% average win rate**—a **15× improvement** over random baseline.

---

## Research Questions

This project addresses four fundamental research questions in multi-agent reinforcement learning:

| ID | Research Question | Method | Key Finding |
|----|-------------------|--------|-------------|
| **RQ1** | Does pure self-play lead to strategy collapse? | Self-Play RL Analysis | ✓ Yes: 89.6% draw rate |
| **RQ2** | Can human data prevent strategy collapse? | Human-Regularized RL (DiL-πKL) | ✓ Yes: 48.6% draw rate |
| **RQ3** | Does opponent diversity improve robustness? | Population-Based Training | ✓ Yes: 45.2% vs Random |
| **RQ4** | What is each component's contribution? | Ablation Study | KL reg. +11.8% (most impactful) |

---

## Background

### The Diplomacy Challenge

Diplomacy presents unique challenges for AI systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WHY DIPLOMACY IS HARD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🎯 MASSIVE ACTION SPACE        │  ⏱️ LONG-TERM PLANNING                    │
│  ~10²⁰ possible move            │  Games last 8-9 years avg                 │
│  combinations per turn          │  (~35 decision points)                    │
│                                 │                                           │
│  🤝 MULTI-AGENT DYNAMICS        │  🔄 SIMULTANEOUS MOVES                    │
│  7 players with competing       │  All players move at once                 │
│  and aligned interests          │  No sequential advantage                  │
│                                 │                                           │
│  📊 PARTIAL OBSERVABILITY       │  🎭 NON-TRANSITIVE STRATEGIES             │
│  Must infer opponent            │  Strategy A beats B, B beats C,           │
│  intentions from actions        │  C beats A (rock-paper-scissors)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prior Work

Our approach builds upon recent advances in Diplomacy AI:

| Paper | Key Contribution | Limitation Addressed |
|-------|-----------------|---------------------|
| **Paquette et al. (2019)** | First competitive No-Press agent using RL | Limited to imitation learning |
| **Bakhtin et al. (2021) - DORA** | Double Oracle RL for strategy diversity | Computational complexity |
| **Bakhtin et al. (2022) - Diplodocus** | Human-regularized RL + planning | Requires extensive compute |
| **Meta AI (2022) - Cicero** | Full-press Diplomacy with language | Focuses on communication |

**Our contribution:** A systematic study of hybrid training approaches accessible with limited computational resources (Google Colab free tier).

---

## Methodology

### Overview

Our training pipeline consists of four interconnected components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │   Human      │
     │   Games      │──────────┐
     │  (33,279)    │          │
     └──────────────┘          ▼
                        ┌─────────────────┐
                        │   BEHAVIORAL    │
                        │    CLONING      │───────────────────┐
                        │   (7.8% Top-5)  │                   │
                        └────────┬────────┘                   │
                                 │                            │
                                 │ Initialize                 │
                                 ▼                            │
     ┌──────────────┐    ┌─────────────────┐                  │
     │   Self vs    │◄───│   SELF-PLAY     │                  │
     │    Self      │    │   RL (RQ1)      │                  │
     │   500 games  │    │  89.6% draws    │                  │
     └──────────────┘    └────────┬────────┘                  │
                                 │                            │
                                 │ + KL Penalty               │ π_human
                                 ▼                            │
                        ┌─────────────────┐                   │
                        │ HUMAN-REGULARIZED│◄──────────────────┘
                        │   RL (RQ2)       │
                        │  48.6% draws     │
                        └────────┬────────┘
                                 │
                                 │ + Diverse Opponents
                                 ▼
                        ┌─────────────────┐
                        │  POPULATION-    │
                        │  BASED (RQ3)    │
                        │  45.2% vs Rand  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  FINAL AGENT    │
                        │  39.9% avg WR   │
                        │  (15× baseline) │
                        └─────────────────┘
```

### Component Details

#### 1. Data Integration & Exploratory Analysis

**Dataset:** 33,279 No-Press games from [webDiplomacy.net](https://github.com/diplomacy/research)

| Statistic | Value |
|-----------|-------|
| Total games | 33,279 |
| Games used (Colab limit) | 5,201 |
| Avg. phases/game | **34.7** |
| Avg. game years | **8.7** |
| Draw rate | **41.9%** |
| Top win rate | Austria (15.0%) |
| Lowest win rate | England (3.2%) |

**Order Type Distribution:**

| Order Type | Percentage | Count |
|------------|------------|-------|
| MOVE | **61.9%** | ~1.6M |
| HOLD | 16.1% | ~0.42M |
| SUPPORT | 12.5% | ~0.33M |
| BUILD | 6.4% | ~0.17M |
| DISBAND | 1.6% | ~0.04M |

#### 2. Behavioral Cloning

**Architecture:** MLP with 3,039,411 parameters

| Layer | Configuration | Output |
|-------|---------------|--------|
| Input | State vector | 1,216 |
| Hidden 1 | FC + LayerNorm + ReLU + Dropout(0.2) | 512 |
| Hidden 2 | FC + LayerNorm + ReLU + Dropout(0.2) | 512 |
| Hidden 3 | FC + LayerNorm + ReLU + Dropout(0.2) | 256 |
| Output | FC (logits) | 7,859 |

**Training Results:**

| Metric | Value |
|--------|-------|
| Training samples | 2,204,602 |
| Vocabulary size | 7,859 |
| Top-1 Accuracy | **1.93%** (150× over random) |
| Top-5 Accuracy | **7.80%** |
| Training time | ~85 minutes |

#### 3. Self-Play Reinforcement Learning (RQ1)

**Algorithm:** Proximal Policy Optimization (PPO)

```python
L_PPO = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)] - c₁·L_VF + c₂·H[π]
```

**Reward Shaping:**

| Event | Reward |
|-------|--------|
| Win (18+ SCs) | +10.0 |
| Gain 1 SC | +0.1 |
| Lose 1 SC | -0.1 |
| Survive 1 phase | +0.02 |
| Elimination | -1.0 |

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Training games | 500 |
| Max game length | 100 phases |
| Learning rate | 2.5 × 10⁻⁴ |
| PPO clip (ε) | 0.2 |
| GAE lambda (λ) | 0.95 |
| Training time | ~4 hours |

#### 4. Human-Regularized RL (RQ2)

**Algorithm:** DiL-πKL (Bakhtin et al., 2022)

```
L_DiL-πKL = L_PPO + β · D_KL(π_θ || π_human)
```

Where:
- `π_θ` — current RL policy (trainable)
- `π_human` — frozen BC policy
- `β = 0.1` — KL penalty coefficient

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Training games | 600 |
| KL coefficient (β) | 0.1 |
| Max KL threshold | 0.5 |
| Entropy coefficient | 0.02 |
| Training time | ~3.5 hours |

#### 5. Population-Based Training (RQ3)

**Opponent Population:**

| Agent Type | Base Weight | Purpose |
|------------|-------------|---------|
| Random | 0.15 | Baseline, prevents catastrophic failures |
| BC (Human-like) | 0.25 | Exposes to human strategies |
| Ckpt_200 | 0.15 | Early training strategies |
| Ckpt_400 | 0.15 | Mid training strategies |
| Ckpt_600 | 0.15 | Late training strategies |
| Ckpt_800 | 0.15 | Near-final strategies |

**Prioritized Fictitious Self-Play (PFSP):**

```
P(opponent_i) ∝ (1 - win_rate_i)^p,  p = 0.5
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Training games | 800 |
| Checkpoint interval | 200 games |
| Final population size | 6 agents |
| Training time | ~5 hours |

---

## Repository Structure

```
Improve_Self-Play_for_Diplomacy/
│
├── 📁 data/
│   └── standard_no_press.jsonl          # 33,279 human games
│
├── 📁 DataIntegration/
│   ├── data_integration_eda.ipynb       # Exploratory Data Analysis
│   └── src/
│       ├── data_loader.py               # Data loading utilities
│       └── eda.py                       # EDA functions
│
├── 📁 BehavioralCloning/
│   ├── bc_training.ipynb                # BC training notebook
│   └── models/
│       └── best_bc_model.pt             # Trained BC model
│
├── 📁 SelfPlay/
│   ├── self_play_diplomacy.ipynb        # Pure self-play (RQ1)
│   └── checkpoints/                     # Training checkpoints
│
├── 📁 HumanRegularizedRL/
│   ├── human_regularized_rl.ipynb       # DiL-πKL implementation (RQ2)
│   └── models/
│       └── hrrl_model.pt                # Trained HR-RL model
│
├── 📁 PopulationBasedTraining/
│   ├── population_based_training.ipynb  # PBT implementation (RQ3)
│   ├── pbt_agent.pt                     # Final PBT agent
│   └── pbt_history.json                 # Training history
│
├── 📁 Evaluation/
│   └── ablation_study.ipynb             # RQ4: Component analysis
│
├── 📁 docs/
│   ├── final_report.pdf                 # Project report
│   └── figures/                         # Generated figures
│       ├── selfplay_training_curves.png
│       ├── hrrl_training_curves.png
│       └── pbt_training_curves.png
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── LICENSE                              # MIT License
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Google Colab (alternative, free tier works)

### Setup

```bash
# Clone repository
git clone https://github.com/GiacomoCoworker/Improve_Self-Play_for_Diplomacy.git
cd Improve_Self-Play_for_Diplomacy

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from diplomacy import Game; print('✓ Diplomacy package installed')"
```

### Running Experiments

Each experiment is contained in a self-sufficient Jupyter notebook designed for Google Colab:

| Notebook | Description | Runtime | Hardware |
|----------|-------------|---------|----------|
| `DataIntegration/data_integration_eda.ipynb` | Data analysis & visualization | ~10 min | CPU |
| `BehavioralCloning/bc_training.ipynb` | Train BC policy on human data | ~85 min | T4 GPU |
| `SelfPlay/self_play_diplomacy.ipynb` | Pure self-play RL (RQ1) | ~4 hours | T4 GPU |
| `HumanRegularizedRL/human_regularized_rl.ipynb` | DiL-πKL training (RQ2) | ~3.5 hours | T4 GPU |
| `PopulationBasedTraining/population_based_training.ipynb` | PBT training (RQ3) | ~5 hours | T4 GPU |

**Quick Start (Google Colab):**

1. Open any notebook in Google Colab
2. Set runtime to GPU: `Runtime → Change runtime type → GPU`
3. Upload `standard_no_press.jsonl` when prompted (or mount Google Drive)
4. Run all cells: `Runtime → Run all`

---

## Results

### RQ1: Self-Play Analysis

**Finding:** Pure self-play leads to strategy collapse with 89.6% draw rate.

| Metric | Games 1-100 | Games 400-500 | Trend |
|--------|-------------|---------------|-------|
| Win Rate | 8.2% | **2.8%** | ↓ |
| Draw Rate | 62.0% | **89.6%** | ↑ |
| Policy Entropy | 2.84 | **1.23** | ↓ (collapse) |
| Avg. Game Length | 45.2 | 78.4 | ↑ |

**Cross-Play Performance:**

| Opponent | Win Rate |
|----------|----------|
| vs Self | 14.3% |
| vs Random | 18.5% |
| vs BC | **8.2%** (catastrophic) |

### RQ2: Human-Regularized RL

**Finding:** KL regularization prevents strategy collapse while enabling improvement.

| Metric | Pure Self-Play | HR-RL (DiL-πKL) | Improvement |
|--------|----------------|-----------------|-------------|
| Draw Rate | 89.6% | **48.6%** | -41.0% |
| Win Rate (vs self) | 2.8% | **15.2%** | +12.4% |
| Policy Entropy | 1.23 | **1.89** | +0.66 |
| Win vs BC | 8.2% | **28.4%** | +20.2% |
| Win vs Random | 18.5% | **42.1%** | +23.6% |

**KL Coefficient Sensitivity:**

| β | Final KL | Win Rate | Behavior |
|---|----------|----------|----------|
| 0.0 | 1.45 | 3.2% | Collapse |
| **0.1** | **0.15** | **15.2%** | **Optimal** |
| 0.5 | 0.03 | 8.6% | Over-regularized |

### RQ3: Population-Based Training

**Finding:** Diverse opponents improve generalization across all opponent types.

| Metric | Games 1-200 | Games 800 | Trend |
|--------|-------------|-----------|-------|
| Episode Reward | 0.18 | **0.78** | ↑ |
| Game Length | 68.4 | **52.6** | ↓ (wins faster) |
| Overall Win Rate | 8.5% | **18.6%** | ↑ |
| Draw Rate | 72.4% | **45.2%** | ↓ |

**Win Rate by Opponent:**

| Opponent Type | Win Rate |
|---------------|----------|
| vs Random | **45.2%** |
| vs BC | **32.8%** |
| vs Checkpoints | **48.6%** |

### RQ4: Ablation Study

**Finding:** KL regularization provides the largest individual contribution (+11.8%).

| Configuration | BC Init | KL Reg | Population | Avg Win Rate | Robustness |
|---------------|---------|--------|------------|--------------|------------|
| Random Baseline | ✗ | ✗ | ✗ | 2.6% | 0.12 |
| Pure Self-Play | ✗ | ✗ | ✗ | 9.8% | 0.32 |
| BC Only | ✓ | ✗ | ✗ | 17.2% | 0.48 |
| BC + Self-Play | ✓ | ✗ | ✗ | 23.6% | 0.56 |
| HR-RL (DiL-πKL) | ✓ | ✓ | ✗ | 35.4% | 0.72 |
| **Full Hybrid (PBT)** | ✓ | ✓ | ✓ | **39.9%** | **0.81** |

**Component Contributions:**

| Component | Win Rate Gain | Relative Improvement |
|-----------|---------------|---------------------|
| Self-Play (vs Random) | +7.2% | Baseline RL |
| BC Initialization | +7.4% | +75% over self-play |
| RL Fine-tuning | +6.4% | +37% over BC |
| **KL Regularization** | **+11.8%** | **+50% over BC+SP** |
| Population Diversity | +4.5% | +13% over HR-RL |

---

## Key Findings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUMMARY OF FINDINGS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ❌ RQ1: Pure self-play FAILS                                               │
│     → 89.6% draw rate (strategy collapse)                                   │
│     → 8.2% win rate vs BC (catastrophic overfitting)                        │
│                                                                              │
│  ✅ RQ2: Human regularization WORKS                                         │
│     → Reduces draw rate to 48.6% (-41%)                                     │
│     → Maintains entropy at 1.89 (vs 1.23 collapse)                          │
│     → 28.4% win rate vs BC (+20.2%)                                         │
│                                                                              │
│  ✅ RQ3: Population diversity IMPROVES robustness                           │
│     → 45.2% vs Random, 32.8% vs BC                                          │
│     → PFSP focuses training on challenging opponents                        │
│                                                                              │
│  📊 RQ4: Component contributions QUANTIFIED                                 │
│     → KL regularization: +11.8% (MOST IMPACTFUL)                            │
│     → BC initialization: +7.4%                                              │
│     → Population diversity: +4.5%                                           │
│     → Total: 39.9% avg win rate (15× over random)                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Computational Constraints

All experiments were conducted on Google Colab free tier:

| Resource | Our Setup | State-of-the-Art |
|----------|-----------|------------------|
| GPU | Tesla T4 (16GB) | TPU v3 pods |
| Training games | 500-800 | 100,000+ (DORA) |
| Training time | 4-5 hours | Days to weeks |
| Dataset used | 5,201 games (16%) | Full dataset |
| Parallel games | 1 | 1,000+ |

Despite operating at <1% of state-of-the-art scale, our results demonstrate the effectiveness of hybrid approaches and provide insights applicable to larger-scale training.

---

## Limitations & Future Work

### Current Limitations

- **Computational resources:** Limited training compared to Diplodocus (1M+ games)
- **Partial dataset:** Only 16% of available data used due to Colab memory
- **No search/planning:** Pure policy network without Monte Carlo Tree Search
- **No-press variant only:** Does not address negotiation in full Diplomacy

### Future Directions

1. **Full dataset training** with increased compute resources
2. **Integration with MCTS** for improved strategic depth
3. **Larger populations** with more diverse agent types
4. **Transfer to full-press Diplomacy** with language models
5. **Distributed training** for scaling experiments

---

## References

### Primary References

1. Bakhtin, A., et al. (2022). *Human-level play in the game of Diplomacy by combining language models with strategic reasoning.* Science, 378(6624).

2. Bakhtin, A., et al. (2021). *No-Press Diplomacy from Scratch.* NeurIPS 2021.

3. Paquette, P., et al. (2019). *No-Press Diplomacy: Modeling Multi-Agent Gameplay.* NeurIPS 2019.

4. Gray, J., et al. (2020). *Human-Level Performance in No-Press Diplomacy via Equilibrium Search.* ICLR 2021.

### Methodological References

5. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

6. Silver, D., et al. (2017). *Mastering the game of Go without human knowledge.* Nature, 550(7676).

7. Vinyals, O., et al. (2019). *Grandmaster level in StarCraft II using multi-agent reinforcement learning.* Nature, 575(7782).

### Dataset

8. WebDiplomacy Research Dataset. https://github.com/diplomacy/research

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Diplomacy game rules are public domain. The dataset is provided under MIT License by diplomacy.org.

---

## Acknowledgments

We thank:
- The **UPC Barcelona** faculty for guidance and support
- Prof. **Ulises Cortes** for project supervision
- The **diplomacy.org** community for providing the dataset
- **Meta AI** for open-sourcing the Diplomacy research codebase

---

<p align="center">
  <strong>Universitat Politècnica de Catalunya</strong><br>
  Department of Computer Science<br>
  Fall 2025/26
</p>

<p align="center">
  <img src="https://www.upc.edu/comunicacio/ca/identitat/descarrega-arxius-grafics/fitxers-marca-principal/upc-positiu-p3005.png" alt="UPC Logo" width="200"/>
</p>