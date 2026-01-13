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

**Supervisor:** Prof. [Supervisor Name], Department of Computer Science, UPC Barcelona

---

## Abstract

Multi-agent reinforcement learning in complex strategic environments remains a fundamental challenge in artificial intelligence. **Diplomacy**, a seven-player game of negotiation and strategy, represents one of the most demanding testbeds for AI research due to its combinatorial action space (~10²⁰ possible actions per turn), need for long-term planning, and multi-agent dynamics with simultaneous moves.

This project investigates **hybrid training approaches** for No-Press Diplomacy (the non-communication variant), combining:

1. **Behavioral Cloning (BC)** from 33,279 human expert games
2. **Self-Play Reinforcement Learning** with PPO
3. **Human-Regularized RL (DiL-πKL)** to prevent strategy collapse
4. **Population-Based Training (PBT)** for robust generalization

Our experiments demonstrate that combining human gameplay data with diverse opponent populations significantly improves agent robustness compared to pure self-play approaches.

---

## Research Questions

This project addresses four fundamental research questions in multi-agent reinforcement learning:

| ID | Research Question | Method |
|----|-------------------|--------|
| **RQ1** | Does pure self-play lead to strategy collapse and overfitting in No-Press Diplomacy? | Self-Play RL Analysis |
| **RQ2** | Can human gameplay data effectively bootstrap the learning process while maintaining performance? | Human-Regularized RL (DiL-πKL) |
| **RQ3** | Does training against a diverse population of opponents improve robustness and generalization? | Population-Based Training |
| **RQ4** | What is the relative contribution of each component (BC, self-play, human regularization, population diversity)? | Ablation Study |

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
│  ~10²⁰ possible move            │  Games last 20+ years                     │
│  combinations per turn          │  (60+ decision points)                    │
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

**Our contribution:** A systematic study of hybrid training approaches accessible with limited computational resources.

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
                        │     (BC)        │                   │
                        └────────┬────────┘                   │
                                 │                            │
                                 │ Initialize                 │
                                 ▼                            │
     ┌──────────────┐    ┌─────────────────┐                  │
     │   Self vs    │◄───│   SELF-PLAY     │                  │
     │    Self      │    │      RL         │                  │
     │   Games      │    │    (PPO)        │                  │
     └──────────────┘    └────────┬────────┘                  │
                                 │                            │
                                 │ + KL Penalty               │ π_human
                                 ▼                            │
                        ┌─────────────────┐                   │
                        │ HUMAN-REGULARIZED│◄──────────────────┘
                        │       RL         │
                        │   (DiL-πKL)      │
                        └────────┬────────┘
                                 │
                                 │ + Diverse Opponents
                                 ▼
                        ┌─────────────────┐
                        │  POPULATION-    │
                        │    BASED        │
                        │   TRAINING      │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  FINAL AGENT    │
                        │   (Robust &     │
                        │  Generalized)   │
                        └─────────────────┘
```

### Component Details

#### 1. Data Integration & Behavioral Cloning

**Dataset:** 33,279 No-Press games from [diplomacy.org](https://github.com/diplomacy/research)

| Statistic | Value |
|-----------|-------|
| Total games | 33,279 |
| Total phases | ~1.16M |
| Avg. phases/game | 34.8 |
| Avg. game length | 11.6 years |
| Draw rate | 42% |

**State Encoding:** 1,216-dimensional vector per game state
- Per-location features (75 × 16 = 1,200): unit presence, SC ownership
- Global features (16): SC counts, unit counts, phase info
- Relative encoding from each power's perspective

**Action Encoding:** Vocabulary-based (~13,000 unique orders)
- Order types: HOLD, MOVE, SUPPORT, CONVOY, BUILD, DISBAND, RETREAT

#### 2. Self-Play Reinforcement Learning (RQ1)

**Algorithm:** Proximal Policy Optimization (PPO)

```python
L_PPO = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)] - c₁·L_VF + c₂·H[π]
```

Where:
- `r(θ) = π_θ(a|s) / π_θold(a|s)` — probability ratio
- `A` — Generalized Advantage Estimation (GAE)
- `ε = 0.2` — clipping parameter
- `H[π]` — entropy bonus for exploration

**Reward Shaping:**
| Event | Reward |
|-------|--------|
| Win (18+ SCs) | +10.0 |
| Gain 1 SC | +0.5 |
| Lose 1 SC | -0.3 |
| Survive 1 phase | +0.01 |
| Elimination | -10.0 |

#### 3. Human-Regularized RL (RQ2)

**Algorithm:** DiL-πKL (Bakhtin et al., 2022)

The key insight is to add a KL divergence penalty that keeps the RL policy close to human behavior:

```
L_DiL-πKL = L_PPO + β · D_KL(π_θ || π_human)
```

Where:
- `π_θ` — current RL policy
- `π_human` — frozen BC policy trained on human data
- `β` — KL penalty coefficient (hyperparameter)

**Intuition:** This prevents "strategy collapse" where the agent discovers narrow strategies that work only against itself but fail against diverse opponents.

#### 4. Population-Based Training (RQ3)

**Opponent Population:**

| Agent Type | Weight | Purpose |
|------------|--------|---------|
| Random | 0.15 | Baseline, prevents catastrophic failures |
| BC (Human-like) | 0.25 | Exposes to human strategies |
| Checkpoints | 0.15 each | Prevents forgetting past strategies |
| Current Self | 0.30 | Continues improvement |

**Prioritized Fictitious Self-Play (PFSP):**

```
P(opponent_i) ∝ (1 - win_rate_i)^p
```

This sampling strategy focuses training on opponents the agent struggles against, accelerating improvement on weaknesses.

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
│   ├── src/
│   │   ├── data_loader.py               # Data loading utilities
│   │   ├── dataset_download.py          # Download scripts
│   │   └── eda.py                        # EDA functions
│   └── README.md
│
├── 📁 BehavioralCloning/
│   ├── bc_training.ipynb                # BC training notebook
│   ├── models/                          # Saved BC models
│   └── README.md
│
├── 📁 SelfPlay/
│   ├── self_play_training.ipynb         # Pure self-play (RQ1)
│   ├── checkpoints/                     # Training checkpoints
│   └── README.md
│
├── 📁 HumanRegularizedRL/
│   ├── human_regularized_rl.ipynb       # DiL-πKL implementation (RQ2)
│   ├── models/                          # HR-RL models
│   └── README.md
│
├── 📁 PopulationBasedTraining/
│   ├── population_based_training.ipynb  # PBT implementation (RQ3)
│   ├── population/                      # Population checkpoints
│   └── README.md
│
├── 📁 Evaluation/
│   ├── ablation_study.ipynb             # RQ4: Component analysis
│   ├── tournament.ipynb                 # Head-to-head evaluation
│   └── results/                         # Evaluation results
│
├── 📁 docs/
│   ├── final_report.pdf                 # Project report
│   ├── presentation.pptx                # Final presentation
│   └── figures/                         # Generated figures
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── LICENSE                              # MIT License
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab (alternative)

### Setup

```bash
# Clone repository
git clone https://github.com/[username]/Improve_Self-Play_for_Diplomacy.git
cd Improve_Self-Play_for_Diplomacy

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from diplomacy import Game; print('✓ Diplomacy package installed')"
```

### Running Experiments

Each experiment is contained in a self-sufficient Jupyter notebook designed for Google Colab:

| Notebook | Description | Runtime |
|----------|-------------|---------|
| `DataIntegration/data_integration_eda.ipynb` | Data analysis & visualization | ~10 min |
| `BehavioralCloning/bc_training.ipynb` | Train BC policy on human data | ~30 min |
| `SelfPlay/self_play_training.ipynb` | Pure self-play RL (RQ1) | ~2-3 hours |
| `HumanRegularizedRL/human_regularized_rl.ipynb` | DiL-πKL training (RQ2) | ~2-3 hours |
| `PopulationBasedTraining/population_based_training.ipynb` | PBT training (RQ3) | ~2-3 hours |

**Quick Start (Google Colab):**

1. Open any notebook in Google Colab
2. Set runtime to GPU: `Runtime → Change runtime type → GPU`
3. Upload `standard_no_press.jsonl` when prompted
4. Run all cells: `Runtime → Run all`

---

## Results

### RQ1: Self-Play Analysis

**Finding:** Pure self-play leads to strategy collapse with 100% draw rate and no decisive victories.

| Metric | Value |
|--------|-------|
| Games trained | 3,000 |
| Win rate | 0% |
| Draw rate | 100% |
| Avg. game length | 200 (max) |

**Interpretation:** Without external pressure, the agent converges to passive equilibrium strategies.

### RQ2: Human-Regularized RL

**Finding:** KL regularization toward human policy significantly improves learning stability.

| Metric | Pure Self-Play | HR-RL (DiL-πKL) |
|--------|---------------|-----------------|
| Win rate vs Random | ~14% | ~35% |
| Strategy diversity | Low | High |
| KL from human | Unbounded | < 0.5 |

### RQ3: Population-Based Training

**Finding:** Training against diverse opponents improves generalization.

| Opponent Type | Win Rate |
|---------------|----------|
| vs Random | 45% |
| vs BC | 38% |
| vs Checkpoints | 42% |
| **Overall** | **41%** |

### RQ4: Ablation Study

| Configuration | Win Rate vs Random | Win Rate vs BC | Robustness |
|---------------|-------------------|----------------|------------|
| BC only | 25% | - | Low |
| Self-Play only | 14% | 10% | Very Low |
| HR-RL | 35% | 28% | Medium |
| **PBT (Full)** | **45%** | **38%** | **High** |

---

## Key Findings

1. **Pure self-play is insufficient** for Diplomacy — leads to exploitable strategies
2. **Human data provides crucial inductive bias** — accelerates learning and improves diversity
3. **KL regularization prevents collapse** — maintains human-like strategic variety
4. **Population diversity is essential** — prevents overfitting to single opponent type
5. **Hybrid approaches outperform pure methods** — combining BC + RL + PBT yields best results

---

## Limitations & Future Work

### Current Limitations

- **Computational resources:** Limited training compared to Diplodocus (1M+ games)
- **No search/planning:** Pure policy network without Monte Carlo Tree Search
- **No press variant only:** Does not address negotiation in full Diplomacy

### Future Directions

1. **Integration with MCTS** for improved strategic depth
2. **Larger-scale training** with distributed computing
3. **Transfer to full-press Diplomacy** with language models
4. **Multi-objective optimization** balancing win rate and human-likeness

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

8. Diplomacy Research Dataset. https://github.com/diplomacy/research

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Diplomacy game rules are public domain. The dataset is provided under MIT License by diplomacy.org.

---

## Acknowledgments

We thank:
- The **UPC Barcelona** faculty for guidance and support
- The **diplomacy.org** community for providing the dataset
- **Meta AI** for open-sourcing the Diplomacy research codebase
- The **Anthropic** team for AI assistance in development

---

<p align="center">
  <strong>Universitat Politècnica de Catalunya</strong><br>
  Department of Computer Science<br>
  Fall 2025/26
</p>

<p align="center">
  <img src="https://www.upc.edu/comunicacio/ca/identitat/descarrega-arxius-grafics/fitxers-marca-principal/upc-positiu-p3005.png" alt="UPC Logo" width="200"/>
</p>
