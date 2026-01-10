# No-Press Diplomacy Data Integration

**Project:** Improve Self-Play for Diplomacy  
**Authors:** Giacomo Colosio, Maciej Tasarz, Jakub Seliga, Luka Ivcevic  
**Course:** Intelligent System Project - UPC Barcelona  
**Semester:** Fall 2025/26

---

## Overview

This module provides tools for loading, preprocessing, and analyzing the diplomacy.org human gameplay dataset for training reinforcement learning agents in No-Press Diplomacy.

## Dataset Information

The primary dataset comes from [diplomacy.org](https://github.com/diplomacy/research):

| Category | Games |
|----------|-------|
| Total games | 156,468 |
| No-press games (standard map) | **33,279** |
| Press games (standard map, no messages) | 106,456 |
| Non-standard map games | 16,633 |

**Our target:** 33,279 no-press games on the standard map

## Project Structure

```
diplomacy_project/
├── data/
│   ├── raw/           # Downloaded raw data
│   ├── processed/     # Preprocessed training data
│   ├── sample/        # Sample data for testing
│   └── synthetic/     # Synthetic data for development
├── notebooks/
│   └── data_integration_eda.ipynb   # Main EDA notebook (Colab-ready)
├── src/
│   ├── data_loader.py      # Core data loading utilities
│   ├── dataset_download.py # Dataset download and preprocessing
│   └── eda.py              # Exploratory data analysis
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install diplomacy pandas matplotlib seaborn
```

### 2. Create Sample Data (for testing)

```python
from src.data_loader import create_sample_data, DiplomacyDataLoader

# Create sample data
create_sample_data("./data/sample")

# Load and analyze
loader = DiplomacyDataLoader("./data/sample")
loader.load_directory()
print(loader.export_summary())
```

### 3. Run EDA

```python
from src.eda import DiplomacyEDA

eda = DiplomacyEDA(loader)
eda.generate_full_report()
```

### 4. Use Jupyter Notebook (Recommended for Colab)

Open `notebooks/data_integration_eda.ipynb` in Google Colab or Jupyter.

## Data Format

### Game Structure

```json
{
  "id": "game_001",
  "map": "standard",
  "is_no_press": true,
  "phases": [
    {
      "name": "S1901M",
      "orders": {
        "AUSTRIA": ["A VIE - GAL", "A BUD - SER", "F TRI - ALB"],
        "ENGLAND": ["F LON - NTH", "F EDI - NWG", "A LVP - YOR"],
        ...
      },
      "state": { ... }
    },
    ...
  ],
  "outcome": {"AUSTRIA": 5, "ENGLAND": 4, ...}
}
```

### Phase Naming Convention

- `S1901M` = Spring 1901 Movement
- `F1901M` = Fall 1901 Movement
- `W1901A` = Winter 1901 Adjustments

### Order Format

- Move: `A VIE - GAL` (Army Vienna to Galicia)
- Hold: `A VIE H` (Army Vienna Hold)
- Support: `A VIE S A BUD - GAL` (Army Vienna Supports Army Budapest to Galicia)
- Convoy: `F NTH C A YOR - NWY` (Fleet North Sea Convoys Army Yorkshire to Norway)

## Key Classes

### `DiplomacyDataLoader`

Main class for loading game data:

```python
loader = DiplomacyDataLoader("./data")
loader.load_directory(pattern="*.json*", max_games=1000)
loader.filter_no_press()
loader.filter_standard_map()
loader.filter_min_phases(5)

stats = loader.compute_statistics()
```

### `Game`, `Phase`, `Order`

Data classes representing game structure:

```python
for game in loader.games:
    print(f"Game {game.game_id}: {game.num_phases} phases, winner: {game.winner}")
    for phase in game.phases:
        for power, orders in phase.orders.items():
            for order_str in orders:
                order = Order.from_string(order_str)
                print(f"{power}: {order.order_type} - {order.raw_order}")
```

## Statistics Computed

- Total games, no-press games, standard map games
- Phase distribution (min, max, mean)
- Year distribution
- Winner distribution per power
- Order type distribution (HOLD, MOVE, SUPPORT, CONVOY)
- Opening analysis (S1901M patterns)

## Visualizations

The EDA module generates:

1. **Game length distribution** (phases and years)
2. **Winner distribution** (bar chart and pie chart)
3. **Order type distribution** 
4. **Power performance over time** (average supply centers)

## Next Steps

After data integration:

1. **Behavioral Cloning (BC)**: Train initial policy via supervised learning on human data
2. **State Encoding**: Implement 887-dimensional board state encoding per Bakhtin et al. (2022)
3. **Policy Network**: Train transformer-based policy with LSTM decoder
4. **Self-Play + Human Regularization**: DiL-πKL algorithm implementation

## References

- Bakhtin et al. (2022). *Mastering the Game of No-Press Diplomacy via Human-Regularized RL and Planning*
- Paquette et al. (2019). *No-Press Diplomacy: Modeling Multi-Agent Gameplay*
- Dataset: https://github.com/diplomacy/research

## License

Dataset: MIT License (diplomacy.org)
Code: MIT License
