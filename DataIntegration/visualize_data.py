"""
Visualization of No-Press Diplomacy Dataset
"""

import json
from collections import Counter
import matplotlib.pyplot as plt

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
MAX_GAMES = 1000

# Load data
print("Loading data...")
games = []
with open("data/standard_no_press.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= MAX_GAMES:
            break
        games.append(json.loads(line))
print(f"Loaded {len(games)} games")

# Collect stats
phase_counts = []
winners = Counter()
order_types = Counter()

for game in games:
    phases = game.get('phases', [])
    phase_counts.append(len(phases))
    
    # Winner
    winner = None
    if phases:
        last_state = phases[-1].get('state', {})
        centers = last_state.get('centers', {})
        for power, cs in centers.items():
            if len(cs) >= 18:
                winner = power
                break
    winners[winner if winner else 'Draw'] += 1
    
    # Orders
    for phase in phases:
        for power, orders in phase.get('orders', {}).items():
            if orders is None:
                continue
            for order in orders:
                if ' H' in order:
                    order_types['HOLD'] += 1
                elif ' - ' in order:
                    order_types['MOVE'] += 1
                elif ' S ' in order:
                    order_types['SUPPORT'] += 1
                elif ' C ' in order:
                    order_types['CONVOY'] += 1
                else:
                    order_types['OTHER'] += 1

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Game length distribution
axes[0].hist(phase_counts, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(sum(phase_counts)/len(phase_counts), color='red', linestyle='--', label=f'Mean: {sum(phase_counts)/len(phase_counts):.1f}')
axes[0].set_xlabel('Number of Phases')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Game Length Distribution')
axes[0].legend()

# 2. Winner distribution
labels = POWERS + ['Draw']
counts = [winners.get(p, 0) for p in labels]
colors = plt.cm.tab10(range(len(labels)))
bars = axes[1].bar(labels, counts, color=colors, edgecolor='black')
axes[1].set_xlabel('Power')
axes[1].set_ylabel('Wins')
axes[1].set_title('Winner Distribution')
axes[1].tick_params(axis='x', rotation=45)

# 3. Order types
order_labels = list(order_types.keys())
order_counts = list(order_types.values())
axes[2].pie(order_counts, labels=order_labels, autopct='%1.1f%%', colors=plt.cm.Set3(range(len(order_labels))))
axes[2].set_title('Order Type Distribution')

plt.tight_layout()
plt.savefig('diplomacy_eda.png', dpi=150)
print("Saved: diplomacy_eda.png")
plt.show()