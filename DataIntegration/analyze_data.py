"""
Quick Analysis - LIGHT VERSION (first 1000 games only)
"""

import json
from collections import Counter

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

MAX_GAMES = 1000

print(f"Loading first {MAX_GAMES} games...")
games = []
with open("data/standard_no_press.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= MAX_GAMES:
            break
        games.append(json.loads(line))
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1} games...")

print(f"Loaded {len(games)} games total")

# Analyze
print("Analyzing...")
phase_counts = []
winners = Counter()
order_types = Counter()
total_orders = 0

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
    
    # Orders (with None check)
    for phase in phases:
        for power, orders in phase.get('orders', {}).items():
            if orders is None:  # <-- FIX: skip None orders
                continue
            for order in orders:
                total_orders += 1
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

# Print report
print("\n" + "="*50)
print(f"ANALYSIS OF {len(games)} NO-PRESS GAMES")
print("="*50)
print(f"\nPhases: min={min(phase_counts)}, max={max(phase_counts)}, avg={sum(phase_counts)/len(phase_counts):.1f}")

print(f"\nWinners:")
for p in POWERS + ['Draw']:
    c = winners.get(p, 0)
    print(f"  {p}: {c} ({c/len(games)*100:.1f}%)")

print(f"\nOrders: {total_orders:,} total")
for t, c in order_types.most_common():
    print(f"  {t}: {c:,} ({c/total_orders*100:.1f}%)")
print("="*50)