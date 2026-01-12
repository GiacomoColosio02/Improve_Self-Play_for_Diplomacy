"""
Action Encoder for Diplomacy
Encodes orders into numerical format for ML.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import re

LOCATIONS = [
    'ANK', 'BEL', 'BER', 'BRE', 'BUD', 'BUL', 'CON', 'DEN', 'EDI', 'GRE',
    'HOL', 'KIE', 'LON', 'LVP', 'MAR', 'MOS', 'MUN', 'NAP', 'NWY', 'PAR',
    'POR', 'ROM', 'RUM', 'SER', 'SEV', 'SMY', 'SPA', 'STP', 'SWE', 'TRI',
    'TUN', 'VEN', 'VIE', 'WAR',
    'ALB', 'APU', 'ARM', 'BOH', 'BUR', 'CLY', 'FIN', 'GAL', 'GAS', 'LVN',
    'NAF', 'PIC', 'PIE', 'PRU', 'RUH', 'SIL', 'SYR', 'TUS', 'TYR', 'UKR',
    'WAL', 'YOR',
    'ADR', 'AEG', 'BAL', 'BAR', 'BLA', 'BOT', 'EAS', 'ENG', 'GOL', 'HEL',
    'ION', 'IRI', 'MAO', 'NAO', 'NTH', 'NWG', 'SKA', 'TYS', 'WES'
]

LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCATIONS)}
IDX_TO_LOC = {i: loc for i, loc in enumerate(LOCATIONS)}

# Order types
ORDER_TYPES = ['HOLD', 'MOVE', 'SUPPORT_HOLD', 'SUPPORT_MOVE', 'CONVOY', 'BUILD', 'DISBAND', 'RETREAT']
ORDER_TO_IDX = {o: i for i, o in enumerate(ORDER_TYPES)}

NUM_LOCATIONS = len(LOCATIONS)
NUM_ORDER_TYPES = len(ORDER_TYPES)


class ActionEncoder:
    """
    Encodes Diplomacy orders into action indices.
    
    Action space structure:
        - Each action is: (order_type, source_loc, target_loc, aux_loc)
        - We use a simplified flat encoding for now
        
    Simplified encoding:
        - HOLD: (source, source)
        - MOVE: (source, target) 
        - SUPPORT: (source, target, supported_unit_loc)
        - CONVOY: (source, target, convoyed_unit_loc)
        
    Action index = order_type * (N^2) + source * N + target
    where N = number of locations
    
    Total actions: 8 * 75 * 75 = 45,000 (we'll use a subset)
    """
    
    def __init__(self):
        self.num_locations = NUM_LOCATIONS
        self.num_order_types = NUM_ORDER_TYPES
        # Simplified: just encode (order_type, source, target)
        self.action_size = NUM_ORDER_TYPES * NUM_LOCATIONS * NUM_LOCATIONS
        
        # For practical use, we'll map orders to a smaller vocabulary
        self.order_to_idx = {}
        self.idx_to_order = {}
        self.vocab_size = 0
        
    def build_vocab(self, games: List[Dict], max_vocab: int = 10000):
        """Build vocabulary from training games."""
        from collections import Counter
        order_counts = Counter()
        
        for game in games:
            for phase in game.get('phases', []):
                for power, orders in phase.get('orders', {}).items():
                    if orders is None:
                        continue
                    for order in orders:
                        # Normalize order
                        norm_order = self._normalize_order(order)
                        if norm_order:
                            order_counts[norm_order] += 1
        
        # Take most common orders
        most_common = order_counts.most_common(max_vocab - 1)
        
        # Index 0 is for unknown/padding
        self.order_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_order = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (order, count) in enumerate(most_common, start=2):
            self.order_to_idx[order] = idx
            self.idx_to_order[idx] = order
        
        self.vocab_size = len(self.order_to_idx)
        print(f"Built vocabulary with {self.vocab_size} orders")
        
    def _normalize_order(self, order: str) -> Optional[str]:
        """Normalize an order string."""
        order = order.strip().upper()
        
        # Remove coast specifications for normalization
        order = re.sub(r'/[A-Z]{2}', '', order)
        
        # Basic validation
        if len(order) < 3:
            return None
            
        return order
    
    def encode_order(self, order: str) -> int:
        """Encode a single order to its index."""
        norm_order = self._normalize_order(order)
        if norm_order is None:
            return self.order_to_idx.get('<UNK>', 1)
        return self.order_to_idx.get(norm_order, self.order_to_idx.get('<UNK>', 1))
    
    def decode_order(self, idx: int) -> str:
        """Decode an index back to order string."""
        return self.idx_to_order.get(idx, '<UNK>')
    
    def encode_orders(self, orders: List[str]) -> List[int]:
        """Encode a list of orders."""
        return [self.encode_order(o) for o in orders]
    
    def parse_order(self, order: str) -> Dict:
        """
        Parse an order string into components.
        
        Returns dict with:
            - type: HOLD, MOVE, SUPPORT, CONVOY, etc.
            - unit: A or F
            - source: location
            - target: location (for moves)
            - aux: auxiliary location (for supports)
        """
        order = order.strip().upper()
        result = {'type': 'UNKNOWN', 'unit': '', 'source': '', 'target': '', 'aux': ''}
        
        # Parse unit type and source
        match = re.match(r'^([AF])\s+([A-Z]{3})', order)
        if match:
            result['unit'] = match.group(1)
            result['source'] = match.group(2)
        
        # Determine order type
        if ' H' in order and ' - ' not in order:
            result['type'] = 'HOLD'
            result['target'] = result['source']
        elif ' - ' in order:
            result['type'] = 'MOVE'
            move_match = re.search(r'-\s+([A-Z]{3})', order)
            if move_match:
                result['target'] = move_match.group(1)
        elif ' S ' in order:
            if ' - ' in order:
                result['type'] = 'SUPPORT_MOVE'
            else:
                result['type'] = 'SUPPORT_HOLD'
            # Find supported unit location
            support_match = re.search(r'S\s+[AF]\s+([A-Z]{3})', order)
            if support_match:
                result['aux'] = support_match.group(1)
        elif ' C ' in order:
            result['type'] = 'CONVOY'
        elif ' B' in order:
            result['type'] = 'BUILD'
        elif ' D' in order:
            result['type'] = 'DISBAND'
            
        return result
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'order_to_idx': self.order_to_idx,
                'vocab_size': self.vocab_size
            }, f)
        print(f"Saved vocabulary to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.order_to_idx = data['order_to_idx']
        self.idx_to_order = {int(v): k for k, v in self.order_to_idx.items()}
        self.vocab_size = data['vocab_size']
        print(f"Loaded vocabulary with {self.vocab_size} orders")


def test_encoder():
    """Test the action encoder."""
    encoder = ActionEncoder()
    
    # Test order parsing
    test_orders = [
        'A PAR H',
        'A PAR - BUR',
        'F BRE - MAO',
        'A MAR S A PAR - BUR',
        'F NTH C A YOR - NWY',
        'A LON B',
        'A PAR D'
    ]
    
    print("Order parsing test:")
    for order in test_orders:
        parsed = encoder.parse_order(order)
        print(f"  {order:25s} -> {parsed}")


if __name__ == '__main__':
    test_encoder()
