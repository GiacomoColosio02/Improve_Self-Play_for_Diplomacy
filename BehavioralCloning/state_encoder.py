"""
State Encoder for Diplomacy
Encodes board state into a vector representation.
"""

import numpy as np
from typing import Dict, List, Tuple

# All 75 locations on the standard Diplomacy map
LOCATIONS = [
    # Supply centers (34)
    'ANK', 'BEL', 'BER', 'BRE', 'BUD', 'BUL', 'CON', 'DEN', 'EDI', 'GRE',
    'HOL', 'KIE', 'LON', 'LVP', 'MAR', 'MOS', 'MUN', 'NAP', 'NWY', 'PAR',
    'POR', 'ROM', 'RUM', 'SER', 'SEV', 'SMY', 'SPA', 'STP', 'SWE', 'TRI',
    'TUN', 'VEN', 'VIE', 'WAR',
    # Non-supply center land (22)
    'ALB', 'APU', 'ARM', 'BOH', 'BUR', 'CLY', 'FIN', 'GAL', 'GAS', 'LVN',
    'NAF', 'PIC', 'PIE', 'PRU', 'RUH', 'SIL', 'SYR', 'TUS', 'TYR', 'UKR',
    'WAL', 'YOR',
    # Sea zones (19)
    'ADR', 'AEG', 'BAL', 'BAR', 'BLA', 'BOT', 'EAS', 'ENG', 'GOL', 'HEL',
    'ION', 'IRI', 'MAO', 'NAO', 'NTH', 'NWG', 'SKA', 'TYS', 'WES'
]

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

SUPPLY_CENTERS = [
    'ANK', 'BEL', 'BER', 'BRE', 'BUD', 'BUL', 'CON', 'DEN', 'EDI', 'GRE',
    'HOL', 'KIE', 'LON', 'LVP', 'MAR', 'MOS', 'MUN', 'NAP', 'NWY', 'PAR',
    'POR', 'ROM', 'RUM', 'SER', 'SEV', 'SMY', 'SPA', 'STP', 'SWE', 'TRI',
    'TUN', 'VEN', 'VIE', 'WAR'
]

LOC_TO_IDX = {loc: i for i, loc in enumerate(LOCATIONS)}
POWER_TO_IDX = {p: i for i, p in enumerate(POWERS)}


class StateEncoder:
    """
    Encodes Diplomacy game state into a fixed-size vector.
    
    Encoding per location (75 locations):
        - 7 bits: which power has a unit here (one-hot)
        - 1 bit: is it an army (1) or fleet (0)
        - 7 bits: which power owns this SC (one-hot, if SC)
        - 1 bit: is this location a supply center
        
    Total per location: 16 features
    Total: 75 * 16 = 1200 features
    
    Plus global features:
        - 7 values: supply center count per power
        - 7 values: unit count per power
        - 1 value: year (normalized)
        - 1 value: season (0=spring, 0.5=fall, 1=winter)
        
    Total global: 16 features
    Grand total: 1216 features
    """
    
    def __init__(self):
        self.num_locations = len(LOCATIONS)
        self.num_powers = len(POWERS)
        self.features_per_loc = 16
        self.global_features = 16
        self.state_size = self.num_locations * self.features_per_loc + self.global_features
        
    def encode(self, state: Dict, phase_name: str = '') -> np.ndarray:
        """
        Encode a game state into a vector.
        
        Args:
            state: Game state dict with 'units' and 'centers'
            phase_name: Phase name like 'S1901M'
            
        Returns:
            numpy array of shape (state_size,)
        """
        features = np.zeros(self.state_size, dtype=np.float32)
        
        units = state.get('units', {})
        centers = state.get('centers', {})
        
        # Encode each location
        for loc_idx, loc in enumerate(LOCATIONS):
            offset = loc_idx * self.features_per_loc
            
            # Check for units at this location
            for power_idx, power in enumerate(POWERS):
                power_units = units.get(power, [])
                for unit in power_units:
                    unit_loc = self._parse_unit_location(unit)
                    if unit_loc == loc:
                        # Unit presence (one-hot for power)
                        features[offset + power_idx] = 1.0
                        # Unit type: 1 for Army, 0 for Fleet
                        features[offset + 7] = 1.0 if unit.startswith('A ') else 0.0
                        break
            
            # Check supply center ownership
            if loc in SUPPLY_CENTERS:
                features[offset + 15] = 1.0  # Is supply center
                for power_idx, power in enumerate(POWERS):
                    power_centers = centers.get(power, [])
                    if loc in power_centers:
                        features[offset + 8 + power_idx] = 1.0
                        break
        
        # Global features
        global_offset = self.num_locations * self.features_per_loc
        
        # Supply center counts (normalized by 18, the victory condition)
        for power_idx, power in enumerate(POWERS):
            sc_count = len(centers.get(power, []))
            features[global_offset + power_idx] = sc_count / 18.0
        
        # Unit counts (normalized by 17, max possible units)
        for power_idx, power in enumerate(POWERS):
            unit_count = len(units.get(power, []))
            features[global_offset + 7 + power_idx] = unit_count / 17.0
        
        # Year and season
        if phase_name:
            year = self._parse_year(phase_name)
            season = self._parse_season(phase_name)
            features[global_offset + 14] = (year - 1901) / 20.0  # Normalize year
            features[global_offset + 15] = season
        
        return features
    
    def _parse_unit_location(self, unit: str) -> str:
        """Extract location from unit string like 'A PAR' or 'F STP/SC'."""
        parts = unit.split()
        if len(parts) >= 2:
            loc = parts[1].split('/')[0]  # Handle STP/SC -> STP
            return loc
        return ''
    
    def _parse_year(self, phase_name: str) -> int:
        """Extract year from phase name like 'S1901M'."""
        try:
            return int(phase_name[1:5])
        except:
            return 1901
    
    def _parse_season(self, phase_name: str) -> float:
        """Extract season from phase name. Returns 0, 0.5, or 1."""
        if phase_name.startswith('S'):
            return 0.0
        elif phase_name.startswith('F'):
            return 0.5
        else:  # Winter
            return 1.0


def test_encoder():
    """Test the state encoder."""
    encoder = StateEncoder()
    print(f"State size: {encoder.state_size}")
    
    # Sample state
    state = {
        'units': {
            'FRANCE': ['A PAR', 'A MAR', 'F BRE'],
            'GERMANY': ['A BER', 'A MUN', 'F KIE'],
            'ENGLAND': ['F LON', 'F EDI', 'A LVP']
        },
        'centers': {
            'FRANCE': ['PAR', 'MAR', 'BRE'],
            'GERMANY': ['BER', 'MUN', 'KIE'],
            'ENGLAND': ['LON', 'EDI', 'LVP']
        }
    }
    
    encoded = encoder.encode(state, 'S1901M')
    print(f"Encoded shape: {encoded.shape}")
    print(f"Non-zero features: {np.count_nonzero(encoded)}")
    print(f"Sample values: {encoded[:20]}")


if __name__ == '__main__':
    test_encoder()
