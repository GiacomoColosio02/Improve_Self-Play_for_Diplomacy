from random import choice

class SelfPlayAgent:
    def __init__(self, strategy):
        self.strategy = strategy

    def choose_orders(self, game, power_name: str) -> list[str]:
        possible_orders = game.get_all_possible_orders()
        orderable_locations = game.get_orderable_locations(power_name)
        selected_orders = []
        for loc in orderable_locations:
            legal = possible_orders.get(loc, [])
            if legal:
                selected_orders.append(self.choose(legal))
        return selected_orders

    def choose(self, options) -> str:
        return self.strategy.select(options)
    
class RandomStrategy:
    def select(self, options: list[str]) -> str:
        return choice(options)
    
class RLAgent(SelfPlayAgent):
    def __init__(self, model):
        super().__init__(strategy=model)
        self.model = model

    def choose_orders(self, game, power_name):
        possible_orders = game.get_all_possible_orders()
        orderable_locations = game.get_orderable_locations(power_name)
        board_state = RLAgent._encode_board_state(game)


