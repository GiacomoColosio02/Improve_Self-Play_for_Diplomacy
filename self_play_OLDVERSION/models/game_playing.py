from diplomacy import Game


def play_game(game: Game, agents: dict, max_turns: int = 100) -> Game:
    turn_count = 0
    while not game.is_game_done and turn_count < max_turns:
        possible_orders = game.get_all_possible_orders()
        for power_name, power in game.powers.items():
            agent = agents[power_name]
            orders = agent.choose_orders(game, power_name)
            game.set_orders(power_name, orders)
        game.process()
        turn_count += 1
    return game