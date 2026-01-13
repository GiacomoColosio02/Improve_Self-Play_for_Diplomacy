from diplomacy import Game
from ..models.self_play_agents import SelfPlayAgent, RandomStrategy, RLAgent
from ..models.game_playing import play_game
from ..models.dilpilk_model import BoardStateEncoder, DilpILKNet
import tensorflow as tf

def set_up_game() -> Game:
    game = Game()
    return game
    
def set_up_agents(game, agent, strategy) -> dict:
    agents = {power[0] : agent(strategy()) for power in game.powers.items()}
    return agents

def test_play_game(agent=SelfPlayAgent, strategy=RandomStrategy):
    game = set_up_game()
    agents = set_up_agents(game, agent, strategy)
    play_game(game, agents, max_turns=10)

    print("Game played successfully up to phase:", game.phase)

def test_encode_board_state():
    game = set_up_game()
    agents = set_up_agents(game, RLAgent, RandomStrategy)
    board_state = BoardStateEncoder.encode_board_state(game)
    print("Encoded board state:", board_state)

def test_model_forward():
    model = DilpILKNet()
    game = set_up_game()
    board_state = BoardStateEncoder.encode_board_state(game)
    embeddings = model.call(board_state)
    if isinstance(embeddings, (list, tuple)):
        for e in embeddings:
            assert isinstance(e, tf.Tensor)
    else:
        assert isinstance(embeddings, tf.Tensor)
    print("Model forward pass successful, output shape:")
    for e in embeddings:
        print(e.shape)

test_model_forward()
test_play_game()
test_encode_board_state()

