import tensorflow as tf
import numpy as np
from diplomacy import Game
from self_play.models.self_play_agents import RLAgent, RandomStrategy, SelfPlayAgent
from self_play.models.game_playing import play_game
from self_play.models.dilpilk_model import DilpILKNet, BoardStateEncoder

# Hyperparameters
learning_rate = 1e-4
num_games = 5
gamma = 0.99  # reward discount

model = DilpILKNet()
optimizer = tf.keras.optimizers.Adam(learning_rate)


# Simple reward function
def compute_reward(game, power_name):
    """Example: reward = number of supply centers controlled by player."""
    centers_by_power = game.get_centers(None)
    return len(centers_by_power.get(power_name, []))


# Self-play and training
for game_idx in range(num_games):
    print(f"\n Starting self-play game {game_idx + 1}/{num_games}")
    game = Game()

    # each player gets an RL agent
    agents = {
        power: RLAgent(model)
        for power in game.powers
    }

    trajectories = {p: [] for p in agents.keys()}

    max_turns = 20
    turn = 0
    while not game.is_game_done and turn < max_turns:
        for power_name in game.powers.keys():
            agent = agents[power_name]

            board_state = BoardStateEncoder.encode_board_state(game)
            policy_logits, value_logits, _ = model.call(board_state, training=False)

            # Pick random action (for now we simulate choice)
            chosen_action = np.random.randint(policy_logits.shape[-1])

            # Save for RL update
            reward = compute_reward(game, power_name)
            trajectories[power_name].append((board_state, policy_logits, value_logits, reward))

        game.process()
        turn += 1

    print(f"Game {game_idx + 1} finished after {turn} turns")

    # RL policy/value update (per game)
    for power_name, steps in trajectories.items():
        if not steps:
            continue

        rewards = [r for (_, _, _, r) in steps]
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            total_loss = 0.0
            for (board_state, policy_logits, value_logits, reward), Gt in zip(steps, returns):
                policy_out, value_out, _ = model.call(board_state, training=True)

                # Policy loss - for now we just use a placeholder since we didn’t store chosen actions yet
                policy_loss = -tf.reduce_mean(value_out)  # mock policy loss

                # Value loss
                value_loss = tf.reduce_mean(tf.square(value_out - Gt))
                total_loss += policy_loss + 0.5 * value_loss

        grads = tape.gradient(total_loss,
                              model.encoder.trainable_variables
                              + model.policy_head.trainable_variables
                              + model.value_head.trainable_variables)
        optimizer.apply_gradients(zip(grads,
                                      model.encoder.trainable_variables
                                      + model.policy_head.trainable_variables
                                      + model.value_head.trainable_variables))

        print(f"Agent {power_name}: loss={float(total_loss):.4f}")

