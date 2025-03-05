import time
import numpy as np
from monopoly_env.envs.monopoly_env import MonopolyEnv


def run_simulation():
    env = MonopolyEnv()
    observation = env.reset()

    total_reward = 0.0
    done = False

    # We'll count each turn as one "round".
    round_count = 0
    forced_purchase_rounds = 5  # For the first 5 turns, force purchase if available.

    print("Starting simulation with forced purchases for the first 5 rounds...\n")
    time.sleep(1)

    while not done:
        # Get current player.
        current_player = env.players[env.game.current_player_index]

        # For the first few rounds, try to force a purchase if possible.
        if round_count < forced_purchase_rounds:
            # Check if the current player's position corresponds to a property.
            forced = False
            for prop in env.game.board.properties_meta:
                # Assuming each property meta has a "position" field.
                if prop.get("id") == current_player.current_position:
                    # Determine if the property is unowned.
                    idx = env.game.board.property_id_to_index[prop["id"]]
                    bank_owner = np.array([1] + [0] * (env.game.board.num_owners - 1), dtype=np.float32)
                    if np.array_equal(env.game.board.state[idx, 0:env.game.board.num_owners], bank_owner):
                        # Force the purchase option.
                        current_player.set_option_to_buy(True)
                        forced = True
                        break
            if forced and current_player.can_buy_property():
                action = (10, 1)  # Buy property.
            else:
                # Otherwise, if purchase isn't applicable, choose a random valid action.
                valid_top = np.nonzero(env.game.get_valid_actions(current_player))[0]
                top_act = np.random.choice(valid_top) if valid_top.size > 0 else 0
                valid_sub_mask = env.game.get_valid_subactions(current_player, top_act)
                valid_sub = np.nonzero(valid_sub_mask)[0]
                sub_act = np.random.choice(valid_sub) if valid_sub.size > 0 else 0
                action = (top_act, sub_act)
        else:
            # After the forced purchase rounds, pick random valid actions.
            valid_top = np.nonzero(env.game.get_valid_actions(current_player))[0]
            top_act = np.random.choice(valid_top) if valid_top.size > 0 else 0
            valid_sub_mask = env.game.get_valid_subactions(current_player, top_act)
            valid_sub = np.nonzero(valid_sub_mask)[0]
            sub_act = np.random.choice(valid_sub) if valid_sub.size > 0 else 0
            action = (top_act, sub_act)

        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(
            f"Round {round_count + 1} | Player: {current_player.player_name} | Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)

        round_count += 1

    print("Simulation ended.")
    print(f"Total reward accumulated: {total_reward:.2f}")


if __name__ == "__main__":
    run_simulation()
