import time
import numpy as np
from monopoly_env.envs.monopoly_env import MonopolyEnv


def decode_trade_subaction(sub):
    """Decodes a trade offer sub-action into (target_index, property_index, price_tier_index)."""
    target_index = sub // (28 * 3)
    remainder = sub % (28 * 3)
    property_index = remainder // 3
    price_tier_index = remainder % 3
    return target_index, property_index, price_tier_index


def choose_trade_subaction_for_target(valid_indices, desired_target=0):
    """
    Given a list of valid sub-action indices for a trade offer (sell),
    choose one that decodes to the desired target.
    If none match, return the first valid index.
    """
    for sub in valid_indices:
        target_index, prop_idx, price_tier = decode_trade_subaction(sub)
        if target_index == desired_target:
            return int(sub)
    # Fallback: return the first valid sub-action index.
    return int(valid_indices[0])


def run_simulation():
    env = MonopolyEnv()
    observation = env.reset()

    total_reward = 0.0
    done = False

    # --------------------------------------------------
    # Round 1: Bob Buys a Property
    # --------------------------------------------------
    bob = env.players[env.game.current_player_index]
    bob.current_position = 2  # Assuming property at position 2 is purchasable.
    bob.update_phase('post-roll')
    bob.set_option_to_buy(True)

    actions_round1 = [
        (10, 1),  # Buy Property (action index 10, sub-action 1 means "buy")
        (7, 0),  # Conclude phase (post-roll -> out-of-turn)
        (7, 0)  # Conclude phase (out-of-turn -> next player's pre-roll)
    ]

    print("Round 1: Bob buys a property...\n")
    time.sleep(1)
    for action in actions_round1:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)
        if done:
            break

    # --------------------------------------------------
    # Round 2: Charlie Goes to Jail and Uses a Get-Out-of-Jail Card
    # --------------------------------------------------
    charlie = env.players[env.game.current_player_index]
    charlie.go_to_jail()
    charlie.has_get_out_of_jail_chance_card = True

    actions_round2 = [
        (8, 0),  # Use Get Out of Jail (action index 8)
        (7, 0),  # Conclude phase (transition phase)
        (7, 0)  # Conclude phase to move to next player's turn.
    ]

    print("Round 2: Charlie goes to jail and uses his Get-Out-of-Jail card...\n")
    time.sleep(1)
    for action in actions_round2:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)
        if done:
            break

    # --------------------------------------------------
    # Round 3: Bob Improves His Property
    # --------------------------------------------------
    env.game.current_player_index = 0  # Force Bob to be current.
    bob = env.players[env.game.current_player_index]
    bob.update_phase('pre-roll')

    valid_sub_mask = env.game.get_valid_subactions(bob, 2)
    valid_indices = np.nonzero(valid_sub_mask)[0]
    if valid_indices.size > 0:
        chosen_sub_improve = int(valid_indices[0])
    else:
        chosen_sub_improve = 0

    actions_round3 = [
        (2, chosen_sub_improve),  # Improve Property (action index 2)
        (7, 0),  # Conclude phase (pre-roll triggers dice roll -> post-roll)
        (7, 0)  # Conclude phase to move to next player's turn.
    ]

    print("Round 3: Bob improves his property...\n")
    time.sleep(1)
    for action in actions_round3:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)
        if done:
            break

    # --------------------------------------------------
    # Round 4: Mortgage/Free Mortgage
    # --------------------------------------------------
    env.game.current_player_index = 0
    bob = env.players[env.game.current_player_index]
    bob.update_phase('post-roll')

    valid_sub_mask = env.game.get_valid_subactions(bob, 5)
    valid_indices = np.nonzero(valid_sub_mask)[0]
    if valid_indices.size > 0:
        chosen_sub_mortgage = int(valid_indices[0])
    else:
        chosen_sub_mortgage = 0

    actions_round4 = [
        (5, chosen_sub_mortgage),  # Mortgage/Free Mortgage (action index 5)
        (7, 0),  # Conclude phase (post-roll -> out-of-turn)
        (7, 0)  # Conclude phase to move to next player's turn.
    ]

    print("Round 4: Bob uses Mortgage/Free Mortgage on a property...\n")
    time.sleep(1)
    for action in actions_round4:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)
        if done:
            break

    # --------------------------------------------------
    # Round 5: Trade Offer Cycle
    # --------------------------------------------------
    # Part A: Bob makes a trade offer (sell offer) targeting Charlie.
    env.game.current_player_index = 0
    bob = env.players[env.game.current_player_index]
    bob.update_phase('pre-roll')

    valid_sub_mask = env.game.get_valid_subactions(bob, 0)
    valid_indices = np.nonzero(valid_sub_mask)[0]
    if valid_indices.size > 0:
        # Change desired_target to 0 to target Charlie based on candidate ordering.
        chosen_sub_trade = choose_trade_subaction_for_target(valid_indices, desired_target=0)
    else:
        chosen_sub_trade = 0

    actions_round5a = [
        (0, chosen_sub_trade),  # Make Trade Offer (Sell) by Bob.
        (7, 0),  # Conclude phase.
        (7, 0)  # End Bob's turn.
    ]

    print("Round 5A: Bob makes a trade offer to sell a property to Charlie...\n")
    time.sleep(1)
    for action in actions_round5a:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)
        if done:
            break

    # Part B: Charlie responds to the trade offer and accepts it.
    env.game.current_player_index = 1
    charlie = env.players[env.game.current_player_index]
    charlie.update_phase('out-of-turn')

    actions_round5b = [
        (11, 1),  # Respond to Trade: Accept (action index 11, sub-action 1 means "accept")
        (7, 0),  # Conclude phase.
        (7, 0)  # End Charlie's turn.
    ]

    print("Round 5B: Charlie responds to the trade offer and accepts it...\n")
    time.sleep(1)
    for action in actions_round5b:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action taken: {action}, Reward: {reward:.2f}, Info: {info}\n")
        time.sleep(1)
        if done:
            break

    print("Simulation ended.")
    print(f"Total reward accumulated: {total_reward:.2f}")


if __name__ == "__main__":
    run_simulation()
