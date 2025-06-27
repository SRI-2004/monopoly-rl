# tests/test_monopoly_env.py

import unittest
import os
import json
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from monopoly_env.envs.monopoly_env import MonopolyEnv
from unittest.mock import patch


class TestMonopolyEnv(unittest.TestCase):
    def setUp(self):
        """
        Set up a temporary board JSON file with dummy property data and initialize the environment.
        """
        # Create a temporary JSON file for board data using the new unified layout.
        self.test_json = "test_board_data.json"
        board_layout = []
        
        # Add special spaces
        board_layout.append({"id": 0, "name": "Go", "type": "go"})
        board_layout.append({"id": 10, "name": "Jail", "type": "jail"})
        board_layout.append({"id": 30, "name": "Go to Jail", "type": "go_to_jail"})
        
        # Create 28 dummy properties with minimal required fields.
        property_ids = [i for i in range(1, 40) if i not in [0, 10, 30]]
        for i in range(28):
            prop_id = property_ids[i]
            board_layout.append({
                "id": prop_id,
                "name": f"Property {prop_id}",
                "color_group": "red",
                "type": "street",
                "price": 60,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2}
            })
            
        board_data = {"board_layout": sorted(board_layout, key=lambda x: x['id'])}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)

        # Initialize the environment with 4 players to match the default config.
        self.env = MonopolyEnv(board_json_path=self.test_json, num_players=4, max_steps=10)

    def tearDown(self):
        """
        Clean up the temporary JSON file.
        """
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_reset(self):
        """
        Test that reset returns an observation with expected keys and shapes.
        """
        obs, info = self.env.reset()
        self.assertIn("player", obs)
        self.assertIn("board", obs)
        self.assertEqual(obs["player"].shape, (16,))
        self.assertEqual(obs["board"].shape, (224,))

    def test_step(self):
        """
        Test that a valid step returns an observation, a float reward, a boolean done flag, and an info dict.
        """
        self.env.reset()
        # Use a simple action, for example: (0, 0) -> top-level action 0 (which becomes 1 internally) with sub-action 0.
        action = (0, 0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(obs, dict)
        self.assertIn("player", obs)
        self.assertIn("board", obs)
        self.assertIsInstance(reward, float)
        self.assertTrue(isinstance(terminated, bool) or isinstance(truncated, bool))
        self.assertIsInstance(info, dict)

    def test_episode_termination(self):
        """
        Test that the environment eventually terminates.
        """
        self.env.reset()
        terminated, truncated = False, False
        step_count = 0
        # Use an action that typically won't change much (e.g., skip turn: top-level action 6).
        action = (6, 0)
        while not (terminated or truncated) and step_count < 20:
            _, _, terminated, truncated, _ = self.env.step(action)
            step_count += 1
        self.assertTrue(terminated or truncated or step_count == self.env.max_steps)

    def test_action_out_of_bounds(self):
        """
        Test that providing an action outside the action space returns the correct error.
        """
        self.env.reset()
        action = (15, 0) # Top-level action 15 is invalid (space is 12).
        _, reward, _, _, info = self.env.step(action)
        self.assertLess(reward, 0)
        self.assertIn("error", info)
        self.assertEqual(info["error"], "Action out of bounds.")

    def test_logically_invalid_top_level_action(self):
        """
        Test that a logically invalid top-level action (e.g., buying in the wrong phase) returns the correct error.
        """
        self.env.reset()
        # Action 10 is "Buy Property", which is invalid in the initial 'pre-roll' phase.
        action = (10, 0)
        _, reward, _, _, info = self.env.step(action)
        self.assertLess(reward, 0)
        self.assertIn("error", info)
        self.assertEqual(info["error"], "Invalid top-level action selected according to valid actions mask.")

    def test_logically_invalid_sub_action(self):
        """
        Test that a logically invalid sub-action (e.g., selling a property not owned) returns the correct error.
        """
        self.env.reset()
        # Action 4 is "Sell Property". Sub-action 0 corresponds to the first property.
        # This is invalid since the player starts with no properties.
        action = (4, 0)
        _, reward, _, _, info = self.env.step(action)
        self.assertLess(reward, 0)
        self.assertIn("error", info)
        self.assertEqual(info["error"], "Invalid sub-action selected according to valid sub-actions mask.")

    def test_property_purchase(self):
        """
        Test that a player can successfully purchase a property.
        """
        self.env.reset()
        player = self.env.game.players[0]
        
        # Manually move player to property 1 and give them the option to buy
        player.current_position = 1
        player.phase = 'post-roll'
        player.set_option_to_buy(True)
        initial_cash = player.current_cash
        
        # Action 10 is Buy, Sub-action 1 is 'yes'
        action = (10, 1) 
        
        _, _, _, _, info = self.env.step(action)
        
        property_meta = self.env.game.board.get_property_meta(1)
        price = property_meta['price']
        
        self.assertIn(1, player.assets)
        self.assertEqual(player.current_cash, initial_cash - price)
        self.assertIn("purchased", info['result'])

    def test_rent_payment(self):
        """
        Test that rent is paid when a player lands on an owned property.
        """
        self.env.reset()
        owner = self.env.game.players[0]
        renter = self.env.game.players[1]
        
        # Give property 1 to the owner
        prop_id = 1
        property_meta = self.env.game.board.get_property_meta(prop_id)
        price = property_meta['price']
        rent = property_meta['rent']['base']
        
        owner.assets.add(prop_id)
        self.env.game.board.purchase_property(prop_id, 1) # player_id is 1-indexed

        # Manually set the game state to force the renter to land on the property
        self.env.game.current_player_index = 1 # It's the renter's turn
        renter.current_position = prop_id
        renter.phase = 'pre-roll'

        owner_initial_cash = owner.current_cash
        renter_initial_cash = renter.current_cash

        # The action that will trigger the rent payment is concluding the pre-roll phase (dice roll)
        # However, to avoid randomness, we manually trigger the landing logic.
        # Let's use the 'conclude_phase' action, which rolls and moves. We can't guarantee landing.
        # A better test is to directly call the internal logic. But we want to test the env.
        # The most deterministic way is to find an action that doesn't move and then check state.
        # Let's re-think. The rent logic happens in _handle_landing after a move.
        # We can't deterministically land. So let's manually call the landing logic and check the result.
        
        landing_message = self.env.game._handle_landing(renter)

        self.assertIn(f"paid {rent} to {owner.player_name}", landing_message)
        self.assertEqual(owner.current_cash, owner_initial_cash + rent)
        self.assertEqual(renter.current_cash, renter_initial_cash - rent)

    def test_go_to_jail(self):
        """
        Test that a player is sent to jail when landing on the 'Go to Jail' space.
        """
        self.env.reset()
        player = self.env.game.players[0]
        
        # Manually place player on the 'Go to Jail' space
        go_to_jail_space = next((s for s in self.env.game.board.board_layout if s['type'] == 'go_to_jail'), None)
        self.assertIsNotNone(go_to_jail_space)
        player.current_position = go_to_jail_space['id']
        
        # Trigger the landing logic
        self.env.game._handle_landing(player)
        
        jail_space = next((s for s in self.env.game.board.board_layout if s['type'] == 'jail'), None)
        self.assertIsNotNone(jail_space)
        
        self.assertTrue(player.currently_in_jail)
        self.assertEqual(player.current_position, jail_space['id'])

    def test_pay_jail_fine(self):
        """
        Test that a player can get out of jail by paying the fine.
        """
        self.env.reset()
        player = self.env.game.players[0]
        
        # Manually put the player in jail
        player.go_to_jail()
        initial_cash = player.current_cash
        
        # Action 9 is "Pay Jail Fine"
        action = (9, 0)
        
        _, _, _, _, info = self.env.step(action)
        
        self.assertFalse(player.currently_in_jail)
        self.assertEqual(player.current_cash, initial_cash - 50) # Assuming fine is 50
        self.assertIn("paid a jail fine", info['result'])

    def test_bankruptcy_and_win_condition(self):
        """
        Test that a player goes bankrupt and another player wins.
        """
        # For this test, we need exactly two players to simplify the win condition
        self.env = MonopolyEnv(board_json_path=self.test_json, num_players=2, max_steps=50)
        
        owner = self.env.game.players[0]
        renter = self.env.game.players[1]
        
        # Give a high-rent property to the owner
        prop_id = 1
        property_meta = self.env.game.board.get_property_meta(prop_id)
        rent = property_meta['rent']['base']
        
        owner.assets.add(prop_id)
        self.env.game.board.purchase_property(prop_id, 1)

        # Set the renter up for failure
        renter.current_cash = rent - 1 # Not enough cash to pay rent
        renter.assets.clear() # No assets to sell

        # Manually trigger the landing
        self.env.game.current_player_index = 1
        renter.current_position = prop_id
        
        # Force the rent event deterministically
        self.env.game._handle_landing(renter)

        # Let the env process one dummy step so it sees statuses
        obs, reward, terminated, truncated, info = self.env.step((6,0))  # Skip

        self.assertEqual(renter.status, 'lost')
        self.assertEqual(owner.status, 'won')
        self.assertTrue(terminated)

    def test_init_with_too_many_players(self):
        """Test that initializing with too many players raises a ValueError."""
        with self.assertRaises(ValueError):
            MonopolyEnv(board_json_path=self.test_json, num_players=5) # Default config only has 4 names

    def test_seed(self):
        """Test that the seed method returns a seed and sets the np_random attribute."""
        seed = self.env.seed(123)
        self.assertIsNotNone(seed)
        self.assertTrue(hasattr(self.env, 'np_random'))

    def test_render(self):
        """Test that the render method executes without errors."""
        try:
            self.env.render()
        except Exception as e:
            self.fail(f"render() raised an exception: {e}")

    def test_close(self):
        """Test that the close method executes without errors."""
        try:
            self.env.close()
        except Exception as e:
            self.fail(f"close() raised an exception: {e}")

    def test_step_with_exception(self):
        """Test that the step function handles exceptions from game logic gracefully."""
        self.env.reset()
        with patch.object(self.env.game, 'process_action', side_effect=Exception("Test Error")):
            obs, reward, terminated, truncated, info = self.env.step((6, 0)) # A valid action
            self.assertLess(reward, 0)
            self.assertIn("error", info)
            self.assertEqual(info["error"], "Test Error")

    def test_bankruptcy_with_multiple_players(self):
        """Test that one player can go bankrupt while the game continues with >2 players."""
        # This test requires 3 players.
        self.env = MonopolyEnv(board_json_path=self.test_json, num_players=3, max_steps=50)
        
        owner = self.env.game.players[0]
        renter = self.env.game.players[1]
        
        # Give property 1 to the owner
        prop_id = 1
        property_meta = self.env.game.board.get_property_meta(prop_id)
        rent = property_meta['rent']['base']
        
        owner.assets.add(prop_id)
        self.env.game.board.purchase_property(prop_id, 1)

        # Set the renter up for failure
        renter.current_cash = rent - 1 
        renter.assets.clear()

        # Manually trigger the landing
        self.env.game.current_player_index = 1
        renter.current_position = prop_id
        
        # Force the rent event deterministically
        self.env.game._handle_landing(renter)

        # Let the env process one dummy step so it sees statuses
        obs, reward, terminated, truncated, info = self.env.step((6,0))

        self.assertEqual(renter.status, 'lost')
        self.assertNotEqual(owner.status, 'won') # Game should not be over
        self.assertFalse(terminated)

    def test_observation_space_with_trade(self):
        """Test that the observation space correctly reflects a pending trade."""
        self.env.reset()
        # Manually create a pending trade
        trade = {
            "from_player": self.env.game.players[0],
            "to_player": self.env.game.players[1],
            "property_index": 0,
            "offer_price": 100
        }
        self.env.game.pending_trade = trade
        
        obs = self.env._get_obs()
        
        self.assertEqual(obs['pending_trade_valid'], 1)
        self.assertNotEqual(np.sum(obs['trade_details']), 0) # Details should be populated
        # Check if the 'from' player in the trade details is correct (player 0)
        self.assertAlmostEqual(obs['trade_details'][0], 0.0)


if __name__ == '__main__':
    unittest.main()
