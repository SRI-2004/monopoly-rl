# tests/test_core.py

import unittest
import os
import json
import numpy as np
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monopoly_env.core.board import Board
from monopoly_env.core.player import Player
from monopoly_env.core.game_logic import GameLogic



class TestBoard(unittest.TestCase):
    def setUp(self):
        # Create a dummy board JSON file with the unified layout.
        self.test_json = "test_board_data.json"
        board_layout = []
        for i in range(1, 29):
            board_layout.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue",
                "type": "street",
                "price": 100 + i,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2}
            })
        board_data = {"board_layout": board_layout}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)
        # Pass the number of players to the constructor.
        self.num_players = 4
        self.board = Board(self.test_json, num_players=self.num_players)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_board_reset(self):
        """Test that after reset each property has initial values."""
        self.board.reset()
        state = self.board.get_state(flatten=False)
        self.assertEqual(state.shape, (28, 8))
        for property_state in state:
            owner_vector = property_state[0:self.num_players + 1]
            expected_bank_owner = np.zeros(self.num_players + 1)
            expected_bank_owner[0] = 1
            self.assertTrue(np.array_equal(owner_vector, expected_bank_owner.astype(np.float32)))
            # Assuming the rest of the indices are fixed relative to the end
            self.assertEqual(property_state[5], 0)  # Monopoly flag
            self.assertEqual(property_state[6], 0)  # House fraction
            self.assertEqual(property_state[7], 0)  # Hotel fraction

    def test_purchase_property(self):
        """Test that purchasing a property updates the owner vector correctly."""
        self.board.reset()
        self.board.purchase_property(1, 1)  # Player with id 1 buys property id 1.
        idx = self.board.property_id_to_index[1]
        owner_vector = self.board.state[idx, 0:self.num_players + 1]
        expected_owner = np.zeros(self.num_players + 1, dtype=np.float32)
        expected_owner[1] = 1
        self.assertTrue(np.array_equal(owner_vector, expected_owner))

    def test_mortgage_and_unmortgage(self):
        """Test that mortgage and unmortgage functions work."""
        self.board.reset()
        self.board.mortgage_property(2)
        idx = self.board.property_id_to_index[2]
        self.assertEqual(self.board.state[idx, 4], 1)
        self.board.unmortgage_property(2)
        self.assertEqual(self.board.state[idx, 4], 0)

    def test_set_buildings(self):
        """Test setting house/hotel fractions."""
        self.board.reset()
        self.board.set_buildings(3, 0.5, 0.0)
        idx = self.board.property_id_to_index[3]
        self.assertAlmostEqual(self.board.state[idx, 6], 0.5)
        self.assertAlmostEqual(self.board.state[idx, 7], 0.0)

    def test_update_monopoly_flag(self):
        """Test that when all properties in a color group are owned, the monopoly flag is updated."""
        self.board.reset()
        # For color "red", assume properties with id 1,2,3.
        for prop_id in [1, 2, 3]:
            self.board.purchase_property(prop_id, 1)
        self.board.update_monopoly_flag("red")
        for prop_id in [1, 2, 3]:
            idx = self.board.property_id_to_index[prop_id]
            self.assertEqual(self.board.state[idx, 5], 1)


class TestPlayer(unittest.TestCase):
    def setUp(self):
        self.player = Player("TestPlayer")

    def test_state_vector_length(self):
        state_vec = self.player.get_state_vector()
        self.assertEqual(state_vec.shape, (16,))

    def test_move_and_cash(self):
        self.player.move(10)
        self.assertEqual(self.player.current_position, 10)
        initial_cash = self.player.current_cash
        self.player.add_cash(100)
        self.assertEqual(self.player.current_cash, initial_cash + 100)
        self.player.subtract_cash(50)
        self.assertEqual(self.player.current_cash, initial_cash + 50)


class TestGameLogic(unittest.TestCase):
    def setUp(self):
        # Create a dummy board JSON file using the unified layout.
        self.test_json = "test_board_data.json"
        board_layout = []
        for i in range(1, 29):
            board_layout.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue",
                "type": "street",
                "price": 100 + i,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2}
            })
        board_data = {"board_layout": board_layout}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)

        # Create dummy players.
        self.players = [Player("Alice"), Player("Bob"), Player("Charlie"), Player("Diana")]
        self.game_logic = GameLogic(self.test_json, self.players)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_roll_dice(self):
        roll = self.game_logic.roll_dice()
        self.assertTrue(2 <= roll <= 12)

    def test_conclude_phase_pre_roll(self):
        # Set current player's phase to pre-roll and position to 0.
        current_player = self.players[self.game_logic.current_player_index]
        current_player.update_phase("pre-roll")
        current_player.current_position = 0
        result = self.game_logic.conclude_phase(current_player)
        self.assertEqual(current_player.phase, "post-roll")
        self.assertNotEqual(current_player.current_position, 0)
        self.assertIn("rolled", result)

    def test_trade_offer_sell_invalid(self):
        # Test that a trade offer sell raises an exception if the player does not own the property.
        current_player = self.players[self.game_logic.current_player_index]
        with self.assertRaises(Exception):
            # sub_action = 0 decodes to property index 0. Player does not own it.
            self.game_logic.make_trade_offer_sell(current_player, 0)


class TestGameLogicTrade(unittest.TestCase):
    def setUp(self):
        # Create a dummy board JSON file using the unified layout.
        self.test_json = "test_board_data.json"
        board_layout = []
        for i in range(1, 29):
            board_layout.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue",
                "type": "street",
                "price": 100 + i,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2}
            })
        board_data = {"board_layout": board_layout}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)

        self.players = [Player("Alice"), Player("Bob"), Player("Charlie"), Player("Diana")]
        self.game_logic = GameLogic(self.test_json, self.players)
        # Let Alice own property id 1.
        # We also need to update the board state to reflect this ownership for trade checks
        self.players[0].assets.add(1)
        self.game_logic.board.purchase_property(1, 1) # player_id 1 owns prop 1

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_trade_offer_sell_valid(self):
        current_player = self.players[0]
        # sub_action = 0 decodes to target_index=0 (Bob), property_index=0 (Prop 1), price_tier=0.
        # This should now be valid since Alice owns property 1.
        result = self.game_logic.make_trade_offer_sell(current_player, 0)
        self.assertIn("Trade Offer (Sell)", result)


class TestGameLogicActions(unittest.TestCase):
    def setUp(self):
        self.test_json = "test_board_data.json"
        board_layout = []
        # Add a "Go to Jail" space for testing jail logic
        board_layout.append({"id": 30, "name": "Go to Jail", "type": "go_to_jail"})
        board_layout.append({"id": 10, "name": "Jail", "type": "jail"})
        
        for i in range(1, 29):
            board_layout.append({
                "id": i, "name": f"Property {i}", "color_group": "red", "type": "street",
                "price": 100, "house_cost": 50, "mortgage_value": 30, "rent": {"base": 2}
            })
        
        board_data = {"board_layout": sorted(board_layout, key=lambda x: x['id'])}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)

        self.players = [Player("Alice"), Player("Bob")]
        self.game_logic = GameLogic(self.test_json, self.players)
        self.game_logic.current_player_index = 0
        self.current_player = self.players[0]

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_buy_property(self):
        """Test that a player can buy an unowned property."""
        self.current_player.current_position = 1
        self.current_player.set_option_to_buy(True)
        self.current_player.update_phase('post-roll') # Player must be in post-roll to buy
        initial_cash = self.current_player.current_cash
        
        result = self.game_logic.buy_property(self.current_player, 1) # sub_action 1 means 'yes'
        
        self.assertIn("purchased", result)
        self.assertEqual(self.current_player.current_cash, initial_cash - 100)
        self.assertIn(1, self.current_player.assets)
        self.assertFalse(self.current_player.can_buy_property())

    def test_sell_property(self):
        """Test that a player can sell a property to the bank."""
        self.current_player.assets.add(1)
        initial_cash = self.current_player.current_cash
        
        result = self.game_logic.sell_property(self.current_player, 0) # sub_action 0 is property index 0 (id 1)
        
        self.assertIn("sold property", result)
        self.assertEqual(self.current_player.current_cash, initial_cash + (30 * 2)) # Sells for 2x mortgage
        self.assertNotIn(1, self.current_player.assets)

    def test_mortgage_property(self):
        """Test that a player can mortgage a property."""
        self.current_player.assets.add(1)
        
        result = self.game_logic.mortgage_free_mortgage(self.current_player, 0)
        
        self.assertIn("mortgaged", result)
        idx = self.game_logic.board.property_id_to_index[1]
        self.assertEqual(self.game_logic.board.state[idx, 4], 1)

    def test_improve_property(self):
        """Test that a player can build a house on a property."""
        self.current_player.assets.add(1) # Owns property 1
        initial_cash = self.current_player.current_cash
        
        # In this test setup, prop with id 1 is the first "street" type
        result = self.game_logic.improve_property(self.current_player, 0) # sub_action 0 is prop_idx 0, building type 0 (house)
        
        self.assertIn("improved property", result)
        self.assertEqual(self.current_player.current_cash, initial_cash - 50)
        idx = self.game_logic.board.property_id_to_index[1]
        self.assertAlmostEqual(self.game_logic.board.state[idx, 6], 0.25)

    def test_conclude_phase_transitions(self):
        """Test the phase transitions from post-roll to out-of-turn and to the next player."""
        # Test post-roll -> out-of-turn
        self.current_player.update_phase('post-roll')
        result = self.game_logic.conclude_phase(self.current_player)
        self.assertIn("concluded post-roll", result)
        self.assertEqual(self.current_player.phase, "out-of-turn")
        
        # Test out-of-turn -> next player's pre-roll
        result = self.game_logic.conclude_phase(self.current_player)
        self.assertIn("Out-of-turn concluded", result)
        self.assertEqual(self.game_logic.current_player_index, 1) # Moved to next player
        self.assertEqual(self.players[1].phase, 'pre-roll') # Next player is in pre-roll

    def test_skip_turn(self):
        """Test that skipping a turn concludes the current phase."""
        self.current_player.update_phase("pre-roll")
        self.game_logic.skip_turn(self.current_player)
        self.assertEqual(self.current_player.phase, 'post-roll')

    def test_get_valid_actions_in_jail(self):
        """Test that a player in jail has the correct valid actions."""
        self.current_player.go_to_jail()
        valid_actions = self.game_logic.get_valid_actions(self.current_player)
        # Actions 8 (Use Get Out of Jail) and 9 (Pay Jail Fine) should be true.
        self.assertTrue(valid_actions[8])
        self.assertTrue(valid_actions[9])
        # All other actions should be false.
        self.assertEqual(sum(valid_actions), 2)

    def test_get_valid_actions_trade_pending(self):
        """Test that a player with a pending trade has the correct valid actions."""
        # Create a pending trade for the second player
        self.game_logic.pending_trade = {"to_player": self.players[1]}
        valid_actions = self.game_logic.get_valid_actions(self.players[1])
        # Action 11 (Respond to Trade) and 6 (Skip) should be true.
        self.assertTrue(valid_actions[11])
        self.assertTrue(valid_actions[6])
        self.assertEqual(sum(valid_actions), 2)

    def test_get_valid_subactions_buy_property(self):
        """Test the sub-action mask for buying a property."""
        # Player is not in a state to buy
        mask = self.game_logic.get_valid_subactions(self.current_player, 10) # Action 10: Buy
        self.assertFalse(mask.any())

        # Player is in a state to buy
        self.current_player.set_option_to_buy(True)
        self.current_player.update_phase('post-roll') # Also need to be in post-roll phase
        mask = self.game_logic.get_valid_subactions(self.current_player, 10)
        self.assertTrue(mask.all()) # Both 'yes' and 'no' should be valid
        self.assertEqual(len(mask), 2)


if __name__ == '__main__':
    unittest.main()
