# tests/test_core.py

import unittest
import os
import json
import numpy as np

from monopoly_env.core.board import Board
from monopoly_env.core.player import Player
from monopoly_env.core.game_logic import GameLogic


class TestBoard(unittest.TestCase):
    def setUp(self):
        # Create a dummy board JSON file with 28 properties.
        self.test_json = "test_board_data.json"
        properties = []
        for i in range(1, 29):
            properties.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue" if i <= 6 else "green" if i <= 9 else "yellow",
                "type": "street",
                "price": 100 + i,  # Distinct prices.
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2},
                # For testing, set a simple position equal to id.
                "position": i
            })
        board_data = {"properties": properties}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)
        self.board = Board(self.test_json)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_board_reset(self):
        """Test that after reset each property has initial values."""
        self.board.reset()
        state = self.board.get_state(flatten=False)
        self.assertEqual(state.shape, (28, 8))
        for property_state in state:
            owner_vector = property_state[0:4]
            # Owner vector should be one-hot for bank: [1, 0, 0, 0]
            self.assertTrue(np.array_equal(owner_vector, np.array([1, 0, 0, 0], dtype=np.float32)))
            self.assertEqual(property_state[4], 0)  # Mortgage flag.
            self.assertEqual(property_state[5], 0)  # Monopoly flag.
            self.assertEqual(property_state[6], 0)  # House fraction.
            self.assertEqual(property_state[7], 0)  # Hotel fraction.

    def test_purchase_property(self):
        """Test that purchasing a property updates the owner vector correctly."""
        self.board.reset()
        self.board.purchase_property(1, 1)  # Player with id 1 buys property id 1.
        idx = self.board.property_id_to_index[1]
        owner_vector = self.board.state[idx, 0:4]
        expected_owner = np.array([0, 1, 0, 0], dtype=np.float32)
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
        # Create a dummy board JSON file.
        self.test_json = "test_board_data.json"
        properties = []
        for i in range(1, 29):
            properties.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue" if i <= 6 else "green" if i <= 9 else "yellow",
                "type": "street",
                "price": 100 + i,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2},
                "position": i
            })
        board_data = {"properties": properties}
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
            # sub_action = 0 decodes to property index 0 (property id 1) with target and price tier = 0.
            self.game_logic.make_trade_offer_sell(current_player, 0)


class TestGameLogicTrade(unittest.TestCase):
    def setUp(self):
        # Create a dummy board JSON file.
        self.test_json = "test_board_data.json"
        properties = []
        for i in range(1, 29):
            properties.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue" if i <= 6 else "green" if i <= 9 else "yellow",
                "type": "street",
                "price": 100 + i,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2},
                "position": i
            })
        board_data = {"properties": properties}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)

        self.players = [Player("Alice"), Player("Bob"), Player("Charlie"), Player("Diana")]
        self.game_logic = GameLogic(self.test_json, self.players)
        # Let Alice own property id 1.
        self.players[0].assets.add(1)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_trade_offer_sell_valid(self):
        current_player = self.players[0]
        # sub_action = 0 decodes to target_index=0, property_index=0, price_tier=0.
        result = self.game_logic.make_trade_offer_sell(current_player, 0)
        self.assertIn("Trade Offer (Sell)", result)


if __name__ == '__main__':
    unittest.main()
