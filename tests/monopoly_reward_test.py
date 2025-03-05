# tests/test_reward_calculator.py

import unittest
import os
import json
import numpy as np

from monopoly_env.core.board import Board
from monopoly_env.core.player import Player
from monopoly_env.utils.reward_calculator import RewardCalculator


class TestRewardCalculator(unittest.TestCase):
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
                "price": 100,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2}
            })
        board_data = {"properties": properties}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)
        self.board = Board(self.test_json)
        self.players = [Player("Alice"), Player("Bob"), Player("Charlie"), Player("Diana")]
        self.rc = RewardCalculator(self.board, self.players)

    def tearDown(self):
        if os.path.exists(self.test_json):
            os.remove(self.test_json)

    def test_compute_networth_undeveloped(self):
        # No properties owned: net worth should equal cash (default 1500) with no penalty.
        networth = self.rc.compute_networth(self.players[0])
        self.assertAlmostEqual(networth, 1500)

    def test_compute_networth_with_property(self):
        # Let Alice own an undeveloped property (id 1).
        self.players[0].assets.add(1)
        networth = self.rc.compute_networth(self.players[0])
        # For undeveloped property: contribution = (price - mortgage) * multiplier (multiplier 1.25 if not full set).
        contribution = (100 - 30) * 1.25
        expected = 1500 + contribution
        self.assertAlmostEqual(networth, expected)

    def test_compute_dense_reward(self):
        current_player = self.players[0]
        initial_reward = self.rc.compute_dense_reward(current_player)
        # Simulate a net worth increase.
        current_player.add_cash(100)
        reward_after = self.rc.compute_dense_reward(current_player)
        self.assertNotEqual(initial_reward, reward_after)

    def test_compute_sparse_reward_win(self):
        current_player = self.players[0]
        current_player.update_status("won")
        sparse = self.rc.compute_sparse_reward(current_player)
        self.assertGreater(sparse, 0)

    def test_compute_sparse_reward_lost(self):
        current_player = self.players[0]
        current_player.update_status("lost")
        sparse = self.rc.compute_sparse_reward(current_player)
        self.assertLess(sparse, 0)


if __name__ == '__main__':
    unittest.main()
