# tests/test_monopoly_env.py

import unittest
import os
import json
import numpy as np

from monopoly_env.envs.monopoly_env import MonopolyEnv


class TestMonopolyEnv(unittest.TestCase):
    def setUp(self):
        """
        Set up a temporary board JSON file with dummy property data and initialize the environment.
        """
        # Create a temporary JSON file for board data.
        self.test_json = "test_board_data.json"
        properties = []
        # Create 28 dummy properties with minimal required fields.
        for i in range(1, 29):
            properties.append({
                "id": i,
                "name": f"Property {i}",
                "color_group": "red" if i <= 3 else "blue" if i <= 6 else "green" if i <= 9 else "yellow",
                "type": "street",
                "price": 60,
                "house_cost": 50,
                "mortgage_value": 30,
                "rent": {"base": 2}
            })
        board_data = {"properties": properties}
        with open(self.test_json, "w") as f:
            json.dump(board_data, f)

        # Initialize the environment using the dummy board JSON.
        self.env = MonopolyEnv(board_json_path=self.test_json,
                               player_names=["Alice", "Bob", "Charlie", "Diana"],
                               max_steps=10)

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
        obs = self.env.reset()
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
        obs, reward, done, info = self.env.step(action)
        self.assertIsInstance(obs, dict)
        self.assertIn("player", obs)
        self.assertIn("board", obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_episode_termination(self):
        """
        Test that the environment eventually terminates.
        """
        self.env.reset()
        done = False
        step_count = 0
        # Use an action that typically won't change much (e.g., skip turn: top-level action 6).
        action = (6, 0)
        while not done and step_count < 20:
            _, _, done, _ = self.env.step(action)
            step_count += 1
        self.assertTrue(done or step_count == self.env.max_steps)

    def test_invalid_action(self):
        """
        Test that providing an invalid action (e.g., negative values) results in a negative reward.
        """
        self.env.reset()
        # Provide an invalid action with negative values.
        action = (-1, -1)
        obs, reward, done, info = self.env.step(action)
        # Expect a negative reward (error is caught and -10.0 is returned).
        self.assertLess(reward, 0)
        self.assertIn("error", info)


if __name__ == '__main__':
    unittest.main()
