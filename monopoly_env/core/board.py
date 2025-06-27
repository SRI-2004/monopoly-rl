import json
import numpy as np
import os
import random


class Board:
    """
    Board representation for the simplified Monopoly environment.

    The board has 28 property locations, each represented by 8 dimensions:
      - Owner (4 dims): One-hot vector indicating who owns the property.
                         Convention: index 0 = bank, indices 1-3 = players.
      - Mortgaged flag (1 dim): 0 or 1.
      - Monopoly flag (1 dim): 0 or 1.
      - House/Hotel count (2 dims): Fractions representing houses and hotels built.
    Total dimensions: 28 * 8 = 224.
    """

    def __init__(self, json_filepath, num_players=3):
        """
        Initialize the Board.

        Parameters:
            json_filepath (str): Path to the JSON file containing board details.
            num_players (int): The number of players in the game.
        """
        self.json_filepath = json_filepath
        self.board_layout = self._load_board_layout(json_filepath)
        self.board_map = {space['id']: space for space in self.board_layout}
        self.board_meta = {str(space['id']): space for space in self.board_layout}
        
        # Filter the full layout to get just the buyable properties for the state vector
        self.properties_meta = [p for p in self.board_layout if p.get("price")]
        
        # Create mappings for quick lookups, essential for the new reward calculator
        self.properties_meta_by_id = {p['id']: p for p in self.properties_meta}
        self.property_id_to_state_idx = {prop["id"]: i for i, prop in enumerate(self.properties_meta)}

        # Create a mapping from property ID to its index in our state array (of 28 properties)
        self.property_id_to_index = {prop["id"]: i for i, prop in enumerate(self.properties_meta)}
        self.num_properties = len(self.properties_meta)
        self.num_dims = 8  # 8 dimensions per property.
        self.num_owners = num_players + 1  # One-hot vector: [bank, player1, player2, ...].

        # Load card decks
        self.chance_cards = []
        self.community_chest_cards = []
        self._load_cards(json_filepath)
        self.chance_discard = []
        self.community_chest_discard = []

        # Initialize the board state.
        self.reset()

    def _load_board_layout(self, filepath):
        """Load the full board layout from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)
        # Sort by id to ensure order
        return sorted(data.get("board_layout", []), key=lambda x: x["id"])

    def _load_cards(self, filepath):
        """Load the card decks from the JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.chance_cards = data.get("chance_cards", [])
        self.community_chest_cards = data.get("community_chest_cards", [])

    def shuffle_decks(self):
        """Shuffle both card decks."""
        # Move any discarded cards back into the main decks
        self.chance_cards.extend(self.chance_discard)
        self.community_chest_cards.extend(self.community_chest_discard)
        self.chance_discard = []
        self.community_chest_discard = []
        
        random.shuffle(self.chance_cards)
        random.shuffle(self.community_chest_cards)

    def get_property_meta_by_board_id(self, board_id):
        """
        Retrieve the metadata for any space on the board given its ID (position).
        """
        return self.board_map.get(board_id)

    def reset(self):
        """
        Reset the board state to its initial configuration.

        For each property:
          - Owner: bank (one-hot vector: [1, 0, 0, 0])
          - Mortgage flag: 0
          - Monopoly flag: 0
          - House fraction: 0.0
          - Hotel fraction: 0.0
        """
        # Create the state array: shape (num_properties, num_dims)
        self.state = np.zeros((self.num_properties, self.num_dims), dtype=np.float32)

        for idx, prop in enumerate(self.properties_meta):
            self.property_id_to_index[prop["id"]] = idx
            # Set owner: bank owns initially.
            owner_vector = np.zeros(self.num_owners, dtype=np.float32)
            owner_vector[0] = 1  # Bank represented at index 0.
            self.state[idx, 0:self.num_owners] = owner_vector
            # Mortgage flag at index 4: initially not mortgaged.
            self.state[idx, 4] = 0
            # Monopoly flag at index 5: initially false.
            self.state[idx, 5] = 0
            # House fraction at index 6 and hotel fraction at index 7: initially 0.
            self.state[idx, 6] = 0.0
            self.state[idx, 7] = 0.0

        self.shuffle_decks()

    def get_state(self, flatten=True):
        """
        Return the current board state.

        Parameters:
            flatten (bool): If True, returns a 1D array of length 224;
                            otherwise returns a 2D array of shape (28, 8).
        """
        if flatten:
            return self.state.flatten()
        return self.state

    def get_property_meta(self, property_id):
        """
        Retrieve the metadata for a property given its ID.

        Parameters:
            property_id (int): The unique ID of the property.

        Returns:
            dict: Property details from the JSON meta.
        """
        if property_id not in self.property_id_to_index:
            raise ValueError(f"Property id {property_id} not found in the state vector.")
        
        # Find the property in the original list of 28.
        idx = self.property_id_to_index.get(property_id)
        return self.properties_meta[idx]

    def purchase_property(self, property_id, player_id):
        """
        Update the board state to reflect a property purchase.

        Parameters:
            property_id (int): The unique ID of the property to purchase.
            player_id (int): The ID of the player purchasing the property.
                             Should be in the range [1, num_owners-1] (0 is reserved for bank).

        Raises:
            ValueError: If player_id is invalid.
            Exception: If the property is not owned by the bank.
        """
        if player_id < 1 or player_id >= self.num_owners:
            raise ValueError(f"Invalid player_id. Must be between 1 and {self.num_owners - 1}.")

        if property_id not in self.property_id_to_index:
            raise ValueError(f"Property id {property_id} not found in the state vector.")
            
        idx = self.property_id_to_index.get(property_id)

        current_owner = self.state[idx, 0:self.num_owners]
        # Only allow purchase if the bank currently owns the property.
        bank_owner_vector = np.zeros(self.num_owners, dtype=np.float32)
        bank_owner_vector[0] = 1
        if not np.array_equal(current_owner, bank_owner_vector):
            raise Exception("Property is already owned by someone.")

        # Update owner: set all zeros then mark the player's index.
        new_owner = np.zeros(self.num_owners, dtype=np.float32)
        new_owner[player_id] = 1
        self.state[idx, 0:self.num_owners] = new_owner

    def mortgage_property(self, property_id):
        """
        Set the mortgage flag for the specified property.

        Parameters:
            property_id (int): The property to mortgage.
        """
        if property_id not in self.property_id_to_index:
            raise ValueError(f"Property id {property_id} not found in the state vector.")
        idx = self.property_id_to_index.get(property_id)
        self.state[idx, 4] = 1  # Mortgage flag.

    def unmortgage_property(self, property_id):
        """
        Remove the mortgage from a property.

        Parameters:
            property_id (int): The property to unmortgage.
        """
        if property_id not in self.property_id_to_index:
            raise ValueError(f"Property id {property_id} not found in the state vector.")
        idx = self.property_id_to_index.get(property_id)
        self.state[idx, 4] = 0

    def set_buildings(self, property_id, house_fraction, hotel_fraction):
        """
        Set the fractions for houses and hotels on a property.

        Parameters:
            property_id (int): The property to update.
            house_fraction (float): Fraction (0.0 to 1.0) representing houses built.
            hotel_fraction (float): Fraction (0.0 to 1.0) representing hotels built.

        Raises:
            ValueError: If the provided fractions are outside [0.0, 1.0].
        """
        if property_id not in self.property_id_to_index:
            raise ValueError(f"Property id {property_id} not found in the state vector.")
        idx = self.property_id_to_index.get(property_id)

        if not (0.0 <= house_fraction <= 1.0 and 0.0 <= hotel_fraction <= 1.0):
            raise ValueError("house_fraction and hotel_fraction must be between 0.0 and 1.0")

        self.state[idx, 6] = house_fraction
        self.state[idx, 7] = hotel_fraction

    def update_monopoly_flag(self, color_group):
        """
        Check all properties in a given color group.
        If all properties are owned by the same non-bank owner and none are mortgaged,
        set their monopoly flag to 1; otherwise, reset it to 0.

        Parameters:
            color_group (str): The color group to check (e.g., "brown", "blue").
        """
        # Get indices of properties in the specified color group.
        indices = [
            self.property_id_to_index[prop["id"]] for prop in self.properties_meta
            if prop.get("color_group") == color_group
        ]
        if not indices:
            return # No properties of this color, or they are not in the state vector.

        # Check owners and mortgage status.
        owners = []
        for idx in indices:
            owner_vector = self.state[idx, 0:self.num_owners]
            owner = np.argmax(owner_vector)  # Find the owner index.
            owners.append(owner)
            # If any property is mortgaged, monopoly cannot be established.
            if self.state[idx, 4] == 1:
                for idx2 in indices:
                    self.state[idx2, 5] = 0
                return

        # All properties must be owned by the same player (and not the bank, i.e. owner != 0).
        if len(set(owners)) == 1 and owners[0] != 0:
            for idx in indices:
                self.state[idx, 5] = 1
        else:
            for idx in indices:
                self.state[idx, 5] = 0

    def get_board_meta(self):
        """Returns the full board metadata dictionary."""
        return self.board_meta


