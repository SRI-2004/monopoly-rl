import random
from .board import Board
from .player import Player
import numpy as np


class GameLogic:
    """
    Implements the core game logic for our simplified Monopoly environment.

    It links the board (loaded from a JSON file) and the players, and provides methods
    to process actions from the agent. The top-level action space (12 discrete choices)
    and sub-action parameter spaces are decoded here.
    """

    def __init__(self, board_json_path, players):
        """
        Initialize the game logic.

        Parameters:
            board_json_path (str): Path to the JSON file with board and card details.
            players (list): A list of Player objects.
        """
        self.board = Board(board_json_path)
        self.players = players
        self.current_player_index = 0  # index into self.players
        self.pending_trade = None  # Holds pending trade offer details

    def roll_dice(self):
        """
        Roll two six-sided dice and return the sum.
        """
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        return dice1 + dice2

    def process_action(self, action, sub_action=None):
        """
        Process a top-level action with an optional sub-action parameter.

        The action parameter is an integer (1 to 12) corresponding to:
          1: Make Trade Offer (Sell)
          2: Make Trade Offer (Buy)
          3: Improve Property
          4: Sell House/Hotel
          5: Sell Property
          6: Mortgage/Free Mortgage
          7: Skip Turn
          8: Conclude
          9: Use Get Out of Jail
          10: Pay Jail Fine
          11: Buy Property
          12: Respond to Trade

        Parameters:
            action (int): Top-level action index.
            sub_action (int, optional): The discrete parameter value (if required).

        Returns:
            A string message summarizing the outcome.
        """
        current_player = self.players[self.current_player_index]
        if action == 1:
            return self.make_trade_offer_sell(current_player, sub_action)
        elif action == 2:
            return self.make_trade_offer_buy(current_player, sub_action)
        elif action == 3:
            return self.improve_property(current_player, sub_action)
        elif action == 4:
            return self.sell_house_hotel(current_player, sub_action)
        elif action == 5:
            return self.sell_property(current_player, sub_action)
        elif action == 6:
            return self.mortgage_free_mortgage(current_player, sub_action)
        elif action == 7:
            return self.skip_turn(current_player)
        elif action == 8:
            return self.conclude_phase(current_player)
        elif action == 9:
            return self.use_get_out_of_jail(current_player)
        elif action == 10:
            return self.pay_jail_fine(current_player)
        elif action == 11:
            return self.buy_property(current_player, sub_action)
        elif action == 12:
            return self.respond_to_trade(current_player, sub_action)
        else:
            raise ValueError("Invalid action index.")

    def get_valid_actions(self, player):
        """
        Determine and return a mask for the 12 top-level actions based on the current player's state.

        Returns:
            A list of booleans of length 12, where True indicates the corresponding action is valid.

        Action indices (0-indexed):
            0: Make Trade Offer (Sell)
            1: Make Trade Offer (Buy)
            2: Improve Property
            3: Sell House/Hotel
            4: Sell Property
            5: Mortgage/Free Mortgage
            6: Skip Turn
            7: Conclude
            8: Use Get Out of Jail
            9: Pay Jail Fine
            10: Buy Property
            11: Respond to Trade
        """
        valid_actions = [False] * 12

        # If the player is in jail, only allow actions 9 (Use Get Out of Jail) and 10 (Pay Jail Fine).
        if player.currently_in_jail:
            valid_actions[8] = True  # Use Get Out of Jail (action index 9)
            valid_actions[9] = True  # Pay Jail Fine (action index 10)
            return valid_actions

        # If a trade is pending and is addressed to the player, only allow Respond to Trade.
        if self.pending_trade is not None and self.pending_trade["to_player"] == player:
            valid_actions[11] = True  # Respond to Trade (action index 12)
            # Optionally, you might allow Skip Turn as well.
            valid_actions[6] = True  # Skip Turn
            return valid_actions

        # Evaluate valid actions based on player's phase.
        if player.phase == 'pre-roll':
            # Allow trade offers.
            valid_actions[0] = True  # Make Trade Offer (Sell)
            valid_actions[1] = True  # Make Trade Offer (Buy)
            # Allow property improvements.
            valid_actions[2] = True  # Improve Property
            valid_actions[3] = True  # Sell House/Hotel
            # Allow mortgaging actions.
            valid_actions[5] = True  # Mortgage/Free Mortgage
            # Allow skipping and concluding phase.
            valid_actions[6] = True  # Skip Turn
            valid_actions[7] = True  # Conclude (which will roll dice)
        elif player.phase == 'post-roll':
            # In post-roll, allow property management.
            valid_actions[0] = True  # Trade Sell
            valid_actions[1] = True  # Trade Buy
            valid_actions[4] = True  # Sell Property to bank
            valid_actions[5] = True  # Mortgage/Free Mortgage
            valid_actions[6] = True  # Skip Turn
            valid_actions[7] = True  # Conclude (transition phase)
            # Allow buying the property if landed on one.
            valid_actions[10] = True  # Buy Property
        elif player.phase == 'out-of-turn':
            # In out-of-turn, primarily respond to trades.
            valid_actions[11] = True  # Respond to Trade
            valid_actions[6] = True  # Skip Turn (if no trade response is needed)
            valid_actions[7] = True  # Conclude phase (to move to next player's turn)
        else:
            # Unknown phase; no actions allowed.
            pass

        return valid_actions

    def get_valid_subactions(self, player, top_action):
        """
        Given a player and a top-level action index (0-indexed), return a boolean mask for the valid subactions.

        The sub-action space dimensions for each top-level action are defined as:
          0: Make Trade Offer (Sell) -> 252 = 3 players x 28 properties x 3 price tiers
          1: Make Trade Offer (Buy)  -> 252 = 3 players x 28 properties x 3 price tiers
          2: Improve Property        -> 44  = 22 properties x 2 building types (house/hotel)
          3: Sell House/Hotel        -> 44  = 22 properties x 2 building types
          4: Sell Property           -> 28  = one-hot over 28 properties
          5: Mortgage/Free Mortgage  -> 28  = one-hot over 28 properties
          6: Skip Turn               -> 1
          7: Conclude                -> 1
          8: Use Get Out of Jail     -> 1
          9: Pay Jail Fine           -> 1
          10: Buy Property           -> 2   (0: decline, 1: buy)
          11: Respond to Trade       -> 2   (0: Reject, 1: Accept)

        Returns:
            A NumPy boolean array of appropriate length.
        """
        # Define dimensions for each top-level action.
        subaction_dims = {0: 252, 1: 252, 2: 44, 3: 44, 4: 28, 5: 28,
                          6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2}
        dim = subaction_dims.get(top_action, 0)

        board = self.board  # Use self.board directly.

        if top_action == 0:
            # Make Trade Offer (Sell)
            mask = np.zeros((3, 28, 3), dtype=bool)
            for prop_idx, prop in enumerate(board.properties_meta):
                if prop["id"] in player.assets:
                    mask[:, prop_idx, :] = True
            return mask.flatten()
        elif top_action == 1:
            # Make Trade Offer (Buy)
            mask = np.zeros((3, 28, 3), dtype=bool)
            candidate_players = [p for p in self.players if p != player]
            for prop_idx, prop in enumerate(board.properties_meta):
                for cand_idx, cand in enumerate(candidate_players):
                    if prop["id"] in cand.assets:
                        mask[cand_idx, prop_idx, :] = True
            return mask.flatten()
        elif top_action == 2:
            # Improve Property
            eligible_props = [prop for prop in board.properties_meta if prop.get("type") == "street"]
            mask = np.zeros((len(eligible_props), 2), dtype=bool)
            for i, prop in enumerate(eligible_props):
                if prop["id"] in player.assets:
                    idx = board.property_id_to_index[prop["id"]]
                    house_frac = board.state[idx, 6]
                    hotel_frac = board.state[idx, 7]
                    if house_frac < 1.0:
                        mask[i, 0] = True
                    if hotel_frac < 1.0:
                        mask[i, 1] = True
            return mask.flatten()
        elif top_action == 3:
            # Sell House/Hotel
            eligible_props = [prop for prop in board.properties_meta if prop.get("type") == "street"]
            mask = np.zeros((len(eligible_props), 2), dtype=bool)
            for i, prop in enumerate(eligible_props):
                if prop["id"] in player.assets:
                    idx = board.property_id_to_index[prop["id"]]
                    house_frac = board.state[idx, 6]
                    hotel_frac = board.state[idx, 7]
                    if house_frac > 0:
                        mask[i, 0] = True
                    if hotel_frac > 0:
                        mask[i, 1] = True
            return mask.flatten()
        elif top_action == 4:
            # Sell Property
            mask = np.zeros(28, dtype=bool)
            for i, prop in enumerate(board.properties_meta):
                if prop["id"] in player.assets:
                    mask[i] = True
            return mask
        elif top_action == 5:
            # Mortgage/Free Mortgage
            mask = np.zeros(28, dtype=bool)
            for i, prop in enumerate(board.properties_meta):
                if prop["id"] in player.assets:
                    mask[i] = True
            return mask
        elif top_action in [6, 7, 8, 9]:
            return np.array([True])
        elif top_action == 10:
            # Buy Property: dimension 2 for yes/no.
            if player.can_buy_property():
                return np.array([True, True])
            else:
                return np.array([False, False])
        elif top_action == 11:
            # Respond to Trade
            if self.pending_trade is not None and self.pending_trade["to_player"] == player:
                return np.array([True, True])
            else:
                return np.array([False, False])
        else:
            return np.zeros(dim, dtype=bool)

    # 1. Make Trade Offer (Sell)
    def make_trade_offer_sell(self, player, sub_action):
        """
        Propose selling a property to another player.

        The 252-dim sub_action parameter is decoded as:
          - target_player_index: sub_action // (28 * 3)
          - property_index: (sub_action % (28 * 3)) // 3
          - price_tier_index: sub_action % 3

        Price multipliers are: [0.75, 1.0, 1.25].
        """
        # Decode sub_action
        target_index = sub_action // (28 * 3)
        remainder = sub_action % (28 * 3)
        property_index = remainder // 3
        price_tier_index = remainder % 3
        price_multipliers = [0.75, 1.0, 1.25]
        multiplier = price_multipliers[price_tier_index]

        # Identify target player (among the other three)
        candidate_players = [p for i, p in enumerate(self.players) if i != self.current_player_index]
        if target_index >= len(candidate_players):
            raise ValueError("Invalid target player index in trade offer.")
        target_player = candidate_players[target_index]

        # Get property details from board meta (assumes properties are sorted by id)
        property_meta = self.board.properties_meta[property_index]
        base_price = property_meta.get("price", 0)
        offer_price = int(multiplier * base_price)

        # Check that the offering player owns the property.
        # (Assuming player's assets are tracked as property IDs.)
        if property_meta["id"] not in player.assets:
            raise Exception("Player does not own the property being offered for sale.")

        # Store pending trade offer
        self.pending_trade = {
            "type": "sell",
            "from_player": player,
            "to_player": target_player,
            "property_index": property_index,
            "offer_price": offer_price
        }
        return (f"Trade Offer (Sell): {player.player_name} offers to sell property "
                f"{property_meta['name']} (ID: {property_meta['id']}) to {target_player.player_name} "
                f"for {offer_price}.")

    # 2. Make Trade Offer (Buy)
    def make_trade_offer_buy(self, player, sub_action):
        """
        Propose buying a property from another player.

        The sub_action is decoded in the same way as in the sell-offer.
        """
        target_index = sub_action // (28 * 3)
        remainder = sub_action % (28 * 3)
        property_index = remainder // 3
        price_tier_index = remainder % 3
        price_multipliers = [0.75, 1.0, 1.25]
        multiplier = price_multipliers[price_tier_index]

        candidate_players = [p for i, p in enumerate(self.players) if i != self.current_player_index]
        if target_index >= len(candidate_players):
            raise ValueError("Invalid target player index in trade offer.")
        target_player = candidate_players[target_index]

        property_meta = self.board.properties_meta[property_index]
        base_price = property_meta.get("price", 0)
        offer_price = int(multiplier * base_price)

        # Check that the target player owns the property.
        if property_meta["id"] not in target_player.assets:
            raise Exception("Target player does not own the property being requested for purchase.")

        self.pending_trade = {
            "type": "buy",
            "from_player": player,
            "to_player": target_player,
            "property_index": property_index,
            "offer_price": offer_price
        }
        return (f"Trade Offer (Buy): {player.player_name} offers to buy property "
                f"{property_meta['name']} (ID: {property_meta['id']}) from {target_player.player_name} "
                f"for {offer_price}.")

    # 3. Improve Property
    def improve_property(self, player, sub_action):
        """
        Build a house or hotel on a color-group property.

        The 44-dim sub_action is decoded as:
          - property_index = sub_action // 2 (from 22 eligible properties)
          - building_type: 0 for house, 1 for hotel.
        """
        property_index = sub_action // 2
        building_type_index = sub_action % 2  # 0: house, 1: hotel

        # Among the 28 board properties, assume only those with type "street" are improvable.
        color_properties = [prop for prop in self.board.properties_meta if prop["type"] == "street"]
        if property_index >= len(color_properties):
            raise ValueError("Invalid property index for improvement.")
        property_meta = color_properties[property_index]

        if property_meta["id"] not in player.assets:
            raise Exception("Player does not own this property, cannot improve.")

        # Determine improvement cost. For a house, use house_cost; for a hotel, assume 4× house_cost.
        if building_type_index == 0:
            cost = property_meta.get("house_cost", 0)
            improvement = "house"
        else:
            cost = property_meta.get("house_cost", 0) * 4
            improvement = "hotel"

        if player.current_cash < cost:
            raise Exception("Not enough cash to improve property.")

        player.subtract_cash(cost)

        # Update board state: increment the building fraction.
        idx = self.board.property_id_to_index[property_meta["id"]]
        current_house_fraction = self.board.state[idx, 6]
        current_hotel_fraction = self.board.state[idx, 7]

        if improvement == "house":
            new_house_fraction = min(1.0, current_house_fraction + 0.25)
            self.board.set_buildings(property_meta["id"], new_house_fraction, current_hotel_fraction)
        else:
            new_hotel_fraction = min(1.0, current_hotel_fraction + 1.0)
            self.board.set_buildings(property_meta["id"], current_house_fraction, new_hotel_fraction)

        return (f"{player.player_name} improved property {property_meta['name']} with a {improvement} "
                f"at cost {cost}.")

    # 4. Sell House/Hotel
    def sell_house_hotel(self, player, sub_action):
        """
        Sell a house or hotel from a color-group property.

        The 44-dim sub_action is decoded identically as in improve_property.
        """
        property_index = sub_action // 2
        building_type_index = sub_action % 2  # 0: house, 1: hotel

        color_properties = [prop for prop in self.board.properties_meta if prop["type"] == "street"]
        if property_index >= len(color_properties):
            raise ValueError("Invalid property index for selling improvement.")
        property_meta = color_properties[property_index]

        if property_meta["id"] not in player.assets:
            raise Exception("Player does not own this property, cannot sell improvement.")

        # Assume refund is half the improvement cost.
        if building_type_index == 0:
            refund = int(property_meta.get("house_cost", 0) / 2)
            improvement = "house"
        else:
            refund = int(property_meta.get("house_cost", 0) * 4 / 2)
            improvement = "hotel"

        idx = self.board.property_id_to_index[property_meta["id"]]
        current_house_fraction = self.board.state[idx, 6]
        current_hotel_fraction = self.board.state[idx, 7]

        if building_type_index == 0:
            new_house_fraction = max(0.0, current_house_fraction - 0.25)
            self.board.set_buildings(property_meta["id"], new_house_fraction, current_hotel_fraction)
        else:
            new_hotel_fraction = max(0.0, current_hotel_fraction - 1.0)
            self.board.set_buildings(property_meta["id"], current_house_fraction, new_hotel_fraction)

        player.add_cash(refund)
        return (f"{player.player_name} sold a {improvement} on {property_meta['name']} for {refund}.")

    # 5. Sell Property
    def sell_property(self, player, sub_action):
        """
        Sell a property to the bank.

        The sub_action is a one-hot index (0–27) corresponding to a property.
        """
        property_index = sub_action
        if property_index < 0 or property_index >= len(self.board.properties_meta):
            raise ValueError("Invalid property index for selling property.")

        property_meta = self.board.properties_meta[property_index]
        if property_meta["id"] not in player.assets:
            raise Exception("Player does not own this property, cannot sell.")

        # For example, sale price is double the mortgage value.
        sale_price = property_meta.get("mortgage_value", 0) * 2
        player.add_cash(sale_price)
        player.remove_asset(property_meta["id"])

        # Update board state: revert ownership to bank.
        idx = self.board.property_id_to_index[property_meta["id"]]
        self.board.state[idx, 0:self.board.num_owners] = [1, 0, 0, 0]
        return (f"{player.player_name} sold property {property_meta['name']} to the bank for {sale_price}.")

    # 6. Mortgage/Free Mortgage
    def mortgage_free_mortgage(self, player, sub_action):
        """
        Toggle the mortgage status of a property.

        The sub_action is a one-hot index (0–27) corresponding to a property.
        """
        property_index = sub_action
        if property_index < 0 or property_index >= len(self.board.properties_meta):
            raise ValueError("Invalid property index for mortgage action.")

        property_meta = self.board.properties_meta[property_index]
        if property_meta["id"] not in player.assets:
            raise Exception("Player does not own this property, cannot mortgage/unmortgage.")

        idx = self.board.property_id_to_index[property_meta["id"]]
        if self.board.state[idx, 4] == 0:
            self.board.mortgage_property(property_meta["id"])
            return f"{player.player_name} mortgaged property {property_meta['name']}."
        else:
            self.board.unmortgage_property(property_meta["id"])
            return f"{player.player_name} unmortgaged property {property_meta['name']}."

    # 7. Skip Turn
    def skip_turn(self, player):
        """
        Skip the current turn and automatically conclude the phase.
        """
        result_msg = f"{player.player_name} skipped their turn."
        # Automatically conclude the phase after skipping.
        conclude_msg = self.conclude_phase(player)
        return f"{result_msg} {conclude_msg}"

    # 8. Conclude
    def conclude_phase(self, player):
        """
        End the current phase.

        - In Pre‐roll: automatically roll dice, move the player, and transition to Post‐roll.
        - In Post‐roll: transition to Out‐of‐Turn.
        - In Out‐of‐Turn: finish any pending trades and move to the next player's Pre‐roll.
        """
        if player.phase == 'pre-roll':
            roll = self.roll_dice()
            new_position = (player.current_position + roll) % 40  # standard Monopoly board has 40 positions
            player.move(new_position)
            player.update_phase('post-roll')
            # Check if landed on a purchasable property.
            for prop in self.board.properties_meta:
                # (Assumes each property meta may include a "position" field.)
                if prop.get("id", None) == new_position:
                    idx = self.board.property_id_to_index[prop["id"]]
                    owner_vector = self.board.state[idx, 0:self.board.num_owners]
                    if (owner_vector == [1, 0, 0, 0]).all():
                        player.set_option_to_buy(True)
                        break
            return (f"{player.player_name} rolled {roll} and moved to position {new_position}. "
                    "Phase changed to post-roll.")
        elif player.phase == 'post-roll':
            player.update_phase('out-of-turn')
            return (f"{player.player_name} concluded post-roll. Phase changed to out-of-turn.")
        elif player.phase == 'out-of-turn':
            player.update_phase('pre-roll')
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            return (f"Out-of-turn concluded. Next player is {self.players[self.current_player_index].player_name}.")
        else:
            raise Exception("Unknown phase encountered during conclude_phase.")

    # 9. Use Get Out of Jail
    def use_get_out_of_jail(self, player):
        """
        Use a Get Out of Jail card, if available.
        """
        if player.has_get_out_of_jail_chance_card or player.has_get_out_of_jail_community_chest_card:
            player.has_get_out_of_jail_chance_card = False
            player.has_get_out_of_jail_community_chest_card = False
            player.leave_jail()
            return f"{player.player_name} used a Get Out of Jail card and is now free."
        else:
            raise Exception("Player does not have a Get Out of Jail card.")

    # 10. Pay Jail Fine
    def pay_jail_fine(self, player):
        """
        Pay the jail fine (assumed to be $50) if the player is in jail.
        """
        jail_fine = 50
        if player.currently_in_jail:
            if player.current_cash < jail_fine:
                raise Exception("Not enough cash to pay the jail fine.")
            player.subtract_cash(jail_fine)
            player.leave_jail()
            return f"{player.player_name} paid a jail fine of {jail_fine} and is now free."
        else:
            raise Exception("Player is not in jail.")

    # 11. Buy Property
    def buy_property(self, player, sub_action):
        """
        Purchase an unowned property on which the player just landed.

        The sub_action here is typically a binary flag (e.g., 1 for yes).
        """
        if not player.can_buy_property():
            raise Exception("Buying property is not permitted at this time.")
        if sub_action != 1:
            return f"{player.player_name} chose not to buy the property."

        landed_position = player.current_position
        property_meta = None
        for prop in self.board.properties_meta:
            if prop.get("id", None) == landed_position:
                property_meta = prop
                break

        if property_meta is None:
            raise Exception("No property available for purchase at this position.")

        idx = self.board.property_id_to_index[property_meta["id"]]
        owner_vector = self.board.state[idx, 0:self.board.num_owners]
        if not (owner_vector == [1, 0, 0, 0]).all():
            raise Exception("Property is already owned.")

        price = property_meta.get("price", 0)
        if player.current_cash < price:
            raise Exception("Not enough cash to purchase the property.")

        player.subtract_cash(price)
        # Use Board's purchase_property to update ownership.
        # (Assumes player IDs start at 1; adjust as needed.)
        self.board.purchase_property(property_meta["id"], self.current_player_index + 1)
        player.add_asset(property_meta["id"])
        player.set_option_to_buy(False)
        return f"{player.player_name} purchased {property_meta['name']} for {price}."

    # 12. Respond to Trade
    def respond_to_trade(self, player, sub_action):
        """
        Respond to a pending trade offer.

        The sub_action is binary: 0 = reject, 1 = accept.
        """
        if self.pending_trade is None:
            raise Exception("There is no pending trade offer to respond to.")
        if self.pending_trade["to_player"] != player:
            raise Exception("This trade offer is not addressed to the current player.")

        if sub_action == 1:
            trade = self.pending_trade
            if trade["type"] == "sell":
                # Player (responder) is buying from the offering player.
                offer_price = trade["offer_price"]
                if player.current_cash < offer_price:
                    raise Exception("Not enough cash to accept the trade offer.")
                player.subtract_cash(offer_price)
                trade["from_player"].add_cash(offer_price)

                # Retrieve property metadata.
                property_meta = self.board.properties_meta[trade["property_index"]]
                idx = self.board.property_id_to_index[property_meta["id"]]

                # Update the board's owner vector directly.
                # Assuming owner vector: index 0 = bank, indices 1...N = players.
                new_owner = self.players.index(player) + 1
                new_owner_vector = np.zeros(self.board.num_owners, dtype=np.float32)
                new_owner_vector[new_owner] = 1
                self.board.state[idx, 0:self.board.num_owners] = new_owner_vector

                # Update asset holdings.
                trade["from_player"].remove_asset(property_meta["id"])
                player.add_asset(property_meta["id"])

                result = (f"{player.player_name} accepted the trade and bought {property_meta['name']} "
                          f"for {offer_price}.")
            elif trade["type"] == "buy":
                # Player (responder) is selling to the offering player.
                offer_price = trade["offer_price"]
                if trade["from_player"].current_cash < offer_price:
                    raise Exception("Offering player lacks sufficient cash for the trade.")
                trade["from_player"].subtract_cash(offer_price)
                player.add_cash(offer_price)

                property_meta = self.board.properties_meta[trade["property_index"]]
                idx = self.board.property_id_to_index[property_meta["id"]]

                # Update the board's owner vector directly for the sale.
                new_owner = self.players.index(trade["from_player"]) + 1
                new_owner_vector = np.zeros(self.board.num_owners, dtype=np.float32)
                new_owner_vector[new_owner] = 1
                self.board.state[idx, 0:self.board.num_owners] = new_owner_vector

                # Update asset holdings.
                player.remove_asset(property_meta["id"])
                trade["from_player"].add_asset(property_meta["id"])

                result = (f"{player.player_name} accepted the trade and sold {property_meta['name']} "
                          f"for {offer_price}.")
            else:
                raise Exception("Unknown trade type encountered.")
            self.pending_trade = None
            return result
        elif sub_action == 0:
            self.pending_trade = None
            return f"{player.player_name} rejected the trade offer."
        else:
            raise ValueError("Invalid response value for trade (must be 0 or 1).")
