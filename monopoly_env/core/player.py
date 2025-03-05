import numpy as np


class Player:
    """
    Player representation for the simplified Monopoly environment.

    The player's state is represented using 16 dimensions, constructed as follows:
      0: current_position (int)
      1-4: status one-hot (waiting_for_move, current_move, won, lost)
      5: has_get_out_of_jail_community_chest_card (bool)
      6: has_get_out_of_jail_chance_card (bool)
      7: current_cash (int)
      8: num_railroads_possessed (int)
      9: num_utilities_possessed (int)
     10: currently_in_jail (bool)
     11: is_property_offer_outstanding (bool)
     12: _option_to_buy (bool)
     13-15: phase one-hot (pre-roll, post-roll, out-of-turn)

    Note: The phase and status controls the set of allowed actions. For example, buying a property
    is only allowed in the post-roll phase (and when _option_to_buy is True), while pre-roll may allow
    trades, improvements, or mortgaging.
    """

    # Allowed statuses and phases
    VALID_STATUSES = ['waiting_for_move', 'current_move', 'won', 'lost']
    PHASES = ['pre-roll', 'post-roll', 'out-of-turn']

    def __init__(
            self,
            player_name: str,
            current_position: int = 0,
            status: str = 'waiting_for_move',
            has_get_out_of_jail_community_chest_card: bool = False,
            has_get_out_of_jail_chance_card: bool = False,
            current_cash: int = 1500,
            num_railroads_possessed: int = 0,
            num_utilities_possessed: int = 0,
            assets: set = None,
            full_color_sets_possessed: set = None,
            currently_in_jail: bool = False,
            outstanding_property_offer=None,
            _option_to_buy: bool = False,
            phase: str = 'pre-roll'
    ):
        self.player_name = player_name
        self.current_position = current_position

        if status not in Player.VALID_STATUSES:
            raise ValueError(f"Invalid status provided. Valid statuses: {Player.VALID_STATUSES}")
        self.status = status

        self.has_get_out_of_jail_community_chest_card = has_get_out_of_jail_community_chest_card
        self.has_get_out_of_jail_chance_card = has_get_out_of_jail_chance_card
        self.current_cash = current_cash
        self.num_railroads_possessed = num_railroads_possessed
        self.num_utilities_possessed = num_utilities_possessed
        self.assets = assets if assets is not None else set()
        self.full_color_sets_possessed = full_color_sets_possessed if full_color_sets_possessed is not None else set()
        self.currently_in_jail = currently_in_jail

        self.outstanding_property_offer = outstanding_property_offer
        self.is_property_offer_outstanding = False  # Initially, no offer is outstanding.
        self.mortgaged_assets = set()  # Assets currently mortgaged.
        self._option_to_buy = _option_to_buy  # Flag to denote if buying the landed property is allowed.

        if phase not in Player.PHASES:
            raise ValueError(f"Invalid phase provided. Valid phases: {Player.PHASES}")
        self.phase = phase

    # --- Movement and Cash Management ---
    def move(self, new_position: int):
        """
        Update the player's current position.
        """
        self.current_position = new_position

    def add_cash(self, amount: int):
        """
        Increase the player's cash balance.
        """
        self.current_cash += amount

    def subtract_cash(self, amount: int):
        """
        Decrease the player's cash balance.
        """
        self.current_cash -= amount

    def update_status(self, new_status: str):
        """
        Update the player's status.
        Allowed statuses: 'waiting_for_move', 'current_move', 'won', 'lost'
        """
        if new_status not in Player.VALID_STATUSES:
            raise ValueError(f"Invalid status. Valid statuses are: {Player.VALID_STATUSES}")
        self.status = new_status

    def update_phase(self, new_phase: str):
        """
        Update the player's current phase.
        Allowed phases: 'pre-roll', 'post-roll', 'out-of-turn'
        """
        if new_phase not in Player.PHASES:
            raise ValueError(f"Invalid phase. Valid phases are: {Player.PHASES}")
        self.phase = new_phase

    # --- Asset Management ---
    def add_asset(self, asset):
        """
        Add an asset (property, railroad, or utility) to the player's holdings.
        Expects asset to have at least an attribute 'type'.
        """
        self.assets.add(asset)
        if hasattr(asset, "type"):
            asset_type = asset.type.lower()
            if asset_type == "railroad":
                self.num_railroads_possessed += 1
            elif asset_type == "utility":
                self.num_utilities_possessed += 1

    def remove_asset(self, asset):
        """
        Remove an asset from the player's holdings.
        """
        if asset in self.assets:
            self.assets.remove(asset)
            if hasattr(asset, "type"):
                asset_type = asset.type.lower()
                if asset_type == "railroad" and self.num_railroads_possessed > 0:
                    self.num_railroads_possessed -= 1
                elif asset_type == "utility" and self.num_utilities_possessed > 0:
                    self.num_utilities_possessed -= 1

    def add_full_color_set(self, color: str):
        """
        Mark a full color set as possessed.
        """
        self.full_color_sets_possessed.add(color)

    def remove_full_color_set(self, color: str):
        """
        Remove a full color set from the player's possession.
        """
        self.full_color_sets_possessed.discard(color)

    def mortgage_asset(self, asset):
        """
        Mortgage an asset by adding it to the mortgaged assets set.
        """
        self.mortgaged_assets.add(asset)

    def unmortgage_asset(self, asset):
        """
        Unmortgage an asset by removing it from the mortgaged assets set.
        """
        self.mortgaged_assets.discard(asset)

    # --- Property Offer Management ---
    def set_outstanding_offer(self, offer):
        """
        Set an outstanding property offer.

        Parameters:
            offer: The details of the offer (could be an object or dict).
        """
        self.outstanding_property_offer = offer
        self.is_property_offer_outstanding = True

    def clear_outstanding_offer(self):
        """
        Clear any outstanding property offer.
        """
        self.outstanding_property_offer = None
        self.is_property_offer_outstanding = False

    def set_option_to_buy(self, flag: bool):
        """
        Set the option-to-buy flag.
        This flag indicates whether the player is allowed to purchase the property they just landed on.
        """
        self._option_to_buy = flag

    def can_buy_property(self) -> bool:
        """
        Determine if the player can buy a property.
        Typically, property purchase is only allowed in the 'post-roll' phase and when the option to buy is active.

        Returns:
            bool: True if buying is allowed, False otherwise.
        """
        return self.phase == 'post-roll' and self._option_to_buy

    # --- Jail Management ---
    def go_to_jail(self):
        """
        Mark the player as being in jail.
        """
        self.currently_in_jail = True

    def leave_jail(self):
        """
        Mark the player as no longer in jail.
        """
        self.currently_in_jail = False

    # --- State Representation ---
    def get_state_vector(self) -> np.ndarray:
        """
        Construct and return a 16-dimensional state vector representing the player's current state.
        The vector is constructed in the following order:
          [current_position (1),
           status one-hot (4): [waiting_for_move, current_move, won, lost],
           has_get_out_of_jail_community_chest_card (1),
           has_get_out_of_jail_chance_card (1),
           current_cash (1),
           num_railroads_possessed (1),
           num_utilities_possessed (1),
           currently_in_jail (1),
           is_property_offer_outstanding (1),
           _option_to_buy (1),
           phase one-hot (3): [pre-roll, post-roll, out-of-turn]]

        Returns:
            np.ndarray: A 1D numpy array of length 16 (dtype: float32)
        """
        state = []

        # 0. current_position
        state.append(float(self.current_position))

        # 1-4. Status one-hot vector (order: waiting_for_move, current_move, won, lost)
        status_vector = [0.0] * 4
        status_vector[Player.VALID_STATUSES.index(self.status)] = 1.0
        state.extend(status_vector)

        # 5. has_get_out_of_jail_community_chest_card (bool as float)
        state.append(1.0 if self.has_get_out_of_jail_community_chest_card else 0.0)

        # 6. has_get_out_of_jail_chance_card (bool as float)
        state.append(1.0 if self.has_get_out_of_jail_chance_card else 0.0)

        # 7. current_cash
        state.append(float(self.current_cash))

        # 8. num_railroads_possessed
        state.append(float(self.num_railroads_possessed))

        # 9. num_utilities_possessed
        state.append(float(self.num_utilities_possessed))

        # 10. currently_in_jail (bool as float)
        state.append(1.0 if self.currently_in_jail else 0.0)

        # 11. is_property_offer_outstanding (bool as float)
        state.append(1.0 if self.is_property_offer_outstanding else 0.0)

        # 12. _option_to_buy (bool as float)
        state.append(1.0 if self._option_to_buy else 0.0)

        # 13-15. Phase one-hot vector (order: pre-roll, post-roll, out-of-turn)
        phase_vector = [0.0] * 3
        phase_vector[Player.PHASES.index(self.phase)] = 1.0
        state.extend(phase_vector)

        return np.array(state, dtype=np.float32)


