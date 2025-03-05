Below are the functions (methods) defined in each file:

---

**player.py**

- `__init__`
- `move`
- `add_cash`
- `subtract_cash`
- `update_status`
- `update_phase`
- `add_asset`
- `remove_asset`
- `add_full_color_set`
- `remove_full_color_set`
- `mortgage_asset`
- `unmortgage_asset`
- `set_outstanding_offer`
- `clear_outstanding_offer`
- `set_option_to_buy`
- `can_buy_property`
- `go_to_jail`
- `leave_jail`
- `get_state_vector`

---

**board.py**

- `__init__`
- `_load_board_data`
- `reset`
- `get_state`
- `get_property_meta`
- `purchase_property`
- `mortgage_property`
- `unmortgage_property`
- `set_buildings`
- `update_monopoly_flag`

---

**game_logic.py**

- `__init__`
- `roll_dice`
- `process_action`
- `get_valid_actions`
- `make_trade_offer_sell`
- `make_trade_offer_buy`
- `improve_property`
- `sell_house_hotel`
- `sell_property`
- `mortgage_free_mortgage`
- `skip_turn`
- `conclude_phase`
- `use_get_out_of_jail`
- `pay_jail_fine`
- `buy_property`
- `respond_to_trade`

---

**Verification of Conflicts/Redundancies:**

- Each file focuses on a distinct component:
  - **player.py:** Manages individual player state and actions.
  - **board.py:** Manages the board state and property-related operations.
  - **game_logic.py:** Implements the game rules and ties together the board and players.
- There are no apparent conflicts or redundant functions between these files; each serves its intended purpose within the overall architecture.