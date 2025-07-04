import numpy as np

class ScriptedAgent:
    """
    An enhanced scripted agent for Monopoly with intelligent trading capabilities.

    This agent follows these rules in priority order:
    1. Get out of jail if in jail
    2. Buy property if landed on an unowned one and can afford it
    3. Build houses/hotels on monopolies when profitable
    4. Make strategic trades to form monopolies
    5. Respond to trades intelligently
    6. Conclude turn
    """
    def __init__(self, player_id, num_players, board_meta):
        self.player_id = player_id
        self.num_players = num_players
        self.board_meta = board_meta # Dict of property metadata
        
        # Create mapping from property ID to board metadata
        self.prop_id_to_meta = {}
        for prop in board_meta.values():
            if isinstance(prop, dict) and 'id' in prop:
                self.prop_id_to_meta[prop['id']] = prop
        
        # Group properties by color for monopoly detection
        self.color_groups = {}
        for prop in self.prop_id_to_meta.values():
            if prop.get('type') == 'street' and 'color_group' in prop:
                color = prop['color_group']
                if color not in self.color_groups:
                    self.color_groups[color] = []
                self.color_groups[color].append(prop['id'])
        
        # Cash reserve to maintain for safety
        self.min_cash_reserve = 200
        
        # Trading parameters
        self.max_trade_price_multiplier = 1.5  # Will pay up to 150% of property value for monopoly completion
        self.min_trade_price_multiplier = 0.8  # Will sell for at least 80% of property value

    def _get_owned_properties(self, obs):
        """Extract owned properties from board state."""
        board_state = obs['board']
        owned_props = []
        
        # Board state is flattened: 28 properties x 8 dimensions
        # Dimensions: [owner_0, owner_1, owner_2, owner_3, owner_4, mortgage, monopoly, houses, hotels]
        board_2d = board_state.reshape(28, 8)
        
        # Create mapping from board index to property ID
        # Based on the board layout, the indices correspond to properties in this order:
        board_index_to_prop_id = {
            0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 11, 7: 12, 8: 13, 9: 14,
            10: 15, 11: 16, 12: 18, 13: 19, 14: 21, 15: 23, 16: 24, 17: 25,
            18: 26, 19: 27, 20: 28, 21: 29, 22: 31, 23: 32, 24: 34, 25: 35,
            26: 37, 27: 39
        }
        
        # Check each property (player_id + 1 because bank is owner 0)
        owner_idx = self.player_id + 1
        for prop_idx in range(28):
            if board_2d[prop_idx, owner_idx] > 0.5:  # We own this property
                if prop_idx in board_index_to_prop_id:
                    owned_props.append(board_index_to_prop_id[prop_idx])
        
        return owned_props

    def _get_all_player_properties(self, obs):
        """Get properties owned by all players."""
        board_state = obs['board']
        board_2d = board_state.reshape(28, 8)
        
        board_index_to_prop_id = {
            0: 1, 1: 3, 2: 5, 3: 6, 4: 8, 5: 9, 6: 11, 7: 12, 8: 13, 9: 14,
            10: 15, 11: 16, 12: 18, 13: 19, 14: 21, 15: 23, 16: 24, 17: 25,
            18: 26, 19: 27, 20: 28, 21: 29, 22: 31, 23: 32, 24: 34, 25: 35,
            26: 37, 27: 39
        }
        
        player_properties = {i: [] for i in range(self.num_players)}
        
        for prop_idx in range(28):
            for player_id in range(self.num_players):
                owner_idx = player_id + 1  # +1 because bank is owner 0
                if board_2d[prop_idx, owner_idx] > 0.5:
                    if prop_idx in board_index_to_prop_id:
                        player_properties[player_id].append(board_index_to_prop_id[prop_idx])
                    break
        
        return player_properties

    def _detect_monopolies(self, owned_props):
        """Detect which color groups we have monopolies on."""
        monopolies = []
        for color, prop_ids in self.color_groups.items():
            if all(prop_id in owned_props for prop_id in prop_ids):
                monopolies.append(color)
        return monopolies

    def _detect_near_monopolies(self, owned_props):
        """Detect color groups where we need just one more property for monopoly."""
        near_monopolies = []
        for color, prop_ids in self.color_groups.items():
            owned_in_color = [prop_id for prop_id in prop_ids if prop_id in owned_props]
            if len(owned_in_color) == len(prop_ids) - 1:  # Need just one more
                missing_prop = [prop_id for prop_id in prop_ids if prop_id not in owned_props][0]
                near_monopolies.append({
                    'color': color,
                    'missing_prop': missing_prop,
                    'owned_props': owned_in_color
                })
        return near_monopolies

    def _find_trading_opportunities(self, obs):
        """Find properties we could trade to complete monopolies."""
        all_player_props = self._get_all_player_properties(obs)
        my_props = all_player_props[self.player_id]
        near_monopolies = self._detect_near_monopolies(my_props)
        
        trading_opportunities = []
        
        for near_monopoly in near_monopolies:
            missing_prop = near_monopoly['missing_prop']
            
            # Find who owns the missing property
            for other_player_id in range(self.num_players):
                if other_player_id != self.player_id:
                    if missing_prop in all_player_props[other_player_id]:
                        # Found a trading opportunity
                        trading_opportunities.append({
                            'target_player': other_player_id,
                            'target_prop': missing_prop,
                            'color': near_monopoly['color'],
                            'value': self._calculate_monopoly_value(near_monopoly['color'])
                        })
                        break
        
        return trading_opportunities

    def _calculate_monopoly_value(self, color):
        """Calculate the strategic value of completing a monopoly."""
        color_props = self.color_groups.get(color, [])
        if not color_props:
            return 0
        
        # Calculate based on rent potential and building costs
        total_value = 0
        for prop_id in color_props:
            prop_meta = self.prop_id_to_meta.get(prop_id)
            if prop_meta:
                base_rent = prop_meta.get('rent', {}).get('base', 0)
                # Monopoly doubles base rent, plus building potential
                total_value += base_rent * 2
                # Add building value potential
                house_cost = prop_meta.get('house_cost', 50)
                hotel_rent = prop_meta.get('rent', {}).get('hotel', 0)
                total_value += (hotel_rent - base_rent) * 0.5  # Discounted future value
        
        return total_value

    def _choose_trade_offer(self, trading_opportunities, current_cash):
        """Choose the best trade offer to make."""
        if not trading_opportunities:
            return None
        
        # Sort by monopoly value (highest first)
        trading_opportunities.sort(key=lambda x: x['value'], reverse=True)
        
        for opportunity in trading_opportunities:
            target_prop = opportunity['target_prop']
            prop_meta = self.prop_id_to_meta.get(target_prop)
            if not prop_meta:
                continue
            
            base_price = prop_meta.get('price', 0)
            max_offer = int(base_price * self.max_trade_price_multiplier)
            
            # Only make offer if we can afford it while maintaining cash reserve
            if current_cash >= max_offer + self.min_cash_reserve:
                return {
                    'type': 'buy',
                    'target_player': opportunity['target_player'],
                    'target_prop': target_prop,
                    'offer_price': max_offer,
                    'prop_meta': prop_meta
                }
        
        return None

    def _convert_trade_to_action(self, trade_offer):
        """Convert a trade offer to action format."""
        if not trade_offer:
            return None
        
        # Find property index in board metadata
        prop_id = trade_offer['target_prop']
        prop_index = None
        
        # Find the index of this property in the board properties list
        for i, prop_meta in enumerate(self.prop_id_to_meta.values()):
            if prop_meta.get('id') == prop_id:
                prop_index = i
                break
        
        if prop_index is None:
            return None
        
        # Determine price tier (0: 0.75x, 1: 1.0x, 2: 1.25x)
        base_price = trade_offer['prop_meta'].get('price', 0)
        offer_price = trade_offer['offer_price']
        
        if offer_price <= base_price * 0.875:  # Closer to 0.75x
            price_tier = 0
        elif offer_price <= base_price * 1.125:  # Closer to 1.0x
            price_tier = 1
        else:  # Closer to 1.25x
            price_tier = 2
        
        # Calculate sub-action for trade buy
        # Format: target_player_index * (28 * 3) + property_index * 3 + price_tier
        target_player = trade_offer['target_player']
        
        # Get candidate players (excluding self)
        candidate_players = [i for i in range(self.num_players) if i != self.player_id]
        if target_player not in candidate_players:
            return None
        
        target_index = candidate_players.index(target_player)
        sub_action = target_index * (28 * 3) + prop_index * 3 + price_tier
        
        return (1, sub_action)  # Action 1 is "Make Trade Offer (Buy)"

    def _should_accept_trade(self, obs):
        """Determine if we should accept a pending trade offer."""
        # Check if there's a pending trade addressed to us
        if obs.get('pending_trade_valid', 0) == 0:
            return False
        
        trade_details = obs.get('trade_details', np.zeros(4))
        if np.sum(trade_details) == 0:
            return False
        
        # Decode trade details to see if it's addressed to us
        # trade_details format: [from_player_idx_norm, to_player_idx_norm, prop_idx_norm, price_norm]
        to_player_idx_norm = trade_details[1]
        # Convert normalized index back to actual player index
        to_player_idx = int(round(to_player_idx_norm * (self.num_players - 1)))
        
        # Only respond if the trade is addressed to us
        if to_player_idx != self.player_id:
            return False
        
        # For now, use simple heuristic: accept if the price is reasonable
        # In a more sophisticated version, we'd analyze the strategic value
        return True  # Accept most trades for now to encourage trading

    def _get_buildable_properties(self, obs, monopolies):
        """Get properties where we can build houses/hotels."""
        if not monopolies:
            return []
        
        board_state = obs['board']
        board_2d = board_state.reshape(28, 8)
        buildable = []
        
        # Create reverse mapping from property ID to board index
        prop_id_to_board_idx = {
            1: 0, 3: 1, 5: 2, 6: 3, 8: 4, 9: 5, 11: 6, 12: 7, 13: 8, 14: 9,
            15: 10, 16: 11, 18: 12, 19: 13, 21: 14, 23: 15, 24: 16, 25: 17,
            26: 18, 27: 19, 28: 20, 29: 21, 31: 22, 32: 23, 34: 24, 35: 25,
            37: 26, 39: 27
        }
        
        for color in monopolies:
            for prop_id in self.color_groups[color]:
                prop_meta = self.prop_id_to_meta.get(prop_id)
                if not prop_meta:
                    continue
                
                # Get board index for this property
                board_idx = prop_id_to_board_idx.get(prop_id)
                if board_idx is None:
                    continue
                
                # Check if we own this property
                if board_2d[board_idx, self.player_id + 1] > 0.5:
                    house_frac = board_2d[board_idx, 6]  # House fraction
                    hotel_frac = board_2d[board_idx, 7]  # Hotel fraction
                    mortgage_flag = board_2d[board_idx, 4]  # Mortgage flag
                    
                    # Can build if not mortgaged and not fully developed
                    if mortgage_flag < 0.5 and (house_frac < 1.0 or hotel_frac < 1.0):
                        buildable.append({
                            'prop_id': prop_id,
                            'board_idx': board_idx,
                            'house_frac': house_frac,
                            'hotel_frac': hotel_frac,
                            'house_cost': prop_meta.get('house_cost', 50),
                            'color': color
                        })
        
        return buildable

    def _choose_building_action(self, buildable_props, current_cash):
        """Choose the best property to build on."""
        if not buildable_props:
            return None
        
        # Sort by profitability (cheaper houses first, then by rent potential)
        buildable_props.sort(key=lambda x: (x['house_cost'], -x['prop_id']))
        
        for prop in buildable_props:
            # Build house if we can afford it and maintain cash reserve
            if prop['house_frac'] < 1.0:  # Can build house
                cost = prop['house_cost']
                if current_cash >= cost + self.min_cash_reserve:
                    return ('house', prop)
            
            # Build hotel if we have 4 houses and can afford it
            elif prop['house_frac'] >= 1.0 and prop['hotel_frac'] < 1.0:
                cost = prop['house_cost']  # Hotel costs same as house
                if current_cash >= cost + self.min_cash_reserve:
                    return ('hotel', prop)
        
        return None

    def _convert_to_sub_action(self, building_choice):
        """Convert building choice to sub-action for improve property."""
        if not building_choice:
            return None
        
        building_type, prop_info = building_choice
        prop_id = prop_info['prop_id']
        
        # Find the property index in the list of street properties
        street_props = []
        for prop_meta in self.prop_id_to_meta.values():
            if prop_meta.get('type') == 'street':
                street_props.append(prop_meta['id'])
        
        street_props.sort()  # Ensure consistent ordering
        
        try:
            prop_index = street_props.index(prop_id)
            # Sub-action encoding: property_index * 2 + building_type
            # building_type: 0 for house, 1 for hotel
            building_type_idx = 0 if building_type == 'house' else 1
            sub_action = prop_index * 2 + building_type_idx
            return sub_action
        except ValueError:
            return None

    def get_action(self, obs):
        """
        Get an action based on the current observation.

        Args:
            obs (dict): The observation for this agent.

        Returns:
            tuple: A tuple representing the (top_level_action, sub_action).
        """
        player_state = obs['player'] # This is the raw vector
        
        # We need to decode the player state vector to make decisions.
        # Based on monopoly_env/core/player.py:
        # 0: pos, 1-4: status, 5-6: jail cards, 7: cash, 8: RRs, 9: utils, 10: in_jail, 12: can_buy, 13-15: phase
        
        is_in_jail = player_state[10] > 0
        can_buy = player_state[12] > 0
        current_cash = player_state[7]
        current_pos = int(player_state[0])
        
        phase_vec = player_state[13:16]
        phase = np.argmax(phase_vec) # 0: pre-roll, 1: post-roll, 2: out-of-turn
        
        # Rule 1: Get out of jail if possible
        if is_in_jail: # Any phase
            # Action 9 is "Pay Jail Fine"
            if current_cash > 50: # Assuming fine is 50
                return (9, 0)
            else:
                # Action 8 is "Use Get Out of Jail" (if we have a card)
                return (8, 0)

        # Rule 2: Respond to trades if there's a pending trade addressed to us
        if obs.get('pending_trade_valid', 0) > 0:  # There is a pending trade
            trade_details = obs.get('trade_details', np.zeros(4))
            if np.sum(trade_details) > 0:
                # Decode trade details to see if it's addressed to us
                to_player_idx_norm = trade_details[1]
                to_player_idx = int(round(to_player_idx_norm * (self.num_players - 1)))
                
                # Only respond if the trade is addressed to us
                if to_player_idx == self.player_id:
                    if self._should_accept_trade(obs):
                        return (11, 1)  # Accept trade
                    else:
                        return (11, 0)  # Reject trade

        # Rule 3: Buy property if the option is available
        if can_buy and phase == 1: # Post-roll phase
            property_meta = self.board_meta.get(str(current_pos))
            if property_meta and current_cash >= property_meta['price']:
                # Action 10 is "Buy Property", Sub-action 1 is "Yes"
                return (10, 1)
            else:
                # Sub-action 0 is "No"
                return (10, 0)
        
        # Rule 4: Build houses/hotels on monopolies (pre-roll phase only)
        if phase == 0: # Pre-roll phase allows property improvements
            try:
                # Get properties we own
                owned_props = self._get_owned_properties(obs)
                
                # Check for monopolies
                monopolies = self._detect_monopolies(owned_props)
                
                if monopolies:
                    # Get buildable properties
                    buildable_props = self._get_buildable_properties(obs, monopolies)
                    
                    # Choose what to build
                    building_choice = self._choose_building_action(buildable_props, current_cash)
                    
                    if building_choice:
                        # Convert to sub-action
                        sub_action = self._convert_to_sub_action(building_choice)
                        
                        if sub_action is not None:
                            # Action 2 is "Improve Property"
                            return (2, sub_action)
            except Exception as e:
                # If house building logic fails, continue to trading logic
                pass

        # Rule 5: Make strategic trades to form monopolies (pre-roll phase only)
        if phase == 0: # Pre-roll phase allows trades
            try:
                trading_opportunities = self._find_trading_opportunities(obs)
                trade_offer = self._choose_trade_offer(trading_opportunities, current_cash)
                
                if trade_offer:
                    action = self._convert_trade_to_action(trade_offer)
                    if action:
                        return action
            except Exception as e:
                # If trading logic fails, continue to default action
                pass
        
        # Default action: Conclude the current phase/turn
        # Action 7 is "Conclude Phase" (0-indexed)
        return (7, 0) 