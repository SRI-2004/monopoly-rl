#!/usr/bin/env python3

import numpy as np
import json
import sys
sys.path.append('.')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def test_trading_agent():
    """
    Test the enhanced scripted agent with trading capabilities.
    """
    print("=== TESTING TRADING-ENABLED SCRIPTED AGENT ===")
    
    # Create environment
    env = MonopolyMAv2(board_json_path="../data.json", num_players=4, max_steps=2000)
    
    # Load board metadata
    with open('../data.json', 'r') as f:
        board_data = json.load(f)
    board_meta = {str(prop['id']): prop for prop in board_data['board_layout']}
    
    # Create scripted agents for all players
    agents = []
    for i in range(4):
        agent = ScriptedAgent(player_id=i, num_players=4, board_meta=board_meta)
        agents.append(agent)
    
    print(f"Created {len(agents)} trading-enabled scripted agents")
    
    # Reset environment
    observations = env.reset()
    
    # Set up a scenario where players can potentially trade
    game = env.internal_env.game
    board = game.board
    
    # Give players some properties to encourage trading
    # Player 0: Mediterranean Ave (1), Baltic Ave (3) - brown monopoly potential
    # Player 1: Oriental Ave (6), Vermont Ave (8) - light blue partial
    # Player 2: Connecticut Ave (9) - light blue partial  
    # Player 3: St. Charles Place (11) - pink partial
    
    print("\n=== SETTING UP TRADING SCENARIO ===")
    
    # Give Player 0 one brown property (Mediterranean Ave)
    player_0 = game.players[0]
    player_0.add_asset(1)
    prop_idx = board.property_id_to_index[1]
    owner_vector = np.zeros(board.num_owners, dtype=np.float32)
    owner_vector[1] = 1.0  # Player 0 is owner index 1
    board.state[prop_idx, 0:board.num_owners] = owner_vector
    
    # Give Player 1 the other brown property (Baltic Ave) 
    player_1 = game.players[1]
    player_1.add_asset(3)
    prop_idx = board.property_id_to_index[3]
    owner_vector = np.zeros(board.num_owners, dtype=np.float32)
    owner_vector[2] = 1.0  # Player 1 is owner index 2
    board.state[prop_idx, 0:board.num_owners] = owner_vector
    
    # Give Player 1 some light blue properties
    player_1.add_asset(6)  # Oriental Ave
    prop_idx = board.property_id_to_index[6]
    owner_vector = np.zeros(board.num_owners, dtype=np.float32)
    owner_vector[2] = 1.0
    board.state[prop_idx, 0:board.num_owners] = owner_vector
    
    # Give Player 2 another light blue property
    player_2 = game.players[2]
    player_2.add_asset(8)  # Vermont Ave
    prop_idx = board.property_id_to_index[8]
    owner_vector = np.zeros(board.num_owners, dtype=np.float32)
    owner_vector[3] = 1.0  # Player 2 is owner index 3
    board.state[prop_idx, 0:board.num_owners] = owner_vector
    
    # Update player counts and monopolies
    for player in game.players:
        game._update_player_counts(player)
        game._update_player_monopolies(player)
    
    print(f"Player 0 assets: {player_0.assets}")
    print(f"Player 1 assets: {player_1.assets}")
    print(f"Player 2 assets: {player_2.assets}")
    print(f"Player 3 assets: {game.players[3].assets}")
    
    # Test trading opportunities detection
    print("\n=== TESTING TRADING LOGIC ===")
    
    # Get observations for each player
    all_obs = env._get_all_observations()
    
    # Test Player 0's trading logic (should want to buy Baltic Ave from Player 1)
    agent_0 = agents[0]
    obs_0 = all_obs['player_0']
    
    print(f"Player 0 cash: ${obs_0['player'][7]}")
    
    # Test trading opportunity detection
    try:
        owned_props = agent_0._get_owned_properties(obs_0)
        print(f"Player 0 owned properties: {owned_props}")
        
        near_monopolies = agent_0._detect_near_monopolies(owned_props)
        print(f"Player 0 near monopolies: {near_monopolies}")
        
        trading_opportunities = agent_0._find_trading_opportunities(obs_0)
        print(f"Player 0 trading opportunities: {trading_opportunities}")
        
        if trading_opportunities:
            trade_offer = agent_0._choose_trade_offer(trading_opportunities, obs_0['player'][7])
            print(f"Player 0 chosen trade offer: {trade_offer}")
            
            if trade_offer:
                action = agent_0._convert_trade_to_action(trade_offer)
                print(f"Player 0 trade action: {action}")
    except Exception as e:
        print(f"Error in Player 0 trading logic: {e}")
    
    # Run a few steps to see if trading happens
    print("\n=== RUNNING SIMULATION ===")
    
    step_count = 0
    max_steps = 100
    trades_made = 0
    houses_built = 0
    
    while step_count < max_steps:
        step_count += 1
        
        # Get current player
        current_player_id = env.internal_env.game.current_player_index
        current_player_name = f'player_{current_player_id}'
        
        # Get action from appropriate agent
        agent = agents[current_player_id]
        obs = env._get_all_observations()[current_player_name]
        
        try:
            action = agent.get_action(obs)
            print(f"Step {step_count}: Player {current_player_id} action: {action}")
            
            # Execute action
            step_result = env.step({current_player_name: action})
            observations, rewards, terminated, truncated, info = step_result
            
            # Check for trades
            if action[0] == 1:  # Trade offer buy
                print(f"  --> Player {current_player_id} made a trade offer!")
                trades_made += 1
            elif action[0] == 11:  # Respond to trade
                print(f"  --> Player {current_player_id} responded to trade: {'Accept' if action[1] == 1 else 'Reject'}")
            elif action[0] == 2:  # Improve property
                print(f"  --> Player {current_player_id} built a house/hotel!")
                houses_built += 1
            
            # Check for monopolies
            current_player_obj = env.internal_env.game.players[current_player_id]
            if current_player_obj.full_color_sets_possessed:
                print(f"  --> Player {current_player_id} has monopolies: {current_player_obj.full_color_sets_possessed}")
            
            # Check if game ended
            if terminated or truncated:
                print(f"Game ended at step {step_count}")
                break
                
        except Exception as e:
            print(f"Error at step {step_count}: {e}")
            break
    
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Total steps: {step_count}")
    print(f"Trades made: {trades_made}")
    print(f"Houses built: {houses_built}")
    
    # Final state analysis
    print(f"\n=== FINAL STATE ===")
    for i, player in enumerate(env.internal_env.game.players):
        print(f"Player {i}:")
        print(f"  Assets: {player.assets}")
        print(f"  Monopolies: {player.full_color_sets_possessed}")
        print(f"  Cash: ${player.current_cash}")
        print(f"  Status: {player.status}")
    
    # Check house fractions on board
    total_houses = np.sum(board.state[:, 6])
    print(f"\nTotal house fraction on board: {total_houses}")
    
    if houses_built > 0:
        print("✅ SUCCESS: Houses were built!")
    elif trades_made > 0:
        print("🔄 PARTIAL SUCCESS: Trades were attempted!")
    else:
        print("❌ No trades or houses - may need more steps or different setup")

if __name__ == "__main__":
    test_trading_agent() 