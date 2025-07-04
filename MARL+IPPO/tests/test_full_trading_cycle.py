#!/usr/bin/env python3

import numpy as np
import json
import sys
sys.path.append('.')
from env_wrapper import MonopolyMAv2
from scripted_agent import ScriptedAgent

def test_full_trading_cycle():
    """
    Test the complete trading cycle with proper turn management.
    """
    print("=== TESTING FULL TRADING CYCLE ===")
    
    # Create environment
    env = MonopolyMAv2(board_json_path="../data.json", num_players=4, max_steps=1000)
    
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
    
    # Update player counts and monopolies
    for player in game.players:
        game._update_player_counts(player)
        game._update_player_monopolies(player)
    
    print(f"Player 0 assets: {player_0.assets}")
    print(f"Player 1 assets: {player_1.assets}")
    print(f"Player 2 assets: {game.players[2].assets}")
    print(f"Player 3 assets: {game.players[3].assets}")
    
    # Set all players to pre-roll phase so they can make trades
    for player in game.players:
        player.update_phase('pre-roll')
    
    print("\n=== RUNNING TRADING SIMULATION ===")
    
    step_count = 0
    max_steps = 50
    trades_made = 0
    houses_built = 0
    trade_responses = 0
    
    while step_count < max_steps:
        step_count += 1
        
        # Get current player
        current_player_id = env.internal_env.game.current_player_index
        current_player_name = f'player_{current_player_id}'
        current_player_obj = env.internal_env.game.players[current_player_id]
        
        print(f"\nStep {step_count}: Player {current_player_id} turn")
        print(f"  Phase: {current_player_obj.phase}")
        print(f"  Assets: {current_player_obj.assets}")
        print(f"  Monopolies: {current_player_obj.full_color_sets_possessed}")
        print(f"  Cash: ${current_player_obj.current_cash}")
        
        # Check for pending trades
        if game.pending_trade:
            print(f"  Pending trade: {game.pending_trade}")
        
        # Get action from appropriate agent
        agent = agents[current_player_id]
        obs = env._get_all_observations()[current_player_name]
        
        try:
            action = agent.get_action(obs)
            print(f"  Chosen action: {action}")
            
            # Execute action
            step_result = env.step({current_player_name: action})
            observations, rewards, terminated, truncated, info = step_result
            
            print(f"  Result: {info.get('result', 'No result message')}")
            
            # Check for trades
            if action[0] == 1:  # Trade offer buy
                print(f"  --> Player {current_player_id} made a BUY trade offer!")
                trades_made += 1
            elif action[0] == 0:  # Trade offer sell
                print(f"  --> Player {current_player_id} made a SELL trade offer!")
                trades_made += 1
            elif action[0] == 11:  # Respond to trade
                response = 'Accept' if action[1] == 1 else 'Reject'
                print(f"  --> Player {current_player_id} responded to trade: {response}")
                trade_responses += 1
            elif action[0] == 2:  # Improve property
                print(f"  --> Player {current_player_id} built a house/hotel!")
                houses_built += 1
            
            # Check for monopolies after action
            if current_player_obj.full_color_sets_possessed:
                print(f"  --> Player {current_player_id} has monopolies: {current_player_obj.full_color_sets_possessed}")
            
            # Check if game ended
            if terminated or truncated:
                print(f"Game ended at step {step_count}")
                break
                
        except Exception as e:
            print(f"Error at step {step_count}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Total steps: {step_count}")
    print(f"Trades made: {trades_made}")
    print(f"Trade responses: {trade_responses}")
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
    
    # Check if any monopolies were formed
    monopolies_formed = 0
    for player in env.internal_env.game.players:
        monopolies_formed += len(player.full_color_sets_possessed)
    
    print(f"Total monopolies formed: {monopolies_formed}")
    
    if houses_built > 0:
        print("✅ SUCCESS: Houses were built!")
    elif monopolies_formed > 0:
        print("🏠 MONOPOLY SUCCESS: Monopolies were formed!")
    elif trades_made > 0:
        print("🔄 PARTIAL SUCCESS: Trades were attempted!")
    else:
        print("❌ No trades or monopolies formed")

if __name__ == "__main__":
    test_full_trading_cycle() 