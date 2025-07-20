#!/usr/bin/env python3
"""
Fantasy Cricket Team Simulation - Efficient Submission Version
Optimized for quick execution while maintaining quality
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_teams_efficient(df, target_teams=5000):
    """Efficient team generation with constraint satisfaction"""
    teams = []
    unique_teams = set()
    player_selections = defaultdict(int)
    
    # Calculate target selections for each player
    player_targets = {}
    for _, player in df.iterrows():
        player_targets[player['player_code']] = int(player['perc_selection'] * target_teams)
    
    # Group players by role
    role_players = {}
    for role in df['role'].unique():
        role_players[role] = df[df['role'] == role]['player_code'].tolist()
    
    print(f"Generating {target_teams:,} teams...")
    
    attempts = 0
    max_attempts = target_teams * 3
    
    while len(teams) < target_teams and attempts < max_attempts:
        attempts += 1
        
        if attempts % 1000 == 0:
            print(f"  Progress: {len(teams):,} teams generated (attempt {attempts:,})")
        
        # Generate team with role constraints
        team = []
        used_players = set()
        
        # Step 1: Select one from each required role
        required_roles = ['Batsman', 'Bowler', 'WK', 'Allrounder']
        
        for role in required_roles:
            if role not in role_players:
                break
                
            available = [p for p in role_players[role] if p not in used_players]
            if not available:
                break
            
            # Calculate priorities based on deficit
            priorities = []
            for player_code in available:
                target = player_targets[player_code]
                current = player_selections[player_code]
                remaining = max(1, target_teams - len(teams))
                
                deficit = target - current
                if deficit > 0:
                    priority = deficit / remaining
                else:
                    priority = 0.1
                priorities.append(priority)
            
            # Select player
            if sum(priorities) > 0:
                probs = np.array(priorities) / sum(priorities)
                selected = np.random.choice(available, p=probs)
            else:
                selected = np.random.choice(available)
            
            team.append(selected)
            used_players.add(selected)
        
        if len(team) != 4:  # Failed to get required roles
            continue
        
        # Step 2: Fill remaining 7 spots
        all_available = [p for p in df['player_code'] if p not in used_players]
        
        if len(all_available) < 7:
            continue
        
        # Calculate priorities for remaining players
        priorities = []
        for player_code in all_available:
            target = player_targets[player_code]
            current = player_selections[player_code]
            remaining = max(1, target_teams - len(teams))
            
            deficit = target - current
            if deficit > 0:
                priority = deficit / remaining
            else:
                priority = 0.1
            priorities.append(priority)
        
        # Select remaining 7 players
        if sum(priorities) > 0:
            probs = np.array(priorities) / sum(priorities)
            remaining_selected = np.random.choice(all_available, size=7, replace=False, p=probs)
        else:
            remaining_selected = np.random.choice(all_available, size=7, replace=False)
        
        team.extend(remaining_selected)
        
        # Check uniqueness
        team_tuple = tuple(sorted(team))
        if team_tuple in unique_teams:
            continue
        
        # Accept team
        teams.append(team)
        unique_teams.add(team_tuple)
        
        # Update player selections
        for player_code in team:
            player_selections[player_code] += 1
    
    print(f"Generated {len(teams):,} unique teams successfully!")
    return teams, player_selections

def create_team_df(teams, df):
    """Create team dataframe"""
    print("Creating team dataframe...")
    rows = []
    
    for team_id, team in enumerate(teams, 1):
        for player_code in team:
            player_info = df[df['player_code'] == player_code].iloc[0]
            rows.append({
                'match_code': player_info['match_code'],
                'player_code': player_code,
                'player_name': player_info['player_name'],
                'role': player_info['role'],
                'team': player_info['team'],
                'perc_selection': player_info['perc_selection'],
                'team_id': team_id
            })
    
    return pd.DataFrame(rows)

def evaluate_accuracy(team_df, original_df):
    """Evaluate and print accuracy"""
    print("\n" + "="*60)
    print("ACCURACY EVALUATION")
    print("="*60)
    
    total_teams = team_df['team_id'].nunique()
    results = []
    
    for _, player in original_df.iterrows():
        player_code = player['player_code']
        expected = player['perc_selection']
        
        team_count = len(team_df[team_df['player_code'] == player_code]['team_id'].unique())
        actual = team_count / total_teams
        
        if expected > 0:
            error = (actual - expected) / expected
        else:
            error = 0
        
        within_5_percent = abs(error) <= 0.05
        
        results.append({
            'player_code': player_code,
            'player_name': player['player_name'],
            'role': player['role'],
            'team': player['team'],
            'expected_perc_selection': expected,
            'team_count': team_count,
            'actual_perc_selection': actual,
            'perc_error': error,
            'within_5_percent': within_5_percent
        })
    
    accuracy_df = pd.DataFrame(results)
    
    # Summary statistics
    players_within_5_percent = accuracy_df['within_5_percent'].sum()
    total_players = len(accuracy_df)
    
    print(f"Total teams: {total_teams:,}")
    print(f"Players within +/-5%: {players_within_5_percent}/{total_players}")
    print(f"Success rate: {players_within_5_percent/total_players*100:.1f}%")
    print(f"Target: 20/22 players within +/-5%")
    
    if players_within_5_percent >= 20:
        print("*** QUALIFICATION: PASSED! ***")
    else:
        print(f"Qualification: FAILED (need {20-players_within_5_percent} more)")
    
    print(f"\nMax error: {accuracy_df['perc_error'].abs().max():.3f}")
    print(f"Mean error: {accuracy_df['perc_error'].abs().mean():.3f}")
    
    # Show successful players
    passed = accuracy_df[accuracy_df['within_5_percent']]
    if len(passed) > 0:
        print(f"\nPlayers within +/-5%:")
        for _, row in passed.iterrows():
            print(f"  {row['player_name']}: {row['perc_error']*100:+.1f}%")
    
    print("="*60)
    
    return accuracy_df

def main():
    """Main execution"""
    print("FANTASY CRICKET TEAM SIMULATION - EFFICIENT SUBMISSION")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/player_data_sample.csv')
    print(f"Loaded {len(df)} players")
    
    # Generate teams (using smaller number for speed)
    teams, player_selections = generate_teams_efficient(df, target_teams=5000)
    
    # Create dataframes
    team_df = create_team_df(teams, df)
    accuracy_df = evaluate_accuracy(team_df, df)
    
    # Save files
    team_df.to_csv('team_df.csv', index=False)
    accuracy_df.to_csv('accuracy_summary.csv', index=False)
    
    print(f"\nFiles saved:")
    print(f"- team_df.csv ({len(team_df):,} rows)")
    print(f"- accuracy_summary.csv ({len(accuracy_df)} players)")
    
    # Create evaluation output
    players_within_5 = accuracy_df['within_5_percent'].sum()
    eval_output = f"""FANTASY CRICKET TEAM SIMULATION - ACCURACY EVALUATION
======================================================================
Total teams generated: {len(teams):,}
Total players: {len(accuracy_df)}
Players within +/-5% error: {players_within_5} out of {len(accuracy_df)}
Success rate: {players_within_5/len(accuracy_df)*100:.1f}%
Qualification threshold: 20 players within +/-5%
QUALIFICATION STATUS: {'PASSED' if players_within_5 >= 20 else 'FAILED'}

Maximum error: {accuracy_df['perc_error'].abs().max():.4f}
Mean absolute error: {accuracy_df['perc_error'].abs().mean():.4f}
Teams missing required roles: 0
======================================================================"""
    
    with open('evaluation_output.txt', 'w') as f:
        f.write(eval_output)
    
    print("- evaluation_output.txt")
    print("\nSubmission ready!")

if __name__ == "__main__":
    main()