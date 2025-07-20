#!/usr/bin/env python3
"""
Final Fantasy Cricket Team Simulation - 20,000 teams
"""

import pandas as pd
import numpy as np
import random
import time
from typing import List, Dict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_team_composition(team_players: List[int], df: pd.DataFrame) -> bool:
    """Validate team composition requirements"""
    if len(team_players) != 11:
        return False
    
    team_df = df[df['player_code'].isin(team_players)]
    roles_in_team = set(team_df['role'].values)
    required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
    
    return required_roles.issubset(roles_in_team)

def generate_teams(df: pd.DataFrame, num_teams: int = 20000) -> List[List[int]]:
    """
    Generate unique teams with role constraints and probability weighting
    """
    teams = []
    unique_teams = set()
    
    # Pre-calculate role groups
    role_groups = {}
    for role in df['role'].unique():
        role_df = df[df['role'] == role].copy()
        role_groups[role] = {
            'players': role_df['player_code'].values,
            'probs': role_df['perc_selection'].values / role_df['perc_selection'].sum()
        }
    
    # All players with normalized probabilities
    all_players = df['player_code'].values
    all_probs = df['perc_selection'].values / df['perc_selection'].sum()
    
    attempts = 0
    max_attempts = num_teams * 3
    
    print(f"Starting generation of {num_teams} teams...")
    start_time = time.time()
    
    while len(teams) < num_teams and attempts < max_attempts:
        attempts += 1
        
        team = []
        
        # Select one from each required role first
        for role in ['Batsman', 'Bowler', 'WK', 'Allrounder']:
            if role in role_groups:
                players = role_groups[role]['players']
                probs = role_groups[role]['probs']
                selected = np.random.choice(players, p=probs)
                team.append(selected)
        
        # Fill remaining 7 spots
        remaining_spots = 11 - len(team)
        available_mask = ~np.isin(all_players, team)
        available_players = all_players[available_mask]
        available_probs = all_probs[available_mask]
        available_probs = available_probs / available_probs.sum()
        
        if len(available_players) >= remaining_spots:
            additional = np.random.choice(
                available_players, 
                size=remaining_spots, 
                replace=False, 
                p=available_probs
            )
            team.extend(additional)
        
        # Check validity and uniqueness
        team_tuple = tuple(sorted(team))
        if (len(team) == 11 and 
            team_tuple not in unique_teams and 
            validate_team_composition(team, df)):
            teams.append(team)
            unique_teams.add(team_tuple)
        
        if len(teams) % 2000 == 0 and len(teams) > 0:
            elapsed = time.time() - start_time
            print(f"Generated {len(teams)} teams in {elapsed:.1f}s...")
    
    total_time = time.time() - start_time
    print(f"Completed: {len(teams)} teams in {total_time:.2f} seconds")
    return teams

def create_team_dataframe(teams: List[List[int]], df: pd.DataFrame) -> pd.DataFrame:
    """Create team_df in required format"""
    print("Creating team dataframe...")
    team_rows = []
    
    for team_id, team in enumerate(teams, 1):
        for player_code in team:
            player_info = df[df['player_code'] == player_code].iloc[0]
            team_rows.append({
                'match_code': player_info['match_code'],
                'player_code': player_code,
                'player_name': player_info['player_name'],
                'role': player_info['role'],
                'team': player_info['team'],
                'perc_selection': player_info['perc_selection'],
                'team_id': team_id
            })
    
    return pd.DataFrame(team_rows)

def evaluate_accuracy(team_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate selection accuracy"""
    print("Evaluating accuracy...")
    
    player_stats = []
    total_teams = team_df['team_id'].nunique()
    
    for _, player in original_df.iterrows():
        player_code = player['player_code']
        expected_selection = player['perc_selection']
        
        # Count appearances
        team_count = len(team_df[team_df['player_code'] == player_code]['team_id'].unique())
        actual_perc_selection = team_count / total_teams
        
        # Calculate error
        if expected_selection > 0:
            perc_error = (actual_perc_selection - expected_selection) / expected_selection
        else:
            perc_error = 0 if actual_perc_selection == 0 else float('inf')
        
        player_stats.append({
            'player_code': player_code,
            'player_name': player['player_name'],
            'role': player['role'],
            'team': player['team'],
            'expected_perc_selection': expected_selection,
            'team_count': team_count,
            'actual_perc_selection': actual_perc_selection,
            'perc_error': perc_error,
            'within_5_percent': abs(perc_error) <= 0.05
        })
    
    accuracy_df = pd.DataFrame(player_stats)
    
    # Print results
    players_within_5_percent = accuracy_df['within_5_percent'].sum()
    total_players = len(accuracy_df)
    
    print("=" * 60)
    print("FANTASY CRICKET TEAM SIMULATION - ACCURACY EVALUATION")
    print("=" * 60)
    print(f"Total teams generated: {total_teams:,}")
    print(f"Total players: {total_players}")
    print(f"Players within ±5% error: {players_within_5_percent} out of {total_players}")
    print(f"Success rate: {players_within_5_percent/total_players*100:.1f}%")
    print(f"Qualification threshold: 20 players within ±5%")
    print(f"QUALIFICATION STATUS: {'PASSED' if players_within_5_percent >= 20 else 'FAILED'}")
    print()
    print(f"Maximum error: {accuracy_df['perc_error'].abs().max():.4f} ({accuracy_df['perc_error'].abs().max()*100:.2f}%)")
    print(f"Minimum error: {accuracy_df['perc_error'].abs().min():.4f} ({accuracy_df['perc_error'].abs().min()*100:.2f}%)")
    print(f"Mean absolute error: {accuracy_df['perc_error'].abs().mean():.4f} ({accuracy_df['perc_error'].abs().mean()*100:.2f}%)")
    
    # Check missing roles
    missing_role_teams = 0
    for team_id in team_df['team_id'].unique():
        team_roles = set(team_df[team_df['team_id'] == team_id]['role'])
        required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
        if not required_roles.issubset(team_roles):
            missing_role_teams += 1
    
    print(f"Teams missing required roles: {missing_role_teams}")
    print("=" * 60)
    
    return accuracy_df

def main():
    print("Fantasy Cricket Team Simulation")
    print("=" * 40)
    
    # Load data
    df = pd.read_csv('player_data_sample.csv')
    print(f"Loaded {df.shape[0]} players")
    print(f"Role distribution: {df['role'].value_counts().to_dict()}")
    
    # Generate teams
    teams = generate_teams(df, num_teams=20000)
    
    if len(teams) < 19000:  # Allow some tolerance
        print(f"Warning: Only generated {len(teams)} teams (target: 20000)")
    
    # Create dataframe
    team_df = create_team_dataframe(teams, df)
    
    # Save team_df
    team_df.to_csv('team_df.csv', index=False)
    print(f"Saved team_df.csv: {len(team_df)} rows")
    
    # Evaluate accuracy
    accuracy_summary = evaluate_accuracy(team_df, df)
    
    # Save accuracy summary
    accuracy_summary.to_csv('accuracy_summary.csv', index=False)
    print("Saved accuracy_summary.csv")
    
    print("\nProject completed successfully!")
    print(f"Files generated:")
    print(f"- team_df.csv ({len(team_df):,} rows)")
    print(f"- accuracy_summary.csv ({len(accuracy_summary)} rows)")

if __name__ == "__main__":
    main()