#!/usr/bin/env python3
"""
Improved Fantasy Cricket Team Simulation with Target-Based Selection
"""

import pandas as pd
import numpy as np
import random
import time
from typing import List, Dict
from collections import defaultdict

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

def generate_teams_with_target_tracking(df: pd.DataFrame, num_teams: int = 20000) -> List[List[int]]:
    """
    Generate teams with explicit target tracking to match perc_selection values
    """
    teams = []
    unique_teams = set()
    
    # Calculate target counts for each player
    player_targets = {}
    player_actual_counts = defaultdict(int)
    
    for _, player in df.iterrows():
        player_code = player['player_code']
        target_count = int(player['perc_selection'] * num_teams)
        player_targets[player_code] = target_count
    
    # Pre-calculate role groups
    role_groups = {}
    for role in df['role'].unique():
        role_df = df[df['role'] == role].copy()
        role_groups[role] = role_df['player_code'].tolist()
    
    print(f"Starting generation of {num_teams} teams with target tracking...")
    start_time = time.time()
    
    attempts = 0
    max_attempts = num_teams * 5
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    while len(teams) < num_teams and attempts < max_attempts and consecutive_failures < max_consecutive_failures:
        attempts += 1
        
        # Calculate current selection probabilities based on remaining targets
        current_probs = {}
        for player_code in df['player_code']:
            remaining_target = max(0, player_targets[player_code] - player_actual_counts[player_code])
            remaining_teams = num_teams - len(teams)
            
            if remaining_teams > 0:
                # Probability based on how much we need to catch up
                base_prob = remaining_target / remaining_teams
                # Add small random factor to avoid deterministic patterns
                current_probs[player_code] = max(0.001, base_prob + np.random.normal(0, 0.01))
            else:
                current_probs[player_code] = 0.001
        
        team = []
        
        # Ensure at least one player from each role
        for role in ['Batsman', 'Bowler', 'WK', 'Allrounder']:
            if role in role_groups:
                # Select from this role based on adjusted probabilities
                role_players = role_groups[role]
                role_probs = [current_probs[p] for p in role_players]
                role_probs = np.array(role_probs)
                role_probs = role_probs / role_probs.sum()
                
                selected = np.random.choice(role_players, p=role_probs)
                team.append(selected)
        
        # Fill remaining spots
        remaining_spots = 11 - len(team)
        available_players = [p for p in df['player_code'] if p not in team]
        
        if len(available_players) >= remaining_spots:
            available_probs = [current_probs[p] for p in available_players]
            available_probs = np.array(available_probs)
            available_probs = available_probs / available_probs.sum()
            
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
            
            # Update actual counts
            for player_code in team:
                player_actual_counts[player_code] += 1
            
            consecutive_failures = 0
            
            if len(teams) % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"Generated {len(teams)} teams in {elapsed:.1f}s...")
        else:
            consecutive_failures += 1
    
    total_time = time.time() - start_time
    print(f"Completed: {len(teams)} teams in {total_time:.2f} seconds")
    
    # Print target vs actual summary
    print("\nTarget vs Actual Summary (top 10 players):")
    print("Player\t\tTarget\tActual\tError%")
    for player_code in sorted(player_targets.keys())[:10]:
        target = player_targets[player_code]
        actual = player_actual_counts[player_code]
        error_pct = ((actual - target) / target * 100) if target > 0 else 0
        player_name = df[df['player_code'] == player_code]['player_name'].iloc[0]
        print(f"{player_name[:12]:<12}\t{target}\t{actual}\t{error_pct:.1f}%")
    
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
    
    print("=" * 70)
    print("FANTASY CRICKET TEAM SIMULATION - ACCURACY EVALUATION")
    print("=" * 70)
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
    
    # Detailed breakdown
    print("\nDetailed Player Accuracy:")
    print("Player\t\tRole\t\tExpected\tActual\t\tError%\tWithin 5%")
    print("-" * 70)
    for _, row in accuracy_df.iterrows():
        status = "✓" if row['within_5_percent'] else "✗"
        print(f"{row['player_name'][:12]:<12}\t{row['role']:<12}\t{row['expected_perc_selection']:.3f}\t\t{row['actual_perc_selection']:.3f}\t\t{row['perc_error']*100:+.1f}%\t{status}")
    
    # Check missing roles
    missing_role_teams = 0
    for team_id in team_df['team_id'].unique():
        team_roles = set(team_df[team_df['team_id'] == team_id]['role'])
        required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
        if not required_roles.issubset(team_roles):
            missing_role_teams += 1
    
    print(f"\nTeams missing required roles: {missing_role_teams}")
    print("=" * 70)
    
    return accuracy_df

def main():
    print("Fantasy Cricket Team Simulation - Improved Algorithm")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('player_data_sample.csv')
    print(f"Loaded {df.shape[0]} players")
    print(f"Role distribution: {df['role'].value_counts().to_dict()}")
    
    # Show target counts
    print("\nTarget selection counts (for 20,000 teams):")
    for _, player in df.iterrows():
        target = int(player['perc_selection'] * 20000)
        print(f"{player['player_name'][:15]:<15}: {target:5d} teams ({player['perc_selection']:.3f})")
    
    # Generate teams
    teams = generate_teams_with_target_tracking(df, num_teams=20000)
    
    if len(teams) < 19000:  # Allow some tolerance
        print(f"Warning: Only generated {len(teams)} teams (target: 20000)")
    
    # Create dataframe
    team_df = create_team_dataframe(teams, df)
    
    # Save team_df
    team_df.to_csv('team_df.csv', index=False)
    print(f"\nSaved team_df.csv: {len(team_df):,} rows")
    
    # Evaluate accuracy
    accuracy_summary = evaluate_accuracy(team_df, df)
    
    # Save accuracy summary
    accuracy_summary.to_csv('accuracy_summary.csv', index=False)
    print("\nSaved accuracy_summary.csv")
    
    print("\nProject completed successfully!")
    print(f"Files generated:")
    print(f"- team_df.csv ({len(team_df):,} rows)")
    print(f"- accuracy_summary.csv ({len(accuracy_summary)} rows)")

if __name__ == "__main__":
    main()