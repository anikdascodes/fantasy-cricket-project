#!/usr/bin/env python3
"""
Fantasy Cricket Team Simulation - Quick Demo for Submission
Optimized version generating 10,000 teams for faster execution
"""

import pandas as pd
import numpy as np
import random
import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class QuickTeamGenerator:
    def __init__(self, df: pd.DataFrame, target_teams: int = 10000):
        self.df = df
        self.target_teams = target_teams
        self.teams = []
        self.unique_teams = set()
        self.player_selections = defaultdict(int)
        
        # Calculate targets
        self.player_targets = {}
        for _, player in self.df.iterrows():
            self.player_targets[player['player_code']] = int(player['perc_selection'] * target_teams)
        
        # Group players by role
        self.role_players = {}
        for role in self.df['role'].unique():
            self.role_players[role] = self.df[self.df['role'] == role]['player_code'].tolist()
    
    def calculate_priorities(self) -> Dict[int, float]:
        """Calculate selection priorities based on deficit"""
        priorities = {}
        teams_generated = len(self.teams)
        remaining_teams = max(1, self.target_teams - teams_generated)
        
        for player_code in self.df['player_code']:
            target = self.player_targets[player_code]
            current = self.player_selections[player_code]
            base_prob = self.df[self.df['player_code'] == player_code]['perc_selection'].iloc[0]
            
            deficit = target - current
            
            if deficit > 0:
                priority = (deficit / remaining_teams) * (1 + base_prob)
                if current < target * 0.8:
                    priority *= 1.5
            else:
                priority = base_prob * 0.2
            
            priorities[player_code] = max(0.001, priority)
        
        return priorities
    
    def generate_team(self, priorities: Dict[int, float]) -> List[int]:
        """Generate a single team using role constraints and priorities"""
        team = []
        used_players = set()
        
        # Ensure at least one from each role
        required_roles = ['Batsman', 'Bowler', 'WK', 'Allrounder']
        
        for role in required_roles:
            if role not in self.role_players:
                continue
                
            available = [p for p in self.role_players[role] if p not in used_players]
            if not available:
                return []
            
            # Select based on priorities
            role_priorities = [priorities[p] for p in available]
            total_priority = sum(role_priorities)
            
            if total_priority > 0:
                probs = [p / total_priority for p in role_priorities]
                selected = np.random.choice(available, p=probs)
            else:
                selected = np.random.choice(available)
            
            team.append(selected)
            used_players.add(selected)
        
        # Fill remaining 7 spots
        all_available = [p for p in self.df['player_code'] if p not in used_players]
        
        if len(all_available) < 7:
            return []
        
        # Select remaining players based on priorities
        remaining_priorities = [priorities[p] for p in all_available]
        total_priority = sum(remaining_priorities)
        
        if total_priority > 0:
            probs = [p / total_priority for p in remaining_priorities]
            remaining_selected = np.random.choice(all_available, size=7, replace=False, p=probs)
        else:
            remaining_selected = np.random.choice(all_available, size=7, replace=False)
        
        team.extend(remaining_selected)
        return team
    
    def validate_team(self, team: List[int]) -> bool:
        """Validate team has required roles"""
        if len(team) != 11 or len(set(team)) != 11:
            return False
        
        team_roles = set()
        for player_code in team:
            role = self.df[self.df['player_code'] == player_code]['role'].iloc[0]
            team_roles.add(role)
        
        required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
        return required_roles.issubset(team_roles)
    
    def generate_all_teams(self) -> List[List[int]]:
        """Generate all teams"""
        print(f"Starting team generation for {self.target_teams:,} teams...")
        start_time = time.time()
        
        attempts = 0
        max_attempts = self.target_teams * 5
        
        while len(self.teams) < self.target_teams and attempts < max_attempts:
            attempts += 1
            
            # Recalculate priorities every 1000 attempts
            if attempts % 1000 == 1:
                priorities = self.calculate_priorities()
                
                # Progress update
                if len(self.teams) > 0:
                    elapsed = time.time() - start_time
                    rate = len(self.teams) / elapsed
                    print(f"Generated {len(self.teams):,} teams in {elapsed:.1f}s ({rate:.0f}/sec)")
            
            team = self.generate_team(priorities)
            
            if not team or not self.validate_team(team):
                continue
            
            team_tuple = tuple(sorted(team))
            if team_tuple in self.unique_teams:
                continue
            
            # Accept team
            self.teams.append(team)
            self.unique_teams.add(team_tuple)
            
            # Update selections
            for player_code in team:
                self.player_selections[player_code] += 1
        
        total_time = time.time() - start_time
        print(f"Completed: {len(self.teams):,} teams in {total_time:.2f} seconds")
        print(f"Success rate: {len(self.teams)/attempts*100:.1f}%")
        
        return self.teams

def create_team_dataframe(teams: List[List[int]], df: pd.DataFrame) -> pd.DataFrame:
    """Create team_df in required format"""
    print(f"Creating team dataframe for {len(teams):,} teams...")
    
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

def evaluate_team_accuracy(team_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate accuracy and create summary"""
    print("Evaluating accuracy...")
    
    player_stats = []
    total_teams = team_df['team_id'].nunique()
    original_df = pd.read_csv('data/player_data_sample.csv')
    
    for _, player in original_df.iterrows():
        player_code = player['player_code']
        expected_selection = player['perc_selection']
        
        team_count = len(team_df[team_df['player_code'] == player_code]['team_id'].unique())
        actual_perc_selection = team_count / total_teams
        
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
    print(f"Players within +/-5% error: {players_within_5_percent} out of {total_players}")
    print(f"Success rate: {players_within_5_percent/total_players*100:.1f}%")
    print(f"Qualification threshold: 20 players within +/-5%")
    
    if players_within_5_percent >= 20:
        print("*** QUALIFICATION STATUS: PASSED! ***")
    else:
        print(f"QUALIFICATION STATUS: FAILED (need {20-players_within_5_percent} more)")
    
    print(f"\nMaximum error: {accuracy_df['perc_error'].abs().max():.4f}")
    print(f"Mean absolute error: {accuracy_df['perc_error'].abs().mean():.4f}")
    
    # Detailed results
    print("\nTop Performers (within +/-5%):")
    passed_players = accuracy_df[accuracy_df['within_5_percent']]
    for _, row in passed_players.iterrows():
        print(f"  {row['player_name']}: {row['perc_error']*100:+.1f}% error")
    
    print("=" * 70)
    
    return accuracy_df

def main():
    """Quick demo version for submission"""
    print("FANTASY CRICKET TEAM SIMULATION - SUBMISSION DEMO")
    print("Quick version generating 10,000 teams for demonstration")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/player_data_sample.csv')
    print(f"Loaded {df.shape[0]} players from dataset")
    
    print(f"\nRole distribution:")
    for role, count in df['role'].value_counts().items():
        print(f"  {role}: {count} players")
    
    # Generate teams
    generator = QuickTeamGenerator(df, target_teams=10000)
    teams = generator.generate_all_teams()
    
    print(f"\nGenerated {len(teams):,} unique teams!")
    
    # Create and save dataframe
    team_df = create_team_dataframe(teams, df)
    team_df.to_csv('team_df.csv', index=False)
    print(f"Saved team_df.csv: {len(team_df):,} rows")
    
    # Evaluate accuracy
    accuracy_summary = evaluate_team_accuracy(team_df)
    accuracy_summary.to_csv('accuracy_summary.csv', index=False)
    print("Saved accuracy_summary.csv")
    
    # Save evaluation output
    players_within_5_percent = accuracy_summary['within_5_percent'].sum()
    evaluation_text = f"""FANTASY CRICKET TEAM SIMULATION - ACCURACY EVALUATION
======================================================================
Total teams generated: {len(teams):,}
Total players: {len(accuracy_summary)}
Players within +/-5% error: {players_within_5_percent} out of {len(accuracy_summary)}
Success rate: {players_within_5_percent/len(accuracy_summary)*100:.1f}%
Qualification threshold: 20 players within +/-5%
QUALIFICATION STATUS: {'PASSED' if players_within_5_percent >= 20 else 'FAILED'}

Maximum error: {accuracy_summary['perc_error'].abs().max():.4f}
Mean absolute error: {accuracy_summary['perc_error'].abs().mean():.4f}
Teams missing required roles: 0
======================================================================"""
    
    with open('evaluation_output.txt', 'w') as f:
        f.write(evaluation_text)
    
    print("Saved evaluation_output.txt")
    
    # Final summary
    print(f"\nSUBMISSION FILES READY:")
    print(f"- quick_submission_demo.py (this script)")
    print(f"- team_df.csv ({len(team_df):,} rows)")
    print(f"- accuracy_summary.csv ({len(accuracy_summary)} players)")
    print(f"- evaluation_output.txt")
    
    if players_within_5_percent >= 20:
        print(f"\nSUCCESS! {players_within_5_percent}/22 players achieved target accuracy!")
    else:
        print(f"\nProgress: {players_within_5_percent}/22 players within target")
    
    print("\nSubmission ready for mahesh@apnacricketteam.com")

if __name__ == "__main__":
    main()