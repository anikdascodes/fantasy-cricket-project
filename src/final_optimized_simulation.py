#!/usr/bin/env python3
"""
Final Optimized Fantasy Cricket Team Simulation
Mathematical approach to achieve 20/22 players within ±5% error target
"""

import pandas as pd
import numpy as np
import random
import time
from typing import List, Dict, Set
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class MathematicalTeamGenerator:
    def __init__(self, df: pd.DataFrame, target_teams: int = 20000):
        self.df = df
        self.target_teams = target_teams
        self.teams = []
        self.unique_teams = set()
        self.player_selections = defaultdict(int)
        
        # Calculate exact targets
        self.player_targets = {}
        for _, player in self.df.iterrows():
            self.player_targets[player['player_code']] = int(player['perc_selection'] * target_teams)
        
        # Group players by role
        self.role_players = {}
        for role in self.df['role'].unique():
            self.role_players[role] = self.df[self.df['role'] == role]['player_code'].tolist()
        
        # Pre-calculate valid team templates
        self.team_templates = self._generate_team_templates()
        
    def _generate_team_templates(self) -> List[Dict]:
        """Generate different team composition templates"""
        templates = []
        
        # Template 1: Minimum requirements + balanced
        templates.append({
            'Batsman': 3, 'Bowler': 3, 'WK': 1, 'Allrounder': 4
        })
        
        # Template 2: Batsman heavy
        templates.append({
            'Batsman': 4, 'Bowler': 3, 'WK': 1, 'Allrounder': 3
        })
        
        # Template 3: Bowler heavy
        templates.append({
            'Batsman': 3, 'Bowler': 4, 'WK': 1, 'Allrounder': 3
        })
        
        # Template 4: Allrounder heavy
        templates.append({
            'Batsman': 2, 'Bowler': 2, 'WK': 2, 'Allrounder': 5
        })
        
        return templates
    
    def calculate_selection_priority(self) -> Dict[int, float]:
        """Calculate priority scores for each player based on target deficit"""
        priorities = {}
        teams_generated = len(self.teams)
        
        for player_code in self.df['player_code']:
            target = self.player_targets[player_code]
            current = self.player_selections[player_code]
            remaining_teams = self.target_teams - teams_generated
            
            if remaining_teams <= 0:
                priorities[player_code] = 0
                continue
            
            # Calculate how far behind/ahead we are
            deficit = target - current
            
            # Priority based on deficit and remaining opportunity
            if deficit > 0:
                # Behind target - high priority
                priority = deficit / remaining_teams
                # Boost if significantly behind
                if current < target * 0.8:
                    priority *= 2.0
            else:
                # Ahead of target - low priority
                priority = 0.1 / remaining_teams
            
            priorities[player_code] = max(0.001, priority)
        
        return priorities
    
    def generate_team_from_template(self, template: Dict[str, int], priorities: Dict[int, float]) -> List[int]:
        """Generate a team using a specific template and priorities"""
        team = []
        
        for role, count in template.items():
            if role not in self.role_players:
                continue
                
            # Get available players for this role
            available_players = [p for p in self.role_players[role] if p not in team]
            
            if len(available_players) < count:
                return []  # Cannot satisfy template
            
            # Calculate weights for role players
            role_priorities = [priorities[p] for p in available_players]
            total_priority = sum(role_priorities)
            
            if total_priority > 0:
                # Convert to probabilities
                probs = [p / total_priority for p in role_priorities]
                
                # Select players for this role
                selected = np.random.choice(
                    available_players, 
                    size=count, 
                    replace=False, 
                    p=probs
                )
                team.extend(selected)
            else:
                # Fallback to random selection
                selected = np.random.choice(available_players, size=count, replace=False)
                team.extend(selected)
        
        return team
    
    def validate_team(self, team: List[int]) -> bool:
        """Validate team composition"""
        if len(team) != 11 or len(set(team)) != 11:
            return False
        
        # Check role requirements
        team_roles = set()
        for player_code in team:
            role = self.df[self.df['player_code'] == player_code]['role'].iloc[0]
            team_roles.add(role)
        
        required_roles = {'Batsman', 'Bowler', 'WK', 'Allrounder'}
        return required_roles.issubset(team_roles)
    
    def generate_teams_batch(self, batch_size: int = 1000) -> List[List[int]]:
        """Generate teams in batches for efficiency"""
        new_teams = []
        max_attempts = batch_size * 5
        attempts = 0
        
        while len(new_teams) < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Calculate current priorities
            priorities = self.calculate_selection_priority()
            
            # Choose template (rotate through them)
            template = self.team_templates[attempts % len(self.team_templates)]
            
            # Generate team
            team = self.generate_team_from_template(template, priorities)
            
            if not team:
                continue
            
            # Check validity and uniqueness
            team_tuple = tuple(sorted(team))
            if (self.validate_team(team) and team_tuple not in self.unique_teams):
                new_teams.append(team)
                self.unique_teams.add(team_tuple)
                self.teams.append(team)
                
                # Update selections
                for player_code in team:
                    self.player_selections[player_code] += 1
        
        return new_teams
    
    def generate_all_teams(self) -> List[List[int]]:
        """Generate all teams using batch approach"""
        print(f"Starting mathematical team generation for {self.target_teams} teams...")
        start_time = time.time()
        
        batch_size = 1000
        total_batches = self.target_teams // batch_size
        
        for batch_num in range(total_batches):
            batch_teams = self.generate_teams_batch(batch_size)
            
            elapsed = time.time() - start_time
            print(f"Batch {batch_num + 1}/{total_batches}: Generated {len(batch_teams)} teams "
                  f"(Total: {len(self.teams)}) in {elapsed:.1f}s")
            
            # Check accuracy every few batches
            if (batch_num + 1) % 5 == 0:
                accuracy = self._check_current_accuracy()
                print(f"  Current accuracy: {accuracy}/22 players within ±5%")
        
        # Generate remaining teams if needed
        remaining = self.target_teams - len(self.teams)
        if remaining > 0:
            final_batch = self.generate_teams_batch(remaining)
            print(f"Final batch: Generated {len(final_batch)} teams")
        
        total_time = time.time() - start_time
        print(f"Completed: {len(self.teams)} teams in {total_time:.2f} seconds")
        
        return self.teams
    
    def _check_current_accuracy(self) -> int:
        """Check current accuracy"""
        if len(self.teams) == 0:
            return 0
        
        accurate_count = 0
        total_teams = len(self.teams)
        
        for player_code in self.df['player_code']:
            expected = self.df[self.df['player_code'] == player_code]['perc_selection'].iloc[0]
            actual = self.player_selections[player_code] / total_teams
            
            if expected > 0:
                error = abs((actual - expected) / expected)
                if error <= 0.05:
                    accurate_count += 1
        
        return accurate_count

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

def evaluate_team_accuracy(team_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate accuracy and print results"""
    print("Evaluating accuracy...")
    
    player_stats = []
    total_teams = team_df['team_id'].nunique()
    
    # Get original player data
    original_df = pd.read_csv('player_data_sample.csv')
    
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
    print("FANTASY CRICKET TEAM SIMULATION - FINAL ACCURACY EVALUATION")
    print("=" * 70)
    print(f"Total teams generated: {total_teams:,}")
    print(f"Total players: {total_players}")
    print(f"Players within ±5% error: {players_within_5_percent} out of {total_players}")
    print(f"Success rate: {players_within_5_percent/total_players*100:.1f}%")
    print(f"Qualification threshold: 20 players within ±5%")
    
    if players_within_5_percent >= 20:
        print("*** QUALIFICATION STATUS: PASSED! ***")
    else:
        print(f"QUALIFICATION STATUS: FAILED (need {20-players_within_5_percent} more)")
    
    print()
    print(f"Maximum error: {accuracy_df['perc_error'].abs().max():.4f} ({accuracy_df['perc_error'].abs().max()*100:.2f}%)")
    print(f"Minimum error: {accuracy_df['perc_error'].abs().min():.4f} ({accuracy_df['perc_error'].abs().min()*100:.2f}%)")
    print(f"Mean absolute error: {accuracy_df['perc_error'].abs().mean():.4f} ({accuracy_df['perc_error'].abs().mean()*100:.2f}%)")
    
    # Detailed results
    print("\nDetailed Player Accuracy:")
    print("Player Name\t\tRole\t\tExpected\tActual\t\tError%\t\tStatus")
    print("-" * 70)
    
    # Sort by accuracy status (pass first, then by error)
    sorted_df = accuracy_df.sort_values(['within_5_percent', 'perc_error'], ascending=[False, True])
    
    for _, row in sorted_df.iterrows():
        status = "PASS" if row['within_5_percent'] else "FAIL"
        print(f"{row['player_name'][:15]:<15}\t{row['role']:<12}\t{row['expected_perc_selection']:.3f}\t\t{row['actual_perc_selection']:.3f}\t\t{row['perc_error']*100:+6.1f}%\t\t{status}")
    
    # Team validation
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
    print("FINAL OPTIMIZED Fantasy Cricket Team Simulation")
    print("Mathematical approach for accuracy target achievement")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('player_data_sample.csv')
    print(f"Loaded {df.shape[0]} players")
    
    # Show targets
    print("\nTarget vs Expected (for 20,000 teams):")
    print("Player\t\tTarget\tExpected%")
    print("-" * 40)
    for _, player in df.iterrows():
        target = int(player['perc_selection'] * 20000)
        print(f"{player['player_name'][:15]:<15}\t{target:5d}\t{player['perc_selection']:.1%}")
    
    # Generate teams
    generator = MathematicalTeamGenerator(df, target_teams=20000)
    teams = generator.generate_all_teams()
    
    print(f"\nGenerated {len(teams)} teams successfully!")
    
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
    with open('evaluation_output.txt', 'w') as f:
        f.write("FANTASY CRICKET TEAM SIMULATION - FINAL ACCURACY EVALUATION\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total teams generated: {len(teams):,}\n")
        f.write(f"Total players: {len(accuracy_summary)}\n")
        f.write(f"Players within ±5% error: {players_within_5_percent} out of {len(accuracy_summary)}\n")
        f.write(f"Success rate: {players_within_5_percent/len(accuracy_summary)*100:.1f}%\n")
        f.write(f"Qualification threshold: 20 players within ±5%\n")
        f.write(f"QUALIFICATION STATUS: {'PASSED' if players_within_5_percent >= 20 else 'FAILED'}\n")
        f.write(f"\nMaximum error: {accuracy_summary['perc_error'].abs().max():.4f}\n")
        f.write(f"Mean absolute error: {accuracy_summary['perc_error'].abs().mean():.4f}\n")
        f.write(f"Teams missing required roles: 0\n")
        f.write("=" * 70 + "\n")
    
    print("Saved evaluation_output.txt")
    
    # Final status
    if players_within_5_percent >= 20:
        print(f"\n*** PROJECT SUCCESS! ***")
        print(f"{players_within_5_percent}/22 players achieved ±5% accuracy target!")
        print("Ready for submission to mahesh@apnacricketteam.com")
    else:
        print(f"\nProgress: {players_within_5_percent}/22 players within target")
        print(f"Need {20-players_within_5_percent} more players to meet qualification")
    
    print(f"\nFinal deliverables ready:")
    print(f"- team_df.csv ({len(team_df):,} rows)")
    print(f"- accuracy_summary.csv ({len(accuracy_summary)} rows)")
    print(f"- evaluation_output.txt")
    print(f"- final_optimized_simulation.py")

if __name__ == "__main__":
    main()