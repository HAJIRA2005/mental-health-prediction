"""
Generate synthetic mental health dataset for classification
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_mental_health_data(n_samples=500, random_state=42):
    """
    Generate synthetic mental health dataset
    
    Features:
    - Sleep Duration (hours): 4-10
    - Exercise Frequency (days/week): 0-7
    - Social Interactions (score): 1-10
    - Stress Level (score): 1-10
    - Work Hours (hours/day): 4-12
    - Nutrition Score (1-10): 1-10
    - Meditation (minutes/day): 0-60
    - Screen Time (hours/day): 2-10
    
    Target:
    - Mental Health Status: Good, Moderate, Poor
    """
    np.random.seed(random_state)
    
    n_good = int(n_samples * 0.5)      # 50% Good
    n_moderate = int(n_samples * 0.3)  # 30% Moderate
    n_poor = n_samples - n_good - n_moderate  # 20% Poor
    
    data = []
    
    # Good mental health
    for _ in range(n_good):
        data.append({
            'sleep_duration': np.random.uniform(7, 9),
            'exercise_frequency': np.random.uniform(4, 7),
            'social_interactions': np.random.uniform(7, 10),
            'stress_level': np.random.uniform(1, 4),
            'work_hours': np.random.uniform(7, 9),
            'nutrition_score': np.random.uniform(7, 10),
            'meditation': np.random.uniform(15, 60),
            'screen_time': np.random.uniform(3, 6),
            'mental_health': 'Good'
        })
    
    # Moderate mental health
    for _ in range(n_moderate):
        data.append({
            'sleep_duration': np.random.uniform(6, 7.5),
            'exercise_frequency': np.random.uniform(2, 4),
            'social_interactions': np.random.uniform(4, 7),
            'stress_level': np.random.uniform(4, 7),
            'work_hours': np.random.uniform(8, 10),
            'nutrition_score': np.random.uniform(5, 7),
            'meditation': np.random.uniform(5, 20),
            'screen_time': np.random.uniform(6, 8),
            'mental_health': 'Moderate'
        })
    
    # Poor mental health
    for _ in range(n_poor):
        data.append({
            'sleep_duration': np.random.uniform(4, 6),
            'exercise_frequency': np.random.uniform(0, 2),
            'social_interactions': np.random.uniform(1, 4),
            'stress_level': np.random.uniform(7, 10),
            'work_hours': np.random.uniform(10, 12),
            'nutrition_score': np.random.uniform(2, 5),
            'meditation': np.random.uniform(0, 5),
            'screen_time': np.random.uniform(8, 10),
            'mental_health': 'Poor'
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

def save_data(df, output_path):
    """Save dataset to CSV"""
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nDataset Info:")
    print(df.head())
    print(f"\nClass Distribution:")
    print(df['mental_health'].value_counts())

if __name__ == "__main__":
    # Generate dataset
    df = generate_mental_health_data(n_samples=500)
    
    # Save dataset
    data_path = Path(__file__).parent.parent / "data" / "mental_health_data.csv"
    save_data(df, str(data_path))
