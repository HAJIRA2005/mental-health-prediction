"""
Make predictions using the trained model
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_mental_health(features, model, scaler, label_encoder):
    """
    Predict mental health status for given features
    
    features: dict with keys - sleep_duration, exercise_frequency, social_interactions,
              stress_level, work_hours, nutrition_score, meditation, screen_time
    """
    # Create dataframe from features
    feature_df = pd.DataFrame([features])
    
    # Scale features
    features_scaled = scaler.transform(feature_df)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Decode prediction
    health_status = label_encoder.inverse_transform([prediction])[0]
    
    return health_status, probability

def main():
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "best_mental_health_model.pkl"
    data_path = Path(__file__).parent.parent / "data" / "mental_health_data.csv"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run model_training.py first")
        return
    
    model = load_model(str(model_path))
    
    # Load and fit scaler from data
    df = pd.read_csv(data_path)
    X = df.drop('mental_health', axis=1)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Create label encoder for decoding
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['mental_health'])
    
    print("\n" + "="*60)
    print("MENTAL HEALTH PREDICTION SYSTEM")
    print("="*60)
    
    # Example predictions
    test_cases = [
        {
            'name': 'Healthy Person',
            'features': {
                'sleep_duration': 8.5,
                'exercise_frequency': 5,
                'social_interactions': 8,
                'stress_level': 2,
                'work_hours': 8,
                'nutrition_score': 8,
                'meditation': 30,
                'screen_time': 4
            }
        },
        {
            'name': 'Moderate Person',
            'features': {
                'sleep_duration': 6.5,
                'exercise_frequency': 3,
                'social_interactions': 5,
                'stress_level': 5,
                'work_hours': 9,
                'nutrition_score': 6,
                'meditation': 10,
                'screen_time': 7
            }
        },
        {
            'name': 'At-Risk Person',
            'features': {
                'sleep_duration': 5,
                'exercise_frequency': 1,
                'social_interactions': 2,
                'stress_level': 9,
                'work_hours': 11,
                'nutrition_score': 3,
                'meditation': 0,
                'screen_time': 9
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 60)
        
        for key, value in test_case['features'].items():
            print(f"  {key:.<25} {value:.1f}")
        
        prediction, probability = predict_mental_health(
            test_case['features'], model, scaler, label_encoder
        )
        
        print(f"\nPrediction: {prediction}")
        print(f"Confidence:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name:.<20} {probability[i]*100:.2f}%")

if __name__ == "__main__":
    main()
