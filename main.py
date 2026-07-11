"""
Main execution script for Mental Health Prediction Classification Project
Run this file to execute the entire pipeline
"""
import sys
from pathlib import Path

# Add scripts to path
scripts_path = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_path))

from data_generator import generate_mental_health_data, save_data
from model_training import MentalHealthClassifier
from visualization import plot_class_distribution, plot_data_distribution, plot_correlation_matrix
from prediction import load_model, predict_mental_health
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def main():
    print("\n" + "="*70)
    print(" "*15 + "MENTAL HEALTH PREDICTION CLASSIFICATION")
    print(" "*20 + "Mini Project Execution")
    print("="*70)
    
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "mental_health_data.csv"
    model_path = project_root / "models" / "best_mental_health_model.pkl"
    
    # Step 1: Generate Data
    print("\n[STEP 1] Generating Synthetic Dataset...")
    print("-" * 70)
    df = generate_mental_health_data(n_samples=500)
    save_data(df, str(data_path))
    
    # Step 2: Create Visualizations
    print("\n[STEP 2] Creating Visualizations...")
    print("-" * 70)
    print("Generating class distribution plot...")
    plot_class_distribution(
        df,
        output_path=project_root / "models" / "class_distribution.png"
    )
    
    print("Generating feature distribution plot...")
    plot_data_distribution(
        df,
        output_path=project_root / "models" / "feature_distribution.png"
    )
    
    print("Generating correlation matrix...")
    plot_correlation_matrix(
        df,
        output_path=project_root / "models" / "correlation_matrix.png"
    )
    
    # Step 3: Train Models
    print("\n[STEP 3] Training Classification Models...")
    print("-" * 70)
    classifier = MentalHealthClassifier()
    best_model, best_model_name, X_test, y_test = classifier.train_all_models(str(data_path))
    classifier.save_model(best_model, best_model_name, str(model_path))
    
    # Step 4: Display Results Summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    for model_name, metrics in classifier.results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Step 5: Make Sample Predictions
    print("\n[STEP 4] Making Sample Predictions...")
    print("-" * 70)
    
    # Load model and scaler
    model = load_model(str(model_path))
    
    # Fit scaler from data
    df = pd.read_csv(data_path)
    X = df.drop('mental_health', axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['mental_health'])
    
    # Test predictions
    test_cases = [
        {
            'name': 'Healthy Individual',
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
            'name': 'Moderate Health Individual',
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
            'name': 'At-Risk Individual',
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
        print(f"\n✓ {test_case['name']}")
        prediction, probability = predict_mental_health(
            test_case['features'], model, scaler, label_encoder
        )
        print(f"  Prediction: {prediction}")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"    {class_name:.<25} {probability[i]*100:>6.2f}%")
    
    # Final Summary
    print("\n" + "="*70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Output Files:")
    print(f"  • Dataset: {data_path}")
    print(f"  • Best Model: {model_path}")
    print(f"  • Visualizations: {project_root / 'models'}")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
