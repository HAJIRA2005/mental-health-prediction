"""
Flask web application for Mental Health Prediction
"""
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load model and preprocessing objects
def load_model_components():
    """Load trained model, scaler, and label encoder"""
    model_path = Path(__file__).parent / "models" / "best_mental_health_model.pkl"
    data_path = Path(__file__).parent / "data" / "mental_health_data.csv"

    if not model_path.exists() or not data_path.exists():
        raise FileNotFoundError("Model or data file not found. Please run main.py first.")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data to fit scaler and encoder
    df = pd.read_csv(data_path)
    X = df.drop('mental_health', axis=1)

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['mental_health'])

    return model, scaler, label_encoder

# Load components at startup
try:
    model, scaler, label_encoder = load_model_components()
    print("Model and preprocessing components loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler, label_encoder = None, None, None

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run main.py first.'}), 500

    try:
        # Get form data
        features = {
            'sleep_duration': float(request.form['sleep_duration']),
            'exercise_frequency': float(request.form['exercise_frequency']),
            'social_interactions': float(request.form['social_interactions']),
            'stress_level': float(request.form['stress_level']),
            'work_hours': float(request.form['work_hours']),
            'nutrition_score': float(request.form['nutrition_score']),
            'meditation': float(request.form['meditation']),
            'screen_time': float(request.form['screen_time'])
        }

        # Create DataFrame and scale
        feature_df = pd.DataFrame([features])
        features_scaled = scaler.transform(feature_df)

        # Make prediction
        prediction_encoded = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        # Prepare response
        result = {
            'prediction': prediction,
            'probabilities': {
                class_name: round(prob * 100, 2)
                for class_name, prob in zip(label_encoder.classes_, probabilities)
            },
            'features': features
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)