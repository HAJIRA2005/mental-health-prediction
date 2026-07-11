# Mental Health Prediction Using Classification

A machine learning project that predicts mental health status using classification algorithms.

## Project Overview

This project builds and evaluates multiple classification models to predict mental health status (Good, Moderate, or Poor) based on lifestyle and health factors.

**Features Used:**
- Sleep Duration (hours)
- Exercise Frequency (days/week)
- Social Interactions (score)
- Stress Level (score)
- Work Hours (hours/day)
- Nutrition Score
- Meditation (minutes/day)
- Screen Time (hours/day)

**Target Variable:**
- Mental Health Status: Good, Moderate, Poor

## Project Structure

```
MLT PROJECT/
├── data/
│   └── mental_health_data.csv          # Generated synthetic dataset
├── models/
│   ├── best_mental_health_model.pkl    # Trained model
│   ├── class_distribution.png          # Visualization
│   ├── feature_distribution.png        # Visualization
│   └── correlation_matrix.png          # Visualization
├── scripts/
│   ├── data_generator.py              # Generate synthetic data
│   ├── model_training.py              # Train classification models
│   ├── visualization.py               # Create visualizations
│   └── prediction.py                  # Make predictions
├── main.py                            # Main execution script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Models Used

1. **Logistic Regression** - Linear classification model
2. **Random Forest** - Ensemble method with multiple decision trees
3. **Support Vector Machine (SVM)** - Non-linear classification model

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start - Run Everything

Execute the complete pipeline:
```bash
python main.py
```

This will:
1. Generate a synthetic dataset (500 samples)
2. Create visualizations
3. Train all models
4. Save the best model
5. Make sample predictions

### Run Individual Components

**Generate Data:**
```bash
python scripts/data_generator.py
```

**Train Models:**
```bash
python scripts/model_training.py
```

**Create Visualizations:**
```bash
python scripts/visualization.py
```

**Make Predictions:**
```bash
python scripts/prediction.py
```

### Web Interface

**Start the Web Application:**
```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`

The web interface provides:
- Interactive form for mental health assessment
- Real-time predictions with confidence scores
- Beautiful, responsive design
- About page with project details

## Results Interpretation

The models output the following metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Ability to find all positive instances
- **F1 Score**: Harmonic mean of precision and recall

## Sample Predictions

Example outputs from three test cases:

**Healthy Individual:**
- Sleep: 8.5 hrs, Exercise: 5 days/week, Social: 8/10
- **Prediction: Good** (High confidence)

**Moderate Health Individual:**
- Sleep: 6.5 hrs, Exercise: 3 days/week, Social: 5/10
- **Prediction: Moderate** (Moderate confidence)

**At-Risk Individual:**
- Sleep: 5 hrs, Exercise: 1 day/week, Social: 2/10
- **Prediction: Poor** (High confidence)

## Key Findings

The models demonstrate that:
- Sleep duration is a strong indicator of mental health
- Exercise frequency significantly impacts mental well-being
- Social interactions are crucial for good mental health
- High stress levels are correlated with poor mental health
- Meditation and healthy nutrition support mental well-being

## Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms
- matplotlib: Data visualization
- seaborn: Statistical visualizations

## Future Enhancements

- [ ] Add more features (e.g., family history, medication)
- [ ] Implement deep learning models
- [ ] Add hyperparameter tuning
- [ ] Create a web interface for predictions
- [ ] Deploy as a REST API
- [ ] Add real patient data (with privacy compliance)
- [ ] Implement SHAP for model interpretability
- [ ] Add confidence intervals for predictions

## Performance

Current Model Performance:
- Best Model: Random Forest
- Average Accuracy: ~85-90%
- Cross-validation Score: Consistent across folds

## Disclaimer

This project is for educational purposes only. Mental health predictions should not be used for clinical diagnosis. Always consult with healthcare professionals for mental health concerns.

## Author

Created as a Machine Learning Techniques Mini Project

## License

Educational Use Only
