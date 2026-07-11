"""
Train and evaluate classification models for mental health prediction
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

class MentalHealthClassifier:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        
    def load_data(self, data_path):
        """Load dataset"""
        df = pd.read_csv(data_path)
        return df
    
    def preprocess_data(self, df):
        """Preprocess data - feature scaling and encoding"""
        # Separate features and target
        X = df.drop('mental_health', axis=1)
        y = df['mental_health']
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(X_train, y_train)
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""
        model = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return metrics, y_pred
    
    def train_all_models(self, data_path):
        """Train all models"""
        # Load and preprocess
        df = self.load_data(data_path)
        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data(df)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Training Set: {X_train.shape[0]}, Test Set: {X_test.shape[0]}")
        
        # Train models
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'], _ = self.evaluate_model(
            lr_model, X_test, y_test, "Logistic Regression"
        )
        
        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'], _ = self.evaluate_model(
            rf_model, X_test, y_test, "Random Forest"
        )
        
        # SVM
        svm_model = self.train_svm(X_train, y_train)
        self.models['SVM'] = svm_model
        self.results['SVM'], _ = self.evaluate_model(
            svm_model, X_test, y_test, "SVM"
        )
        
        # Save best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        return best_model, best_model_name, X_test, y_test
    
    def save_model(self, model, model_name, output_path):
        """Save trained model"""
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nBest Model ({model_name}) saved to {output_path}")

def main():
    data_path = Path(__file__).parent.parent / "data" / "mental_health_data.csv"
    model_output_path = Path(__file__).parent.parent / "models" / "best_mental_health_model.pkl"
    
    # Initialize classifier
    classifier = MentalHealthClassifier()
    
    # Train all models
    best_model, best_model_name, X_test, y_test = classifier.train_all_models(str(data_path))
    
    # Save best model
    classifier.save_model(best_model, best_model_name, str(model_output_path))
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {classifier.results[best_model_name]['accuracy']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
