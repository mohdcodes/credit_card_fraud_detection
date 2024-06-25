# scripts/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Evaluate model
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    test_data = load_data('../data/fraudTest.csv')
    X_test = test_data.drop('isFraud', axis=1)
    y_test = test_data['isFraud']

    model = joblib.load('../models/fraud_detector.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    accuracy, report = evaluate_model(model, scaler, X_test, y_test)
    
    print(f'Model Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{report}')
