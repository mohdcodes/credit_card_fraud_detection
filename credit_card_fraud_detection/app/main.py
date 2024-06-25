# app/main.py
#done
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    model = joblib.load('../models/fraud_detector.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Streamlit app
st.title('Credit Card Fraud Detection')
uploaded_file = st.file_uploader('Upload a CSV file for prediction', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X = data.drop('isFraud', axis=1, errors='ignore')
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    data['isFraud_Prediction'] = predictions
    
    st.write(data)
    st.download_button('Download Predictions', data.to_csv(index=False), file_name='predictions.csv')
