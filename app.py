# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model_path = 'random_forest_model.joblib'
    scaler_path = 'scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error('Model or scaler file not found. Please ensure that you have run train_model.py and the files are in the correct directory.')
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to preprocess uploaded data
def preprocess_data(df, scaler):
    X = df.drop('Class', axis=1)
    X_scaled = scaler.transform(X)
    return X_scaled

def main():
    st.title('Credit Card Fraud Detection')

    # File upload
    st.subheader('Upload Transaction Data')
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.subheader('Uploaded Data')
        st.write(df.head())

        # Check if 'Class' column is present
        if 'Class' not in df.columns:
            st.error('Uploaded data does not contain the "Class" column.')
            st.stop()

        # Load model and scaler
        model, scaler = load_model()

        # Preprocess data
        X_test = preprocess_data(df, scaler)

        # Check the preprocessed data
        st.subheader('Preprocessed Data')
        st.write(X_test)

        # Make predictions
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # Display results
        st.subheader('Prediction Results')
        results_df = pd.DataFrame({
            'Prediction': y_pred,
            'Probability': y_probs
        })
        st.write(results_df)

        # ROC Curve for uploaded data
        st.subheader('ROC Curve')
        if 'Class' in df.columns:
            fpr, tpr, thresholds = roc_curve(df['Class'], y_probs)
            roc_auc = roc_auc_score(df['Class'], y_probs)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)
        else:
            st.warning('No "Class" column in uploaded data, cannot compute ROC curve.')

if __name__ == '__main__':
    main()
