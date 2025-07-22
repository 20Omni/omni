
import streamlit as st
import joblib
import pandas as pd

# Load all models and components
tfidf_vectorizer = joblib.load('priority_tfidf_vectorizer.pkl')
label_encoder = joblib.load('priority_label_encoder.pkl')
rf_model = joblib.load('priority_random_forest.pkl')
xgb_model = joblib.load('priority_xgboost.pkl')
optimized_rf_model = joblib.load('optimized_rf_model.pkl')

st.title("🧠 AI-Powered Task Priority Predictor")

# Text input
task_description = st.text_area("📝 Enter your task description")

# Model selection
model_choice = st.selectbox("🤖 Choose Model for Prediction", 
                            ['Random Forest', 'XGBoost', 'Optimized Random Forest'])

# Predict button
if st.button("🎯 Predict Priority"):
    if task_description.strip() == "":
        st.warning("Please enter a task description.")
    else:
        # Vectorize the input
        vectorized_input = tfidf_vectorizer.transform([task_description])

        # Predict based on model selected
        if model_choice == 'Random Forest':
            prediction = rf_model.predict(vectorized_input)
        elif model_choice == 'XGBoost':
            prediction = xgb_model.predict(vectorized_input)
        else:
            prediction = optimized_rf_model.predict(vectorized_input)

        # Decode label
        priority_label = label_encoder.inverse_transform(prediction)[0]

        # Show result
        st.success(f"🔔 Predicted Priority: **{priority_label}**")
