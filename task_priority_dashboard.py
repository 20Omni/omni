code = '''import streamlit as st
import joblib

# Load vectorizer and label encoder
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
label_encoder = joblib.load("priority_label_encoder.pkl")

# Load models
models = {
    "Optimized Random Forest": joblib.load("optimized_rf_model.pkl"),
    "Random Forest": joblib.load("priority_random_forest.pkl"),
    "XGBoost": joblib.load("priority_xgboost.pkl")
}

# UI
st.title("🧠 AI-Powered Task Priority Predictor")
st.subheader("📝 Enter your task description")

task_input = st.text_area("Task", placeholder="e.g. Send status update to manager")

selected_model_name = st.selectbox("🤖 Choose Model for Prediction", list(models.keys()))

if st.button("🔍 Predict Priority"):
    if task_input.strip() == "":
        st.warning("Please enter a task description.")
    else:
        try:
            # Vectorize input
            input_vector = vectorizer.transform([task_input])

            # Predict using selected model
            model = models[selected_model_name]
            prediction = model.predict(input_vector)

            # Decode label
            predicted_label = label_encoder.inverse_transform(prediction)[0]

            st.success(f"✅ Predicted Priority: {predicted_label}")
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")
'''
with open("task_priority_dashboard.py", "w") as f:
    f.write(code)

print("✅ File saved as task_priority_dashboard.py")

