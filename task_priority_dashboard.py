import streamlit as st
import pandas as pd
import datetime
import joblib

# --- Load Models ---
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_model = joblib.load("priority_xgboost.pkl")
label_encoder = joblib.load("priority_label_encoder.pkl")

# --- Sample Users List ---
users = ["Alice", "Bob", "Charlie", "David"]

# --- Initialize Workload Tracker ---
if "user_workload" not in st.session_state:
    st.session_state.user_workload = {user: 0 for user in users}

# --- Title ---
st.title("🧠 AI Task Assignment Assistant")

# --- Task Form ---
with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Assign Task")

if submitted:
    if not task_desc.strip():
        st.warning("Please enter a task description.")
    else:
        # --- Preprocess and Predict ---
        vectorized = vectorizer.transform([task_desc])
        encoded_pred = priority_model.predict(vectorized)[0]
        predicted_priority = label_encoder.inverse_transform([encoded_pred])[0]

        # --- Deadline Urgency ---
        today = datetime.date.today()
        days_left = (deadline - today).days
        urgency_score = max(0, 10 - days_left)

        # --- Assign Task to User with Minimum (Workload + Urgency) ---
        best_user = min(
            users,
            key=lambda u: st.session_state.user_workload[u] + urgency_score
        )
        st.session_state.user_workload[best_user] += 1

        # --- Output ---
        st.success(f"✅ Task assigned to **{best_user}**")
        st.info(f"🧠 Predicted Priority: **{predicted_priority}**")
        st.write(f"⏳ Days Left: `{days_left}`")

        st.subheader("📊 Current Workload")
        st.write(pd.DataFrame.from_dict(st.session_state.user_workload, orient="index", columns=["Tasks Assigned"]))

# --- Reset Button ---
if st.button("🔄 Reset Workload"):
    for user in users:
        st.session_state.user_workload[user] = 0
    st.success("Workload reset!")


