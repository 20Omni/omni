import streamlit as st
import pandas as pd
import joblib
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models and vectorizer
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_model = joblib.load("priority_xgboost.pkl")
label_encoder = joblib.load("priority_label_encoder.pkl")

# Load dataset with real users
@st.cache_data
def load_user_data():
    df = pd.read_csv("final_task_dataset_balanced.csv")
    return df['assigned_user'].dropna().unique().tolist()

users = load_user_data()

# Initialize session state for workload tracking
if "user_workload" not in st.session_state:
    st.session_state.user_workload = {user: 0 for user in users}

# Title
st.title("📋 AI Task Assignment Dashboard")

# Task input form
with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign Task")

if submitted:
    if not task_desc:
        st.error("❗ Please enter a task description.")
    else:
        try:
            task_vector = vectorizer.transform([task_desc])
            pred_encoded = priority_model.predict(task_vector)[0]
            pred_priority = label_encoder.inverse_transform([pred_encoded])[0]

            # Deadline urgency score
            today = datetime.date.today()
            days_left = (deadline - today).days
            deadline_score = max(0, 10 - days_left)

            # Compute workload score
            user_scores = []
            for user in users:
                load = st.session_state.user_workload.get(user, 0)
                total_score = load + deadline_score
                user_scores.append((user, total_score))

            assigned_user = sorted(user_scores, key=lambda x: x[1])[0][0]

            # Update workload
            st.session_state.user_workload[assigned_user] += 1

            st.success(f"✅ Task assigned to **{assigned_user}** with predicted priority: **{pred_priority}**")
            st.info(f"🕐 Days until deadline: {days_left}")
            st.write("📊 Current Workload per User:")
            st.write(pd.DataFrame.from_dict(st.session_state.user_workload, orient='index', columns=["Tasks Assigned"]))

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")

# Reset workload
if st.button("🔁 Reset Workload"):
    st.session_state.user_workload = {user: 0 for user in users}
    st.success("Workload has been reset.")
