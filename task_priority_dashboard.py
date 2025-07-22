import streamlit as st
import pandas as pd
import datetime
import joblib

# Load models
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_model = joblib.load("priority_xgboost.pkl")
label_encoder = joblib.load("priority_label_encoder.pkl")

# Load dataset containing users and workload
@st.cache_data
def load_data():
    return pd.read_csv("final_task_dataset_balanced.csv")

df = load_data()

# Title
st.title("📋 AI Task Assignment Dashboard (with Real Workload)")

# Form to input new task
with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict Priority & Assign")

if submitted:
    # Predict priority
    vectorized_task = vectorizer.transform([task_desc])
    encoded_pred = priority_model.predict(vectorized_task)[0]
    priority = label_encoder.inverse_transform([encoded_pred])[0]

    # Calculate urgency score
    today = datetime.date.today()
    days_left = (deadline - today).days
    urgency_score = max(0, 10 - days_left)

    # Use current user workload from dataset
    user_load_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_load_df["urgency_score"] = urgency_score
    user_load_df["combined_score"] = user_load_df["user_current_load"] + urgency_score

    # Choose user with lowest combined score
    assigned_user = user_load_df.sort_values("combined_score").iloc[0]["assigned_user"]

    # Add task to the dataframe
    new_task = {
        "task_id": f"task_{len(df)+1}",
        "task_description": task_desc,
        "deadline": deadline.strftime("%Y-%m-%d"),
        "priority": priority,
        "assigned_user": assigned_user,
        "user_current_load": user_load_df[user_load_df["assigned_user"] == assigned_user]["user_current_load"].values[0] + 1,
        "category": "",  # optional
        "status": "Pending"
    }

    df = pd.concat([df, pd.DataFrame([new_task])], ignore_index=True)

    # Show results
    st.success(f"✅ Task assigned to **{assigned_user}** with predicted priority: **{priority}**")
    st.info(f"🕐 Days left until deadline: {days_left}")
    st.dataframe(df.tail(1))

    # Allow download of updated dataset
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button("📥 Download Updated Dataset", data=csv, file_name="updated_task_data.csv", mime="text/csv")

# Show current workload table
st.subheader("📊 Current Average Workload per User")
workload_summary = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
st.dataframe(workload_summary)

