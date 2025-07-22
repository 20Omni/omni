import streamlit as st
import pandas as pd
import datetime
import joblib

# 🔹 Load models
vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_model = joblib.load("priority_xgboost.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

rf_model = joblib.load("optimized_rf_model.pkl")  # For category prediction
category_label_encoder = joblib.load("category_label_encoder.pkl")

# 🔹 Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("final_task_dataset_balanced.csv")

df = load_data()

# 🔹 Title
st.title("🧠 AI Task Assignment Dashboard")

# 🔹 Task Input Form
with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign")

if submitted:
    # 👉 Vectorize task description
    task_vector = vectorizer.transform([task_desc])

    # 🔹 Predict Priority
    pred_priority_enc = priority_model.predict(task_vector)[0]
    pred_priority = priority_label_encoder.inverse_transform([pred_priority_enc])[0]

    # 🔹 Predict Category
    pred_category_enc = rf_model.predict(task_vector)[0]
    pred_category = category_label_encoder.inverse_transform([pred_category_enc])[0]

    # 🔹 Compute deadline urgency
    today = datetime.date.today()
    days_left = (deadline - today).days
    urgency_score = max(0, 10 - days_left)

    # 🔹 Determine best user to assign based on workload
    user_workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_workload_df["urgency_score"] = urgency_score
    user_workload_df["combined_score"] = user_workload_df["user_current_load"] + urgency_score

    assigned_user = user_workload_df.sort_values("combined_score").iloc[0]["assigned_user"]

    # 🔹 Display assignment result
    st.success(f"✅ Task Assigned to: **{assigned_user}**")
    st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days to deadline: {days_left}")

    # 🔹 Show current load for only the assigned user
    current_load = user_workload_df[user_workload_df["assigned_user"] == assigned_user]["user_current_load"].values[0]
    st.write("📊 **Current Workload of Assigned User:**")
    st.write(f"**{assigned_user}** has **{int(current_load)}** tasks currently.")

    # 🔹 Optionally save updated task (code here only if needed)
    # new_task = {...}
    # df = pd.concat([df, pd.DataFrame([new_task])], ignore_index=True)

# (Optional) Reset session or reload can be added if needed
