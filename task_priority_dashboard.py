import streamlit as st
import pandas as pd
import datetime
import joblib

# =====================
# 🔹 Load Models & Encoders
# =====================
priority_model = joblib.load("priority_xgboost.pkl")  # or priority_random_forest.pkl
priority_label_encoder = joblib.load("priority_label_encoder.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")

rf_model = joblib.load("optimized_rf_model.pkl")
category_label_encoder = joblib.load("category_label_encoder.pkl")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# =====================
# 🔹 Load Dataset (to access user workload)
# =====================
@st.cache_data
def load_data():
    return pd.read_csv("final_task_dataset_balanced.csv")

df = load_data()

# =====================
# 🔹 Streamlit UI
# =====================
st.title("🧠 AI Task Assignment Dashboard")

with st.form("task_form"):
    task_desc = st.text_area("📝 Enter Task Description")
    deadline = st.date_input("📅 Deadline", min_value=datetime.date.today())
    submitted = st.form_submit_button("Predict & Assign")

if submitted:
    # 🔸 Vectorize task description
    task_vector_priority = priority_vectorizer.transform([task_desc])
    task_vector_category = task_vectorizer.transform([task_desc])

    # 🔸 Predict priority
    pred_priority_enc = priority_model.predict(task_vector_priority)[0]
    pred_priority = priority_label_encoder.inverse_transform([pred_priority_enc])[0]

    # 🔸 Predict category
    pred_category_enc = rf_model.predict(task_vector_category)[0]
    pred_category = category_label_encoder.inverse_transform([pred_category_enc])[0]

    # 🔸 Compute deadline urgency
    today = datetime.date.today()
    days_left = (deadline - today).days
    urgency_score = max(0, 10 - days_left)

    # 🔸 Determine user workload
    user_workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_workload_df["urgency_score"] = urgency_score
    user_workload_df["combined_score"] = user_workload_df["user_current_load"] + urgency_score

    # 🔸 Assign task to user with lowest combined score
    assigned_user = user_workload_df.sort_values("combined_score").iloc[0]["assigned_user"]

    # ✅ Final Output
    st.success(f"✅ Task Assigned to: **{assigned_user}**")
    st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days to Deadline: {days_left}")

    # 🔸 Show only assigned user’s workload
    current_load = user_workload_df[user_workload_df["assigned_user"] == assigned_user]["user_current_load"].values[0]
    st.write("📊 **Current Workload of Assigned User:**")
    st.write(f"**{assigned_user}** has **{int(current_load)}** tasks currently.")
