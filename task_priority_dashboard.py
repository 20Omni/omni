import streamlit as st
import pandas as pd
import datetime
import joblib

# =====================
# 🔹 Load Models & Encoders
# =====================
priority_model = joblib.load("priority_xgboost.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")

rf_model = joblib.load("optimized_rf_model.pkl")
category_label_encoder = joblib.load("category_label_encoder.pkl")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# =====================
# 🔹 Load Dataset (for user workload + historical data)
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

    # 🔸 User workload
    workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    workload_df.columns = ["user", "user_current_load"]

    # 🔸 Filter users who handled same category
    category_matches = df[df["category"] == pred_category]
    matching_users = category_matches["assigned_user"].value_counts().index.tolist()

    # 🔸 Filter out high workload users (> 20)
    matching_users_filtered = []
    for user in matching_users:
        current_load = workload_df[workload_df["user"] == user]["user_current_load"]
        if not current_load.empty and current_load.values[0] <= 20:
            matching_users_filtered.append((user, current_load.values[0]))

    # 🔸 Choose user
    if matching_users_filtered:
        # Select user with lowest workload among filtered matches
        assigned_user = sorted(matching_users_filtered, key=lambda x: x[1])[0][0]
    else:
        # Fall back to lowest workload + urgency score
        workload_df["combined_score"] = workload_df["user_current_load"] + urgency_score
        assigned_user = workload_df.sort_values("combined_score").iloc[0]["user"]

    # ✅ Display assignment
    st.success(f"✅ Task Assigned to: **{assigned_user}**")
    st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days to Deadline: {days_left}")

    # 🔸 Show only the assigned user’s workload
    assigned_load = workload_df[workload_df["user"] == assigned_user]["user_current_load"].values[0]
    st.write("📊 **Current Workload of Assigned User:**")
    st.write(f"**{assigned_user}** has **{int(assigned_load)}** tasks currently.")

