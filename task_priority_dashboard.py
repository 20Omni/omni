import streamlit as st
import pandas as pd
import datetime
import joblib

# =====================
# 🔹 Load Models & Encoders
# =====================
priority_model = joblib.load("priority_xgboost.pkl")  # or use priority_random_forest.pkl
priority_label_encoder = joblib.load("priority_label_encoder.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")

rf_model = joblib.load("optimized_rf_model.pkl")
category_label_encoder = joblib.load("category_label_encoder.pkl")
task_vectorizer = joblib.load("task_tfidf_vectorizer.pkl")

# =====================
# 🔹 Load Dataset
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
    urgency_score = max(0, 10 - days_left)  # closer deadline → more urgency

    # 🔸 Group workload by user from dataset
    user_workload_df = df.groupby("assigned_user")["user_current_load"].mean().reset_index()
    user_workload_df.rename(columns={"user_current_load": "avg_workload"}, inplace=True)

    # 🔸 Filter users who handled the predicted category
    matching_users = df[df["category"] == pred_category]["assigned_user"].unique()
    matching_users_filtered = user_workload_df[
        (user_workload_df["assigned_user"].isin(matching_users)) &
        (user_workload_df["avg_workload"] <= 20)  # avoid overloaded users
    ].copy()

    matching_users_filtered["urgency_score"] = urgency_score
    matching_users_filtered["combined_score"] = matching_users_filtered["avg_workload"] + urgency_score

    # Show matching users for debugging
    st.write("🕵️ Matching Users (category + workload ≤ 20):")
    st.dataframe(matching_users_filtered)

    if not matching_users_filtered.empty:
        # Assign user with lowest combined score
        assigned_user = matching_users_filtered.sort_values("combined_score").iloc[0]["assigned_user"]
    else:
        assigned_user = "No available user"

    # ✅ Final Result
    if assigned_user != "No available user":
        st.success(f"✅ Task Assigned to: **{assigned_user}**")
        st.info(f"🔺 Priority: **{pred_priority}** | 📁 Category: **{pred_category}** | 🗓 Days to Deadline: {days_left}")

        current_load = matching_users_filtered[
            matching_users_filtered["assigned_user"] == assigned_user
        ]["avg_workload"].values[0]

        st.write("📊 **Current Workload of Assigned User:**")
        st.write(f"**{assigned_user}** has **{round(current_load, 2)}** average tasks.")
    else:
        st.warning("⚠️ No suitable user found to assign this task. All are overloaded or unmatched.")


