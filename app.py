# ==============================
# ADVANCED EMPLOYEE PERFORMANCE APP
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# CONFIG
# ------------------------------

st.set_page_config(page_title="HR Analytics System", layout="wide")

# ------------------------------
# CONSTANTS
# ------------------------------

DEPT_MAPPING = {"HR": 0, "IT": 1, "Sales": 2, "Finance": 3}
LABEL_MAPPING = {0: "Low", 1: "Medium", 2: "High"}

# ------------------------------
# DATA GENERATION
# ------------------------------

@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "Age": np.random.randint(22, 60, n),
        "Experience": np.random.randint(1, 20, n),
        "Department": np.random.choice(list(DEPT_MAPPING.keys()), n),
        "Salary": np.random.randint(20000, 150000, n),
        "Training_Hours": np.random.randint(5, 100, n),
        "Projects": np.random.randint(1, 10, n),
        "Attendance": np.random.uniform(60, 100, n)
    })

    df["Performance"] = (
        df["Experience"] * 0.3 +
        df["Training_Hours"] * 0.2 +
        df["Projects"] * 0.3 +
        df["Attendance"] * 0.2
    )

    df["Performance_Label"] = pd.cut(
        df["Performance"],
        bins=[0, 30, 60, 100],
        labels=[0, 1, 2]
    )

    df["Department"] = df["Department"].map(DEPT_MAPPING)

    return df

# ------------------------------
# MODEL
# ------------------------------

@st.cache_resource
def train_model(df):
    X = df.drop(["Performance", "Performance_Label"], axis=1)
    y = df["Performance_Label"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model

# ------------------------------
# LOAD
# ------------------------------

data = generate_data()
model = train_model(data)

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------

st.sidebar.title("📊 HR Analytics System")
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📂 Data Explorer", "🤖 Model Info", "🎯 Prediction", "📈 Insights"]
)

# ------------------------------
# HOME PAGE
# ------------------------------

if menu == "🏠 Home":
    st.title("💼 Employee Performance Prediction System")

    st.markdown("""
    ### 🎯 Objective
    Predict employee performance using machine learning.

    ### 💡 Business Use
    - Identify high performers  
    - Detect low performers  
    - Improve HR decision-making  
    """)

    st.metric("Total Employees", len(data))
    st.metric("Avg Salary", int(data["Salary"].mean()))

# ------------------------------
# DATA EXPLORER
# ------------------------------

elif menu == "📂 Data Explorer":
    st.title("📂 Employee Dataset")

    st.dataframe(data.head(50))

    dept_filter = st.selectbox("Filter by Department", list(DEPT_MAPPING.values()))

    filtered = data[data["Department"] == dept_filter]

    st.write("Filtered Data")
    st.dataframe(filtered.head())

# ------------------------------
# MODEL INFO
# ------------------------------

elif menu == "🤖 Model Info":
    st.title("🤖 Model Details")

    st.write("Model Used: Random Forest Classifier")

    st.write("### Features Used:")
    st.write(list(data.columns[:-2]))

    st.write("### Training Size:", data.shape)

# ------------------------------
# PREDICTION PAGE
# ------------------------------

elif menu == "🎯 Prediction":

    st.title("🎯 Predict Employee Performance")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 22, 60, 30)
        experience = st.slider("Experience", 1, 20, 5)
        department = st.selectbox("Department", list(DEPT_MAPPING.keys()))

    with col2:
        salary = st.number_input("Salary", 20000, 150000, 50000)
        training = st.slider("Training Hours", 5, 100, 20)
        projects = st.slider("Projects", 1, 10, 3)
        attendance = st.slider("Attendance", 60, 100, 80)

    if st.button("🚀 Predict Now"):

        input_data = np.array([[ 
            age, experience, DEPT_MAPPING[department],
            salary, training, projects, attendance
        ]])

        pred = model.predict(input_data)[0]
        result = LABEL_MAPPING[int(pred)]

        st.subheader(f"Prediction: {result}")

        if result == "High":
            st.success("🌟 High Performer")
        elif result == "Medium":
            st.warning("⚠️ Average Performer")
        else:
            st.error("❗ Needs Improvement")

# ------------------------------
# INSIGHTS DASHBOARD
# ------------------------------

elif menu == "📈 Insights":

    st.title("📈 HR Insights Dashboard")

    # Performance distribution
    fig1, ax1 = plt.subplots()
    data["Performance_Label"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_title("Performance Distribution")
    st.pyplot(fig1)

    # Salary vs Performance
    fig2, ax2 = plt.subplots()
    ax2.scatter(data["Salary"], data["Performance_Label"])
    ax2.set_title("Salary vs Performance")
    st.pyplot(fig2)