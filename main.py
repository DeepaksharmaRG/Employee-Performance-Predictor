# ==============================
# BACKEND MODEL SCRIPT
# ==============================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEPT_MAPPING = {"HR": 0, "IT": 1, "Sales": 2, "Finance": 3}
LABEL_MAPPING = {0: "Low", 1: "Medium", 2: "High"}

np.random.seed(42)

df = pd.DataFrame({
    "Age": np.random.randint(22, 60, 500),
    "Experience": np.random.randint(1, 20, 500),
    "Department": np.random.choice(list(DEPT_MAPPING.keys()), 500),
    "Salary": np.random.randint(20000, 150000, 500),
    "Training_Hours": np.random.randint(5, 100, 500),
    "Projects": np.random.randint(1, 10, 500),
    "Attendance": np.random.uniform(60, 100, 500)
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

X = df.drop(["Performance", "Performance_Label"], axis=1)
y = df["Performance_Label"]

model = RandomForestClassifier()
model.fit(X, y)

sample = [[35, 7, 1, 70000, 40, 5, 90]]
pred = model.predict(sample)[0]

print("Prediction:", LABEL_MAPPING[int(pred)])