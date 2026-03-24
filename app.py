# ============================================================
# 🚀 Ad Sale Prediction - Premium Streamlit App
# Glassmorphism UI + Model Comparison + ROC Curve
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# =======================
# 🎨 Page Config
# =======================
st.set_page_config(page_title="Ad Sale Predictor", layout="wide")

# =======================
# 🧊 Glassmorphism CSS
# =======================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# =======================
# 📂 Load Dataset
# =======================
@st.cache_data
def load_data():
    return pd.read_csv(r"E:\linkdin projects ML\Ad Sale Prediction - Logistic Regression Validation\DigitalAd_dataset.csv")

data = load_data()

# =======================
# 🎯 Prepare Data
# =======================
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =======================
# 🤖 Train Models
# =======================
lr = LogisticRegression()
rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)

# =======================
# 🎯 UI Layout
# =======================
st.title("🚀 Ad Sale Prediction Dashboard")

col1, col2 = st.columns(2)

# =======================
# 🧾 Input Section
# =======================
with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🧾 Customer Input")

    age = st.slider("Age", 18, 60, 30)
    salary = st.slider("Salary", 15000, 150000, 50000)

    input_data = np.array([[age, salary]])
    input_scaled = sc.transform(input_data)

    if st.button("Predict"):
        pred_lr = lr.predict(input_scaled)[0]

        if pred_lr == 1:
            st.success("✅ Likely to Purchase")
        else:
            st.error("❌ Not Likely to Purchase")

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 📊 Model Comparison
# =======================
with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Model Comparison")

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Decision Tree": dt
    }

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    for name, acc in results.items():
        st.write(f"**{name}:** {acc*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 📈 ROC Curve Section
# =======================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📈 ROC Curve Comparison")

plt.figure()

for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

st.pyplot(plt)

st.markdown('</div>', unsafe_allow_html=True)

# =======================
# 📂 Dataset Preview
# =======================
if st.checkbox("Show Dataset"):
    st.write(data.head())