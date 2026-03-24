# ============================================================
# Ad Sale Prediction using Logistic Regression
# Model Validation Project
# ============================================================

# =======================
# 📦 Import Libraries
# =======================
import pandas as pd          # Data handling
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Visualization

# =======================
# 📂 Load Dataset
# =======================
dataset = pd.read_csv('data/digital_ad_dataset.csv')

# Display dataset info
print("Dataset Shape:", dataset.shape)
print(dataset.head())

# =======================
# 🎯 Split Features & Target
# =======================
X = dataset.iloc[:, :-1].values   # Independent variables
y = dataset.iloc[:, -1].values    # Target variable

# =======================
# 🔀 Train-Test Split
# =======================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# =======================
# ⚙️ Feature Scaling
# =======================
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)   # Fit + transform training data
X_test = sc.transform(X_test)         # Only transform test data

# =======================
# 🤖 Train Logistic Regression Model
# =======================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# =======================
# 🔮 Predictions
# =======================
y_pred = model.predict(X_test)

# Compare predicted vs actual values
print("\nPredicted vs Actual:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1),
                      y_test.reshape(len(y_test), 1)), axis=1))

# =======================
# 📊 Confusion Matrix
# =======================
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# =======================
# 🎯 Accuracy Score
# =======================
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# =======================
# 📈 ROC Curve & AUC Score
# =======================
from sklearn.metrics import roc_auc_score, roc_curve

# No-skill probabilities (baseline)
ns_probs = [0 for _ in range(len(y_test))]

# Logistic regression probabilities
lr_probs = model.predict_proba(X_test)[:, 1]

# Calculate AUC scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

print(f"\nNo Skill AUC: {ns_auc:.3f}")
print(f"Logistic Regression AUC: {lr_auc:.3f}")

# Compute ROC curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# Plot ROC Curve
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='*', label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("outputs/roc_curve.png")
plt.show()

# =======================
# 🔁 Cross Validation (K-Fold)
# =======================
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=10)
cv_results = cross_val_score(model, X, y, cv=kfold)

print(f"\nK-Fold CV Accuracy: {cv_results.mean() * 100:.2f}%")

# =======================
# 🔁 Stratified K-Fold
# =======================
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=3)
sk_results = cross_val_score(model, X, y, cv=skfold)

print(f"Stratified K-Fold Accuracy: {sk_results.mean() * 100:.2f}%")

# =======================
# 📊 CAP Curve (Cumulative Accuracy Profile)
# =======================
total = len(y_test)
class_1_count = np.sum(y_test)

# Random model line
plt.plot([0, total], [0, class_1_count],
         linestyle='--', label='Random Model')

# Perfect model line
plt.plot([0, class_1_count, total],
         [0, class_1_count, class_1_count],
         linewidth=2, label='Perfect Model')

# Model probabilities
probs = model.predict_proba(X_test)[:, 1]

# Sort probabilities
model_y = [y for _, y in sorted(zip(probs, y_test), reverse=True)]

# Cumulative sum
y_values = np.append([0], np.cumsum(model_y))
x_values = np.arange(0, total + 1)

# Plot model curve
plt.plot(x_values, y_values, linewidth=3, label='Logistic Model')

# 50% cutoff
index = int(0.5 * total)

plt.plot([index, index], [0, y_values[index]], linestyle='--')
plt.plot([0, index], [y_values[index], y_values[index]], linestyle='--')

plt.xlabel('Total Samples')
plt.ylabel('Positive Class')
plt.title('CAP Curve')
plt.legend()
plt.savefig("outputs/cap_curve.png")
plt.show()