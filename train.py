import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Heart_disease_cleveland_new.csv")

# Display basic information
print("Dataset Overview:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Visualizing class distribution
sns.countplot(x='target', data=df)
plt.title("Class Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Split features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=500, verbose=1)

# Training with progress visualization
print("\nTraining Logistic Regression Model...")
for _ in tqdm(range(1), desc="Training Progress"):
    log_reg.fit(X_train, y_train)

# Predictions
y_pred_lr = log_reg.predict(X_test)

# Model Evaluation
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Identifying misclassified samples
misclassified = np.where(y_test != y_pred_lr)[0]
print(f"\nTotal Misclassified Samples: {len(misclassified)}")

# Save the trained model and scaler
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(scaler, "scaler_lr.pkl")

print("\nModel and scaler saved successfully!")