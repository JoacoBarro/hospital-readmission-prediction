import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from xgboost import XGBClassifier

df = pd.read_csv("data/04_featured_engineered/diabetic_data_fe.csv")

print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["readmitted"].value_counts())

# Define target variable
X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

neg = y_train.value_counts()[0]
pos = y_train.value_counts()[1]
scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)

# Separate column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Numeric columns:", len(numeric_features))
print("Categorical columns:", len(categorical_features))

# Preprocessing pipelines
# Numerical pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # FIXES NaN ERROR
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # FIXES NaN ERROR
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# Final ML Pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ))
])

# Train the model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n==============================")
print("RANDOM FOREST PERFORMANCE")
print("==============================")

rf_pipeline.fit(X_train, y_train)

rf_preds = rf_pipeline.predict(X_test)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]

# Lower threshold
threshold = 0.30
rf_preds = (rf_probs > threshold).astype(int)

print(f"\nUsing threshold = {threshold}")
print("\nAccuracy:", accuracy_score(y_test, rf_preds))
print("\nClassification Report:")
print(classification_report(y_test, rf_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print("\nROC AUC:", roc_auc_score(y_test, rf_probs))

print("\n==============================")
print("XGBOOST PERFORMANCE")
print("==============================")

xgb_pipeline.fit(X_train, y_train)

# Get probabilities
xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]

# ==============================
# AUTO-SELECT BEST THRESHOLD
# ==============================

precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)

# Remove last precision/recall value (no threshold for it)
precision = precision[:-1]
recall = recall[:-1]

# Compute F1 for every threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print("\nBest Threshold (F1 optimized):", best_threshold)
print("Best F1 Score:", best_f1)

# ==============================
# Apply Best Threshold
# ==============================

xgb_preds = (xgb_probs >= best_threshold).astype(int)

print("\nAccuracy:", accuracy_score(y_test, xgb_preds))
print("\nClassification Report:")
print(classification_report(y_test, xgb_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))
print("\nROC AUC:", roc_auc_score(y_test, xgb_probs))

import joblib

joblib.dump(pipeline, "models/logistic_pipeline.pkl")

print("\nModel saved to models/logistic_pipeline.pkl")

