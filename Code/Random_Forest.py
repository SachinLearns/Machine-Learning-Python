# Random Forest for breast cancer detection (scikit-learn)

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from joblib import dump


# 1) Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 1 = malignant? (Actually: 0 = malignant, 1 = benign)

# 2) Train/Val split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Build model (RF doesn’t need scaling)
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # handle slight class imbalance
)

pipe = Pipeline(steps=[("rf", rf)])

# 4) Hyperparameter search (keep grid modest for speed)
param_grid = {
    "rf__n_estimators": [200, 400],
    "rf__max_depth": [None, 8, 16],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", 0.5],
    "rf__bootstrap": [True]
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=0
)

# 5) Train
grid.fit(X_train, y_train)

# 6) Evaluate
best_model = grid.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("Best params:", grid.best_params_)
print(f"CV best ROC AUC: {grid.best_score_:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test ROC AUC:  {auc:.4f}\n")

print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# 7) Feature importances
rf_model: RandomForestClassifier = best_model.named_steps["rf"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
topk = importances.sort_values(ascending=False).head(15)
print("\nTop 15 features by importance:")
for feat, val in topk.items():
    print(f"{feat:35s} {val:.4f}")

# 8) Save model
dump(best_model, "breast_cancer_rf.joblib")
print("\nSaved trained model to breast_cancer_rf.joblib")
