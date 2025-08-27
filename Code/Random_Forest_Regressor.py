# Random Forest Regressor Example
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(200, 3)  # 200 samples, 3 features
y = 5*X[:, 0] + 2*X[:, 1]**2 - 3*X[:, 2] + np.random.randn(200) * 0.2

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Feature importance
feature_names = ["Feature1", "Feature2", "Feature3"]
importances = rf.feature_importances_

plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.show()
