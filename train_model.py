
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
data = pd.read_excel("Concrete_Data.xls")

# Rename columns
data.columns = [
    "cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate",
    "fine_aggregate", "age", "strength"
]

# Separate features and target
X = data.drop("strength", axis=1)
y = data["strength"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regression
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", round(r2, 4))
print("MSE:", round(mse, 4))
print("MAE:", round(mae, 4))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": list(X.columns),
        "model_name": "Random Forest Regression"
    },
    "models/concrete_model.joblib"
)

# Graph 1: Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind="barh", figsize=(9, 6))
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Graph 2: Actual vs Predicted
plt.figure(figsize=(7, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

# Ideal prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Ideal Prediction"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Strength")
plt.legend()
plt.tight_layout()
plt.show()
