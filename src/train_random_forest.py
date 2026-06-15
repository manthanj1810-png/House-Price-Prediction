import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

print("Current Working Directory:")
print(os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("data/train.csv")

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))

print("\nFeature Importance")

for feature, importance in zip(features, model.feature_importances_):
    print(feature, ":", round(importance, 4))

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted")

plt.show()
plt.savefig("actual_vs_predicted_rf.png")


joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nModel saved successfully!")
