import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

model = Lasso(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
print("R2  :", round(r2_score(y_test, y_pred), 4))

print("\nCoefficients:")

for feature, coef in zip(features, model.coef_):
    print(feature, ":", round(coef, 2))