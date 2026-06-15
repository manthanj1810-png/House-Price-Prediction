from pathlib import Path
import pandas as pd
import joblib

model_path = (
    Path(__file__).parent.parent
    / "models"
    / "random_forest_model.pkl"
)

model = joblib.load(model_path)

house = pd.DataFrame([{
    "OverallQual": 7,
    "GrLivArea": 1700,
    "GarageCars": 2,
    "GarageArea": 500,
    "TotalBsmtSF": 1000,
    "FullBath": 2,
    "YearBuilt": 2005
}])

prediction = model.predict(house)

print(
    f"Predicted House Price: ${prediction[0]:,.2f}"
)

# df = pd.read_csv("data/train.csv")

# features = [
#     "OverallQual",
#     "GrLivArea",
#     "GarageCars",
#     "GarageArea",
#     "TotalBsmtSF",
#     "FullBath",
#     "YearBuilt"
# ]

# house = df[features].iloc[[0]]

# print(house)

# prediction = model.predict(house)

# print("Predicted:", prediction[0])

# print("Actual:", df["SalePrice"].iloc[0])