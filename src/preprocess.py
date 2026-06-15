import pandas as pd

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

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print(X.isnull().sum())

