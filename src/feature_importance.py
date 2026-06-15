import matplotlib.pyplot as plt

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

importance = [
    0.5792,
    0.1881,
    0.0220,
    0.0437,
    0.1015,
    0.0131,
    0.0525
]

plt.figure(figsize=(8, 5))
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()