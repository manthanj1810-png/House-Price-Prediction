import pandas as pd

df = pd.read_csv("data/train.csv")

# print("Shape:", df.shape)

# print("\nFirst 5 rows:")
# print(df.head())

# print("\nData Types:")
# print(df.dtypes.value_counts())

# missing = (
#     df.isnull()
#       .sum()
#       .sort_values(ascending=False)
# )

# print(missing.head(15))
print(df["SalePrice"].describe())

corr = df.corr(numeric_only=True)

sale_corr = corr["SalePrice"].sort_values(
    ascending=False
)

print(sale_corr.head(15))