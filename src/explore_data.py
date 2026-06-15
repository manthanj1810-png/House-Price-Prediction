import pandas as pd

df = pd.read_csv("data/train.csv")

missing = (
    df.isnull()
      .sum()
      .sort_values(ascending=False)
)

print(missing.head(15))
