import pandas as pd

# Load the dataset
df = pd.read_csv('GasProperties.csv')

# Feature selection
features = ["T", "P", "TC", "SV"]

print("Before:")
print(df[features].agg(["mean", "std"]))

#Standardize
df_std = df.copy()
df_std[features] = (df[features] - df[features].mean()) / df[features].std()

print("\nAfter:")
print(df_std[features].agg(["mean", "std"]))

df_std.to_csv("StdGasProperties.csv", index=False)











