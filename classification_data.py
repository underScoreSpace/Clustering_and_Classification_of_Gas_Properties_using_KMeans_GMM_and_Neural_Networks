import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("StdGasProperties.csv")

features = ["T", "P", "TC", "SV"]
X = df[features]
wobbe_col = "Idx"

# Create quality classes using percentiles
low_thresh = df[wobbe_col].quantile(0.33)
high_thresh = df[wobbe_col].quantile(0.66)

def classify_wobbe(val):
    if val <= low_thresh:
        return 0   # Regular
    elif val <= high_thresh:
        return 1   # Medium
    else:
        return 2   # Premium

y = df[wobbe_col].apply(classify_wobbe)

# 70 / 15 / 15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))