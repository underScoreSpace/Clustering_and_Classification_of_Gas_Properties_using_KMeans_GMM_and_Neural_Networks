import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# Load data
df = pd.read_csv("StdGasProperties.csv")
features = ["T", "P", "TC", "SV"]
X = df[features]
wobbe_col = "Idx"

# Create classes
low_thresh = df[wobbe_col].quantile(0.33)
high_thresh = df[wobbe_col].quantile(0.66)

def classify_wobbe(val):
    if val <= low_thresh:
        return 0
    elif val <= high_thresh:
        return 1
    else:
        return 2

y = df[wobbe_col].apply(classify_wobbe)

# Split: 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),   # topology
    activation="relu",             # activation function
    solver="adam",                 # optimizer
    learning_rate_init=0.001,      # learning rate
    batch_size=256,                # batch size
    alpha=0.0001,                  # L2 regularization
    max_iter=225,                   # epochs for sklearn
    random_state=42
)

# Train
mlp.fit(X_train, y_train)

# Validation accuracy
val_pred = mlp.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

print("Validation Accuracy:", round(val_acc, 4))

# Test evaluation
y_test_pred = mlp.predict(X_test)

print("\nMLP Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nMLP Test Accuracy:")
print(round(accuracy_score(y_test, y_test_pred) * 100, 2), "%")

print("\nMLP Test F1-score (macro):")
print(round(f1_score(y_test, y_test_pred, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Regular", "Medium", "Premium"]))