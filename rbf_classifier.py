import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from scipy.spatial.distance import cdist

# Load data
df = pd.read_csv("StdGasProperties.csv")
features = ["T", "P", "TC", "SV"]
X = df[features].values
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

y = df[wobbe_col].apply(classify_wobbe).values

# Split: 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Number of hidden units / centers
n_centers = 150

# K-means for RBF centers
kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

# Kernel width sigma
dists_between_centers = cdist(centers, centers)
sigma = np.mean(dists_between_centers[dists_between_centers > 0])

def rbf_transform(X_input, centers, sigma):
    dists = cdist(X_input, centers, metric="euclidean")
    return np.exp(-(dists ** 2) / (2 * sigma ** 2))

# Transform features
X_train_rbf = rbf_transform(X_train, centers, sigma)
X_val_rbf = rbf_transform(X_val, centers, sigma)
X_test_rbf = rbf_transform(X_test, centers, sigma)

# Output layer classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_rbf, y_train)

# Validation accuracy
val_pred = clf.predict(X_val_rbf)
val_acc = accuracy_score(y_val, val_pred)

print("Validation Accuracy:", round(val_acc, 4))

# Test evaluation
y_test_pred = clf.predict(X_test_rbf)

print("\nRBF Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nRBF Test Accuracy:")
print(round(accuracy_score(y_test, y_test_pred) * 100, 2), "%")

print("\nRBF Test F1-score (macro):")
print(round(f1_score(y_test, y_test_pred, average="macro"), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Regular", "Medium", "Premium"]))

print("\nRBF Parameters:")
print("Kernel width (sigma):", round(float(sigma), 4))
print("Number of hidden units:", n_centers)