import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score

df = pd.read_csv("StdGasProperties.csv")
features = ["T", "P", "TC", "SV"]
X = df[features].values

kmeans_labels = np.load("kmeans_labels.npy")
gmm_labels = np.load("gmm_labels.npy")
som_labels = np.load("som_labels.npy")

# Silhouette Scores (sampled)
np.random.seed(42)
sample_size = 10000
indices = np.random.choice(len(X), sample_size, replace=False)

X_sample = X[indices]
kmeans_sample = kmeans_labels[indices]
gmm_sample = gmm_labels[indices]
som_sample = som_labels[indices]

kmeans_score = silhouette_score(X_sample, kmeans_sample)
gmm_score = silhouette_score(X_sample, gmm_sample)
som_score = silhouette_score(X_sample, som_sample)

print("Silhouette Scores (10k samples):")
print(f"K-Means: {kmeans_score:.4f}")
print(f"EM (GMM): {gmm_score:.4f}")
print(f"SOM-derived: {som_score:.4f}")

# Wobbe Index → Quality Classes
wobbe_col = "Idx"

low_thresh = df[wobbe_col].quantile(0.33)
high_thresh = df[wobbe_col].quantile(0.66)

def classify_wobbe(val):
    if val <= low_thresh:
        return "Regular"
    elif val <= high_thresh:
        return "Medium"
    else:
        return "Premium"

df["Quality"] = df[wobbe_col].apply(classify_wobbe)

print("\nWobbe Index thresholds:")
print(f"Regular: <= {low_thresh:.4f}")
print(f"Medium: <= {high_thresh:.4f}")
print(f"Premium: > {high_thresh:.4f}")

# Mean and Variance per Class
means = df.groupby("Quality")[features].mean()
variances = df.groupby("Quality")[features].var()

print("\nMean of features for each class:")
print(means)

print("\nVariance of features for each class:")
print(variances)

# Adjusted Rand Index (ARI)
quality_map = {"Regular": 0, "Medium": 1, "Premium": 2}
true_labels = df["Quality"].map(quality_map).values

kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
gmm_ari = adjusted_rand_score(true_labels, gmm_labels)
som_ari = adjusted_rand_score(true_labels, som_labels)

print("\nAdjusted Rand Index (ARI):")
print(f"K-Means: {kmeans_ari:.4f}")
print(f"GMM: {gmm_ari:.4f}")
print(f"SOM-derived: {som_ari:.4f}")