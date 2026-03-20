import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('StdGasProperties.csv')
features = ["T", "P", "TC", "SV"]
X = df[features]

# Manually choose 3 initial centroids from the dataset
initial_centroids = X.sample(n=3, random_state=42).to_numpy()

kmeans = KMeans(
    n_clusters=3,
    init=initial_centroids,
    n_init=1,
    max_iter=300,
    random_state=42
)

# Fit model
kmeans.fit(X)

print("\nInitialization method: manual selection from standardized dataset")

print("\nInitial centroids:")
for i, row in enumerate(initial_centroids):
    print(f"Cluster {i}: {row}")

# Results
labels = kmeans.labels_
final_centroids = kmeans.cluster_centers_
iterations = kmeans.n_iter_
cluster_sizes = pd.Series(labels).value_counts().sort_index()
total_wcss = kmeans.inertia_

cluster_wcss = []
for i in range(kmeans.n_clusters):
    cluster_points = X[labels == i]
    centroid = final_centroids[i]
    wcss_i = np.sum((cluster_points - centroid) ** 2)
    cluster_wcss.append(wcss_i)

print("\nNumber of iterations until convergence:", iterations)

print("\nFinal centroids:")
for i, centroid in enumerate(final_centroids):
    print(f"Cluster {i}: {centroid}")

print("\nCluster variances (Within-Cluster Sum of Squares):")
for i, wcss in enumerate(cluster_wcss):
    print(f"Cluster {i}: {wcss}")
print(f"Total WCSS: {total_wcss}")

print("\nNumber of samples assigned to each cluster:")
for cluster_id, size in cluster_sizes.items():
    print(f"Cluster {cluster_id}: {size}")

np.save("kmeans_labels.npy", labels)