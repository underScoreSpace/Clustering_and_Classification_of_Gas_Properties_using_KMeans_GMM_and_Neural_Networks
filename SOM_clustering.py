import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans


df = pd.read_csv('StdGasProperties.csv')
features = ['T', 'P', "TC", "SV"]
X = df[features].values

som_rows = 15
som_cols = 15
input_len = X.shape[1]
sigma = 1.0
learning_rate = 0.5
num_iterations = 1000

som = MiniSom(
    x = som_rows,
    y = som_cols,
    input_len=input_len,
    sigma=sigma,
    learning_rate=learning_rate,
    neighborhood_function='gaussian',
    random_seed=42
)

# Randomly initialize SOM weights
som.random_weights_init(X)

initial_weights = som.get_weights().copy()

som.train_random(X, num_iterations)

# Extract neuron prototypes
prototypes = som.get_weights().reshape(som_rows * som_cols, input_len)

# K-Means on SOM prototypes
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)

prototype_labels = kmeans.fit_predict(prototypes)

# Final cluster centroids in feature space
final_centroids = kmeans.cluster_centers_

# Assign each data point to cluster of its BMU

data_cluster_labels = []

for x in X:
    bmu = som.winner(x)  # returns (row, col)
    bmu_index = bmu[0] * som_cols + bmu[1]  # flatten
    cluster_label = prototype_labels[bmu_index]
    data_cluster_labels.append(cluster_label)

data_cluster_labels = np.array(data_cluster_labels)

# Cluster sizes
cluster_sizes = pd.Series(data_cluster_labels).value_counts().sort_index()


print("SOM Clustering Results")
print("-" * 50)

print("Grid size:")
print(f"{som_rows}x{som_cols}")


print("\nNeighborhood function:")
print("gaussian")

print("\nLearning rate schedule:")
print(f"Initial learning rate = {learning_rate}")

print("\nNumber of iterations:")
print(num_iterations)

print("\nSimilarity metric used:")
print("Euclidean distance")

print("\nNeuron weight vectors (prototypes):")
print("Prototype matrix shape:", prototypes.shape)
print("First 5 prototypes:")
print(np.round(prototypes[:5], 4))

print("\nK-Means on SOM prototypes:")
print("Number of prototype clusters: 3")

print("\nFinal cluster centroids:")
for i, centroid in enumerate(final_centroids):
    print(f"Cluster {i}: {np.round(centroid, 4)}")

print("\nCluster sizes (after assigning each data point to BMU cluster):")
for cluster_id, size in cluster_sizes.items():
    print(f"Cluster {cluster_id}: {size}")

np.save("som_labels.npy", data_cluster_labels)