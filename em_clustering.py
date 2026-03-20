import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Load
df = pd.read_csv('StdGasProperties.csv')
features = ["T", "P", "TC", "SV"]
X = df[features]

gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    init_params="kmeans",
    max_iter=300,
    tol=1e-3,
    random_state=42
)

# Fit model
gmm.fit(X)

gmm_labels = gmm.predict(X)
np.save("gmm_labels.npy", gmm_labels)

# Results
weights = gmm.weights_
means = gmm.means_
covariances = gmm.covariances_
iterations = gmm.n_iter_
converged = gmm.converged_

# Probabilities for 3 samples
sample_indices = [0, 1, 2]
sample_probs = gmm.predict_proba(X.iloc[sample_indices])

print("EM Clustering Results (GMM, K=3)")


print("\nInitialization method: kmeans")

print("\nConvergence criterion:")
print(f"Tolerance: {gmm.tol}")
print(f"Maximum iterations: {gmm.max_iter}")
print(f"Converged: {converged}")
print(f"Number of iterations until convergence: {iterations}")

print("\nCovariance type:")
print(f"{gmm.covariance_type}")

print("\nMixture weights:")
for i, w in enumerate(weights):
    print(f"Cluster {i}: {w}")

print("\nMeans:")
for i, mean in enumerate(means):
    print(f"Cluster {i}: {mean}")

print("\nCovariances:")
for i, cov in enumerate(covariances):
    print(f"\nCluster {i}:")
    print(cov)

print("\nProbabilities p(z = k | x) for 3 samples:")
for row_num, probs in zip(sample_indices, sample_probs):
    print(f"Sample {row_num}: {probs}")