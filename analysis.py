import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf # C extension

# Constants
EPSILON = 1e-4
MAX_ITER = 300

def parse_data_points(file_name):
    """Load data points from file_name into a NumPy array and return it. Raise ValueError if empty."""
    X = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
    if X.size == 0:
        raise ValueError()
    
    return X

def compute_symnmf(X, k):
    """Run SymNMF on data X with k clusters and return integer labels."""
    np.random.seed(1234)

    A = symnmf.sym(X.tolist())
    D = symnmf.ddg(A)
    W = symnmf.norm(A, D)

    initial_H = np.random.uniform(0, 2 * np.sqrt(np.mean(W) / k), (X.shape[0], k))

    final_H = symnmf.symnmf(W, initial_H.tolist(), k, EPSILON)

    clusters = np.argmax(final_H, axis=1)
    
    return clusters

# ======================== K-means functions (from HW1) =========================

def calculate_euclidean_distance(p, q):
    """ Returns the Euclidean distance between two points p and q. """
    return math.sqrt(sum((p_i - q_i) ** 2 for p_i, q_i in zip(p, q)))

def assign_clusters(k, datapoints, centroids):
    """ Returns array assignments where assignments[i] stores the index of the cluster datapoints[i] is assigned to. """
    return [min(range(k), key=lambda i: calculate_euclidean_distance(datapoint, centroids[i])) for datapoint in datapoints]

def calculate_updated_centroids(k, d, datapoints, assignments, old_centroids):
    """
    Returns the updated centroids.
    If a centroid is empty, the previous centroid is reused.
    """
    updated_centroids = [[0.0] * d for _ in range(k)] 
    counts = [0] * k

    for point, cluster_index in zip(datapoints, assignments):
        for i in range(d):
            updated_centroids[cluster_index][i] += point[i]
        counts[cluster_index] += 1

    for i in range(k):
        if counts[i] == 0:
            # Reuse old centroid
            updated_centroids[i] = old_centroids[i][:] 
        else:
            updated_centroids[i] = [x / counts[i] for x in updated_centroids[i]]
    
    return updated_centroids

def calculate_centroids_using_kmeans(k,datapoints):
    """ Returns centroids calculated using the k-means algorithm. """
    n, d = len(datapoints), len(datapoints[0])
    previous_centroids = [[float('inf')] * d for _ in range(k)]
    centroids = [datapoints[i][:] for i in range(k)]
    assignments = [0] * n

    for i in range(MAX_ITER):
        if all(calculate_euclidean_distance(curr, prev) < EPSILON for curr, prev in zip(centroids, previous_centroids)):
            break
        
        previous_centroids = [centroid[:] for centroid in centroids]
        
        assignments = assign_clusters(k, datapoints, centroids)

        centroids = calculate_updated_centroids(k, d, datapoints, assignments, previous_centroids)

    return assignments

def parse_input_args():
    """Parse CLI args (k, file_name) from sys.argv and return them."""
    if len(sys.argv) != 3:
        raise ValueError()
    
    k = int(sys.argv[1])
    if k <= 1:
        raise ValueError()
    
    file_name = sys.argv[2]

    return k, file_name

def main():
    try:
        k, file_name = parse_input_args()
        X = parse_data_points(file_name)

        if k >= X.shape[0]:
            raise ValueError()
        
        symnmf_labels = compute_symnmf(X, k)
        symnmf_score = silhouette_score(X, symnmf_labels)

        kmeans_labels = calculate_centroids_using_kmeans(k, X.tolist())
        kmeans_score = silhouette_score(X, kmeans_labels)

        print(f"nmf: {symnmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")
    
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
