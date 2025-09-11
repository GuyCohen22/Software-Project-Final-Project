import sys
import numpy as np
import symnmf # C extension

def parse_input_args():
    """Parse CLI args (k, goal, file_name) from sys.argv and return them."""
    if len(sys.argv) != 4:
        raise ValueError()
    
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]

    return k, goal, file_name

def parse_data_points(file_name):
    """Load data points from file_name into a numpy array and return it. raise an error if empty."""
    X = np.loadtxt(file_name, delimiter=',', dtype=np.float64)
    if X.size == 0:
        raise ValueError()
    
    return X

def init_H(W, k):
    """Initialize H with values sampled uniformly from [0, 2 * sqrt(mean(W)/k)] using a fixed seed and return it."""
    np.random.seed(1234)
    W = np.array(W)
    m = np.mean(W)
    H = np.random.uniform(0, 2 * np.sqrt(m / k), size=(W.shape[0], k)).astype(np.float64)
    return H

def compute_sym(X, k):
    """Compute and return the similarity matrix A by calling the symnmf C extension."""
    return symnmf.sym(X.tolist())

def compute_ddg(X, k):
    """Compute and return the diagonal degree matrix D by calling the symnmf C extension."""
    A = compute_sym(X, k)
    return symnmf.ddg(A)

def compute_norm(X, k):
    """Compute and return the normalized similarity matrix W by calling the symnmf C extension."""
    A = compute_sym(X, k)
    D = symnmf.ddg(A)
    return symnmf.norm(A, D)

def compute_symnmf(X, k):
    """Compute the SymNMF algorithm on X using the symnmf C extension and an initialized H."""
    if k <= 1 or k >= X.shape[0]:
        raise ValueError()
    
    W = compute_norm(X, k)
    initial_H = init_H(W, k)
    return symnmf.symnmf(np.array(W).tolist(), initial_H.tolist())
    

def print_result(result):
    """Print the relevant result based on the selected goal."""
    for vector in result:
        print(",".join(f"{coordinate:.4f}" for coordinate in vector))

def main():
    goal_to_method = {
    "symnmf": compute_symnmf,
    "sym": compute_sym,
    "ddg": compute_ddg,
    "norm": compute_norm,
    }

    try:
        k, goal, file_name = parse_input_args()
        X = parse_data_points(file_name)
        
        if goal in goal_to_method:
            method = goal_to_method[goal]
            result = method(X, k)
            print_result(result)
        
        else:
            raise ValueError()

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()