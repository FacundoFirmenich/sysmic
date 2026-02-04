import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
import time

# Settings
csv_path = r"c:\Users\User\3D Objects\PanAmericanPaper\paper02\hinet_hypo_2001_2005_extracted.csv"
SCALE_FACTOR = 200.0  # 1 unit approx 200 km
R_MIN_KM, R_MAX_KM = 2.0, 500.0 # From 2km to 500km
N_R_STEPS = 30

def compute_correlation_dimension():
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Extract and scale
    coords = df[['x', 'y', 'z']].values * SCALE_FACTOR
    
    # Filter out potential artifacts (z=0 often artifact in some catalogs, checking distribution)
    # The stats showed z max near 0 (surface) and min -2.96 (depth).
    # Since Z is negative in the file, depth increases negatively.
    # Just using distance, so sign doesn't matter.
    
    N = len(coords)
    print(f"Data loaded: {N} events.")
    print(f"Scaled ranges (km):")
    print(f"  X: {coords[:,0].min():.1f} to {coords[:,0].max():.1f}")
    print(f"  Y: {coords[:,1].min():.1f} to {coords[:,1].max():.1f}")
    print(f"  Z: {coords[:,2].min():.1f} to {coords[:,2].max():.1f}")

    # Build Tree
    print("Building KDTree...")
    t0 = time.time()
    tree = cKDTree(coords)
    print(f"Tree built in {time.time()-t0:.2f}s")
    
    # Define radii
    r_values = np.logspace(np.log10(R_MIN_KM), np.log10(R_MAX_KM), N_R_STEPS)
    C_r = []
    
    print("Calculating Correlation Integral C(r)...")
    # Using count_neighbors for efficiency
    # Computation cost is high for large r. 
    # For N=200k, r=500km is huge.
    # Subsampling might be necessary if this hangs.
    # Let's try full dataset but monitor time.
    
    # If N is huge, we can limit the query points but query against the full tree.
    # This estimates C(r) correctly but with slightly higher variance (negligible for N=200k).
    SAMPLE_SIZE = 5000
    if N > SAMPLE_SIZE:
        print(f"Subsampling {SAMPLE_SIZE} query points for speed (tree uses all {N} points)...")
        query_indices = np.random.choice(N, SAMPLE_SIZE, replace=False)
        query_points = coords[query_indices]
        normalization = SAMPLE_SIZE * (N - 1) # Assumes query points are part of tree
    else:
        query_points = coords
        normalization = N * (N - 1) / 2 # Count pairs
        
    for i, r in enumerate(r_values):
        t_step = time.time()
        
        # count_neighbors returns number of pairs (i, j) with d(i,j) <= r
        # If querying specific points against tree:
        if N > SAMPLE_SIZE:
            # query_ball_point is better? no, count_neighbors is for two trees usually
            # tree.count_neighbors(other_tree, r)
            # Create tree for query points
            # Actually tree.query_ball_point returns lists, memory heavy.
            # Using count_neighbors with two trees is efficient.
            q_tree = cKDTree(query_points)
            count = tree.count_neighbors(q_tree, r, cumulative=False)
            # Correct for self-matches if query set overlaps
            # Since query is subset, each point matches itself.
            # We want pairs distinct.
            count = count - SAMPLE_SIZE # Remove self-matches
        else:
            # Self-correlation
            count = tree.count_neighbors(tree, r, cumulative=False)
            count = (count - N) / 2 # Pairs
        
        C_val = count / normalization
        C_r.append(C_val)
        print(f"  r={r:.2f} km: C(r)={C_val:.2e} ({time.time()-t_step:.2f}s)")
        
    # Fit line
    valid = np.array(C_r) > 0
    log_r = np.log10(r_values[valid])
    log_C = np.log10(np.array(C_r)[valid])
    
    # We want scaling region. Usually 10km to 100km?
    # Let's try fitting whole range first, then refining.
    slope, intercept, low, high = theilslopes(log_C, log_r, 0.95)
    
    print("\n--- RESULTS ---")
    print(f"Global Slope (D3 estimate): {slope:.3f}")
    
    # Save Plot Data
    results_df = pd.DataFrame({'r': r_values, 'C_r': C_r})
    results_df.to_csv('hinet_fractal_results.csv', index=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(r_values, C_r, 'o-', label=f'Data (D3 ~ {slope:.2f})')
    
    # Plot fit
    fit_vals = 10**(intercept + slope * log_r)
    plt.loglog(10**log_r, fit_vals, 'r--', label='Theil-Sen Fit')
    
    plt.xlabel('Distance r (km)')
    plt.ylabel('Correlation Integral C(r)')
    plt.title('Fractal Dimension Analysis - Hi-Net 2001-2005')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('hinet_fractal_plot.png')
    print("Plot saved to hinet_fractal_plot.png")

if __name__ == "__main__":
    compute_correlation_dimension()
