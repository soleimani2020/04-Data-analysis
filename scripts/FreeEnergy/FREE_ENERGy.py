import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# --- File setup ---
file_path = "mean_segment_df_V2.txt"  
num_columns = 39
column_names = [f'Col{i+1}' for i in range(num_columns)]
df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)

# --- Quadratic function ---
def quadratic(r, F0, k, R0):
    return F0 + 0.5 * k * (r - R0)**2

# --- Plot setup: 2 rows (probability + free energy), 5 cols (first 10 segments) ---
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=False, sharey=False)

for i in range(3): 
    col = column_names[i]
    R = df[col].dropna().values 

    # --- Histogram → probability distribution ---
    bins = 100
    counts, edges = np.histogram(R, bins=bins, density=True) # Normalised histo
    centers = 0.5 * (edges[:-1] + edges[1:])
    P_R = counts

    # --- Free energy ---
    kB = 1.380649e-23  # Boltzmann constant, J/K
    T = 303            # temperature, K
    F_R = -np.log(P_R + 1e-12)
    
    F_R -= np.min(F_R)
    # --- Top row: Probability distribution ---
    ax_prob = axes[0, i % 5]   # row 0
    mu, sigma = np.mean(R), np.std(R)

    # Plot histogram as bars (raw data)
    ax_prob.bar(centers, P_R, width=edges[1]-edges[0], color='skyblue', alpha=0.6, label="Data: P(R)")

    # Plot Gaussian fit as a smooth red dashed line
    ax_prob.plot(centers, norm.pdf(centers, mu, sigma), 'r--', lw=2, label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}')

    # Styling
    ax_prob.set_title(f"Segment {i+1}")
    ax_prob.set_xlabel("R")
    ax_prob.set_ylabel("P(R)")
    ax_prob.legend(fontsize=8)

    # --- Bottom row: Free energy ---
    ax_free = axes[1, i % 5]   # row 1
    ax_free.plot(centers, F_R, lw=2, label="F(R)")

    # --- Try harmonic fit near minimum ---
    min_index = np.argmin(F_R)
    R0 = centers[min_index]
    F0 = F_R[min_index]
    
    
    mask = (centers > R0 - 5) & (centers < R0 + 5)
    
    
    
    popt, _ = curve_fit(quadratic, centers[mask], F_R[mask], p0=[F0, 1.0, R0])
    F_fit = quadratic(centers, *popt)
    F0_fit, k_fit, R0_fit = popt
    ax_free.plot(centers, F_fit, 'r--', lw=1.5, label=f'Harmonic Fit\nR0={R0_fit:.2f}, k={k_fit:.2f}, F0={F0_fit:.2f}')
    ax_free.set_xlabel("R")
    ax_free.set_ylabel("F(R) [kBT]")
    ax_free.legend(fontsize=8)

plt.tight_layout()

# --- Save the figure ---
plt.savefig("segments_free_energy.png", dpi=300, bbox_inches="tight")
plt.savefig("segments_free_energy.pdf", bbox_inches="tight")
plt.show()
plt.close()
