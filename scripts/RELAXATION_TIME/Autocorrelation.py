import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.signal import correlate
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# awk '{print $5}' fft_magnitudes_output.txt >  data_First_mode.txt

#####################
#####################
#####################
#####################
#####################


def autocorrelation_python(data):
    data_mean = np.mean(data)  
    data_fluctuate=data-data_mean
    result = np.correlate(data_fluctuate, data_fluctuate, mode='full')
    corr_init = result.size//2     # This is the lag=0 position
    result = result[corr_init:]    # Keep only the second half (positive lags)
    result /= result[0]
    return result


# Function to calculate autocorrelation 
def autocorrelation_PBC(data,num_segments):
    data_mean = np.mean(data)  
    data_fluctuate=data-data_mean
    #print("data_fluctuate:",data_fluctuate)
    #print("\n\n\n")
    C_t = []
    for k in range(num_segments):
        #print("Lag Number:",k)
        autocorr = np.sum(data_fluctuate * np.roll(data_fluctuate, -k)) / num_segments
        Shifted_array=np.roll(data_fluctuate, -k)
        #print("Shifted_array:",Shifted_array)
        #print("Autocorrelation:",autocorr)
        C_t.append(autocorr)
        #print("\n\n\n")
    # Normalize the autocorrelation
    C_t = np.array(C_t)
    C_t_normalized = C_t / C_t[0]
    return C_t_normalized



#  Non-periodic
def autocorrelation_NOPBC(data, num_segments):
    N = len(data)
    data = np.asarray(data)
    data_mean = np.mean(data)
    data_fluctuate = data #- data_mean

    C_t = [] # store autocorrelation values at each lag k
    
    
    # Compute the autocorrelation function C(k) of a 1D array for lags k=0,1,2,… up to num_segments
    for k in range(N):
        # Only compute where the shifted signal overlaps
        numerator = np.sum(data_fluctuate[:N-k] * data_fluctuate[k:])
        denominator = N-k   # number of terms used in the sum
        autocorr = numerator / denominator
        C_t.append(autocorr)

    C_t = np.array(C_t)
    C_t_normalized = C_t / C_t[0]  # Normalize by C(0)
    return C_t_normalized




def autocorrelation_nature(data, max_lag=None):
    data = np.asarray(data)
    N = len(data)
    data_mean = np.mean(data)
    data_fluctuate = data - data_mean

    if max_lag is None:
        max_lag = N

    C0 = []
    for lag in range(max_lag):
        valid_length = N - lag
        if valid_length <= 0:
            break
        corr = np.dot(data_fluctuate[:valid_length], data_fluctuate[lag:]) / valid_length
        C0.append(corr)

    C0 = np.array(C0)
    C_normalized = C0 / C0[0]
    return C_normalized




def autocorrelation0(data, num_segments):
    N = len(data)
    data = np.asarray(data)
    data_mean = np.mean(data)
    data_fluctuate = data #- data_mean  # Now subtract the mean to center the data

    C_t = []  # Store autocorrelation values at each lag k

    # Compute the autocorrelation function C(k) for k = 0 to num_segments - 1
    for k in range(num_segments):
        # i from 1 to N - k
        numerator = np.sum(data_fluctuate[1:N - k] * data_fluctuate[1 + k:N])
        denominator = N - k - 1  # Since i = 1 to N-k → total terms = N-k-1
        autocorr = numerator / denominator
        C_t.append(autocorr)

    C_t = np.array(C_t)
    C_t_normalized = C_t / C_t[0]  # Normalize by C(0)
    return C_t_normalized



# File path
file_path = "First_Mode.txt"
data = np.loadtxt(file_path, dtype=float)

A = autocorrelation_python(data)
A= A[:200]


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

def exp_decay(x, y0, A, tau):
    """Exponential decay function with offset"""
    return y0 + A * np.exp(-x/tau)


lags = np.arange(len(A))

# Perform the fit with initial guesses
p0 = [0.1, 0.9, 10.0]  # Initial guess for [y0, A, tau]
popt, pcov = curve_fit(exp_decay, lags, A, p0=p0, maxfev=5000)

# Calculate statistics
y_pred = exp_decay(lags, *popt)
residuals = A - y_pred
chi_sq = np.sum(residuals**2)
ss_tot = np.sum((A-np.mean(A))**2)
r_sq = 1 - np.sum(residuals**2)/ss_tot
perr = np.sqrt(np.diag(pcov))  # Standard errors

# Generate comprehensive report
report = f"""\n
[FIT REPORT - {datetime.now().strftime('%A, %d %B %Y %H:%M:%S %Z')}]
Exponential decay fit of autocorrelation function
Model: y = y0 + A*exp(-x/tau)
Fitting range: lags 0 to {len(A)-1}
------------------------------------------------------------
y0 (baseline)    = {popt[0]:.6f} ± {perr[0]:.6f}
A (amplitude)    = {popt[1]:.6f} ± {perr[1]:.6f}
τ (decay time)   = {popt[2]:.6f} ± {perr[2]:.6f}
------------------------------------------------------------
Goodness-of-fit:
χ² = {chi_sq:.6f}
R² = {r_sq:.6f}
Reduced χ² = {chi_sq/(len(A)-3):.6f}
------------------------------------------------------------
"""

print(report)

# Save results
np.savetxt('autocorrelation_fit_params.txt', 
           np.column_stack((popt, perr)),
           header='Value\tError',
           fmt='%.6f',
           comments=f'Fit parameters:\ny0\tA\ttau\n{report}')

# Plot with inset for parameters
plt.figure(figsize=(10, 6))
plt.plot(lags, A, 'b.', label='Autocorrelation data')
plt.plot(lags, y_pred, 'r-', label='Exponential fit')
plt.xlabel('Lag time/1ns')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function Fit')

# Add parameter box
param_text = f'y0 = {popt[0]:.3f} ± {perr[0]:.3f}\n' \
             f'A = {popt[1]:.3f} ± {perr[1]:.3f}\n' \
             f'τ = {popt[2]:.3f} ± {perr[2]:.3f}\n' \
             f'R² = {r_sq:.4f}'
plt.annotate(param_text, xy=(0.6, 0.6), xycoords='axes fraction',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend()
plt.savefig('autocorrelation_fit.png', dpi=300, bbox_inches='tight')
plt.show()

























