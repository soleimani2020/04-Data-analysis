import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.signal import correlate
from scipy.optimize import curve_fit



# Define exponential decay function
def exp_decay(t, A, Tau):
    return A * np.exp(-t / Tau)



def autocorrelation0(data):
    data = np.asarray(data)
    data_mean = np.mean(data)
    data_fluctuate = data - data_mean

    C_t = [] # store autocorrelation values at each lag kk
    N = len(data)
    
    # Compute the autocorrelation function C(k) of a 1D array for lags k=0,1,2,… up to num_segments
    for k in range(N):
        numerator = np.sum(data_fluctuate[:N-k] * data_fluctuate[k:])
        denominator = N - k  # number of terms used in the sum
        autocorr = numerator / denominator
        C_t.append(autocorr)

    C_t = np.array(C_t)
    C_t_normalized = C_t / C_t[0]  # Normalize by C(0)
    return C_t_normalized





data = [-1,0,1]
Auto = autocorrelation0(data)
print("Auto:",Auto)
plt.figure(figsize=(10, 6))
lags = np.arange(len(Auto))
plt.plot(lags, Auto,'o')
plt.show()











