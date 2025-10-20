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
    data_mean = np.mean(data)  
    data_fluctuate=data-data_mean
    result = np.correlate(data_fluctuate, data_fluctuate, mode='full')
    corr_init = result.size//2   # This is the lag=0 position
    result = result[corr_init:]    # Keep only the second half (positive lags)
    result /= result[0]
    return result


data = [-1,0,1]
Auto = autocorrelation0(data)
print("Auto:",Auto)
plt.figure(figsize=(10, 6))
lags = np.arange(len(Auto))
plt.plot(lags, Auto,'o')
plt.show()











