import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from scipy.stats import norm
from scipy.signal import correlate
from scipy.optimize import curve_fit
from datetime import datetime


# ---------- FFT Analysis ----------

file_path = "mean_segment_df_V2.txt"  
num_columns = 37
column_names = [f'Col{i+1}' for i in range(num_columns)]
df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names, usecols=range(num_columns))
print("df:\n", df)


# Calculate wave_vector and sorting indices for sorting from lowest to highest frequency
dx = 2.1 # nm
N = num_columns
frequencies = np.fft.fftfreq(N, d=dx) * 2 * np.pi  
sorted_idx = np.argsort(frequencies)
frequencies_sorted = frequencies[sorted_idx]
#print("wave_vector:\n", frequencies_sorted)


# List to store sorted magnitudes per row

fft_magnitudes_sorted = []

for index, row in df.head(5000).iterrows():
    #print("row.values:\n",row.values)
    # np.fft.fft
    # Converts a time-domain signal to frequency domain
    # (input) Real → (output) Fourier 
    fft_result = np.fft.fft(row.values)
    #print("fft_result:\n",fft_result)
    magnitude = fft_result.real
    ##magnitude = magnitude**2
    #print("Real:\n",magnitude)
    magnitude_sorted = magnitude[sorted_idx]
    #print("Real_sorted:\n",magnitude_sorted)
    fft_magnitudes_sorted.append(magnitude_sorted)
    # plt.figure(figsize=(6,4))
    # plt.stem(frequencies_sorted, magnitude_sorted, basefmt=" ")
    # plt.title(f'FFT Magnitude vs Wave vector (Conf {index})')
    # plt.xlabel('Wave_vector')
    # plt.ylabel('Magnitude')
    # plt.ylim(0, 50)
    # plt.tight_layout()
    # filename = f"fft_magnitude_row_{index}.png"
    # plt.savefig(filename)


    

# Convert the list of lists into a DataFrame
fft_magnitude_df = pd.DataFrame(fft_magnitudes_sorted, columns=frequencies_sorted)
# Optional: sort columns by q (wave vector)
#fft_magnitude_df = fft_magnitude_df.reindex(sorted(fft_magnitude_df.columns), axis=1)


fft_magnitude_df = pd.DataFrame(
    fft_magnitudes_sorted,
    columns=[f"q={q:.5f}" for q in frequencies_sorted]
)


#print(fft_magnitude_df)

fft_magnitude_df.to_csv("fft_magnitudes_all.csv", index=False)


# ---- NEW: Plot q vs mean magnitude ----
# Extract numeric q values from column names (remove 'q=')
column_means = fft_magnitude_df.mean(axis=0)
q_values = [float(col.replace("q=", "")) for col in column_means.index]
# Compute the mean of each column
mean_values = column_means.values

plt.figure(figsize=(7, 5))
plt.stem(q_values, mean_values, basefmt=" ")
plt.title("Average FFT Real vs Wave Vector (q)")
plt.xlabel("Wave Vector q (1/nm)")
plt.ylabel("Mean Real")
#plt.ylim(0, 5)
plt.tight_layout()
plt.savefig("fft_real_mean_vs_q.png", dpi=300)
plt.show()


# ---------- Autocorrelation Analysis ----------

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
    data_fluctuate = data - data_mean  # Now subtract the mean to center the data

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


# ---------- Exponential decay Analysis ----------

def exp_decay(x, y0, A, tau):
    """Exponential decay function with offset"""
    return y0 + A * np.exp(-x/tau)



# ---------- Relaxation time Analysis ----------


num_segments = 500

results = []

# Loop over each column in the dataframe
for col in fft_magnitude_df.columns:
    data = fft_magnitude_df[col].values
    print("Processing column:", col)
    
    # Compute autocorrelation
    A = autocorrelation0(data, num_segments)
    lags = np.arange(len(A))
    
    # Fit exponential decay with initial guess
    p0 = [0.1, 0.9, 10.0]  # [y0, A, tau]
    try:
        popt, pcov = curve_fit(exp_decay, lags, A, p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))  # Standard errors
    except Exception as e:
        print(f"Fit failed for column {col}: {e}")
        continue
    
    # Calculate statistics
    y_pred = exp_decay(lags, *popt)
    residuals = A - y_pred
    ss_tot = np.sum((A - np.mean(A))**2)
    r_sq = 1 - np.sum(residuals**2) / ss_tot
    
    # Store results (column name, tau, tau_error)
    results.append([col, popt[2], perr[2], r_sq])
    
    # Plot autocorrelation and fit
    plt.figure(figsize=(10, 6))
    plt.plot(lags, A, 'b.', label='Autocorrelation data')
    plt.plot(lags, y_pred, 'r-', label='Exponential fit')
    plt.xlabel('Lag time / 1 ns')
    plt.ylabel('Autocorrelation')
    
    # Add parameter box
    param_text = f'y0 = {popt[0]:.3f} ± {perr[0]:.3f}\n' \
                 f'A = {popt[1]:.3f} ± {perr[1]:.3f}\n' \
                 f'τ = {popt[2]:.3f} ± {perr[2]:.3f}\n' \
                 f'R² = {r_sq:.4f}'
    plt.annotate(param_text, xy=(0.6, 0.6), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.title(f'Autocorrelation and Fit for Column: {col}')
    plt.savefig(f'autocorrelation_fit_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save all relaxation times to a single text file
results = np.array(results, dtype=object)
np.savetxt('relaxation_times.txt', results, fmt='%s', 
           header='Column\tTau\tTau_error\tR_squared', comments='')




# List to store sorted magnitudes per row
fft_magnitudes_sorted = []

for index, row in df.iterrows():
    #print("row.values:\n",row.values)
    # np.fft.fft
    # Converts a time-domain signal to frequency domain
    # (input) Real → (output) Fourier 
    fft_result = np.fft.fft(row.values)
    #print("fft_result:\n",fft_result)
    magnitude = np.abs(fft_result)
    #print("magnitude:\n",magnitude)
    magnitude_sorted = magnitude[sorted_idx]
    print("magnitude_sorted:\n",magnitude_sorted)
    fft_magnitudes_sorted.append(magnitude_sorted)
    # plt.figure(figsize=(6,4))
    # plt.stem(frequencies_sorted, magnitude_sorted, basefmt=" ")
    # plt.title(f'FFT Magnitude vs Wave vector (Conf {index})')
    # plt.xlabel('Wave_vector')
    # plt.ylabel('Magnitude')
    # plt.xlim(-0.5, 0.5)
    # plt.tight_layout()
    # filename = f"fft_magnitude_row_{index}.png"
    # plt.savefig(filename)
    # plt.show() 
    


    

fft_real_parts_sorted = []

for index, row in df.iterrows():
    signal = row.values
    N = len(signal)
    # Perform FFT
    fft_result = np.fft.fft(signal)
    #print("fft_result2:\n",fft_result)
    real_part_sorted = fft_result.real
    #print("real_part_sorted:\n",real_part_sorted)
    real_part_sorted = real_part_sorted[sorted_idx]
    fft_real_parts_sorted.append(real_part_sorted)    
    # plt.figure(figsize=(6,4))
    # plt.stem(frequencies_sorted, magnitude_sorted, basefmt=" ")
    # plt.title(f'Real Part of FFT vs Wave Vector (Conf {index})')
    # plt.xlabel('Wave_vector')
    # plt.ylabel('Real(FFT)')
    # plt.xlim(-0.5, 0.5)
    # plt.tight_layout()
    # filename = f"fft_real_row_{index}.png"
    # plt.savefig(filename)
    # plt.show()   


# Convert list of arrays into a new DataFrame
fft_magnitude_df = pd.DataFrame(fft_real_parts_sorted, columns=[f'q:{freq:.3f}' for freq in frequencies_sorted])
print("New DataFrame with sorted FFT magnitudes for each wave vector:\n", fft_magnitude_df)



for column in fft_magnitude_df.columns:
    plt.figure()
    plt.plot(fft_magnitude_df[column])
    plt.title(f'FFT Real for  {column}')
    plt.xlabel('Frame number')
    plt.ylabel('Real')
    plt.tight_layout()
    # Clean the column name for a valid filename
    safe_column_name = column.replace(":", "_").replace(".", "")
    plt.savefig(f'{safe_column_name}.png')  # Save plot as image
    plt.close()  # Close the figure to avoid memory issues




plt.figure(figsize=(10, 6))

# Plot all columns in the same figure
for column in fft_magnitude_df.columns:
    plt.plot(fft_magnitude_df[column])

plt.title('FFT Real for All Wave vectors')
plt.xlabel('Frame number')
plt.ylabel('Real')
plt.legend(loc='upper right', fontsize='small')  # Show legend with column names
plt.tight_layout()
plt.savefig('fft_all_columns.png')
plt.show()


# Select columns starting from index 3
selected_columns = fft_magnitude_df.columns[3:]

# Create subplots
n_cols = 2  # for example
n_rows = int(np.ceil(len(selected_columns) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axs = axs.flatten()  # Flatten in case it's a 2D array

for i, column in enumerate(selected_columns):
    data = fft_magnitude_df[column].dropna()
    mu, std = norm.fit(data)

    axs[i].hist(data, bins=30, density=True, alpha=0.6, color='skyblue')
    xmin, xmax = axs[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[i].plot(x, p, 'k', linewidth=2)
    axs[i].set_title(f'{column}\nμ={mu:.2f}, σ={std:.2f}')

# Hide any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

fig.suptitle('Real Part Gaussian Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('gaussian_fits_selected_columns.png')
plt.show()




























