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



#####################
#####################
#####################
#####################
#####################


def autocorrelation_python(data):
    data_mean = np.mean(data)  
    data_fluctuate=data-data_mean
    result = np.correlate(data_fluctuate, data_fluctuate, mode='full')
    corr_init = result.size//2   # This is the lag=0 position
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


#####################
#####################
#####################
#####################
#####################




# 1. Read the text file into a DataFrame
file_path = "mean_segment_df_V2.txt"  # Replace with your file path
num_columns = 33
column_names = [f'Col{i+1}' for i in range(num_columns)]
df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
print("df:\n",df)


# Calculate wave_vector and sorting indices for sorting from lowest to highest frequency
dx = 9.54 # nm
N = num_columns
frequencies = np.fft.fftfreq(N, d=dx) * 2 * np.pi  
sorted_idx = np.argsort(frequencies)
frequencies_sorted = frequencies[sorted_idx]
print("wave_vector:\n", frequencies_sorted)


# List to store sorted magnitudes per row
fft_magnitudes_sorted = []

for index, row in df.iterrows():
    #print(row.values)
    # np.fft.fft
    # Converts a time-domain signal to frequency domain
    # (input) Real → (output) Fourier 
    fft_result = np.fft.fft(row.values)
    magnitude = np.abs(fft_result)
    
    magnitude_sorted = magnitude[sorted_idx]
    #print("magnitude_sorted:\n",magnitude_sorted)
    fft_magnitudes_sorted.append(magnitude_sorted)
    #Plot magnitude vs frequency
#     plt.figure(figsize=(6,4))
#     plt.stem(frequencies_sorted, magnitude_sorted, basefmt=" ")
#     plt.title(f'FFT Magnitude vs Wave vector (Conf {index})')
#     plt.xlabel('Wave_vector')
#     plt.ylabel('Magnitude')
#     plt.xlim(-0.5, 0.5)
#     plt.tight_layout()
#     filename = f"fft_magnitude_row_{index}.png"
#     plt.savefig(filename)
#     plt.show()   
#     


# Convert list of arrays into a new DataFrame
fft_magnitude_df = pd.DataFrame(fft_magnitudes_sorted, columns=[f'q:{freq:.3f}' for freq in frequencies_sorted])
print("New DataFrame with sorted FFT magnitudes for each wave vector:\n", fft_magnitude_df)



for column in fft_magnitude_df.columns:
    plt.figure()
    plt.plot(fft_magnitude_df[column])
    plt.title(f'FFT Magnitude for  {column}')
    plt.xlabel('Frame number')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    # Clean the column name for a valid filename
    safe_column_name = column.replace(":", "_").replace(".", "")
    plt.savefig(f'{safe_column_name}.png')  # Save plot as image
    plt.close()  # Close the figure to avoid memory issues




plt.figure(figsize=(10, 6))

# Plot all columns in the same figure
for column in fft_magnitude_df.columns:
    plt.plot(fft_magnitude_df[column], label=column)

plt.title('FFT Magnitudes for All Wave vectors')
plt.xlabel('Frame number')
plt.ylabel('Magnitude')
plt.legend(loc='upper right', fontsize='small')  # Show legend with column names
plt.tight_layout()
plt.savefig('fft_all_columns.png')
plt.show()


###########
###########
###########



# Dictionary to hold autocorrelation results for each wave vector
autocorr_results = {}

print(fft_magnitude_df)

for freq_col in fft_magnitude_df.columns:
    data = fft_magnitude_df[freq_col].values
    num= len(data)
    ac = autocorrelation0(data,num)
    ac = autocorrelation0(data, num)[:-1000]
    print(ac)
    autocorr_results[freq_col] = ac
    #print("\n\n\n")
    
    

# Convert autocorrelation results into a DataFrame for easier analysis/plotting
autocorr_df = pd.DataFrame(autocorr_results)
#print(autocorr_df.columns)
#print("autocorr_df:\n",autocorr_df["q:0.000"])
# Plot all autocorrelation curves
plt.figure(figsize=(10, 6))

# Dictionary to store fit parameters
fit_params = {}

# Time/lag axis
lags = np.arange(len(autocorr_df))
num_cols = len(autocorr_df.columns)
selected_columns = autocorr_df.columns[num_cols // 2:]

#for freq_col in autocorr_df.columns: # all modes 
for freq_col in selected_columns:
    y_data = autocorr_df[freq_col].values
    # Plot autocorrelation
    plt.plot(lags, y_data, label=freq_col)

    # Fit the autocorrelation with exponential decay
    popt, _ = curve_fit(exp_decay, lags, y_data, p0=(1, 5))
    A, Tau = popt
    fit_params[freq_col] = {"A": A, "Tau": Tau}
    # Plot the fitted curve
    y_fit = exp_decay(lags, A, Tau)
    #plt.plot(lags, y_fit, '--', label=f'{freq_col} (fit)', linewidth=1)
    

# Plot formatting
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Functions for All wave vectors')
# Set x-axis limits
#plt.xlim(0, 4000)  # Force x-axis to show 0 to 4000
#plt.ylim(-1, 1)  # Force x-axis to show 0 to 4000
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig('fft_all_VV.png')
plt.show()

# Convert fit parameters to DataFrame
fit_df = pd.DataFrame(fit_params).T  # Transpose so frequencies are rows
fit_df.index.name = "Frequency"
fit_df.reset_index(inplace=True)

#print("\nExtracted Fit Parameters:")
#print(fit_df)

# Save the fit parameters to a text file
with open("fit_parameters.txt", "w") as f:
    f.write("Extracted Fit Parameters:\n\n")
    f.write(fit_df.to_string(index=True))

#####
#####
#####



for freq_col in autocorr_df.columns:
    y_data = autocorr_df[freq_col].values


    popt, _ = curve_fit(exp_decay, lags, y_data, p0=(1, 5))
    A, Tau = popt
    fit_params[freq_col] = {"A": A, "Tau": Tau}
    y_fit = exp_decay(lags, A, Tau)


    # Create a new figure
    plt.figure(figsize=(8, 5))
    plt.plot(lags, y_data, label=f'{freq_col} Autocorr', linewidth=2)
    plt.plot(lags, y_fit, '--', label=f'Fit: A={A:.2f}, Tau={Tau:.2f}', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    plt.xlabel('Lag / 500ps')
    plt.ylabel('Normalisied Autocorrelation')
    plt.title(f'Autocorrelation and Fit for {freq_col}')
    plt.legend()
    plt.tight_layout()
    
    # Save each plot as a file
    plt.savefig(f'autocorr_fit_{freq_col.replace(":", "_")}.png')
    plt.close()























