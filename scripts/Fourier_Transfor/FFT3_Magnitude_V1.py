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
file_path = "mean_segment_df_V2.txt"  
num_columns = 37
column_names = [f'Col{i+1}' for i in range(num_columns)]
df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
#print("df:\n",df)


# Calculate wave_vector and sorting indices for sorting from lowest to highest frequency
# 681.35986
dx = 2.1 # nm
N = num_columns
frequencies = np.fft.fftfreq(N, d=dx) * 2 * np.pi  
#print("frequencies:\n", frequencies)
sorted_idx = np.argsort(frequencies)
#print("sorted_idx:\n", sorted_idx)
frequencies_sorted = frequencies[sorted_idx]
print("wave_vector:\n", frequencies_sorted)


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
    #print("magnitude_sorted:\n",magnitude_sorted)
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
fft_magnitude_df = pd.DataFrame(fft_magnitudes_sorted, columns=[f'q:{freq:.3f}' for freq in frequencies_sorted])
#print("New DataFrame with sorted FFT magnitudes for each wave vector:\n", fft_magnitude_df)
fft_magnitude_df.to_csv('fft_magnitude_results.csv', index=False)


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
    plt.plot(fft_magnitude_df[column])

plt.title('FFT Magnitude for All Wave vectors')
plt.xlabel('Frame number')
plt.ylabel('Magnitude')
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

fig.suptitle('Magnitude Gaussian Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('gaussian_fits_selected_columns.png')
plt.show()


###########
###########
###########








