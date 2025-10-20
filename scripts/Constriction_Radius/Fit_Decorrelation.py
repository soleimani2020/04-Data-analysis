import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Define the model function to fit
def model(x, l, xi, a):
    return (np.exp(-x / xi) + np.exp(-(l - x) / xi)) / (1 + np.exp(-l / xi)) + a

# Load your data
data = pd.read_csv("mean_segment_values.csv")

# Select only the columns that start with "Average"
segment_columns = [col for col in data.columns if col.startswith("Average")]
radii = data[segment_columns].values  # Shape: (num_times, num_segments)

# Create a matching time array for the x-axis
num_times, num_segments = radii.shape
time = np.tile(np.arange(num_times).reshape(-1, 1), (1, num_segments))

# Plot the raw radii values
plt.plot(time, radii, color='green', alpha=0.5, zorder=1)

# Flatten the 2D arrays for curve fitting
x_data = time.flatten()
y_data = radii.flatten()

# Initial guess for parameters: l, xi, a
initial_guess = [num_times, 10.0, 0.0]  # You can tweak these if fitting fails

# Perform the curve fitting
params, covariance = curve_fit(model, x_data, y_data, p0=initial_guess)
l, xi, a = params

print(f"Fitted parameters:\nl = {l}\nxi = {xi}\na = {a}")

# Compute values at x = 0 and x = l using the fitted parameters
x = 0
X0 = model(x, l, xi, a)
print("X0:", X0)

x = l
XL = model(x, l, xi, a)
print("XL:", XL)

# Plot the raw data and the fitted model
plt.scatter(x_data, y_data, label='Data', color='blue', s=5)
x_fit = np.linspace(0, np.max(x_data), 500)
plt.plot(x_fit, model(x_fit, *params), label='Fitted function', color='red')
plt.xlabel(r'$\Delta x$ (Length shift)')
plt.ylabel('Autocorrelation')
plt.legend()
plt.tight_layout()
plt.show()
