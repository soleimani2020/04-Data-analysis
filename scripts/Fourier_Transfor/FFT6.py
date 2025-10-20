import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.signal import correlate
from scipy.optimize import curve_fit


import matplotlib.pyplot as plt
import numpy as np

# Given data (first 4 points)
x = np.array([0, 0.094, 0.188, 0.282])
y = np.array([412.549, 132.13, 18.38, 9.35])

plt.scatter(x, y, color='red', label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of y vs x')
plt.grid(True)
plt.legend()
plt.show()





# Fit a quadratic polynomial
coefficients = np.polyfit(x, y, 2)  # 2 means quadratic
a, b, c = coefficients

# Generate predicted y values
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit**2 + b * x_fit + c

# Plot data and fitted curve
plt.scatter(x, y, color='red', label='Data')
plt.plot(x_fit, y_fit, 'b-', label=f'Quadratic Fit: y = {a:.2f}x² + {b:.2f}x + {c:.2f}')
plt.xlabel('q')
plt.ylabel('Tau')
plt.title('Quadratic Regression Fit')
plt.legend()
plt.show()

# Print the coefficients
print(f"Coefficients: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

