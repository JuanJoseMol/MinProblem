"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

f = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi)
x = np.linspace(-1,1,240)
data = f(x)
kde = gaussian_kde(data, bw_method='scott')
samples = kde.resample(2000)
plt.hist(samples.T, bins=50, density=True, alpha=0.6)
plt.plot(x, f(x), 'k', linewidth=2)
plt.show()"""

#########################################################
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

# Discrete points of the PDF (example)
x_values = np.linspace(-3, 3, 100)
pdf_values = np.exp(-x_values**2/2) / np.sqrt(2*np.pi)  # Example: Gaussian PDF values at those points

# Step 1: Interpolate the PDF
#pdf_interp = interp1d(x_values, pdf_values, bounds_error=False, fill_value=0)

# Step 2: Normalize the PDF
#area = np.trapz(pdf_values, x_values)  # Area under the curve
#pdf_values_normalized = pdf_values / area  # Normalize to make it integrate to 1

# Step 3: Construct the CDF by integrating the normalized PDF
cdf_values = cumtrapz(pdf_values, x_values, initial=0)
#cdf_interp = interp1d(x_values, cdf_values, bounds_error=False, fill_value=(0, 1))  # CDF interpolator

# Step 4: Sample from the CDF (Inverse transform sampling)
uniform_samples = np.random.rand(1000)  # Generate 1000 random samples from U(0,1)
sampled_x_values = np.interp(uniform_samples, cdf_values, x_values)  # Map uniform samples to CDF

# Plot the results
plt.hist(sampled_x_values, bins=50, density=True, alpha=0.6, label='Sampled Points')
plt.plot(x_values, pdf_values, 'k', linewidth=2, label='Original PDF')
plt.legend()
plt.show()"""

##########################################################################
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.integrate import cumtrapz

# Step 1: Define a 2D grid and the PDF values (example)
x_values = np.linspace(-3, 3, 100)
y_values = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_values, y_values)

# Example: 2D Gaussian PDF (you can replace this with your 2D PDF values)
pdf_values = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)

# Step 2: Normalize the PDF
area = np.trapz(np.trapz(pdf_values, x_values, axis=0), y_values)
pdf_values_normalized = pdf_values / area  # Ensure it integrates to 1

# Step 3: Compute the 2D CDF
# First, integrate over the x-axis
cdf_x = cumtrapz(pdf_values_normalized, x_values, axis=0, initial=0)

# Now, integrate the result over the y-axis to get the full 2D CDF
cdf_2d = cumtrapz(cdf_x, y_values, axis=1, initial=0)

# Step 4: Inverse transform sampling
# Generate uniform random samples
uniform_samples_x = np.random.rand(10000)
uniform_samples_y = np.random.rand(10000)

# Map the uniform samples to the 2D CDF using searchsorted
sampled_indices_x = np.searchsorted(cdf_2d[-1, :], uniform_samples_x)
sampled_indices_y = np.searchsorted(cdf_2d[:, -1], uniform_samples_y)

# Get the corresponding X and Y values
sampled_x_values = x_values[sampled_indices_x]
sampled_y_values = y_values[sampled_indices_y]

# Step 5: Plot the sampled points
plt.figure(figsize=(8, 8))
plt.hist2d(sampled_x_values, sampled_y_values, bins=50, density=True, alpha=0.6, cmap='Blues')

# Plot the original PDF as contours
plt.contour(X, Y, pdf_values_normalized, levels=10, linewidths=2, colors='k')
plt.title("2D Histogram of Sampled Points and Original PDF Contours")
#plt.xlim([-3, 3])
#plt.ylim([-3, 3])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()"""

################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Step 1: Define a 2D grid and the PDF values (example)
x_values = np.linspace(-3, 3, 100)
y_values = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_values, y_values)

# Example: 2D Gaussian PDF (replace with your actual PDF data)
pdf_values = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)

# Step 2: Normalize the PDF
pdf_values_normalized = pdf_values / np.sum(pdf_values)

# Step 3: Interpolate the 2D PDF
pdf_interpolator = RectBivariateSpline(x_values, y_values, pdf_values_normalized)

# Step 4: Perform rejection sampling to ensure exactly 1000 samples
target_num_samples = 2000
sampled_x_values = []
sampled_y_values = []

while len(sampled_x_values) < target_num_samples:
    # Generate random samples in batches
    batch_size = 2000  # Larger batch size ensures efficiency
    samples_x = np.random.uniform(-3, 3, batch_size)
    samples_y = np.random.uniform(-3, 3, batch_size)

    # Evaluate the interpolated PDF at these sample points
    pdf_values_at_samples = pdf_interpolator(samples_x, samples_y, grid=False)

    # Generate uniform random values for rejection sampling
    random_values = np.random.uniform(0, np.max(pdf_values_at_samples), batch_size)

    # Keep only the points where the random value is less than the interpolated PDF
    accepted_samples = random_values < pdf_values_at_samples

    # Append the accepted samples
    sampled_x_values.extend(samples_x[accepted_samples])
    sampled_y_values.extend(samples_y[accepted_samples])

# Trim to exactly 1000 samples
sampled_x_values = np.array(sampled_x_values[:target_num_samples])
sampled_y_values = np.array(sampled_y_values[:target_num_samples])
print (sampled_x_values.shape, sampled_y_values.shape)

# Step 5: Plot the sampled points
plt.figure(figsize=(8, 8))
plt.hist2d(sampled_x_values, sampled_y_values, bins=50, density=True, alpha=0.6, cmap='Blues')

# Plot the original PDF as contours
plt.contour(X, Y, pdf_values_normalized, levels=10, linewidths=2, colors='k')
plt.title("2D Histogram of Sampled Points and Original PDF Contours")
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()