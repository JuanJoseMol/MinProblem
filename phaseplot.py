import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm

l = np.linspace(0.001,1,1000)
data1 = np.zeros((400,1000))
data2 = np.zeros((400,1000))
data3 = np.zeros((400,1000))
data = [data1, data2, data3]
sigma_W =  180

for k, tipo in enumerate(["NoBN","BNClassic","BNMatias"]):
    for i, sigma_a in enumerate(l[:400]):
        file_name = 'resultdis/w'+str(sigma_W)+'a' + format(sigma_a, ".3f") + 'norm-' + tipo + '.txt'
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                data[k][i,:] = np.loadtxt(f)
        #else:
        #    with open(f'resultdis/w'+str(sigma_W)+'a' + format(sigma_a, ".2f") + 'norm-' + tipo + '.txt', 'r') as f:
        #        data[k][i,:] = np.loadtxt(f)

    """
    # Plotting a heatmap of the phase map
    plt.figure(figsize=(10, 6))

    # Create a heatmap where rows are arrays and columns are indices
    plt.imshow(data_matrix, norm=LogNorm(), cmap='plasma')

    # Add colorbar
    plt.colorbar(label='Log-scaled colorbar')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Array Index')
    plt.title('2D Phase Map')
    plt.yticks([0, 100, 200, 300, 400], ['0.001', '0.1', '0.2', '0.3', '0.4'])

    #plt.xscale('log')
    plt.show()"""

vmin = min(data1.min(), data2.min(), data3.min())
vmax = max(data1.max(), data2.max(), data3.max())

# Create subplots
fig, axs = plt.subplots(3,1, figsize=(5, 10))

# Plot the first image
im1 = axs[0].imshow(data1, norm=LogNorm(), cmap='plasma', origin='lower')
#axs[0].set_xlabel('time')
axs[0].set_yticks([0, 100, 200, 300, 400])  # Set custom x-ticks
axs[0].set_yticklabels(['0.001', '0.1', '0.2', '0.3', '0.4'])  # Set custom x-tick labels
axs[0].set_title('No BN')
#axs[0].set_yscale('log')

# Plot the second image
im2 = axs[1].imshow(data2, norm=LogNorm(), cmap='plasma', origin='lower')
#axs[1].set_xlabel('time')
axs[1].set_yticks([0, 100, 200, 300, 400])  # Set custom x-ticks
axs[1].set_yticklabels(['0.001', '0.1', '0.2', '0.3', '0.4'])  # Set custom x-tick labels
axs[1].set_title('BN Classic')
#axs[1].set_yscale('log')

# Plot the third image
im3 = axs[2].imshow(data3, norm=LogNorm(), cmap='plasma', origin='lower')
axs[2].set_xlabel('time')
axs[2].set_yticks([0, 100, 200, 300, 400])  # Set custom x-ticks
axs[2].set_yticklabels(['0.001', '0.1', '0.2', '0.3', '0.4'])  # Set custom x-tick labels
axs[2].set_title('BN Matias')
#axs[2].set_yscale('log')

# Add a single colorbar to the entire figure
fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

plt.show()