import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
"""
im = nib.load("../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
print (im.shape)
im = np.swapaxes(im[6:188,16:216,4:186], 0, 1)
maximun = np.max(im)
im = im/maximun
im -= im.mean()
print (im.shape)
plt.imshow(im[:,:,100])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(im[100,:,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(im[:,100,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
"""

"""
# Assume you have function values over the domain [-1, 1] x [0, 1]
# For example, let's define some data in this domain
N = 100  # Number of grid points in each direction
x = np.linspace(0, 1, N//2)
y = np.linspace(-1, 1, N)  # Only half of the y domain (0 to 1)

# Create a meshgrid for the half domain [-1, 1] x [0, 1]
X, Y = np.meshgrid(y, x)

# Example: define a function over this half domain
# (This is just an example; replace it with your actual function)
Z =  np.cos(np.pi * Y*X)
print (Z.shape)

# Now, reflect the function values over y = 0 to get the full [-1, 1] x [-1, 1] domain
# First, flip the array along the y-axis (to reflect across y=0)
Z_reflected = np.flip(Z, axis=0)

# Combine the original and reflected data to get the full domain
Z_full = np.concatenate((Z_reflected, Z), axis=0)
#Z_full = np.vstack((Z_reflected,Z ))

print (Z.shape)
print (Z_full.shape)
plt.imshow(Z)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

plt.imshow(Z_full)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

"""

"""
a = np.zeros((4,4))
b = np.ones((4,4))
c = np.concatenate((a,b), axis=1)

N=100
x = np.linspace(0, 1, N//2)
y = np.linspace(-1, 1, N) 
X, Y = np.meshgrid(y, x)

# Example: define a function over this half domain
# (This is just an example; replace it with your actual function)
Z =  X+Y#np.cos(np.pi * X+Y)
Z1 = np.flip(Z, axis=1)

refle = np.flip(Z1, axis=0)
final = np.concatenate((Z,refle), axis=0)
plt.imshow(Z)
plt.show()
plt.imshow(final)
plt.show()
"""
"""
import numpy as np

def calculate_error_rings(images, num_rings):
    # Get the number of images and the image dimensions
    n_images, height, width = images.shape
    center = (height // 2, width // 2)

    # Calculate the maximum radius (half the width or height)
    max_radius = min(center)

    # Generate a coordinate grid
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Prepare to store the mean absolute errors for each image and ring
    ring_errors = np.zeros((n_images, num_rings))

    # Calculate the radius for each ring
    radii = np.linspace(0, max_radius, num_rings + 1)

    for i in range(num_rings):
        # Define the inner and outer radii of the current ring
        inner_radius = radii[i]
        outer_radius = radii[i + 1]

        # Create a mask for the current ring
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Iterate over each image and calculate the mean absolute error for the current ring
        for img_idx in range(n_images):
            ring_values = images[img_idx][mask]
            ring_errors[img_idx, i] = np.mean(np.abs(ring_values))

    return ring_errors



from skimage import data
from skimage.color import rgb2gray
# Load the camera image (classic black and white image)
image = rgb2gray(data.astronaut())
b = min(image.shape)
print (b//2)
#image = image[None,...]
image = np.concatenate((image[None,:,:],image[None,:,:]), axis=0)

a = calculate_error_rings(image, int(image.shape[1]/2))
print (a.shape)
"""
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from skimage.transform import resize
A = nib.load("../../../dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
#im = np.swapaxes(A[5:189,16:216,100:106], 0, 1)
#im = resize(im, (50,46,10))
#im =  np.swapaxes(A[5:191,14:218,98:108], 0, 1)
#im = resize(im, (68,62,10))
im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
im = resize(im, (50,45,45))
A= resize(A, (90,90,90))
IMG_DIM = 90

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320):
    cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM*2)
    ax.set_ylim(top=IMG_DIM*2)
    ax.set_zlim(top=IMG_DIM*2)
    ax.set_axis_off()
    
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

plot_cube(A[:60,::-1,:50])
"""
"""

from scipy.ndimage import gaussian_filter1d
import numpy as np
rng = np.random.default_rng()
a = rng.standard_normal(101).cumsum()

b = gaussian_filter1d(a, 3)
c = gaussian_filter1d(a, 0.00001)

plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.legend()
plt.show()

from skimage import data
from skimage.color import rgb2gray
import jax.numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
# Load the camera image (classic black and white image)
im = rgb2gray(data.astronaut())
im = resize(im, (256,256))
maximun = np.max(im)
im = im/maximun
im -= im.mean()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im)))))
plt.colorbar()
plt.show()
"""
"""
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

# Parameters of the Gaussian distribution
mu = 0      # Mean
sigma = 1   # Standard deviation
a, b = -2, 2  # Truncation limits (finite domain)

# Calculate the truncated normal parameters
a_std = (a - mu) / sigma  # Standardized lower bound
b_std = (b - mu) / sigma  # Standardized upper bound

# Create a truncated normal distribution
trunc_gaussian = truncnorm(a_std, b_std, loc=mu, scale=sigma)

# Sampling from the truncated Gaussian
samples = trunc_gaussian.rvs(size=10000)
samples1 = trunc_gaussian.rvs(size=10000)
samples2 = trunc_gaussian.rvs(size=(2, 1000))
print (samples[0]==samples1[0])

# Plot the truncated Gaussian distribution
x = np.linspace(a, b, 1000)
pdf = trunc_gaussian.pdf(x)
plt.plot(x, pdf, label="Truncated Gaussian PDF")
plt.hist(samples, bins=50, density=True, alpha=0.5, label="Samples")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
"""