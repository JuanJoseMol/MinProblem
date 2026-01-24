import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import math


fm.fontManager.addfont('../../../dataset/Helvetica Neue Bold.ttf')
matplotlib.rc('font', family='Helvetica Neue')

# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue']})
# ## for Palatino and other serif fonts use:
# #matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rc('text', usetex=True)


#plt.rc('font', family='sans-serif')
#plt.rcParams['font.family'] = u'Helvetica Neue'


plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] ='bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18

def ellenify(xlabel, ylabel, xlabelpad = -10, ylabelpad = -20, minXticks = True, minYticks = True):
    plt.set_xlabel(xlabel, labelpad = xlabelpad)
    plt.set_ylabel(ylabel, labelpad = ylabelpad)

    if minXticks:
        plt.xticks(plt.xlim())
        rang, labels = plt.xticks()
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")

    if minYticks:
        plt.yticks(plt.ylim())
        rang, labels = plt.yticks()
        labels[0].set_verticalalignment("bottom")
        labels[-1].set_verticalalignment("top")


import numpy as onp
from matplotlib import pyplot as plt
from jax import random, grad, jit, vmap
import jax
import jax.numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib
from numpy.fft import fft2, ifft2, fftshift, ifftshift




def calculate_radial_error(images, num_rings):
    # Get the number of images and the image dimensions
    n_images, height, width = images.shape
    center = (height // 2, width // 2)

    # Calculate the maximum radius (half the width or height)
    max_radius = min(center)

    # Generate a coordinate grid
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Calculate the logarithmically spaced radii
    radii = onp.linspace(0, max_radius, num_rings+1)
    area = onp.pi * (radii[1:]**2 - radii[:-1]**2)

    # Initialize the array to store errors for all images and rings
    ring_errors = onp.zeros((n_images, num_rings), dtype=onp.complex64)

    # Vectorized computation of errors for all rings and all images
    for i in range(1,num_rings):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]
        
        # Create the mask for the current ring
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Apply the mask to all images and compute the mean absolute error for the current ring
        ring_values = images[:, mask]  # Shape: (n_images, number_of_pixels_in_ring)
        ring_errors[:, i] = np.mean(np.abs(ring_values), axis=1)#/np.sum(mask)#area[i]


    

    # Normalize by the first image's errors
    return ring_errors 



def calculate_error_spheres(images, num_spheres):
    # Get the number of images and the 3D dimensions
    n_images, width, length, height = images.shape
    center = (width // 2, length // 2, height // 2)

    # Calculate the maximum radius (half the width, length, or height)
    max_radius = min(center)

    # Generate a 3D coordinate grid with correct axis order
    x, y, z = onp.ogrid[:width, :length, :height]
    distance_from_center = onp.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Prepare to store the mean absolute errors for each image and spherical shell
    sphere_errors = onp.zeros((n_images, num_spheres), dtype=onp.complex64)

    # Calculate the logarithmically spaced radii
    radii = np.linspace(0, max_radius, num_spheres+1)
    area = onp.pi * (radii[1:]**2 - radii[:-1]**2)

    # Vectorized computation for each spherical shell
    for i in range(1,num_spheres):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]

        # Create the mask for the current spherical shell
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Apply the mask to all images and compute the mean absolute error for the current spherical shell
        shell_values = images[:, mask]  # Shape: (n_images, number_of_voxels_in_shell)
        
        # Calculate mean absolute error for each image in the current shell
        sphere_errors[:, i] = np.mean(np.abs(shell_values), axis=1)#/np.sum(mask)#area[i]

    return sphere_errors


def calculate_radial(images, num_rings):
    # Get the number of images and the image dimensions
    n_images, height, width = images.shape
    center = (height // 2, width // 2)

    # Calculate the maximum radius (half the width or height)
    max_radius = min(center)

    # Generate a coordinate grid
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Calculate the logarithmically spaced radii
    radii = onp.linspace(0, max_radius, num_rings+1)
    area = onp.pi * (radii[1:]**2 - radii[:-1]**2)

    # Initialize the array to store errors for all images and rings
    ring_errors = onp.zeros((n_images, num_rings))
    ring_std = onp.zeros((n_images, num_rings))

    # Vectorized computation of errors for all rings and all images
    for i in range(num_rings):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]
        
        # Create the mask for the current ring
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Apply the mask to all images and compute the mean absolute error for the current ring
        ring_values = np.abs(images[:, mask])  # Shape: (n_images, number_of_pixels_in_ring)
        ring_errors[:, i] = np.mean(np.abs(ring_values), axis=1).real#/np.mean(mask)#area[i]
        ring_std[:, i] = np.std(np.abs(ring_values)-np.mean(np.abs(ring_values), axis=1)[:,None], axis=1).real


    

    # Normalize by the first image's errors
    return ring_errors[0,:], ring_std[0,:]



def calculate_error(images, num_spheres):
    # Get the number of images and the 3D dimensions
    n_images, width, length, height = images.shape
    center = (width // 2, length // 2, height // 2)

    # Calculate the maximum radius (half the width, length, or height)
    max_radius = min(center)

    # Generate a 3D coordinate grid with correct axis order
    x, y, z = onp.ogrid[:width, :length, :height]
    distance_from_center = onp.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Prepare to store the mean absolute errors for each image and spherical shell
    sphere_errors = onp.zeros((n_images, num_spheres))
    sphere_std = onp.zeros((n_images, num_spheres))

    # Calculate the logarithmically spaced radii
    radii = np.linspace(0, max_radius, num_spheres+1)
    area = onp.pi * (radii[1:]**2 - radii[:-1]**2)

    # Vectorized computation for each spherical shell
    for i in range(num_spheres):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]

        # Create the mask for the current spherical shell
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Apply the mask to all images and compute the mean absolute error for the current spherical shell
        shell_values = np.abs(images[:, mask])  # Shape: (n_images, number_of_voxels_in_shell)
        
        # Calculate mean absolute error for each image in the current shell
        sphere_errors[:, i] = np.mean(np.abs(shell_values), axis=1).real
        sphere_std[:, i] = np.std(np.abs(shell_values)-np.mean(np.abs(shell_values), axis=1)[:,None], axis=1).real


    return sphere_errors[0,:], sphere_std[0,:]

smo = .1
smo2 = .1
n_test = 5
v = np.pi/2
start = 0
n1 = [i for i in range(n_test)]
n2 = [i for i in range(n_test)]
m1 = [i for i in range(n_test)]
m2 = [i for i in range(n_test)]
m3 = [i for i in range(n_test)]
m4 = [i for i in range(n_test)]
n = 2 + math.pi/100
it=10
T = 1#it*500
colors = plt.cm.plasma(np.linspace(0,.8,4))


def slope(u):
    return np.polyfit(np.linspace(0,1,u.shape[0]), np.abs(u), 1)[0]



fig, axs = plt.subplots(2, 4, figsize=(12, 6))


tipo2 = "astro"
from skimage import data
from skimage.color import rgb2gray


tipo3 = '19octuni-' ##'19octnorm-'#"3Nobias4multilayer-norm-""2Nobias4multilayer-uni-"#
tipo = "2Nobias4multilayer-uni-"#
#ej = data.camera()
ej2 = data.astronaut()
ej2 = rgb2gray(ej2)#[20:220,120:320]
N_s = 256
size = (N_s, N_s)
ima = Image.fromarray(ej2)
resized = ima.resize(size, Image.LANCZOS)
im = np.array(resized)
size = im.shape

im = im/255.0
im -= im.mean()
ft = np.fft.fftshift(np.fft.fft2(im))
#im = im[:,:,None]
del ej2
del ima
del resized
layers = [2,20000,1]


aN = np.sqrt(2/(layers[-1] + layers[-2]))
lr =1e-04
if tipo3 == "19octnorm-":
    v = 0.75*np.pi
    SW = [64*v,86*v,128*v]#[21*v,42*v,64*v]##[60,90,120]
    L = "$\sigma_{w}$"
    sigmaA = 0.009000001 #0.08800001#
elif tipo3 == "19octuni-":
    v = np.pi
    SW = [64*v,86*v,128*v]#[21*v,42*v,64*v]##[60,90,120]
    sigmaA = 0.009000001 #0.08800001#
    L = "R"
elif tipo == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [64*v,86*v,128*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [64*v,86*v,128*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "$\sigma_{w}$"

r = int(min(im.shape[0]/n, im.shape[1]/n))
r1 = int(min(im.shape[0], im.shape[1]))
for j, w in enumerate(SW):           
        for k in range(n_test):

            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.009000001-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
            u = u.reshape((11,size[0],size[1]))
            u1 = u.copy()
            path = tipo+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.08800001-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u2 = onp.loadtxt(f)
            u2 = u2.reshape((11,size[0],size[1]))
            u3 = u2.copy()
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft#[int(size[0]/2),:]
            m1[k], m2[k] = calculate_radial(u1, r)

            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft#[int(size[0]/2),:]
            m3[k], m4[k] = calculate_radial(u3, r)


            u = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u) - ft#[int(size[0]/2),:]
            u = -np.log(np.abs(u)/(np.abs(u[0])+0.0000001))
            u = calculate_radial_error(u, r)

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])

            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft#[int(size[0]/2),:]
            u2 = -np.log(np.abs(u2)/(np.abs(u2[0])+0.0000001))
            u2 = calculate_radial_error(u2, r)

            n2[k] = vmap(slope)(u2[:it,:].T)#np.abs(u[it,:])
            
        print (n1[0].shape, n2[0].shape, m1[0].shape, m2[0].shape)
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T   for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
        axs[0, 0].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$') #[:int(size[1]/2)]
        if j == 0:
            maximun = math.floor((y[start:].max()+.4)* 10) / 10

        y = np.mean(np.array([gaussian_filter1d(n2[l],smo)/T   for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
        axs[1, 0].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$') #[:int(size[1]/2)]
        if j == 0:
            maximun2 = math.floor((y[start:].max()+.1)* 10) / 10
        """
        y = np.mean(np.array([gaussian_filter1d(m2[l]/(m1[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[1, 0].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun3 = math.floor((y[start:].max()+.1)* 10) / 10
        y = np.mean(np.array([gaussian_filter1d(m4[l]/(m3[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[3, 0].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun4 = math.floor((y[start:].max()+.1)* 10) / 10
        """
        

axs[0, 0].set_title('Astronaut')
axs[0, 0].axvline(x=32,color='blue', linestyle='--', linewidth=.5)
axs[0, 0].axvline(x=43, color='purple', linestyle='--', linewidth=.5)
axs[0, 0].legend(frameon=False,loc='upper right',fontsize=8)
axs[0, 0].set_xticks([0,64])
axs[0, 0].set_xlim([0,64])
axs[0, 0].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[0, 0].set_ylabel('2 layers', labelpad=-20, fontsize=16)
axs[0, 0].set_yticks([0,maximun])

"""
axs[1, 0].axvline(x=32,color='blue', linestyle='--', linewidth=.5)
axs[1, 0].axvline(x=43, color='purple', linestyle='--', linewidth=.5)
axs[1, 0].set_xticks([0,64])
axs[1, 0].set_xlim([0,64])
axs[1, 0].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 0].set_ylabel('2L noise', labelpad=-20, fontsize=16)
axs[1, 0].set_yticks([0,maximun3])
"""

axs[1, 0].axvline(x=32, color='blue', linestyle='--', linewidth=.5)
axs[1, 0].axvline(x=43, color='purple', linestyle='--', linewidth=.5)
axs[1, 0].set_xticks([0,64])
axs[1, 0].set_xlim([0,64])
axs[1, 0].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 0].set_ylabel('4 layers', labelpad=-20, fontsize=16)
axs[1, 0].set_yticks([0,maximun2])

"""
axs[3, 0].axvline(x=32, color='blue', linestyle='--', linewidth=.5)
axs[3, 0].axvline(x=43, color='purple', linestyle='--', linewidth=.5)
axs[3, 0].set_ylabel('L4 noise', labelpad=-20, fontsize=16)
axs[3, 0].set_xticks([0,64])
axs[3, 0].set_xlim([0,64])
axs[3, 0].set_yticks([0,maximun4])
axs[3, 0].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
"""


n1 = [i for i in range(n_test)]
n2 = [i for i in range(n_test)]
m1 = [i for i in range(n_test)]
m2 = [i for i in range(n_test)]

tipo2 = "brain"
import nibabel as nib
#../../../
A = nib.load("../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
im = onp.swapaxes(A[7:187,16:216,120], 0, 1)

size = im.shape
im = im/255.0
im -= im.mean()
print (size)
#im -= im.mean()
#im = im[:,:,None]
del A
layers = [2,20000,1]

ft = np.fft.fftshift(np.fft.fft2(im))
print (ft.shape)

aN = np.sqrt(2/(layers[-1] + layers[-2]))
lr =1e-04
#tipo = "norm"
#tipo3 = '19octnorm-'
#v = 4*np.pi
#SW = [10*v,30*v,60*v]
#v = np.pi/2
if tipo3 == "19octnorm-":
    v = 0.75*np.pi
    SW = [45*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.006
    L = "$\sigma_{w}$"
elif tipo3 == "19octuni-":
    v = np.pi
    SW = [45*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.006 #0.08800001#
    L = "R"
elif tipo == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [45*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [45*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "$\sigma_{w}$"

r = int(min(im.shape[0]/n, im.shape[1]/n))
r1 = int(min(im.shape[0], im.shape[1]))
n1 = [i for i in range(n_test)]
colors = plt.cm.plasma(np.linspace(0,.8,4))
for j, w in enumerate(SW):           
        for k in range(n_test):
            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.006-'+str(k)#0.0050000004-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
            u = u.reshape((11,size[0],size[1]))
            u1 = u.copy()
            path = tipo+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.08800001-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u2 = onp.loadtxt(f)
            u2 = u2.reshape((11,size[0],size[1]))
            u3 = u2.copy()

            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft#[int(size[0]/2),:]
            m1[k], m2[k] = calculate_radial(u1, r)

            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft#[int(size[0]/2),:]
            m3[k], m4[k] = calculate_radial(u3, r)

            u = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u) - ft
            u = -np.log(np.abs(u)/(np.abs(u[0])+0.0000001))
            u = calculate_radial_error(u, r)

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])

            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft
            u2 = -np.log(np.abs(u2)/(np.abs(u2[0])+0.0000001))
            u2 = calculate_radial_error(u2, r)

            n2[k] = vmap(slope)(u2[:it,:].T)#np.abs(u[it,:])
        print (n1[0].shape, n2[0].shape, m1[0].shape, m2[0].shape)
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T   for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[0, 1].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$')
        if j == 0:
            maximun = math.floor((y[start:].max()+.5)* 10) / 10

        y = np.mean(np.array([gaussian_filter1d(n2[l],smo)/T   for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[1, 1].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$')
        if j == 0:
            maximun2 = math.floor((y[start:].max()+.1)* 10) / 10
        """
        y = np.mean(np.array([gaussian_filter1d(m2[l]/(m1[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[1, 1].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun3 = math.floor((y[start:].max()+.1)* 10) / 10
        y = np.mean(np.array([gaussian_filter1d(m4[l]/(m3[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[3, 1].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun4 = math.floor((y[start:].max()+.1)* 10) / 10
        """

axs[0, 1].set_title('ATLAS 2D')
axs[0, 1].axvline(x=22.5,color='blue', linestyle='--', linewidth=.5)
axs[0, 1].axvline(x=30, color='purple', linestyle='--', linewidth=.5)
axs[0, 1].legend(frameon=False,loc='upper right',fontsize=8)
axs[0, 1].set_xticks([0,45])
axs[0, 1].set_xlim([0,45])
axs[0, 1].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[0, 1].set_yticks([0,maximun])
"""
axs[1, 1].axvline(x=22.5,color='blue', linestyle='--', linewidth=.5)
axs[1, 1].axvline(x=30, color='purple', linestyle='--', linewidth=.5)
axs[1, 1].set_xticks([0,45])
axs[1, 1].set_xlim([0,45])
axs[1, 1].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 1].set_yticks([0,maximun3])
"""

axs[1, 1].axvline(x=22.5, color='blue', linestyle='--', linewidth=.5)
axs[1, 1].axvline(x=30, color='purple', linestyle='--', linewidth=.5)
axs[1, 1].set_xticks([0,45])
axs[1, 1].set_xlim([0,45])
axs[1, 1].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 1].set_yticks([0,maximun2])

"""
axs[3, 1].axvline(x=22.5, color='blue', linestyle='--', linewidth=.5)
axs[3, 1].axvline(x=30, color='purple', linestyle='--', linewidth=.5)
axs[3, 1].set_xticks([0,45])
axs[3, 1].set_xlim([0,45])
axs[3, 1].set_yticks([0,maximun4])
axs[3, 1].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
"""




n1 = [i for i in range(n_test)]
n2 = [i for i in range(n_test)]
m1 = [i for i in range(n_test)]
m2 = [i for i in range(n_test)]

tipo2 = "human"
#tipo3 = '19octnorm-'
from skimage.color import rgb2gray
#../../../
img = Image.open('../../../dataset/1 (107).jpg')
# Convert the image to a Numpy array
img_array = onp.array(img)
s = (140,184)
im1 = Image.fromarray(img_array[:,:,0])
resized1 = im1.resize(s, Image.LANCZOS)
im2 = Image.fromarray(img_array[:,:,1])
resized2 = im2.resize(s, Image.LANCZOS)
im3 = Image.fromarray(img_array[:,:,2])
resized3 = im3.resize(s, Image.LANCZOS)
im = onp.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None], np.array(resized3)[:,:,None]), axis=2)

im = onp.array(im,dtype = 'float32')
im[:,:,0] = im[:,:,0]/255.0
im[:,:,1] = im[:,:,1]/255.0
im[:,:,2] = im[:,:,2]/255.0
im[:,:,0] -= im[:,:,0].mean()
im[:,:,1] -= im[:,:,1].mean()
im[:,:,2] -= im[:,:,2].mean()
size = im.shape
#im = im[:,:,None]
print (im.shape)
del img
del img_array
del resized1
del resized2
del resized3
del im1
del im2
del im3
layers = [2,20000,3]

#ft = np.fft.fftshift(np.fft.fftn(im))#2(im, axes=(0,1)), axes=(0,1))
ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)), axes=(0,1))
print (ft.shape)

aN = np.sqrt(2/(layers[-1] + layers[-2]))
lr =1e-04


if tipo3 == "19octnorm-":
    v = 0.75*np.pi
    SW = [35*v,46*v,70*v]#[7,14,20]
    sigmaA = 0.009000001
    L = "$\sigma_{w}$"
elif tipo3 == "19octuni-":
    v = np.pi
    SW = [35*v,46*v,70*v]
    sigmaA = 0.009000001
    L = "R"
elif tipo == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [35*v,46*v,70*v]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [35*v,46*v,70*v]#[7,14,20]
    sigmaA = 0.08800001#0.009000001
    L = "$\sigma_{w}$"


r = int(min(im.shape[0]/n, im.shape[1]/n))
r1 = int(min(im.shape[0], im.shape[1]))
colors = plt.cm.plasma(np.linspace(0,.8,4))
for j, w in enumerate(SW):
    
        for k in range(n_test):
            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.009000001-'+str(k)
            #path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.010000001-'+str(k)
            #path = 'best'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a'+str(aN.round(2))+tipo+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
                #print (u.shape)
            u = u.reshape((11,size[0],size[1],3))
            p = u.copy()
            path = tipo+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.08800001-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                q = onp.loadtxt(f)
            q = q.reshape((11,size[0],size[1],3))
            q1 = q.copy()
            u1 = p[:,:,:,0]
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft[:,:,0]

            u2 = p[:,:,:,1]
            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft[:,:,1]

            u3 = p[:,:,:,2]
            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft[:,:,2]
            
            p = np.mean(np.array([np.abs(u1),np.abs(u2),np.abs(u3)]), axis=0)

            m1[k], m2[k] = calculate_radial(p, r)

            u1 = q1[:,:,:,0]
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft[:,:,0]

            u2 = q1[:,:,:,1]
            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft[:,:,1]

            u3 = q1[:,:,:,2]
            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft[:,:,2]
            
            q1 = np.mean(np.array([np.abs(u1),np.abs(u2),np.abs(u3)]), axis=0)

            m3[k], m4[k] = calculate_radial(q1, r)


            u1 = u[:,:,:,0]
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft[:,:,0]
            u1 = -np.log(np.abs(u1)/(np.abs(u1[0])+0.0000001))

            u2 = u[:,:,:,1]
            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft[:,:,1]
            u2 = -np.log(np.abs(u2)/(np.abs(u2[0])+0.0000001))

            u3 = u[:,:,:,2]
            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft[:,:,2]
            u3 = -np.log(np.abs(u3)/(np.abs(u3[0])+0.0000001))
            
            u = np.mean(np.array([u1,u2,u3]), axis=0)
            
            u = calculate_radial_error(u, r)          
          

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])

            u1 = q[:,:,:,0]
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft[:,:,0]
            u1 = -np.log(np.abs(u1)/(np.abs(u1[0])+0.0000001))

            u2 = q[:,:,:,1]
            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft[:,:,1]
            u2 = -np.log(np.abs(u2)/(np.abs(u2[0])+0.0000001))

            u3 = q[:,:,:,2]
            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft[:,:,2]
            u3 = -np.log(np.abs(u3)/(np.abs(u3[0])+0.0000001))
            
            q = np.mean(np.array([u1,u2,u3]), axis=0)
            
            q = calculate_radial_error(q, r)          
          

            n2[k] = vmap(slope)(q[:it,:].T)#np.abs(u[it,:])
        print (n1[0].shape, n2[0].shape, m1[0].shape, m2[0].shape)
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[0,2].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$')
        if j == 0:
            maximun = math.floor((y[start:].max()+.4)* 10) / 10
        y = np.mean(np.array([gaussian_filter1d(n2[l],smo)/T  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[1,2].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$')
        if j == 0:
            maximun2 = math.floor((y[start:].max()+.1)* 10) / 10
        """
        y = np.mean(np.array([gaussian_filter1d(m2[l]/(m1[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[1, 2].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun3 = math.floor((y[start:].max()+.1)* 10) / 10
        y = np.mean(np.array([gaussian_filter1d(m4[l]/(m3[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[3, 2].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun4 = math.floor((y[start:].max()+.1)* 10) / 10
        """

axs[0, 2].set_title('Human Face')
axs[0, 2].axvline(x=17.5,color='blue', linestyle='--', linewidth=.5)
axs[0, 2].axvline(x=23, color='purple', linestyle='--', linewidth=.5)
axs[0, 2].legend(frameon=False,loc='upper right',fontsize=8)
axs[0, 2].set_xticks([0,35])
axs[0, 2].set_xlim([0,35])
axs[0, 2].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[0, 2].set_yticks([0,maximun])

"""
axs[1, 2].axvline(x=17.5,color='blue', linestyle='--', linewidth=.5)
axs[1, 2].axvline(x=23, color='purple', linestyle='--', linewidth=.5)
axs[1, 2].set_xticks([0,35])
axs[1, 2].set_xlim([0,35])
axs[1, 2].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 2].set_yticks([0,maximun3])
"""

axs[1, 2].axvline(x=17.5, color='blue', linestyle='--', linewidth=.5)
axs[1, 2].axvline(x=23, color='purple', linestyle='--', linewidth=.5)
axs[1, 2].set_xticks([0,35])
axs[1, 2].set_xlim([0,35])
axs[1, 2].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 2].set_yticks([0,maximun2])

"""
axs[3, 2].axvline(x=17.5, color='blue', linestyle='--', linewidth=.5)
axs[3, 2].axvline(x=23, color='purple', linestyle='--', linewidth=.5)
axs[3, 2].set_xticks([0,35])
axs[3, 2].set_xlim([0,35])
axs[3, 2].set_yticks([0,maximun4])
axs[3, 2].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
"""


n1 = [i for i in range(n_test)]
n2 = [i for i in range(n_test)]
m1 = [i for i in range(n_test)]
m2 = [i for i in range(n_test)]


tipo2 = "3d"
from skimage.color import rgb2gray
#../../../
import nibabel as nib
from skimage.transform import resize
A = nib.load("../../../dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()

im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
im = resize(im, (50,45,45))
maximun = np.max(im)
im = im/255.0
im -= im.mean()
size = im.shape

mip= Image.open('../../../dataset/brain3D.png')



#im = im[:,:,None]
del A
layers = [3,15000,1]
grilla = meshgrid_from_subdiv(im.shape, (-1,1))
x_train = grilla
y_train = im[:,:,:,None]
ft = np.fft.fftshift(np.fft.fftn(im))
print (size)
aN = np.sqrt(2/(layers[-1] + layers[-2]))
lr = 1e-4

if tipo3 == "19octnorm-":
    v = 0.75*np.pi
    SW = [11.25*v,15*v,22.5*v]#[60,80,100]
    sigmaA = 0.008
    L = "$\sigma_{w}$"
elif tipo3 == "19octuni-":
    v = np.pi
    SW = [11.25*v,15*v,22.5*v]
    sigmaA = 0.008
    L = "R"
elif tipo == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [11.25*v,15*v,22.5*v]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [11.25*v,15*v,22.5*v]
    sigmaA = 0.08800001#0.009000001
    L = "$\sigma_{w}$"




r1 = int(min(im.shape[0], im.shape[1]))
r = int(min(im.shape[0]/n, im.shape[1]/n, im.shape[2]/n))
for j, w in enumerate(SW):           
        for k in range(n_test):
            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.008-'+str(k)

            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
            u = u.reshape((11,size[0],size[1], size[2]))
            u1 = u.copy()
            path = tipo+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.08800001-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u2 = onp.loadtxt(f)
            u2 = u2.reshape((11,size[0],size[1], size[2]))
            u3 = u2.copy()
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fftn(x)))(u1) - ft#[int(size[0]/2),:]
            m1[k], m2[k] = calculate_error(u1, r)

            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fftn(x)))(u3) - ft#[int(size[0]/2),:]
            m3[k], m4[k] = calculate_error(u3, r)
            #u = u[:,int(size[0]/2),:,5]
            u = vmap(lambda x: np.fft.fftshift(np.fft.fftn(x)))(u) - ft
            u = -np.log(np.abs(u)/(np.abs(u[0])+0.0000001))
            u = calculate_error_spheres(u, r)

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])

            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fftn(x)))(u2) - ft
            u2 = -np.log(np.abs(u2)/(np.abs(u2[0])+0.0000001))
            u2 = calculate_error_spheres(u2, r)

            n2[k] = vmap(slope)(u2[:it,:].T)#np.abs(u[it,:])
        print (n1[0].shape, n2[0].shape, m1[0].shape, m2[0].shape)
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[0, 3].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$')
        if j == 0:
            maximun = math.floor((y[start:].max()+.4)* 10) / 10
        y = np.mean(np.array([gaussian_filter1d(n2[l],smo)/T  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[1, 3].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = L+' = '+str(int(w))+'$/2\pi$')
        if j == 0:
            maximun2 = math.floor((y[start:].max()+.1)* 10) / 10
        """
        y = np.mean(np.array([gaussian_filter1d(m2[l]/(m1[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[1, 3].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun3 = math.floor((y[start:].max()+.1)* 10) / 10
        y = np.mean(np.array([gaussian_filter1d(m4[l]/(m3[l]+0.0000001),smo2)  for l in range(n_test)]), axis=0) 
        axs[3, 3].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$Mean R$ = '+str(int(w)))
        if j == 0:
            maximun4 = math.floor((y[start:].max()+.1)* 10) / 10
        """

axs[0, 3].set_title("ATLAS 3D")
axs[0, 3].axvline(x=5.625,color='blue', linestyle='--', linewidth=.5)
axs[0, 3].axvline(x=7.5, color='purple', linestyle='--', linewidth=.5)
axs[0, 3].legend(frameon=False,loc='upper right',fontsize=8)
axs[0, 3].set_xticks([0,11])
axs[0, 3].set_xlim([0,11.25])
axs[0, 3].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[0, 3].set_yticks([0,maximun])

"""
axs[1, 3].axvline(x=5.625,color='blue', linestyle='--', linewidth=.5)
axs[1, 3].axvline(x=7.5, color='purple', linestyle='--', linewidth=.5)
axs[1, 3].set_xticks([0,11.25])
axs[1, 3].set_xlim([0,11.25])
axs[1, 3].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 3].set_yticks([0,maximun3])
"""

axs[1, 3].axvline(x=5.625, color='blue', linestyle='--', linewidth=.5)
axs[1, 3].axvline(x=7.5, color='purple', linestyle='--', linewidth=.5)
axs[1, 3].set_xticks([0,11])
axs[1, 3].set_xlim([0,11.25])
axs[1, 3].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 3].set_yticks([0,maximun2])

"""
axs[3, 3].axvline(x=5.625, color='blue', linestyle='--', linewidth=.5)
axs[3, 3].axvline(x=7.5, color='purple', linestyle='--', linewidth=.5)
axs[3, 3].set_xticks([0,11.25])
axs[3, 3].set_xlim([0,11.25])
axs[3, 3].set_yticks([0,maximun4])
axs[3, 3].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
"""

for i in range(2):
    for k in range(3):
        axs[i,k].set_xticks(axs[i,k].get_xlim())
        ticks = axs[i,k].get_xticks()
        labels = axs[i,k].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")




plt.tight_layout()
plt.savefig('results/Anexo'+tipo3+'-LR_others_dataset'+str(smo)+'.pdf', bbox_inches='tight')
plt.show()


