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
    plt.xlabel(xlabel, labelpad = xlabelpad)
    plt.ylabel(ylabel, labelpad = ylabelpad)

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
    for i in range(num_rings):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]
        
        # Create the mask for the current ring
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Apply the mask to all images and compute the mean absolute error for the current ring
        ring_values = images[:, mask]  # Shape: (n_images, number_of_pixels_in_ring)
        ring_errors[:, i] = np.mean(np.abs(ring_values), axis=1)#/np.mean(mask)#area[i]


    

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
    for i in range(num_spheres):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]

        # Create the mask for the current spherical shell
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)

        # Apply the mask to all images and compute the mean absolute error for the current spherical shell
        shell_values = images[:, mask]  # Shape: (n_images, number_of_voxels_in_shell)
        
        # Calculate mean absolute error for each image in the current shell
        sphere_errors[:, i] = np.mean(np.abs(shell_values), axis=1)#/np.mean(mask)#area[i]

    return sphere_errors

smo = .1
n_test = 5
v = np.pi/2
start = 0
n1 = [i for i in range(n_test)]
n = 2
it=10
T = 1#it*500
colors = plt.cm.plasma(np.linspace(0,.8,4))


def slope(u):
    return np.polyfit(np.linspace(0,1,u.shape[0]), np.log(np.abs(u[0]/(u+0.000001))), 1)[0]



fig, axs = plt.subplots(2, 4, figsize=(12, 6))


tipo2 = "astro"
from skimage import data
from skimage.color import rgb2gray


tipo3 = '19octnorm-'#"2Nobias4multilayer-uni-"#'19octuni-' ##'19octnorm-'#"3Nobias4multilayer-norm-"
#ej = data.camera()
ej2 = data.astronaut()
ej2 = rgb2gray(ej2)#[20:220,120:320]
N_s = 256
size = (N_s, N_s)
ima = Image.fromarray(ej2)
resized = ima.resize(size, Image.LANCZOS)
im = np.array(resized)
size = im.shape

axs[0, 0].imshow(im, cmap="gray")
axs[0, 0].set_title('Astronaut')#('Astro')
axs[0, 0].axis('off')
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
    SW = [43*v,86*v,128*v]#[21*v,42*v,64*v]##[60,90,120]
    sigmaA = 0.009000001 #0.08800001#
    L = "R"
elif tipo3 == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [64*v,86*v,128*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [64*v,86*v,128*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "$\sigma_{w}$"

r = int(min(im.shape[0], im.shape[1])/n)
for j, w in enumerate(SW):           
        for k in range(n_test):

            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a'+str(sigmaA)+'-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
            u = u.reshape((11,size[0],size[1]))


            u = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u) - ft#[int(size[0]/2),:]
            u = calculate_radial_error(u, r)

            n1[k] = vmap(slope)(u[:it,:].T)
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T   for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
        axs[1, 0].plot(np.linspace(0,(int(size[0]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$\sigma_{w}$ = '+str(int(w))) #[:int(size[1]/2)]
        if j == 0:
            maximun = math.floor((y[start:].max()+.1)* 10) / 10
#plt.ylim([-0.5, 2])
axs[1, 0].plot(np.linspace(0,(int(size[0]/4)),int(len(y)))[start:], np.array([0 for i in range(int(len(y)))])[start:], color="k")
axs[1, 0].legend(fontsize=8)
axs[1, 0].set_xticks([0,64])
axs[1, 0].set_xlim([0,64])
axs[1, 0].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
axs[1, 0].set_ylabel('$FLR$', labelpad=-15, fontsize=16)
#ellenify('$\\xi$','frequency learning rate',xlabelpad=-50, ylabelpad=-27, minYticks=False)
#axs[1, 0].set_title('Astronaut')
axs[1, 0].set_yticks([0,maximun])





tipo2 = "brain"
import nibabel as nib
#../../../
A = nib.load("../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
im = onp.swapaxes(A[7:187,16:216,120], 0, 1)
axs[0, 1].imshow(im, cmap='gray')
axs[0, 1].set_title('ATLAS 2D')#('R001s001 ATLAS')
axs[0, 1].axis('off')
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
    SW = [30*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.006 #0.08800001#
    L = "R"
elif tipo3 == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [45*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [45*v,60*v,90*v]#[60,90,120]
    sigmaA = 0.08800001
    L = "$\sigma_{w}$"


r = int(min(im.shape[0], im.shape[1])/n)
n_test =5
n1 = [i for i in range(n_test)]
colors = plt.cm.plasma(np.linspace(0,.8,4))
for j, w in enumerate(SW):           
        for k in range(n_test):
            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a'+str(sigmaA)+'-'+str(k)#0.0050000004-'+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
            u = u.reshape((11,size[0],size[1]))
            u = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u) - ft
            u = calculate_radial_error(u, r)

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T   for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[1, 1].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$\sigma_{w}$ = '+str(int(w)))
        if j == 0:
            maximun = math.floor((y[start:].max()+.1)* 10) / 10
#plt.ylim([-0.5, 2])
axs[1, 1].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], np.array([0 for i in range(int(len(y)))])[start:], color="k")
axs[1, 1].legend(fontsize=8)
axs[1, 1].set_xticks([0,45])
axs[1, 1].set_xlim([0,45])
axs[1, 1].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
#ellenify('$\\xi$','frequency learning rate',xlabelpad=-3, ylabelpad=-27, minYticks=False)
#axs[1, 1].set_title('ATLAS 2D')
axs[1, 1].set_yticks([0,maximun])





n1 = [i for i in range(n_test)]
v = np.pi/2
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

axs[0, 2].imshow(im)
axs[0, 2].set_title('Human Face')#('107 HUMAN FACES')
axs[0, 2].axis('off')
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
    SW = [23*v,46*v,70*v]
    sigmaA = 0.009000001
    L = "R"
elif tipo3 == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [35*v,46*v,70*v]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [35*v,46*v,70*v]#[7,14,20]
    sigmaA = 0.08800001#0.009000001
    L = "$\sigma_{w}$"


r = int(min(im.shape[0], im.shape[1])/n)
colors = plt.cm.plasma(np.linspace(0,.8,4))
for j, w in enumerate(SW):
    
        for k in range(n_test):
            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a'+str(sigmaA)+'-'+str(k)
            #path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a0.010000001-'+str(k)
            #path = 'best'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a'+str(aN.round(2))+tipo+str(k)
            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
                #print (u.shape)
            u = u.reshape((11,size[0],size[1],3))

            u1 = u[:,:,:,0]
            u1 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u1) - ft[:,:,0]
            u1 = calculate_radial_error(u1, r)

            u2 = u[:,:,:,1]
            u2 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u2) - ft[:,:,1]
            u2 = calculate_radial_error(u2, r)

            u3 = u[:,:,:,2]
            u3 = vmap(lambda x: np.fft.fftshift(np.fft.fft2(x)))(u3) - ft[:,:,2]
            u3 = calculate_radial_error(u3, r)
            
            u = np.mean(np.array([u1,u2,u3]), axis=0)
            

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[1,2].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$\sigma_{w}$ = '+str(int(w)))
        if j == 0:
            maximun = math.floor((y[start:].max()+.1)* 10) / 10
#plt.ylim([-0.5, 2])
axs[1, 2].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], np.array([0 for i in range(int(len(y)))])[start:], color="k")
axs[1,2].legend(fontsize=8)
#axs[1, 2].set_xlim((0,35))
axs[1, 2].set_xticks([0,35])
axs[1, 2].set_xlim([0,35])
axs[1, 2].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
#ellenify('$\\xi$','frequency learning rate',xlabelpad=-3, ylabelpad=-27, minYticks=False)
#axs[1, 2].set_title('Human Face')
axs[1, 2].set_yticks([0,maximun])


n_test =5
n1 = [i for i in range(n_test)]
#tipo3 = '19octnorm-'
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

axs[0, 3].imshow(mip)
axs[0, 3].set_title("ATLAS 3D")#'3D R027s001 ATLAS')
axs[0, 3].axis('off')

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
    SW = [7.5*v,15*v,22.5*v]
    sigmaA = 0.008
    L = "R"
elif tipo3 == "2Nobias4multilayer-uni-": 
    v = np.pi
    SW = [11.25*v,15*v,22.5*v]
    sigmaA = 0.08800001
    L = "R"
elif tipo3 == "3Nobias4multilayer-norm-": 
    v = 0.75*np.pi
    SW = [11.25*v,15*v,22.5*v]
    sigmaA = 0.08800001#0.009000001
    L = "$\sigma_{w}$"



r = int(min(im.shape[0], im.shape[1])/n)
for j, w in enumerate(SW):           
        for k in range(n_test):
            path = tipo3+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(round(w,3))+'a'+str(sigmaA)+'-'+str(k)

            with open('results/'+path+'.txt', 'r') as f:
                u = onp.loadtxt(f)
            u = u.reshape((11,size[0],size[1], size[2]))
            #u = u[:,int(size[0]/2),:,5]
            u = vmap(lambda x: np.fft.fftshift(np.fft.fftn(x)))(u) - ft
            u = calculate_error_spheres(u, r)

            n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
        y = np.mean(np.array([gaussian_filter1d(n1[l],smo)/T  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)

        axs[1, 3].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], y[start:], color = colors[j], label = '$\sigma_{w}$ = '+str(int(w)))
        if j == 0:
            maximun = math.floor((y[start:].max()+.1)* 10) / 10

axs[1, 3].legend(fontsize=8)
axs[1, 3].plot(np.linspace(0,(int(size[1]/4)),int(len(y)))[start:], np.array([0 for i in range(int(len(y)))])[start:], color="k")
#axs[1, 3].set_xlim((0,11.25))
axs[1, 3].set_xticks([0,11.25])
axs[1, 3].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)
#ellenify('$\\xi$','frequency learning rate',xlabelpad=-10, ylabelpad=-10, minYticks=False)
#axs[1, 3].set_title('ATLAS 3D')
axs[1, 3].set_yticks([0,maximun])
axs[1, 3].set_xlim((0,11.25))





plt.tight_layout()
plt.savefig('results/Int'+tipo3+'-LR_others_dataset'+str(smo)+'.pdf')
plt.show()





