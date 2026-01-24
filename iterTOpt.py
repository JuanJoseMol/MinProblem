import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from PIL import Image
import nibabel as nib
from skimage.color import rgb2gray
from scipy.optimize import minimize
import pickle
from jax import random, grad, jit, vmap
#import jax.numpy as np


def load_function(tipo2, ej):
  

  
  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    if ej =="1":
      A = nib.load("../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    else:
      A = nib.load("../../../dataset/sub-r039s002_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    maximun = np.max(im)
    size = im.shape
    print (size)
    im = im/255.0
    im -= im.mean()
    del A
    layers = [2,15000,1]
    ft = np.fft.fftshift(np.fft.fft2(im))
    #plt.imshow(im)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()


  if tipo2 == "human":
    from skimage.color import rgb2gray
    #../../../
    if ej =="1":
      im = Image.open('../../../dataset/1 (107).jpg')
    else:
      im = Image.open('../../../dataset/1 (1487).jpg')
    # Convert the image to a Numpy array
    im = np.array(im)
    s = (140,184)
    im1 = Image.fromarray(im[:,:,0])
    resized1 = im1.resize(s, Image.LANCZOS)
    im2 = Image.fromarray(im[:,:,1])
    resized2 = im2.resize(s, Image.LANCZOS)
    im3 = Image.fromarray(im[:,:,2])
    resized3 = im3.resize(s, Image.LANCZOS)
    im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None], 
            np.array(resized3)[:,:,None]), axis=2),dtype = 'float32')
    print (type(im))
    
    im[:,:,0] = im[:,:,0]/255.0
    im[:,:,1] = im[:,:,1]/255.0
    im[:,:,2] = im[:,:,2]/255.0
    im[:,:,0] -= im[:,:,0].mean()
    im[:,:,1] -= im[:,:,1].mean()
    im[:,:,2] -= im[:,:,2].mean()
    size = im.shape
    ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
    #plt.imshow(rgb2gray(im))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()

  if tipo2 == "3d":
    from skimage.color import rgb2gray
    #../../../
    import nibabel as nib
    from skimage.transform import resize
    if ej =="1":
      A = nib.load("../../../dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    else:
      A = nib.load("../../../dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    #im = np.swapaxes(A[5:189,16:216,100:106], 0, 1)
    #im = resize(im, (50,46,10))
    #im =  np.swapaxes(A[5:191,14:218,98:108], 0, 1)
    #im = resize(im, (68,62,10))
    im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
    im = resize(im, (50,45,45))
    maximun = np.max(im)
    im = im/255.0
    im -= im.mean()
    size = im.shape
    ft = np.fft.fftshift(np.fft.fftn(im))
    
    #im = im[:,:,None]
    del A
    """
    plt.imshow(im[:,:,5])
    plt.show()
    """
  return ft
# Given function




# Gaussian function
def gaussianbrain(x, y, T, sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2)/ (2 * sigma**2))/(2*np.pi*sigma**2))

# Objective function to minimize
def deviationbrain(sigma, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = gaussianbrain(X, Y, T,sigma)


    h_array = np.array(f_array) * g_array
    deviation = np.trapz(np.trapz((h_array), x, axis=0), y, axis=0)
    return deviation

def gaussianhuman(x, y, T,sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2)/ (2 * sigma**2))/(2*np.pi*sigma**2))

# Objective function to minimize
def deviationhuman(sigma, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = gaussianhuman(X, Y, T, sigma)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(h_array, x, axis=0), y, axis=0)
    return deviation

def gaussian3d(x, y, z, T,sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2+ z**2)/ (2 * sigma**2))/((2*np.pi*sigma**2)*np.sqrt(2*np.pi*sigma**2)))

# Objective function to minimize
def deviation3d(sigma, x, y, z, T,f_array):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    g_array = gaussian3d(X, Y, Z, T, sigma)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(np.trapz(h_array, x, axis=0), y, axis=0), z, axis=0)
    return deviation



if __name__ == "__main__":
    slides = 11
    
    tipo2 = "brain"
    target = load_function(tipo2, "1")
    print (type(target))
    size = target.shape
    ft = (np.abs(target)**2)

    print (ft.shape)
    Nx, Ny = ft.shape
    print (Nx, Ny)
    # Interval and discretization
    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)

    # Initial guess for sigma
    sigma = 90*0.75*np.pi

    path = "19octnorm-brainlr0.0001Ns200-w"+str(np.round(sigma,3))+"a0.006-0"
    #path = "ej1-brainlr0.05m15000-w212.058"
    with open('results/'+path+'.txt', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides,size[0],size[1]))
    for i in range(slides):
      u[i,:,:] = np.abs(np.fft.fftshift(np.fft.fft2(u[i,:,:])) - target)

    print (u.shape)
    a = [np.trapz(np.trapz((u[i,:,:]), x, axis=0), y, axis=0) for i in range(slides)]
    b = []

    times = np.linspace(0,5000,slides)*170
    timesNN = np.linspace(0,1000,slides)
    for T in times:
      

      # Minimize deviation
      result = deviationbrain(sigma, x, y, T, ft)
      b.append(result)

    plt.plot(timesNN,a, label="NN")
    plt.plot(timesNN, b, label="funcional")
    plt.yscale('log')
    plt.legend()
    plt.show()


    



    """
    tipo2 = "human"
    target = load_function(tipo2, "1")
    print (type(target))
    size = target.shape
    ft = (np.abs(target)**2)

    print (ft.shape)
    Nx, Ny, _ = ft.shape
    print (Nx, Ny)
    # Interval and discretization
    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)

    # Initial guess for sigma
    sigma = 70*0.75*np.pi

    #path = "19octnorm-humanlr0.0001Ns184-w"+str(np.round(sigma,3))+"a0.009000001-0"
    path = "ej1-humanlr0.05m15000-w164.934"
    with open('results/'+path+'.txt', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides,size[0],size[1],3))
    for i in range(slides):
      u[i,:,:,0] = np.abs(np.fft.fftshift(np.fft.fft2(u[i,:,:,0])) - target[:,:,0])

    print (u.shape)
    a = [np.trapz(np.trapz((u[i,:,:,0]), x, axis=0), y, axis=0) for i in range(slides)]
    b = []

          
    times = np.linspace(0,5000,slides)*160
    timesNN = np.linspace(0,1000,slides)
    for T in times:
      

      # Minimize deviation
      result = deviationhuman(sigma, x, y, T, ft[:,:,0])
      b.append(result)

    plt.plot(timesNN,a, label="NN")
    plt.plot(timesNN, b, label="funcional")
    plt.yscale('log')
    plt.legend()
    plt.show()

    
    

    
    tipo2 = "3d"
    target = load_function(tipo2, "1")
    print (type(target))
    size = target.shape
    ft = (np.abs(target)**2)

    print (ft.shape)
    Nx, Ny, Nz = ft.shape
    print (Nx, Ny)
    # Interval and discretization
    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    z = np.linspace(-int(Nz/4), int(Nz/4), Nz)

    # Initial guess for sigma
    sigma = 22.5*0.75*np.pi

    #path = "19octnorm-3dlr0.0001Ns50-w53.014a0.008-0"
    path = "ej1-3dlr0.005m15000-w53.014"
    with open('results/'+path+'.txt', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides,size[0],size[1], size[2]))
    for i in range(slides):
      u[i,:,:,:] = np.abs(np.fft.fftshift(np.fft.fftn(u[i,:,:,:])) - target)

    print (u.shape)
    a = [np.trapz(np.trapz(np.trapz((u[i,:,:,:]), x, axis=0), y, axis=0), z, axis=0) for i in range(slides)]
    b = []

    times = np.linspace(0,5000,slides)*1000
    timesNN = np.linspace(0,1000,slides)
    for T in times:
      

      # Minimize deviation
      result = deviation3d(sigma, x, y, z, T, ft)
      b.append(result)

    plt.plot(timesNN,a, label="NN")
    plt.plot(timesNN, b, label="funcional")
    #plt.yscale('log')
    plt.legend()
    plt.show()"""
    

    
    
    
      
    
    

      