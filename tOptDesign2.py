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
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
#import jax.numpy as np
from scipy.optimize import differential_evolution
from randomSampling3 import pdf3dRhoMean, pdfBrainRhoMean, pdfHumanRhoMean
from scipy.interpolate import griddata



print ("asdasd")
def target_function(x, g_array, T, tipo2):

    if tipo2 == "3d":
        area = 8
    else:
        area = 4
    
    # Compute the integrand values for the filtered g_array
    integrand_values = (np.maximum((np.log(g_array) - np.log(x)) / (2 * T), 0))
    
    # Approximate the integral as a Riemann sum
    integral_value = np.sum(integrand_values)*area/np.size(g_array)
    
    return integral_value

# Bisection method
def bisection_method(g_array, T, tipo2, tol=1e-6, max_iter=100):
    a, b = 1e-8, 100000  # Define the search interval

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint

        # Evaluate the function at the midpoint
        f_c = target_function(c, g_array, T, tipo2)

        # Check if the function value is close to 0.5
        if np.abs(f_c - 1) < tol:
            return c
        
        # Decide which half of the interval to search in
        f_a = target_function(a, g_array, T, tipo2)
        if np.sign(f_c - 1) == np.sign(f_a - 1):
            a = c  # Narrow the interval to [c, b]
        else:
            b = c  # Narrow the interval to [a, c]
    
    raise ValueError("Bisection method did not converge")


def load_function(tipo2, ej):
  

  
  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    A = nib.load(ej).get_fdata()

    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    maximun = np.max(im)
    size = im.shape
    im = im/255.0
    im -= im.mean()
    del A
    layers = [2,15000,1]
    ft = np.fft.fftshift(np.fft.fft2(im))



  if tipo2 == "human":
    from skimage.color import rgb2gray
    im = Image.open(ej)
    # Convert the image to a Numpy array
    img_array = np.array(im)
    s = (150,300)
    im1 = Image.fromarray(img_array[:,:,0])
    resized1 = im1.resize(s, Image.LANCZOS)
    im2 = Image.fromarray(img_array[:,:,1])
    resized2 = im2.resize(s, Image.LANCZOS)
    im3 = Image.fromarray(img_array[:,:,2])
    resized3 = im3.resize(s, Image.LANCZOS)
    im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                  , np.array(resized3)[:,:,None]), axis=2), dtype = 'float32')

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

    A = nib.load(ej).get_fdata()

    im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
    im = resize(im, (50,45,45))
    maximun = np.max(im)
    im = im/255.0
    im -= im.mean()
    size = im.shape
    ft = np.fft.fftshift(np.fft.fftn(im))
   
    del A

  return ft

"""
def pdfBrain(z1, z2, T):
  return pdfBrainRhoMean

def pdf3d(z1, z2, z3, T):

  return pdf3dRhoMean(z1, z2, z3, T)


def pdfHuman(z1, z2,T):
  
  return pdfHumanRhoMean(z1, z2, T)
"""

def pdfBrain(z1, z2, T):



    file_path = "dataset/"
    with open(file_path +'T'+str(T)+'Mean-rhoBrain.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))

    Nx, Ny = rhotemp.shape
    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x, y, rhotemp)

    return r(z1, z2, grid=False)




def pdfHuman(z1, z2, T):

    file_path = "dataset/"
    with open('dataset/T'+str(T)+'Mean-rhoHuman.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
   
    Nx, Ny = rhotemp.shape
    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    r = RectBivariateSpline(x, y, rhotemp)

    return r(z1, z2, grid=False)



def pdf3d(z1, z2, z3,T):

    
    with open('dataset/T'+str(T)+'Mean-rho3d.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
    rhotemp = rhotemp.reshape((50,45,45))

    Nx, Ny, Nz = rhotemp.shape
    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    z = np.linspace(-int(Nz/4), int(Nz/4), Nz)
    r = RegularGridInterpolator((x, y, z), rhotemp)

    return r((z1, z2, z3))


def desginBrain(i, x, y, T):
    return np.exp(-i*2*pdfBrain(x, y, T))
def desginHuman(i,x, y, T):
    return np.exp(-i*2*pdfHuman(x, y, T))
def desgin3d(i, x, y, z, T):
    return np.exp(-i*2*pdf3d(x, y, z, T))




def deviationbrain(i, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = desginBrain(i, X, Y, T)


    h_array = f_array * g_array
    deviation = np.trapz(np.trapz((h_array), x, axis=0), y, axis=0)
    return deviation



def deviationhuman(i, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = desginHuman(i, X, Y, T)
    g_array = np.repeat(g_array[:,:,None], 3, axis=2)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(np.sqrt(h_array[:,:,0]**2 +h_array[:,:,1]**2+h_array[:,:,2]**2), x, axis=0), y, axis=0)
    return deviation




def deviation3d(i, x, y, z, T, f_array):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    g_array = desgin3d(i, X, Y, Z, T)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(np.trapz(h_array, x, axis=0), y, axis=0), z, axis=0)
    return deviation

def load_and_preprocess(tipo2, path, path2, slides):
  if tipo2 == "brain":
    target = load_function(tipo2, path2)
    size = target.shape
    ft = (np.abs(target)**2)
    Nx, Ny = ft.shape
    x = np.linspace(-int(Nx / 4), int(Nx / 4), Nx)
    y = np.linspace(-int(Ny / 4), int(Ny / 4), Ny)
    
    with open(f'{path}', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides, size[0], size[1]))
    
    for i in range(slides):
        u[i, :, :] = np.abs(np.fft.fftshift(np.fft.fft2(u[i, :, :])) - target)
    
    return target, ft, u, x, y, 1

  if tipo2 == "human":
    target = load_function(tipo2, path2)
    size = target.shape
    ft = (np.abs(target)**2)

    Nx, Ny, _ = ft.shape

    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)


    
    with open(f'{path}', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides,size[0],size[1],3))
    for i in range(slides):
      u[i,:,:,:] = np.abs(np.fft.fftshift(np.fft.fft2(u[i,:,:,:], axes=(0,1))) - target)

   
    return target, ft, u, x, y, 1

  if tipo2 == "3d":
    target = load_function(tipo2, path2)
    size = target.shape
    ft = (np.abs(target)**2)


    Nx, Ny, Nz = ft.shape


    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    z = np.linspace(-int(Nz/4), int(Nz/4), Nz)

    with open(f'{path}', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides,size[0],size[1], size[2]))
    for i in range(slides):
      u[i,:,:,:] = np.abs(np.fft.fftshift(np.fft.fftn(u[i,:,:,:])) - target)
    
    return target, ft, u, x, y, z


def compute_vectors(u, x, y, z, tipo2,slides, ft,times):
  #times = times[5:]
  if tipo2 == "brain":
    a = [np.trapz(np.trapz(u[i, :, :], x, axis=0), y, axis=0) for i in range(slides)]
    b = [deviationbrain(i, x, y, 30, ft) for i in times]
    return a, b
  if tipo2 == "human":
    a = [np.trapz(np.trapz(np.sqrt(u[i, :, :,0]**2+u[i, :, :,1]**2+u[i, :, :,2]**2), x, axis=0), y, axis=0) for i in range(slides)]
    b = [deviationhuman(i, x, y, 30, ft) for i in times]
    return a, b
  if tipo2 == "3d":
    a = [np.trapz(np.trapz(np.trapz((u[i,:,:,:]), x, axis=0), y, axis=0), z, axis=0) for i in range(slides)]
    b = [deviation3d(i, x, y, z, 80, ft) for i in times]
    return a, b




def objective(parameter, slides, tipo2, path, path2, times):
    target, ft, u, x, y ,z= load_and_preprocess(tipo2, path, path2, slides)
    times_scaled = times * parameter[0]
    a, b = compute_vectors(u, x, y, z, tipo2, slides, ft, times_scaled)
    a = a/a[0]
    b = b/b[0]
    
    return np.sum((np.array(a) - np.array(b))**2)


def optimize_parameter(slides, tipo2, path, path2, times, initial_guess):

    result = minimize(
        objective,
        initial_guess,
        args=(slides, tipo2, path, path2, times),
        bounds=[(0.000000001, 10000000)], 
        method='L-BFGS-B'
    )
    return result
"""

def optimize_parameter(slides, tipo2, path, path2, times, bounds):
    result = differential_evolution(
        objective,
        bounds,
        args=(slides, tipo2, path, path2, times),
        strategy='best1bin',  # Default strategy
        maxiter=1000,         # Maximum number of iterations
        tol=1e-6,             # Convergence tolerance
        seed=42               # For reproducibility
    )

    return result
  """



if __name__ == "__main__":

  for tipo2 in ["3d"]:
    
      if tipo2 == "human": 
        base_dirs = ["dataset/Humans"]
        all_files = []  # To store all matching file paths

        for base_dir in base_dirs:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        gz_path = os.path.join(root, file)
                        all_files.append(gz_path)
        values = [(0.05,20)]#, (0.005,20), (0.0005,100)]
          
      if tipo2 == "brain":
        base_dirs = ["dataset/ATLAS_2/"]
        all_files = []  # To store all matching file paths
        values = [(0.05,20)]#, (0.005,20), (0.0005,100)]
        for base_dir in base_dirs:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('T1w.nii.gz'):
                        gz_path = os.path.join(root, file)
                        all_files.append(gz_path)
                        
      if tipo2 == "3d":
        base_dirs = ["dataset/ATLAS_2/"]
        all_files = []  # To store all matching file paths
        values = [(0.005,200)]#, (0.0005,200), (0.00005,500)]
        for base_dir in base_dirs:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('T1w.nii.gz'):
                        gz_path = os.path.join(root, file)
                        all_files.append(gz_path)
                        
      
      
      lr, it = values[0]
      print ("LR", lr)
      lis = []
      
      
      
      for k, path2 in enumerate(all_files):
        if k==20:
          break
        slides = 21
                
        path = 'results/tEstimation/data-Dej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt'
          # Fixed sigma
        times = np.linspace(0, it, slides)
        initial_guess = [1]  # Initial guess for the parameter

        result = optimize_parameter(slides, tipo2, path, path2,  times, initial_guess)
        best_parameter = result.x[0]

        if k % 4 == 0:
          print(f"Optimized parameter: {best_parameter}")
        lis.append(best_parameter)
      lis = np.array(lis)  
      best = np.mean(lis)   
      print ("mean proporsional constant", best)
      # Recompute vectors a and b with the optimized parameter
      print ()
      """
      target, ft, u, x, y, z = load_and_preprocess(tipo2, path, path2, slides)
      for it in [5,10,20]:
        sigma_initial = 1
        if tipo2 == "brain":
          result = minimize(deviationbrain, sigma_initial, args=(best*it, x, y, best*it, ft), bounds=[(0.001, 10000)])
        if tipo2 == "human":
          result = minimize(deviationhuman, sigma_initial, args=(best*it, x, y, best*it, ft), bounds=[(0.001, 10000)])
        if tipo2 == "3d":
          result = minimize(deviation3d, sigma_initial, args=(best*it, x, y, z, best*it, ft), bounds=[(0.001, 10000)])
        best_sigma = result.x[0]
        print(f"For lr {lr}, Const {best}, it {it} the optimun sigma: {best_sigma}")
      """

  """
    tipo2 = "3d"
      
    if tipo2 == "human": 
      base_dir = "dataset/testhuman"#+tipo2 ../../../
      values = [(0.05,20), (0.005,20), (0.0005,100)] #
    if tipo2 == "brain":
      base_dir = "dataset/testbrain"#+tipo2
      values = [(0.05,20), (0.005,20), (0.0005,100)]
    if tipo2 == "3d":
      base_dir = "dataset/testbrain"#+tipo2
      values = [(0.005,100), (0.0005,300), (0.00005,500)]
    for lr, it in values:
      print ("LR", lr)
      lis = []
      for root, dirs, files in os.walk(base_dir):

        for k, file in enumerate(files):
          if file.endswith('T1w.nii.gz'):
            path2 = os.path.join(root, file)
      
            slides = 21
            
            path = 'results/tEstimation/D-ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt'
              # Fixed T
            times = np.linspace(0, it, slides)
 

            initial_guess = [1]
            result = optimize_parameter(slides, tipo2, path, path2,  times, initial_guess)
            best_parameter = result.x[0]

            if k%8==0:
              print(f"Optimized parameter: {best_parameter}")
            lis.append(best_parameter)


          if file.endswith('.jpg') or file.endswith('.png'):
            gz_path = os.path.join(root, file)
            path2 = os.path.join(root, file)
            path = 'results/tEstimation/D-ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt'
              # Fixed T
            
      
            slides = 21
            times = np.linspace(0, it, slides)
            
            initial_guess = [1]
            result = optimize_parameter(slides, tipo2, path, path2,  times, initial_guess)
            best_parameter = result.x[0]

            if k%8==0:
              print(f"Optimized parameter: {best_parameter}")
            lis.append(best_parameter)

      lis = np.array(lis)  
      best = np.mean(lis)   
      print ("mean proporsional constant", best)
      # Recompute vectors a and b with the optimized parameter
      print ()"""


    
    
    
      
    
    

      