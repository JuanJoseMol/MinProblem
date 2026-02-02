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
import scipy.ndimage as ndi
from loadRealistic import load_image, make_image_dataset
#import jax.numpy as np

def add_poisson_readout_noise(img, max_photons=30, readout_std=2.0):

    img = np.clip(img, 0.0, 1.0)
    lam = max_photons * img
    poisson_noise = np.random.poisson(lam)

    readout_noise = np.random.normal(loc=0.0, scale=readout_std, size=img.shape)
    noisy_counts = poisson_noise + readout_noise
    noisy_img = noisy_counts / max_photons

    return np.clip(noisy_img, 0.0, 1.0)



def downsample_antialiased(img, factor, sigma=None):
    if sigma is None:
        sigma = factor / 2
    img_blur = ndi.gaussian_filter(img, sigma=(sigma, sigma, 0))
    return img_blur[::factor, ::factor, :]

def collect_brain_files(base_dirs, suffix="T1w.nii.gz"):
    all_files = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(suffix):
                    all_files.append(os.path.join(root, file))
    return sorted(all_files)

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
    #plt.imshow(im)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()


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
  if tipo2 == "realistic":
    if ej == "noise":
        img = Image.open("../data/noiseRaro0649.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = np.array(img, dtype = 'float32')

    if ej == "super":
        img = Image.open("../data/mariposa0829.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = np.array(img, dtype = 'float32')

    if ej == "fitting":
        img = Image.open("../data/playa0823.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = np.array(img)/255.0

    if ej == "fitting2":
        img = Image.open("../data/loros23.png").convert("RGB") 
        img = np.array(img, dtype = 'float32')

    if ej == "fitting3":
        img = Image.open("../data/castle19.png").convert("RGB") 
        img = np.array(img, dtype = 'float32')

    if ej == "super2":
        img = Image.open("../data/pinguino0344.png").convert("RGB") 
        img = np.array(img)[70:-70,48:-48]
        img = downsample_antialiased(img, factor=2)
        img = np.array(img, dtype = 'float32')

    if ej == "super3":
        img = Image.open("../data/Coral0026.png").convert("RGB") 
        img = np.array(img)[48:-48,70:-70]
        img = downsample_antialiased(img, factor=2)
        img = np.array(img, dtype = 'float32')

    if ej == "noise2":
        img = Image.open("../data/limones0802.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = np.array(img, dtype = 'float32')
    if ej == "noise3":
        img = Image.open("../data/hongo0858.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = np.array(img, dtype = 'float32')

    img[:,:,0] = img[:,:,0]/255.0
    img[:,:,1] = img[:,:,1]/255.0
    img[:,:,2] = img[:,:,2]/255.0
    #img[:,:,0] -= img[:,:,0].mean()
    #img[:,:,1] -= img[:,:,1].mean()
    #img[:,:,2] -= img[:,:,2].mean()
    

    size = img.shape
    ft = np.fft.fftshift(np.fft.fft2(img, axes=(0,1)))

  if tipo2 == "3d":
    from skimage.color import rgb2gray
    #../../../
    import nibabel as nib
    from skimage.transform import resize

    A = nib.load(ej).get_fdata()
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



def gaussianbrain(x, y, T, sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2)/ (2 * sigma**2))/(2*np.pi*sigma**2))


def deviationbrain(sigma, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = gaussianbrain(X, Y, T,sigma)


    h_array = np.array(f_array) * g_array
    deviation = np.trapz(np.trapz((h_array), x, axis=0), y, axis=0)
    return deviation

def gaussianhuman(x, y, T,sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2)/ (2 * sigma**2))/(2*np.pi*sigma**2))


def deviationhuman(sigma, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = gaussianhuman(X, Y, T, sigma)
    g_array = np.repeat(g_array[:,:,None], 3, axis=2)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(np.sqrt(h_array[:,:,0]**2 +h_array[:,:,1]**2+h_array[:,:,2]**2), x, axis=0), y, axis=0)
    return deviation

def gaussian3d(x, y, z, T,sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2+ z**2)/ (2 * sigma**2))/((2*np.pi*sigma**2)*np.sqrt(2*np.pi*sigma**2)))


def deviation3d(sigma, x, y, z, T,f_array):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    g_array = gaussian3d(X, Y, Z, T, sigma)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(np.trapz(h_array, x, axis=0), y, axis=0), z, axis=0)
    return deviation

def load_and_preprocess(tipo2, path, path2, slides, sigma):
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


    sigma = 10
    
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


    sigma = 10
    with open(f'{path}', 'r') as f:
        u = np.loadtxt(f)
    u = u.reshape((slides,size[0],size[1], size[2]))
    for i in range(slides):
      u[i,:,:,:] = np.abs(np.fft.fftshift(np.fft.fftn(u[i,:,:,:])) - target)
    
    return target, ft, u, x, y, z

  if tipo2 == "realistic":
    target = load_function(tipo2, path2)
    size = target.shape
    ft = (np.abs(target)**2)

    Nx, Ny, _ = ft.shape

    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    
    u = np.load(f"realistic/short-{path}-lr0.1-sigmaW{sigma}.npy")
    u = u.reshape((101,size[0],size[1],3))
    #print (u.shape)
    for i in range(slides):
      u[i,:,:,:] = np.abs(np.fft.fftshift(np.fft.fft2(u[i,:,:,:], axes=(0,1))) - target)

   
    return target, ft, u, x, y, 1


def compute_vectors(u, x, y, z, tipo2,slides, sigma, ft, times):
  if tipo2 == "brain":
    a = [np.trapz(np.trapz(u[i, :, :], x, axis=0), y, axis=0) for i in range(slides)]
    b = [deviationbrain(sigma, x, y, T, ft) for T in times]
    return a, b
  if tipo2 == "human" or tipo2 == "realistic":
    a = [np.trapz(np.trapz(np.sqrt(u[i, :, :,0]**2+u[i, :, :,1]**2+u[i, :, :,2]**2), x, axis=0), y, axis=0) for i in range(slides)]
    b = [deviationhuman(sigma, x, y, T, ft) for T in times]
    return a, b
  if tipo2 == "3d":
    a = [np.trapz(np.trapz(np.trapz((u[i,:,:,:]), x, axis=0), y, axis=0), z, axis=0) for i in range(slides)]
    b = [deviation3d(sigma, x, y, z, T, ft) for T in times]
    return a, b


def objective(parameter, slides, tipo2, path, path2, sigma, times):
    target, ft, u, x, y ,z= load_and_preprocess(tipo2, path, path2, slides, sigma)
    times_scaled = times * parameter
    a, b = compute_vectors(u, x, y, z, tipo2, slides, sigma, ft, times_scaled)
    a = a/a[0]
    b = b/b[0]
    return np.sum((np.array(a) - np.array(b))**2)


def optimize_parameter(slides, tipo2, path, path2, sigma, times, initial_guess):
    result = minimize(
        objective,
        initial_guess,
        args=(slides, tipo2, path, path2, sigma, times),
        bounds=[(1, 100000)], 
        method='L-BFGS-B'
    )
    return result



if __name__ == "__main__":
    import time

    start_time = time.time()
    
    """
    for tipo2 in ["brain", "human", "3d"]:
    
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
        values = [(0.005,100)]#, (0.0005,200), (0.00005,500)]
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
        if k==1:
          break
        slides = 21
                
        if tipo2 == "brain":
          sigma = 0.75*90 *  np.pi
        if tipo2 == "human":
          sigma = 0.75*70 *  np.pi
        if tipo2 == "3d":
          sigma = 0.75*22.5* np.pi
        path = 'results/tEstimation/data-ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt'
          # Fixed sigma
        times = np.linspace(0, it, slides)
        initial_guess = [10]  # Initial guess for the parameter

        result = optimize_parameter(slides, tipo2, path, path2, sigma, times, initial_guess)
        best_parameter = result.x[0]

        if k % 4 == 0:
          print(f"Optimized parameter: {best_parameter}")
        lis.append(best_parameter)
      lis = np.array(lis)  
      best = np.mean(lis)   
      print ("mean proporsional constant", best)
      # Recompute vectors a and b with the optimized parameter
      print ()
      target, ft, u, x, y, z = load_and_preprocess(tipo2, path, path2, slides)
      for it in [10, 20, 30]:
        sigma_initial = 1
        if tipo2 == "brain":
          result = minimize(deviationbrain, sigma_initial, args=(x, y, best*it, ft), bounds=[(0.001, 10000)])
        if tipo2 == "human":
          result = minimize(deviationhuman, sigma_initial, args=(x, y, best*it, ft), bounds=[(0.001, 10000)])
        if tipo2 == "3d":
          result = minimize(deviation3d, sigma_initial, args=(x, y, z, best*it, ft), bounds=[(0.001, 10000)])
        best_sigma = result.x[0]
        print(f"For lr {lr}, Const {best}, it {it} the optimun sigma: {best_sigma}")
    
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.6f} seconds")
    """
    ####################################### sigma find realistic #######################################
    
    slides = 21
    it = 21
    lr = 0.1
              
    tipo2 = "realistic"
    path = "noise"
    path2 = path
    ej = path
    times = np.linspace(0, it, slides)
    initial_guess = [20]  # Initial guess for the parameter
    print (path)
    if ej == "fitting":
      sigma = 79 #optuna
    elif ej == "super":
      sigma = 42 #optuna
    elif ej == "noise":
      sigma = 39 #optuna
    if ej == "fitting2":
      sigma = 44
    if ej == "super2":
      sigma = 46
    if ej == "noise2":
      sigma = 28
    if ej == "fitting3":
      sigma =67
    if ej == "super3":
      sigma = 70
    if ej == "noise3":
      sigma = 28#/(2*np.pi)

    result = optimize_parameter(slides, tipo2, path, path2, sigma, times, initial_guess)
    best = np.array(result.x[0])
    print ("mean proporsional constant", best)

    
    
    #target, ft, u, x, y, z = load_and_preprocess(tipo2, path, path2, slides, sigma)

    img, _ = load_image(ej)
    #_, img = load_image(ej)
    img = np.array(img)
    #img = Image.open("../data/0563.png").convert("RGB") 
    #img = downsample_antialiased(img, factor=2)
    #img = np.array(img, dtype = 'float32')/255.0


    size = img.shape
    ft = np.fft.fftshift(np.fft.fft2(img, axes=(0,1)))
    ft = (np.abs(ft)**2)
    print (ft.shape)
    Nx, Ny, _ = ft.shape

    x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
    y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
    for it in [it,500,1000,2000,5000,10000,50000]:
      sigma_initial = 30
      result = minimize(deviationhuman, sigma_initial, args=(x, y, best*it, ft), bounds=[(0.001, 10000)])

      best_sigma = result.x[0]
      print(f"For lr {lr}, Const {best}, it {it} the optimun sigma: {best_sigma}")
    
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.6f} seconds")
    
    ##############################################################################
    """
    for tipo2 in ["brain", "human", "3d"]: #
        
      if tipo2 == "human": 
        base_dir = "dataset/testhuman"#+tipo2 ../../../
        values = [(0.05,20)]#, (0.005,20), (0.0005,100)]
      if tipo2 == "brain":
        base_dir = "dataset/testbrain"#+tipo2
        values = [(0.05,20)]#, (0.005,20), (0.0005,100)]
      if tipo2 == "3d":
        base_dir = "dataset/testbrain"#+tipo2
        values = [(0.005,100)]#, (0.0005,200), (0.00005,500)]
      for lr, it in values:
        print ("LR", lr)
        lis = []
        k = 0
        for root, dirs, files in os.walk(base_dir):

          for k, file in enumerate(files):
            if file.endswith('T1w.nii.gz'):
              path2 = os.path.join(root, file)

        
              slides = 21
              
              if tipo2 == "brain":
                sigma = 0.75*90 *  np.pi
              if tipo2 == "human":
                sigma = 0.75*70 *  np.pi
              if tipo2 == "3d":
                sigma = 0.75*22.5* np.pi
              path = 'results/tEstimation/ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt'
                # Fixed sigma
              times = np.linspace(0, it, slides)
              initial_guess = [10]  # Initial guess for the parameter

              result = optimize_parameter(slides, tipo2, path, path2, sigma, times, initial_guess)
              best_parameter = result.x[0]

              if k % 8 == 0:
                print(f"Optimized parameter: {best_parameter}")
              lis.append(best_parameter)



            if file.endswith('.jpg') or file.endswith('.png'):
              if k==20:
                break
              gz_path = os.path.join(root, file)
              path2 = os.path.join(root, file)


              slides = 21
              
              if tipo2 == "brain":
                sigma = 0.75*90 *  np.pi
              if tipo2 == "human":
                sigma = 0.75*70 *  np.pi
              if tipo2 == "3d":
                sigma = 0.75*22.5* np.pi
              path = 'results/tEstimation/data-ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt'
                # Fixed sigma
              times = np.linspace(0, it, slides)
              initial_guess = [10]  # Initial guess for the parameter

              result = optimize_parameter(slides, tipo2, path, path2, sigma, times, initial_guess)
              best_parameter = result.x[0]

              if k % 8 == 0:
                print(f"Optimized parameter: {best_parameter}")
              lis.append(best_parameter)


        lis = np.array(lis)  
        best = np.mean(lis)   
        print ("mean proporsional constant", best)
        # Recompute vectors a and b with the optimized parameter
        print ()
    """
    ####################################### sigmas viejos ##########################################  

    """
    for tipo2 in ["brain", "human","3d"]:
      #tipo2 = "3d"
      if tipo2 == "brain":
        ej = "../../../dataset/testbrain/sub-r040s085_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
        #ej = "../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
      if tipo2 == "human":
        ej = "../../../dataset/testhuman/1 (2949).jpg"
        #ej ="../../../dataset/1 (107).jpg"
      if tipo2 == "3d":
        ej = "../../../dataset/testbrain/sub-r040s085_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"
        #ej = "../../../dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"#
      
      
      g = load_function(tipo2, ej)
      ft = (np.abs(g)**2)
      


      sigma_initial = 1
      print (tipo2)

      if tipo2 == "brain":

        for lr, best, it in [(0.05, 9605, 20), (0.005, 1152, 100), (0.0005, 118, 100)]:
          Nx, Ny = ft.shape
          x = np.linspace(-int(Nx / 4), int(Nx / 4), Nx)
          y = np.linspace(-int(Ny / 4), int(Ny / 4), Ny)
          result = minimize(deviationbrain, sigma_initial, args=(x, y, best*it, ft), bounds=[(0.001, 10000)])
          best_sigma = result.x[0]
          print(f"For lr {lr}, Const {best}, T={lr*best}, it {it} the optimun sigma: {best_sigma}")
      if tipo2 == "human":
        for lr, best, it in [(0.05, 2286, 20), (0.005, 240, 100), (0.0005, 24.84, 1)]:
        
          Nx, Ny, _ = ft.shape
          x = np.linspace(-int(Nx / 4), int(Nx / 4), Nx)
          y = np.linspace(-int(Ny / 4), int(Ny / 4), Ny)
          result = minimize(deviationhuman, sigma_initial, args=(x, y, best*it, ft), bounds=[(0.001, 10000)])
          best_sigma = result.x[0]
          print(f"For lr {lr}, Const {best}, T={lr*best}, it {it} the optimun sigma: {best_sigma}")
      if tipo2 == "3d":
        for lr, best, it in [(0.005, 6552, 20), (0.0005, 707, 100), (0.00005, 72, 1)]:
        
          Nx, Ny, Nz = ft.shape
          x = np.linspace(-int(Nx / 4), int(Nx / 4), Nx)
          y = np.linspace(-int(Ny / 4), int(Ny / 4), Ny)
          z = np.linspace(-int(Nz / 4), int(Ny / 4), Nz)
          result = minimize(deviation3d, sigma_initial, args=(x, y, z, best*it, ft), bounds=[(0.001, 10000)])
          best_sigma = result.x[0]
          print(f"For lr {lr}, Const {best}, T={lr*best}, it {it} the optimun sigma: {best_sigma}")
    
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.6f} seconds")
    """
    #################################################################################
    """
    if ej == "noise":
        img = Image.open("../data/noiseRaro0649.png").convert("RGB") 
        img = jnp.array(img)/255.0
        img = img ** 2.2
        img = add_poisson_readout_noise(img,max_photons=30,readout_std=2.0)
        img = img ** (1/2.2)
        img = img*255.0

    if ej == "super":
        img = Image.open("../data/mariposa0829.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = jnp.array(img)

    if ej == "fitting":
        img = np.load('../data/data_div2k.npz')
        img = img['test_data'][2]

    for lr, best, it in [(0.05, 2286, 20), (0.005, 240, 100), (0.0005, 24.84, 1)]:
        
      Nx, Ny, _ = ft.shape
      x = np.linspace(-int(Nx / 4), int(Nx / 4), Nx)
      y = np.linspace(-int(Ny / 4), int(Ny / 4), Ny)
      result = minimize(deviationhuman, sigma_initial, args=(x, y, best*it, ft), bounds=[(0.001, 10000)])
      best_sigma = result.x[0]
      print(f"For lr {lr}, Const {best}, T={lr*best}, it {it} the optimun sigma: {best_sigma}")

    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.6f} seconds")
    """
    ######################################## sigmas y errores #########################################  
    """
    sigma_initial = 1.0
    N_values = [1,2,4,8,16,32,64,128]
    reps = 10
    rng = np.random.default_rng(0)
    results = {}
    sigma_refs = {}

    for tipo2 in ["brain", "human", "3d"]:

      print("\n========================")
      print("Processing:", tipo2)
      print("========================")
      if tipo2 == "brain" or tipo2 == "3d":
        base_dirs = ["dataset/ATLAS_2/"]
        files = collect_brain_files(base_dirs)#[:450]
        print (len(files))
      if tipo2 == "human":
        data_folder = "dataset/Humans" 
        files = sorted(os.listdir(data_folder))[:450]

      sigma_list = []

      for path in files:
          if tipo2 == "brain" or tipo2 == "3d":
            g = load_function(tipo2, path)
          if tipo2 == "human":
            g = load_function(tipo2, os.path.join(data_folder, path))
          ft = np.abs(g)**2

          if tipo2 == "brain":
              Nx, Ny = ft.shape
              x = np.linspace(-Nx//4, Nx//4, Nx)
              y = np.linspace(-Ny//4, Ny//4, Ny)
              T = 9605 * 20  # best * it

              res = minimize(
                  deviationbrain,
                  sigma_initial,
                  args=(x, y, T, ft),
                  bounds=[(1e-3, 1e4)]
              )

          elif tipo2 == "human":
              Nx, Ny, _ = ft.shape
              x = np.linspace(-Nx//4, Nx//4, Nx)
              y = np.linspace(-Ny//4, Ny//4, Ny)
              T = 2286 * 20

              res = minimize(
                  deviationhuman,
                  sigma_initial,
                  args=(x, y, T, ft),
                  bounds=[(1e-3, 1e4)]
              )

          elif tipo2 == "3d":
              Nx, Ny, Nz = ft.shape
              x = np.linspace(-Nx//4, Nx//4, Nx)
              y = np.linspace(-Ny//4, Ny//4, Ny)
              z = np.linspace(-Nz//4, Nz//4, Nz)
              T = 6552 * 20

              res = minimize(
                  deviation3d,
                  sigma_initial,
                  args=(x, y, z, T, ft),
                  bounds=[(1e-3, 1e4)]
              )

          sigma_list.append(res.x[0])

      sigma_list = np.array(sigma_list)
      sigma_ref = sigma_list.mean()
      

      print(f"Reference sigma ({tipo2}, 450): {sigma_ref:.4f}")

      results[tipo2] = {}
      sigma_refs[tipo2] = {}

      for N in N_values:
          errors = []
          for _ in range(reps):
              N_eff = min(N, len(sigma_list))
              idx = rng.choice(len(sigma_list), size=N_eff, replace=False)
              sigma_hat = sigma_list[idx].mean()
              errors.append(abs(sigma_hat - sigma_ref))

          errors = np.array(errors)
          results[tipo2][N] = (errors.mean(), errors.std())
          sigma_refs[tipo2][N] = sigma_hat

    output_file = "results2/sigma_estimation_error.tex"

    with open(output_file, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Finite-sample estimation error for the Gaussian design parameter "
            "$\\sigma_w$. Each entry reports "
            "$\\mathbb{E}[|\\hat\\sigma_N - \\bar\\sigma_{450}|] \\pm$ std over 10 random draws. "
            "The reference values $\\bar\\sigma_{450}$ are shown explicitly.}\n"
        )
        f.write("\\label{tab:sigma-estimation-error}\n")
        f.write("\\begin{tabular}{c c c c c c c}\n")
        f.write("\\toprule\n")
        f.write(
            "$N$ & "
            "$\\bar\\sigma_{\\text{brain}}$ & Error & "
            "$\\bar\\sigma_{\\text{human}}$ & Error & "
            "$\\bar\\sigma_{\\text{3D}}$ & Error \\\\\n"
        )
        f.write("\\midrule\n")

        for N in N_values:
            b_mean, b_std = results["brain"][N]
            h_mean, h_std = results["human"][N]
            d_mean, d_std = results["3d"][N]

            f.write(
                f"{N} & "
                f"${sigma_refs['brain'][N]:.3f}$ & ${b_mean:.3f} \\pm {b_std:.4f}$ & "
                f"${sigma_refs['human'][N]:.3f}$ & ${h_mean:.3f} \\pm {h_std:.4f}$ & "
                f"${sigma_refs['3d'][N]:.3f}$ & ${d_mean:.3f} \\pm {d_std:.4f}$ \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to {output_file}")
    """    

######################################## sigmas estimation #########################################  
    """    
    sigma_initial = 1.0
    N_values = [1,2,4,8,16,32,64,128,450]
    reps = 10
    rng = np.random.default_rng(0)
    results = {}
    sigma_refs = {}

    for tipo2 in ["brain", "human", "3d"]:

      print("\n========================")
      print("Processing:", tipo2)
      print("========================")
      if tipo2 == "brain" or tipo2 == "3d":
        base_dirs = ["dataset/ATLAS_2/"]
        files = collect_brain_files(base_dirs)#[:450]
        print (len(files))
      if tipo2 == "human":
        data_folder = "dataset/Humans" 
        files = sorted(os.listdir(data_folder))[:450]

      sigma_list = []

      for path in files:
          if tipo2 == "brain" or tipo2 == "3d":
            g = load_function(tipo2, path)
          if tipo2 == "human":
            g = load_function(tipo2, os.path.join(data_folder, path))
          ft = np.abs(g)**2

          if tipo2 == "brain":
              Nx, Ny = ft.shape
              x = np.linspace(-Nx//4, Nx//4, Nx)
              y = np.linspace(-Ny//4, Ny//4, Ny)
              T = 9605 * 20  # best * it

              res = minimize(
                  deviationbrain,
                  sigma_initial,
                  args=(x, y, T, ft),
                  bounds=[(1e-3, 1e4)]
              )

          elif tipo2 == "human":
              Nx, Ny, _ = ft.shape
              x = np.linspace(-Nx//4, Nx//4, Nx)
              y = np.linspace(-Ny//4, Ny//4, Ny)
              T = 2286 * 20

              res = minimize(
                  deviationhuman,
                  sigma_initial,
                  args=(x, y, T, ft),
                  bounds=[(1e-3, 1e4)]
              )

          elif tipo2 == "3d":
              Nx, Ny, Nz = ft.shape
              x = np.linspace(-Nx//4, Nx//4, Nx)
              y = np.linspace(-Ny//4, Ny//4, Ny)
              z = np.linspace(-Nz//4, Nz//4, Nz)
              T = 6552 * 20

              res = minimize(
                  deviation3d,
                  sigma_initial,
                  args=(x, y, z, T, ft),
                  bounds=[(1e-3, 1e4)]
              )

          sigma_list.append(res.x[0])

      sigma_list = np.array(sigma_list)
      sigma_ref = sigma_list.mean()
      

      print(f"Reference sigma ({tipo2}, 450): {sigma_ref:.4f}")

      results[tipo2] = {}
      sigma_refs[tipo2] = sigma_ref

      for N in N_values:
        sigmas_hat = []
        for _ in range(reps):
            N_eff = min(N, len(sigma_list))
            idx = rng.choice(len(sigma_list), size=N_eff, replace=False)
            sigmas_hat.append(sigma_list[idx].mean())

        sigmas_hat = np.array(sigmas_hat)
        results[tipo2][N] = (sigmas_hat.mean(), sigmas_hat.std())


    output_file = "results2/sigma_estimation.tex"

    with open(output_file, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Finite-sample estimates of the optimal Gaussian design parameter "
            "$\\sigma_w$. Each entry reports "
            "$\\mathbb{E}[\\hat\\sigma_N] \\pm \\mathrm{std}$ over 10 random draws.}\n"
        )
        f.write("\\label{tab:sigma-estimation}\n")
        f.write("\\begin{tabular}{c c c c}\n")
        f.write("\\toprule\n")
        f.write("$N$ & Brain & Human & 3D \\\\\n")
        f.write("\\midrule\n")

        for N in N_values:
            b_mean, b_std = results["brain"][N]
            h_mean, h_std = results["human"][N]
            d_mean, d_std = results["3d"][N]
            r_brain = sigma_refs["brain"]
            r_3d = sigma_refs["3d"]
            r_human = sigma_refs["human"]

            f.write(
                f"{N} & "
                f"${b_mean:.3f} \\pm {b_std:.3f}$ & "
                f"${h_mean:.3f} \\pm {h_std:.3f}$ & "
                f"${d_mean:.3f} \\pm {d_std:.3f}$ \\\\\n"
            )

        f.write(
            "$\\bar\\sigma_{450}$ & "
            f"{r_brain:.3f} & "
            f"{r_human:.3f} & "
            f"{r_3d:.3f} \\\\\n"
            )
        

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX table saved to {output_file}")
    """
    
    
    
      
    
    

      