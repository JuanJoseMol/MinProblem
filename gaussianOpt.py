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
    #plt.imshow(g_array)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz((h_array), x, axis=0), y, axis=0)
    return deviation

def gaussianhuman(x, y, T,sigma):
    return np.exp(-T*2*np.exp(-(x**2 + y**2)/ (2 * sigma**2))/(2*np.pi*sigma**2))

# Objective function to minimize
def deviationhuman(sigma, x, y, T, f_array):
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_array = gaussianhuman(X, Y, T, sigma)
    g_array = np.repeat(g_array[:,:,None], 3, axis=2)
    h_array = f_array * g_array
    deviation = np.trapz(np.trapz(np.sqrt(h_array[:,:,0]**2 +h_array[:,:,1]**2+h_array[:,:,2]**2), x, axis=0), y, axis=0)
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


if __name__ == "__main__":
    tipo2 = "3d"
    if tipo2 == "brain":
      target = load_function(tipo2, "1")
      f = (np.abs(target)**2)

      print (f.shape)
      Nx, Ny = f.shape
      # Interval and discretization
      x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
      y = np.linspace(-int(Ny/4), int(Ny/4), Ny)

      # Initial guess for sigma
      sigma_initial = 1

      # Minimize deviation
      result = minimize(deviationbrain, sigma_initial, args=(x, y, 40.929*1000, f), bounds=[(0.001, 10000)])
      best_sigma = result.x[0]
      print(f"Best sigma: {best_sigma}")
    

    if tipo2 == "human":
      target = load_function(tipo2, "1")
      f = (np.abs(target)**2)
      print (f.shape)
      Nx, Ny, _ = f.shape
      # Interval and discretization
      x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
      y = np.linspace(-int(Ny/4), int(Ny/4), Ny)

      # Initial guess for sigma
      sigma_initial = 1

      # Minimize deviation
      result = minimize(deviationhuman, sigma_initial, args=(x, y, 7.9*1000, f), bounds=[(0.001, 10000)])
      best_sigma = result.x[0]
      print(f"Best sigma: {best_sigma}")



    if tipo2 == "3d":
      target = load_function(tipo2, "1")
      f = (np.abs(target)**2)
      print (f.shape)
      Nx, Ny, Nz = f.shape
      # Interval and discretization
      x = np.linspace(-int(Nx/4), int(Nx/4), Nx)
      y = np.linspace(-int(Ny/4), int(Ny/4), Ny)
      z = np.linspace(-int(Nz/4), int(Ny/4), Nz)

      # Initial guess for sigma
      sigma_initial = 1

      # Minimize deviation
      result = minimize(deviation3d, sigma_initial, args=(x, y, z, 333.333*1000, f), bounds=[(0.001, 10000)])
      best_sigma = result.x[0]
      print(f"Best sigma: {best_sigma}")

    """
    ej = "2"
    dicresults = {}
    for tipo2 in ["brain", "human", "3d"]:
      target = load_function(tipo2, ej)
      g = (np.abs(target)**2)

      #print (np.min(g), np.max(g))
      for k, T in enumerate([0.5,0.25]+[i for i in range(1,33)]):
        
        if tipo2 == "3d":

            try:
              x_root = bisection_method(g[:,:,0], T, tipo2)
              print("if T==", T,":")
              print( "x_root =", x_root)
              rhotemp = np.where(g[:,:,0] >= x_root, np.log(g[:,:,0]/x_root)/(2*T), 0)
              tem = np.count_nonzero(rhotemp==0)
              temp = tem/np.size(g[:,:,0])*100
              print (temp)
            except ValueError as e:
              print(e)
              
        else:
            try:
                x_root = bisection_method(g, T, tipo2)
                print("if T==", T,":")
                print( "x_root =", x_root)
                rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
                tem = np.count_nonzero(rhotemp==0)
                temp = tem/np.size(g)*100
                print (temp)
            except ValueError as e:
                print(e)

        
        dicresults[T] = [T,x_root, temp]
      with open('speedres/fej'+ej+'Design'+tipo2+'.pickle', 'wb') as handle:
        pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)"""
    
    
      
    
    

      