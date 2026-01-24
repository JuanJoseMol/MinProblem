import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from PIL import Image
import nibabel as nib
from skimage.color import rgb2gray


def load_function(tipo2):
  #N_s = 151
  if tipo2 == "1d":
    def func(x):
        freq = 2.1
        sin = np.sin(2*np.pi*freq*x)
        return np.sign(sin)*(np.abs(sin) > 0.5)

    n = 240
    y = np.linspace(-1,1,n)
    ft  = np.fft.fftshift(np.fft.fft(func(y)))

  if tipo2 == "astro":

    from skimage import data
    from skimage.color import rgb2gray


    #ej = data.camera()
    ej2 = data.astronaut()
    ej2 = rgb2gray(ej2)#[20:220,120:320]
    N_s = 200
    size = (N_s, N_s)
    ima = Image.fromarray(ej2)
    resized = ima.resize(size, Image.LANCZOS)
    im = np.array(resized)
    im = im/255.0
    im -= im.mean()
    ft = np.fft.fftshift(np.fft.fft2(im))


  if tipo2 == "numbers":
    from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
    (traind, _), (_, _) = mnist.load_data()
    traind = traind.astype('float32') / 255.
    im = traind[100,:,:]
    im -= im.mean()
    N_s = im.shape[0]
    size = (N_s, N_s)
    ft = np.fft.fftshift(np.fft.fft2(im))
    """
    plt.imshow(im)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    """

  
  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    A = nib.load("../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
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
    im = Image.open('../../../dataset/1 (107).jpg')
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
    A = nib.load("../../../dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
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
  if tipo2 == "Mbrain":
    #../../../
    with open('../../../dataset/Mean-197-233-189-brain.txt', 'r') as f:
        im = np.array(np.loadtxt(f))
    im = im.reshape((197,233,189))
    im = im[7:187,16:216,120]
    maximun = np.max(im)
    size = im.shape
    print (size)
    im = im/255.0
    im -= im.mean()
    layers = [2,15000,1]
    ft = np.fft.fftshift(np.fft.fft2(im))

  if tipo2 == "M3d":
    #../../../
    import nibabel as nib
    from skimage.transform import resize
    with open('../../../dataset/Mean-197-233-189-brain.txt', 'r') as f:
        A = np.array(np.loadtxt(f))
    A = A.reshape((197,233,189))
    im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
    im = resize(im, (50,45,45))
    maximun = np.max(im)
    im = im/255.0
    im -= im.mean()
    size = im.shape
    ft = np.fft.fftshift(np.fft.fftn(im))
    del A
    layers = [3,10000,1]
    print (size)
    """
    plt.imshow(im[:,:,5])
    plt.show()
    """
    

  if tipo2 == "Mhuman":
    from skimage.transform import resize
    #../../../
    with open('../../../dataset/Mean-300-150-human.txt', 'r') as f:
        im = np.array(np.loadtxt(f))
    im = im.reshape((300,150,3))
    im = np.array(im)
    print (im.shape)
    #img = Image.open('./gdrive/MyDrive/Colab Notebooks/Mean-500-333-human.txt')
    # Convert the image to a Numpy array
    #im = np.array(im)/255.

    #im -= im.mean()
    im[:,:,0] = im[:,:,0]/255.0
    im[:,:,1] = im[:,:,1]/255.0
    im[:,:,2] = im[:,:,2]/255.0
    im[:,:,0] -= im[:,:,0].mean()
    im[:,:,1] -= im[:,:,1].mean()
    im[:,:,2] -= im[:,:,2].mean()
    
    size = im.shape
    ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
    print (im.shape)
    layers = [2,15000,3]
    """
    plt.imshow(im[:,:,0])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()"""
    

  if tipo2 == "Mnumbers":
    from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
    (traind, _), (_, _) = mnist.load_data()
    traind = traind.astype('float32') / 255.
    im = np.mean(traind, axis=0)
    #im -= im.mean()
    im -= im.mean()
    N_s = im.shape[0]
    size = (N_s, N_s)
    ft = np.fft.fftshift(np.fft.fft2(im))
    """
    plt.imshow(im)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    """
    ft = np.fft.fftshift(np.fft.fft2(im))

  if tipo2 == "brain-mediod":
    #../../../
    import nibabel as nib
    A = nib.load("../../../dataset/mediod.nii.gz").get_fdata()
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

  if tipo2 == "human-mediod":
    #../../../
    im = Image.open('../../../dataset/mediodhuman.jpg')
    # Convert the image to a Numpy array
    im = np.array(im)
    s = (150,250)
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

  if tipo2 == "3d-mediod":
    #../../../
    import nibabel as nib
    from skimage.transform import resize
    A = nib.load("../../../dataset/mediod.nii.gz").get_fdata()
    im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
    im = resize(im, (50,45,45))
    maximun = np.max(im)
    im = im/255.0
    im -= im.mean()
    size = im.shape
    ft = np.fft.fftshift(np.fft.fftn(im))
    #plt.imshow(im)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()


    
  return ft

# Define the target function to be zeroed in the Bisection method
def target_function(x, g_array, T, tipo2):
    """
    Approximates the integral for a given x using a sum, but only sums elements of g_array greater than x.
    
    Parameters:
    - x: variable in the bisection method
    - g_array: array of 120 points representing the function values
    - T: fixed positive constant
    
    Returns:
    - integral_value: the approximate value of the integral over [0, 60], summing only if g_array[i] > x
    """
    if tipo2 == "1d":
        area = 2
    elif tipo2 == "3d" or tipo2 == "M3d"  or tipo2 == "3d-mediod":
        area = 8
    else:
        area = 4
    
    # Compute the integrand values for the filtered g_array
    integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
    
    # Approximate the integral as a Riemann sum
    integral_value = np.sum(integrand_values)*area/np.size(g)
    
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





# Example usage
if __name__ == "__main__":
    # Example g_array with 120 points
    #tipo2 = "1d"
    #tipo2 = "brain"
    #tipo2 = "human"
    #tipo2 = "numbers"
    #tipo2 = "3d"
    #tipo2 = "Mbrain"
    #tipo2 = "Mhuman"
    #tipo2 = "Mnumbers"
    #tipo2 = "M3d"
    #tipo2 = "brain-mediod"
    #tipo2 = "human-mediod"
    tipo2 = "3d-mediod"
    target = load_function(tipo2)
    g = (np.abs(target)**2)

    #print (np.min(g), np.max(g))
    for T in [5]:
    #print ("tiempo", T)
    #print (g.shape)
    # Call the bisection method to find the root
      if tipo2 == "human" or tipo2 == "Mhuman" or tipo2=="human-mediod":

          try:
            x_root = bisection_method(g[:,:,0], T, tipo2)
            print("if T==", T,":")
            print( "x_root =", x_root)
          except ValueError as e:
            print(e)
          """
          try:
              for i in range(3):
                  x_root = bisection_method(g[:,:,i], T, tipo2)
                  print(f"Root found {i}: x = {x_root}")
          except ValueError as e:
              print(e)
          """
      else:
          try:
              x_root = bisection_method(g, T, tipo2)
              print("if T==", T,":")
              print( "x_root =", x_root)
          except ValueError as e:
              print(e)
    #print (np.size(g), np.count_nonzero(np.abs(g)**2<x_root), np.count_nonzero(np.abs(g)**2<x_root)*100/np.size(g))
    
    if tipo2 =="1d":
        x = np.linspace(-1,1,240)
        #vmin, vmax =np.min(np.abs(g)),np.max(np.abs(g))
        rho = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
        #rho = rho/(integrate.simpson(rho, x=x))
        I = integrate.simpson(rho, x=x)
        print (I)
    elif tipo2 == "3d" or tipo2 == "M3d" or tipo2 == "3d-mediod":
        rho = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
        Nx = np.linspace(-1,1,target.shape[0])
        Ny = np.linspace(-1,1,target.shape[1])
        Nz = np.linspace(-1,1,target.shape[2])
        print (np.trapz(np.trapz(np.trapz(rho, Nz), Ny),Nx))
    elif tipo2 == "human" or tipo2 == "Mhuman" or tipo2=="human-mediod":
        rho = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
        Nx = np.linspace(-1,1,target.shape[0])
        Ny = np.linspace(-1,1,target.shape[1])
        print (np.trapz(np.trapz(rho[:,:,0], Ny), Nx))
    else:
        rho = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
        Nx = np.linspace(-1,1,target.shape[0])
        Ny = np.linspace(-1,1,target.shape[1])
        print (np.trapz(np.trapz(rho, Ny), Nx))
    """
    r = interp1d(x, rho)
    


    plt.plot(x,g)
    #plt.plot(x,rho)
    plt.plot(x,r(x))
    plt.ylim(vmin,vmax)
    plt.yscale('log')
    plt.show()"""