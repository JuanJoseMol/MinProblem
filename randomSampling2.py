import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from PIL import Image

def pdfbasic(x, sigma_a, s=0):
    if 0 <= x <= 1:
        return (a[0] * x**9 + a[1] * x**8 + a[2] * x**7+ a[3] * x**6 + a[4] * x**5 + a[5]* x**4
                + a[6] * x**3 + a[7] * x**2 + a[8] * x + a[9])
    elif -1 <= x < 0:
        return (-a[0] * x**9 + a[1] * x**8 - a[2] * x**7+ a[3] * x**6 - a[4] * x**5 + a[5]* x**4
                - a[6] * x**3 + a[7] * x**2 - a[8] * x + a[9])
    else:
        return 0

# T = 0.05 low=0.4
b = [-3.582844776231999, 5.567490490466377, 3.3617723802778663, -8.398791800202753, -3.570662962831477, 
    25.612236962416674, -36.56689804536846, 24.567712356213715, -7.739445771459263, 1.3141520903081738]

def pdfbasicfull(x, sigma_a, s=0):
    if 0 <= x <= 1:
        return (b[0] * x**9 + b[1] * x**8 + b[2] * x**7+ b[3] * x**6 + b[4] * x**5 + b[5]* x**4
                + b[6] * x**3 + b[7] * x**2 + b[8] * x + b[9])
    elif -1 <= x < 0:
        return (-b[0] * x**9 + b[1] * x**8 - b[2] * x**7+ b[3] * x**6 - b[4] * x**5 + b[5]* x**4
                - b[6] * x**3 + b[7] * x**2 - b[8] * x + b[9])
    else:
        return 0


R=0.1634992499425288 
def pdfExp(x, sigma_a, s=0):
    if -1 <= x <= 1:
        return np.exp(-x**2 / (2))/0.855624367074333
    else:
        return 0

def pdfPDE(x, sigma_a, s=0):
    """
    with open('optRes/scale500NewMeanrhoestimation-T+500-dt0.5-w376.99111843077515-a'+str(sigma_a.round(3))+'.txt', 'r') as f:
        d = np.array(np.loadtxt(f))
    
    data = np.concatenate((d[-2::-1], d))
    y = np.linspace(-1, 1, len(data))
    if s == 0:
        f_interp = interp1d(y, data, kind='linear', fill_value="extrapolate")
        return f_interp(x)/quad(f_interp, -1, 1)[0]
    else:
        smoothed_data = gaussian_filter1d(data, sigma=s)
        f_interp = interp1d(y, smoothed_data, kind='linear', fill_value="extrapolate")
        return f_interp(x)/quad(f_interp, -1, 1)[0]
    """
    with open('optRes/1dFuncRho-T+500-dt0.5-a0.0316-k50.txt', 'r') as f:
        d = np.array(np.loadtxt(f))
    rho = np.concatenate((d[-1::-1], d[1:]))
    y = np.linspace(-1,1,239)
    r = interp1d(y, rho)

    return r(x)


def pdfPDEhuman(z1, z2, sigma_a=0, s=0):

    
    with open('optRes/Final-MhumanRho-Ns125-T+10-dt0.1-a0.0115k15.txt', 'r') as f:
        data = np.array(np.loadtxt(f)).T
    Nx, Ny = (250,166)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, data)

    return r(z1, z2, grid=False)

def pdfPDEnumbers(z1, z2, sigma_a=0, s=0):

    with open('optRes/Final-MnumbersRho-Ns14-T+20-dt0.2-a0.02k15.txt', 'r') as f:
        data = np.array(np.loadtxt(f)).T
    Nx, Ny = 28, 28

    
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, data)

    return r(z1, z2, grid=False)


def pdfPDEbrain(z1, z2, sigma_a=0, s=0):

    with open('optRes/Final-MbrainRho-Ns100-T+20-dt0.2-a0.0115k15.txt', 'r') as f:
        data = np.array(np.loadtxt(f)).T
    Nx, Ny = 200, 180

    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, data)

    return r(z1, z2, grid=False)



def pdfPDE3d(z1, z2, z3, sigma_a=0, s=0):
    with open('optRes/Final-M3dRho-Ns25-T+20-dt0.2-a0.0115k10.txt', 'r') as f:
        da = np.array(np.loadtxt(f))
    
    Nx, Ny, Nz = 50, 45, 45

    Nsx, Nsy, Nsz = (int(Nx/2), int(Ny), int(Nz))
    da = da.reshape((Nsx, Nsy, Nsz))
    da = da.T

    da1 = np.flip(da, axis=0)
    da2 = np.flip(da1, axis=1)
    da3 = np.flip(da2, axis=2)
    data = np.concatenate((da3,da), axis=2)
    data = data.T

    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), data)

    return r((z1, z2, z3))


def pdfCons(z, sigma_w, s=0):

    x_root = 0.02859143114175198
    T= 5 #1
    def func(x):
        freq = 2.1
        sin = np.sin(2*np.pi*freq*x)
        return np.sign(sin)*(np.abs(sin) > 0.5)

    n = 240
    
    y = np.linspace(-1,1,n)
    g  = np.abs(np.fft.fft(func(y))[:120])**2 + 0.0000001

    x = np.linspace(-1,1,239)
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    rho = np.concatenate((rhotemp[-1::-1], rhotemp[1:]))
    r = interp1d(x, rho)

    return r(z)

def pdfBrain(z1, z2, s=0):

    x_root = 171.3618636231115
    T=5

    file_path = "dataset/"
    with open('dataset/Mean-197-233-189-brain.txt', 'r') as f:
        A = np.array(np.loadtxt(f))
    A = A.reshape((197,233,189))
    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    size = im.shape
    #print (size)
    im -= im.mean()
    Nx, Ny = im.shape
    #im = im[:,:,None]
    del A
    ft = np.fft.fftshift(np.fft.fft2(im))
    g  = np.abs(ft)**2 + 0.0000001
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def pdfNumbers(z1, z2, s=0):

    x_root = 7.231126725708067e-08
    T=20
    
    from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
    (traind, _), (_, _) = mnist.load_data()
    traind = traind.astype('float32') / 255.
    im = np.mean(traind, axis=0)
    #im -= im.mean()
    Nx, Ny = im.shape
    del traind
    ft = np.fft.fftshift(np.fft.fft2(im))
    g  = np.abs(ft)**2 + 0.0000001
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def pdfHuman(z1, z2, s=0):

    x_root = 1.2570860191397374e-08
    T=10

    from skimage.transform import resize
    #../../../
    with open('dataset/Mean-500-333-human.txt', 'r') as f:
        im = np.array(np.loadtxt(f))
    im = im.reshape((500,333,3))
    im = im[:,1:,:]
    im = resize(im, (250,166,3))
    im = im/255.0
    #im -= im.mean()
    size = im.shape
    Nx, Ny = im.shape[0:2]
    #im = im[:,:,None]
    #print (im.shape)

    ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
    g  = np.abs(ft)**2 + 0.0000001
    
    
    rhotemp = np.where(g[:,:,0] >= x_root, np.log(g[:,:,0]/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def pdf3d(z1, z2, z3, s=0):

    x_root = 0.09979413425308423
    T=20


    from skimage.color import rgb2gray
    #../../../
    from skimage.transform import resize
    with open('dataset/Mean-197-233-189-brain.txt', 'r') as f:
        A = np.array(np.loadtxt(f))
    A = A.reshape((197,233,189))
    #im = np.swapaxes(A[5:189,16:216,100:106], 0, 1)
    #im = resize(im, (50,46,10))
    #im =  np.swapaxes(A[5:191,14:218,98:108], 0, 1)
    #im = resize(im, (68,62,10))
    im =  np.swapaxes(A[7:187,16:216,98:108], 0, 1)
    im = resize(im, (100,90,10))
    im -= im.mean()
    Nx, Ny, Nz = im.shape
    
    #im = im[:,:,None]
    del A
    ft = np.fft.fftshift(np.fft.fftn(im))
    g  = np.abs(ft)**2 + 0.0000001
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

    return r((z1, z2, z3))




def sample_from_pdf_rejection(n_samples, func, sigma_a, s=0, batch_size=1000):
    x_range = (-1, 1)
    
    # Precompute max_pdf by sampling in the range
    x_samples = np.linspace(x_range[0], x_range[1], 1000)
    max_pdf = max(func(x, sigma_a, s=0) for x in x_samples)
    
    samples = np.empty(n_samples)  # Pre-allocate array for samples
    count = 0
    
    while count < n_samples:
        # Generate batch of random samples
        x_batch = np.random.uniform(x_range[0], x_range[1], batch_size)
        y_batch = np.random.uniform(0, max_pdf, batch_size)
        
        # Evaluate func for the entire batch
        pdf_batch = np.array([func(x, sigma_a, s=0) for x in x_batch])
        
        # Accept/reject samples
        accepted = x_batch[y_batch <= pdf_batch]
        n_accepted = len(accepted)
        
        # Add the accepted samples to the result
        if count + n_accepted >= n_samples:
            samples[count:n_samples] = accepted[:n_samples - count]
            count = n_samples
        else:
            samples[count:count + n_accepted] = accepted
            count += n_accepted
    
    return samples

def sample2d(n_samples, func, sigma_a, s=0, batch_size=1000):
    sampled_x_values = []
    sampled_y_values = []

    while len(sampled_x_values) < n_samples and len(sampled_y_values)< n_samples:

        samples_x = jax.random.uniform(jax.random.PRNGKey(0), shape=(40000,), minval=-1, maxval=1)
        samples_y = jax.random.uniform(jax.random.PRNGKey(1), shape=(40000,), minval=-1, maxval=1)

        pdf_values_at_samples = func(samples_x, samples_y)
        random_values = jax.random.uniform(jax.random.PRNGKey(2), shape=(40000,)) * jnp.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples]),axis=0)
    
    return samples

def sample3d(n_samples, func, sigma_a, s=0, batch_size=1000):
    sampled_x_values = []
    sampled_y_values = []
    sampled_z_values = []

    while len(sampled_x_values)< n_samples and len(sampled_y_values)< n_samples and len(sampled_z_values)< n_samples:

        samples_x = jax.random.uniform(jax.random.PRNGKey(0), shape=(40000,), minval=-1, maxval=1)
        samples_y = jax.random.uniform(jax.random.PRNGKey(1), shape=(40000,), minval=-1, maxval=1)
        samples_z = jax.random.uniform(jax.random.PRNGKey(2), shape=(40000,), minval=-1, maxval=1)

        pdf_values_at_samples = func(samples_x, samples_y, samples_z)
        random_values = jax.random.uniform(jax.random.PRNGKey(3), shape=(40000,)) * jnp.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
        sampled_z_values.extend(samples_z[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples],
                            np.array(sampled_z_values)[None,:n_samples]),axis=0)
    
    return samples


if __name__ == "__main__":
    # Generate samples
    """
    n = 1000
    s = 10
    sigma_a = np.sqrt(2/2000)
    samples1 = sample_from_pdf_rejection(n, pdfCons,sigma_a, s=s)
    #samples2 = sample_from_pdf_rejection(n, pdf2,sigma_a, s=0)

    # Optionally, plot the results
    plt.hist(samples1, bins=50, density=True, alpha=0.6)
    #plt.hist(samples2, bins=50, density=True, alpha=0.6)

    # Overlay the original PDF
    x = np.linspace(-1, 1, n)
    plt.plot(x, [pdfCons(val,sigma_a, s=s) for val in x], 'k', linewidth=2)
    plt.show()
    #print (quad(lambda x: pdf2(x,sigma_a, s=s), -1, 1))
    
    from scipy.stats import gaussian_kde
    import numpy as np 

    def f(x):
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)

    y = np.linspace(-1,1,1000)
    data = f(y)
    kde = kde = gaussian_kde(data, bw_method='scott')
    samples1 = kde.resample(100)
    samples2 = kde.resample(100)
    print (np.array_equal(np.sort(samples1), np.sort(samples2)))"""

    print (pdfPDE3d(0,0,0))
    print (pdfPDEhuman(0,0))
    print (pdfPDEnumbers(0,0))
    print (pdfPDEbrain(0,0))
    