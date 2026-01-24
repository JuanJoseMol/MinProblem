import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu
from randomSampling3 import sample_from_pdf_rejection, sample2d, sample3d, pdfHuman,  pdfBrain, pdf3d
from randomSampling3 import pdf3dRhoMean, pdfBrainRhoMean, pdfHumanRhoMean
from randomSampling3 import pdfBrainMediod, pdfHumanMediod, pdf3dMediod
from randomSampling3 import pdf3dRhoMediod, pdfBrainRhoMediod, pdfHumanRhoMediod
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib
from skimage.color import rgb2gray
import os
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def target_function(x, g_array, T, tipo2):

    if tipo2 == "3d":
        area = 8
    else:
        area = 4
    
    # Compute the integrand values for the filtered g_array
    integrand_values = (onp.maximum((onp.log(g_array) - onp.log(x)) / (2 * T), 0))
    
    # Approximate the integral as a Riemann sum
    integral_value = onp.sum(integrand_values)*area/onp.size(g_array)
    
    return integral_value

# Bisection method
def bisection_method(g_array, T, tipo2, tol=1e-6, max_iter=100):
    a, b = 1e-8, 100000  # Define the search interval

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint

        # Evaluate the function at the midpoint
        f_c = target_function(c, g_array, T, tipo2)

        # Check if the function value is close to 0.5
        if onp.abs(f_c - 1) < tol:
            return c
        
        # Decide which half of the interval to search in
        f_a = target_function(a, g_array, T, tipo2)
        if onp.sign(f_c - 1) == onp.sign(f_a - 1):
            a = c  # Narrow the interval to [c, b]
        else:
            b = c  # Narrow the interval to [a, c]
    
    raise ValueError("Bisection method did not converge")




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

def sample2d(n_samples, func, sigma_a, k, s=0, batch_size=1000):
    sampled_x_values = []
    sampled_y_values = []

    while len(sampled_x_values) < n_samples and len(sampled_y_values)< n_samples:

        samples_x = jax.random.uniform(jax.random.PRNGKey(3*k+0), shape=(40000,), minval=-1, maxval=1)
        samples_y = jax.random.uniform(jax.random.PRNGKey(3*k+1), shape=(40000,), minval=-1, maxval=1)

        pdf_values_at_samples = func(samples_x, samples_y)
        random_values = jax.random.uniform(jax.random.PRNGKey(3*k+2), shape=(40000,)) * np.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples]),axis=0)
    
    return samples

def sample3d(n_samples, func, sigma_a, k, s=0, batch_size=1000):
    sampled_x_values = []
    sampled_y_values = []
    sampled_z_values = []

    while len(sampled_x_values)< n_samples and len(sampled_y_values)< n_samples and len(sampled_z_values)< n_samples:

        samples_x = jax.random.uniform(jax.random.PRNGKey(4*k+0), shape=(40000,), minval=-1, maxval=1)
        samples_y = jax.random.uniform(jax.random.PRNGKey(4*k+1), shape=(40000,), minval=-1, maxval=1)
        samples_z = jax.random.uniform(jax.random.PRNGKey(4*k+2), shape=(40000,), minval=-1, maxval=1)

        pdf_values_at_samples = func(samples_x, samples_y, samples_z)
        random_values = jax.random.uniform(jax.random.PRNGKey(4*k+3), shape=(40000,)) * np.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
        sampled_z_values.extend(samples_z[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples],
                            np.array(sampled_z_values)[None,:n_samples]),axis=0)
    
    return samples

# Sampling from the truncated Gaussian


def init_params_JJ(layers, key, sigma_W, sigma_a):
  Ws = []
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[0], layers[1]))*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws


def init_params_JJ_uni(layers, key, Wmax, sigma_a):
  Ws = []
  key, subkey = random.split(key)
  Ws.append(random.uniform(subkey, shape=(layers[0], layers[1]), minval=-Wmax, maxval=Wmax))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws


################################################################################################

@jit
def forward_passJJ(H, params):
  Ww = params[0]
  Wa = params[1]
  #print ("asdasd", Ww.shape, Wa.shape, H.shape)
  H = np.matmul(H, Ww)
  #print ("fdhgdf", H.shape)
  H = np.concatenate((np.sin(H), np.cos(H)),axis = -1)
  #print ("cvbvc", H.shape)

  Y = np.matmul(H,Wa)
  return Y




######################################################################

@jit
def loss(params, X, Y):
  MSE_data = np.average((forward_passJJ(X, params) - Y)**2)
  return  MSE_data

def learnigspeed(tipo2, listw,ite, lr):

  import nibabel as nib
  from skimage.transform import resize

  @partial(jit, static_argnums=(0,))
  def step(loss, i, opt_state, X_batch, Y_batch):
      params = get_params(opt_state)
      g = grad(loss)(params, X_batch, Y_batch)
      return opt_update(i, g, opt_state)



  def train(loss, X, Y, opt_state, key, nIter = 10000, batch_size = 10):
      train_loss = []
      for it in range(nIter):
          key, subkey = random.split(key)
          #idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,))
          opt_state = step(loss, it, opt_state, X, Y)
      
          params = get_params(opt_state)
          train_loss_value = loss(params, X, Y)
          train_loss.append(train_loss_value)
          to_print = "it %i, train loss = %e" % (it, train_loss_value)
          if it % 100 == 0:
            print(to_print)
      return opt_state, train_loss




  if tipo2 == "3d":
    base_dir = "dataset/testbrain"
    for root, dirs, files in os.walk(base_dir):

      for k, file in enumerate(files):
        if file.endswith('T1w.nii.gz'):
          gz_path = os.path.join(root, file)

          im = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
          im =  np.swapaxes(im[7:187,16:216,5:185], 0, 1)
          im = resize(im, (50,45,45))
          maximun = 255#np.max(im)
          im = im/maximun
          im -= im.mean()
          layers = [3,15000,1]
          grilla = meshgrid_from_subdiv(im.shape, (-1,1))
          x_train = grilla
          y_train = im[:,:,:,None]
          print (im.shape, layers)
          sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))
          ft = onp.fft.fftshift(onp.fft.fftn(im))
          g = onp.abs(ft)**2
          T = 0.25
          x_root = bisection_method(g, T, tipo2)

          key = random.PRNGKey(k)

          LU = []


          for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
              
            grilla = meshgrid_from_subdiv(im.shape, (-1,1))
            x_train = grilla
            y_train = im[:,:,:,None]

            def pdf3d(z1, z2, z3, s=0):
              Nx, Ny, Nz = g.shape
              rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)

              x_values = np.linspace(-1,1,Nx)
              y_values = np.linspace(-1,1,Ny)
              z_values = np.linspace(-1,1,Nz)
              r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

              return r((z1, z2, z3))

            def init_params(layers, key, sigma_W, sigma_a, k, s=0):
              Ws = []
              Ws.append(sample3d(layers[1], lambda x,y,z: pdf3d(x,y,z), sigma_W, k, s=s).reshape(layers[0], layers[1])*sigma_W)
              key, subkey = random.split(key)
              Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

              return Ws
                      
            params = init_params(layers, key, listw[0], sigmaA, k)

            opt_init, opt_update, get_params = optimizers.sgd(lr)
            opt_state = opt_init(params)

            #for i in range(10):
            opt_state_design, train_loss_design = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

            with open('speedres/final/2Design-'+tipo2+'-T'+str(T)+'lr'+str(lr)+'-'+str(k)+'.txt', 'w') as f:
                onp.savetxt(f, np.array(train_loss_design))





  if tipo2 == "brain":
    base_dir = "dataset/testbrain"
    for root, dirs, files in os.walk(base_dir):

      for k, file in enumerate(files):
        if file.endswith('T1w.nii.gz'):
          gz_path = os.path.join(root, file)

          im = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
          im = np.swapaxes(im[7:187,16:216,120], 0, 1)
          maximun = 255#np.max(im)
          size = im.shape
          print (size)
          im = im/maximun
          im -= im.mean()

          ft = onp.fft.fftshift(onp.fft.fft2(im))
          g = onp.abs(ft)**2
          T = 2
          x_root = bisection_method(g, T, tipo2)

          layers = [2,15000,1]
          grilla = meshgrid_from_subdiv(im.shape, (-1,1))
          x_train = grilla
          y_train = im[:,:,None]
          print (im.shape, layers)
          sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))

          key = random.PRNGKey(k)
          for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:

            grilla = meshgrid_from_subdiv(im.shape, (-1,1))
            x_train = grilla
            y_train = im[:,:,None]
            def pdfBrain(z1, z2, s=0):
              Nx, Ny = g.shape
              rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
              #rhotemp[0] = np.max(rhotemp)
              #rhotemp[1] = np.max(rhotemp)
              x_values = np.linspace(-1,1,Nx)
              y_values = np.linspace(-1,1,Ny)
              #print (x_values.shape, y_values.shape, rhotemp.shape)
              r = RectBivariateSpline(x_values, y_values, rhotemp)

              return r(z1, z2, grid=False)

            def init_params(layers, key, sigma_W, sigma_a, k, s=0):
              Ws = []
              Ws.append(sample2d(layers[1], lambda x,y: pdfBrain(x,y), sigma_W, k, s=s).reshape(layers[0], layers[1])*sigma_W)
              key, subkey = random.split(key)
              Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

              return Ws
            params = init_params(layers, key, listw[0], sigmaA, k)

            opt_init, opt_update, get_params = optimizers.sgd(lr)
            opt_state = opt_init(params)

            #for i in range(10):
            opt_state_design, train_loss_design = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

            with open('speedres/final/2Design-'+tipo2+'-T'+str(T)+'lr'+str(lr)+'-'+str(k)+'.txt', 'w') as f:
                onp.savetxt(f, np.array(train_loss_design))

            
     

  if tipo2 == "human":
    base_dir = "dataset/testhuman"
    for root, dirs, files in os.walk(base_dir):

      for k, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.png'):
          gz_path = os.path.join(root, file)

          img = Image.open(gz_path)
          # Convert the image to a Numpy array
          img_array = onp.array(img)
          size = img_array.shape
          print (size)
          s = (150,300)
          im1 = Image.fromarray(img_array[:,:,0])
          resized1 = im1.resize(s, Image.LANCZOS)
          im2 = Image.fromarray(img_array[:,:,1])
          resized2 = im2.resize(s, Image.LANCZOS)
          im3 = Image.fromarray(img_array[:,:,2])
          resized3 = im3.resize(s, Image.LANCZOS)
          im = onp.asarray(onp.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                        , np.array(resized3)[:,:,None]), axis=2), dtype = 'float32')
          
          im[:,:,0] = im[:,:,0]/255.0
          im[:,:,1] = im[:,:,1]/255.0
          im[:,:,2] = im[:,:,2]/255.0
          im[:,:,0] -= im[:,:,0].mean()
          im[:,:,1] -= im[:,:,1].mean()
          im[:,:,2] -= im[:,:,2].mean()
          print (im.shape)
          del img_array
          del resized1
          del resized2
          del resized3
          del im1
          del im2
          del im3

          ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
          g = onp.abs(ft)**2
          T = 0.5
          x_root = bisection_method(g[:,:,0], T, tipo2)

          layers = [2,15000,3]
          grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
          x_train = grilla
          y_train = im
          sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))      

          key = random.PRNGKey(k)


          for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
              

            grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
            x_train = grilla
            y_train = im
            def pdfHuman(z1, z2, s=0):
              
              Nx, Ny = g.shape[0:2]
              rhotemp = np.where(g[:,:,0] >= x_root, np.log(g[:,:,0]/x_root)/(2*T), 0)
              #rhotemp[0] = np.max(rhotemp)
              #rhotemp[1] = np.max(rhotemp)
              x_values = np.linspace(-1,1,Nx)
              y_values = np.linspace(-1,1,Ny)
              #print (x_values.shape, y_values.shape, rhotemp.shape)
              r = RectBivariateSpline(x_values, y_values, rhotemp)

              return r(z1, z2, grid=False)

            def init_params(layers, key, sigma_W, sigma_a, k, s=0):
              Ws = []
              Ws.append(sample2d(layers[1], lambda x,y: pdfHuman(x,y), sigma_W, k, s=s).reshape(layers[0], layers[1])*sigma_W)
              key, subkey = random.split(key)
              Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

              return Ws
            
            params = init_params(layers, key, listw[0], sigmaA, k)

            opt_init, opt_update, get_params = optimizers.sgd(lr)
            opt_state = opt_init(params)

            #for i in range(10):
            opt_state_design, train_loss_design = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

            with open('speedres/final/2Design-'+tipo2+'-T'+str(T)+'lr'+str(lr)+'-'+str(k)+'.txt', 'w') as f:
                onp.savetxt(f, np.array(train_loss_design))
                



                

  return None
          
  

if __name__ == "__main__":
  
  v = np.pi
  lr = 5*1e-2
  #learnigspeed("3d", [22.5*v] ,1000, lr/10)
  learnigspeed("human", [75*v] ,1000, lr)
  #learnigspeed("brain", [90*v] ,1000, lr)
  """
  
  learnigspeed("3d", [22.5*v] ,1000, 1e-3)
  learnigspeed("human", [75*v] ,1000, 1e-3)
  learnigspeed("brain", [90*v] ,1000, 1e-3)
  learnigspeed("3d", [[0.5*v,1*v, 2*v,3*v,4*v, 5*v, 6*v, 7*v,8*v] ,1000, 1e-3) #[ 4*v, 5*v, 6*v, 7*v, 8*v]
  learnigspeed("human", [2*v,4*v] ,1000, 1e-3) #[10*v,20*v,30*v,40*v,50*v]
  learnigspeed("brain", [2*v,4*v] ,1000, 1e-3) #[10*v,20*v,30*v,40*v,50*v]
  """
  #learnigspeed("3d", [8*v, 9*v, 20*v] ,1000, lr) #[ 4*v, 5*v, 6*v, 7*v, 8*v], 
  #learnigspeed("human", [35*v,40*v,45*v,50*v] ,1000, lr)
  #learnigspeed("brain", [35*v,40*v,45*v,50*v] ,1000, lr)

