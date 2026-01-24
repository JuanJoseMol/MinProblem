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
import pickle


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
        random_values = jax.random.uniform(jax.random.PRNGKey(2), shape=(40000,)) * np.max(pdf_values_at_samples)
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
        random_values = jax.random.uniform(jax.random.PRNGKey(3), shape=(40000,)) * np.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
        sampled_z_values.extend(samples_z[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples],
                            np.array(sampled_z_values)[None,:n_samples]),axis=0)
    
    return samples

def init_params_JJ(layers, key, sigma_W):
  Ws = []
  for i in range(len(layers) - 2):
    if i==0:
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[0], layers[1]))*sigma_W)
    elif i==1:
      std_glorot = np.sqrt(2/(layers[i]*2 + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i]*2, layers[i + 1]))*std_glorot)
    else:
      std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
  std_glorot = np.sqrt(2/(layers[-2] + layers[-1]))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[-2], layers[-1]))*std_glorot)

  return Ws


def init_params_JJ_uni(layers, key, Wmax, sigma_a):
  Ws = []
  key, subkey = random.split(key)
  Ws.append(random.uniform(subkey, shape=(layers[0], layers[1]), minval=-Wmax, maxval=Wmax))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws


def load_function(tipo2,ej, m):
    
  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    if ej =="1":
      A = nib.load("dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
    else:
      A = nib.load("dataset/sub-r039s002_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    
    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    maximun = np.max(im)
    size = im.shape
    print (size)
    im = im/255.0
    im -= im.mean()
    del A

    ft = np.fft.fftshift(np.fft.fft2(im))
    #plt.imshow(im)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()
    layers = [2,m,200,200,1]
    


  if tipo2 == "human":
    from skimage.color import rgb2gray
    #../../../
    #im = Image.open('dataset/1 (1487).jpg') #
    if ej =="1":
      im = Image.open('dataset/1 (107).jpg')
    else:
      im = Image.open('dataset/1 (1487).jpg')
    # Convert the image to a Numpy array
    im = onp.array(im)
    s = (140,184)
    im1 = Image.fromarray(im[:,:,0])
    resized1 = im1.resize(s, Image.LANCZOS)
    im2 = Image.fromarray(im[:,:,1])
    resized2 = im2.resize(s, Image.LANCZOS)
    im3 = Image.fromarray(im[:,:,2])
    resized3 = im3.resize(s, Image.LANCZOS)
    im = onp.asarray(onp.concatenate((onp.array(resized1)[:,:,None], onp.array(resized2)[:,:,None], 
            onp.array(resized3)[:,:,None]), axis=2),dtype = 'float32')
    print (type(im))
    
    im[:,:,0] = im[:,:,0]/255.0
    im[:,:,1] = im[:,:,1]/255.0
    im[:,:,2] = im[:,:,2]/255.0
    im[:,:,0] -= im[:,:,0].mean()
    im[:,:,1] -= im[:,:,1].mean()
    im[:,:,2] -= im[:,:,2].mean()
    size = im.shape
    ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
    layers = [2,m,200,200,3]

    #plt.imshow(rgb2gray(im))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()

  if tipo2 == "3d":
    from skimage.color import rgb2gray
    #../../../
    import nibabel as nib
    from skimage.transform import resize
    if ej =="1":
      A = nib.load("dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
    else:
      A = nib.load("dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    
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
    layers = [3,m,200,200,1]

    
    #im = im[:,:,None]
    del A
    """
    plt.imshow(im[:,:,5])
    plt.show()
    """
  return im, np.abs(ft)**2, layers

################################################################################################

@jit
def forward_passJJ(H, params):
  Ws = params
  N_layers = len(Ws)
  #print (H.shape, Ws[0].shape)
  H = np.matmul(H, Ws[0])
  H = np.concatenate((np.sin(H), np.cos(H)),axis = -1)
  #print (H.shape, Ws[0].shape, bs[0].shape)
  for i in range(1,N_layers-1):
      H = np.matmul(H, Ws[i])
      H = relu(H)
  Y = np.matmul(H, Ws[-1])
  return Y




######################################################################

@jit
def loss(params, X, Y):
  MSE_data = np.average((forward_passJJ(X, params) - Y)**2)
  return  MSE_data

def learnigspeed(tipo2, listw,ite, lr, m, ej):

  import nibabel as nib
  from skimage.transform import resize

  @partial(jit, static_argnums=(0,))
  def step(loss, i, opt_state, X_batch, Y_batch):
      params = get_params(opt_state)
      g = grad(loss)(params, X_batch, Y_batch)
      return opt_update(i, g, opt_state)



  def train(loss, X, Y, opt_state, key, nIter = 10000, batch_size = 10):
      train_loss = []
      sigma = []
      for it in range(nIter):
          key, subkey = random.split(key)
          #idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,))
          opt_state = step(loss, it, opt_state, X, Y)
      
          params = get_params(opt_state)
          #sigma.append(np.std(params[1]))
          train_loss_value = loss(params, X, Y)
          train_loss.append(train_loss_value)
          to_print = "it %i, train loss = %e" % (it, train_loss_value)
          if it % 100 == 0:
            print(to_print)
      return opt_state, train_loss#, sigma



  im, g, layers = load_function(tipo2,ej,m)

  
  
  sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))      

  key = random.PRNGKey(0)
  if tipo2 == "brain":
    with open('speedres/4norBrain.pickle', 'rb') as file:
      data1 = pickle.load(file)
    if ej =="1":
      with open('speedres/fej1Designbrain.pickle', 'rb') as file:
        data2 = pickle.load(file)
    else:
      with open('speedres/fej2Designbrain.pickle', 'rb') as file:
        data2 = pickle.load(file)
  if tipo2 == "human":
    with open('speedres/4norHuman.pickle', 'rb') as file:
      data1 = pickle.load(file)
    if ej =="1":
      with open('speedres/fej1Designhuman.pickle', 'rb') as file:
        data2 = pickle.load(file)
    else:
        with open('speedres/fej2Designhuman.pickle', 'rb') as file:
          data2 = pickle.load(file)
  if tipo2 == "3d":
    with open('speedres/4nor3d.pickle', 'rb') as file:
      data1 = pickle.load(file)
    if ej =="1":
      with open('speedres/fej1Design3d.pickle', 'rb') as file:
        data2 = pickle.load(file)
    else:
      with open('speedres/fej2Design3d.pickle', 'rb') as file:
        data2 = pickle.load(file)


  for T, x_root, p in data2.values():#, (sigmaA,180), (sigmaA*1000,180)]:
    
    if tipo2 == "brain":
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

      
      def init_params(layers, key, sigma_W, s=0):
        Ws = []
        for i in range(len(layers) - 2):
          if i==0:
            Ws.append(sample2d(layers[1], lambda x,y: pdfBrain(x,y), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
          elif i==1:
            std_glorot = np.sqrt(2/(layers[i]*2 + layers[i + 1]))
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i]*2, layers[i + 1]))*std_glorot)
          else:
            std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
        std_glorot = np.sqrt(2/(layers[-2] + layers[-1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[-2], layers[-1]))*std_glorot)

        return Ws
      
    
    if tipo2 == "human":
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


      def init_params(layers, key, sigma_W, s=0):
        Ws = []
        for i in range(len(layers) - 2):
          if i==0:
            Ws.append(sample2d(layers[1], lambda x,y: pdfHuman(x,y), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
          elif i==1:
            std_glorot = np.sqrt(2/(layers[i]*2 + layers[i + 1]))
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i]*2, layers[i + 1]))*std_glorot)
          else:
            std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
        std_glorot = np.sqrt(2/(layers[-2] + layers[-1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[-2], layers[-1]))*std_glorot)

        return Ws
      
    if tipo2 == "3d":
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


      def init_params(layers, key, sigma_W, s=0):
        Ws = []
        for i in range(len(layers) - 2):
          if i==0:
            Ws.append(sample3d(layers[1], lambda x,y,z: pdf3d(x,y,z), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
          elif i==1:
            std_glorot = np.sqrt(2/(layers[i]*2 + layers[i + 1]))
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i]*2, layers[i + 1]))*std_glorot)
          else:
            std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
            key, subkey = random.split(key)
            Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
        std_glorot = np.sqrt(2/(layers[-2] + layers[-1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[-2], layers[-1]))*std_glorot)

        return Ws
              
    params = init_params(layers, key, listw[0])

    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_design, train_loss_design = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    with open('hiper/MLej'+ej+'-'+str(m)+'design2-'+tipo2+'T'+str(T)+'por'+str(p)+'.txt', 'w') as f:
      onp.savetxt(f, np.array(train_loss_design))
  

  if tipo2 == "3d":
    listw = [(k)*np.pi for k in range(1,9)]
  else:
    listw = [5*(k+1)*np.pi for k in range(1,9)]

  for sigma_W in listw:#T, sigma_W in data1.values():


    if tipo2 == "brain":
      grilla = meshgrid_from_subdiv(im.shape, (-1,1))
      x_train = grilla
      y_train = im[:,:,None]

    
    if tipo2 == "human":
      grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
      x_train = grilla
      y_train = im

      
    if tipo2 == "3d":
      grilla = meshgrid_from_subdiv(im.shape, (-1,1))
      x_train = grilla
      y_train = im[:,:,:,None]

          
    params = init_params_JJ(layers, key, sigma_W)

    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_normal, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    with open('hiper/MLej'+ej+'-'+str(m)+'normal-'+tipo2+'sigma'+str(sigma_W)+'.txt', 'w') as f:
      onp.savetxt(f, np.array(train_loss_normal))
    

    
  


  
                

  return None
          
  

if __name__ == "__main__":
  v = np.pi
  lr = 5*1e-2
  m = 15000
  ej = "1"
  #learnigspeed("3d", [22.5*v] ,1000, 5*1e-3, m, ej)
  learnigspeed("human", [75*v] ,1000, lr, m, ej)
  learnigspeed("brain", [90*v] ,1000, lr, m, ej)

