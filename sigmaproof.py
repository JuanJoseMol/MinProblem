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


def load_function(tipo2,ej, m):
    
  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    if ej =="1":
      A = nib.load("dataset/testbrain/sub-r040s085_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
      #A = nib.load("dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
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
    layers = [2,m,1]
    


  if tipo2 == "human":
    from skimage.color import rgb2gray
    #../../../
    #im = Image.open('dataset/1 (1487).jpg') #
    if ej =="1":
      im = Image.open('dataset/testhuman/1 (2949).jpg')
      #im = Image.open('dataset/1 (107).jpg')
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
    layers = [2,m,3]

    #plt.imshow(rgb2gray(im))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()

  if tipo2 == "3d":
    from skimage.color import rgb2gray
    #../../../
    import nibabel as nib
    from skimage.transform import resize
    if ej =="1":
      A = nib.load("dataset/testbrain/sub-r040s085_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
      #A = nib.load("dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
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
    layers = [3,m,1]

    
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
      return opt_state, train_loss, sigma



  im, g, layers = load_function(tipo2,ej,m)

  
  
  sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))      

  key = random.PRNGKey(0)
 

  v = np.pi

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

          
    params = init_params_JJ(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_normal, train_loss_normal, sigma_nor = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    with open('hiper/sigma3-ej'+ej+'-normal-'+tipo2+'sigma'+str(sigma_W)+'.txt', 'w') as f:
      onp.savetxt(f, np.array(train_loss_normal))
    
    

    
  


  
                

  return None
          
  

if __name__ == "__main__":
  v = np.pi
  #lr = 5*1e-2
  m = 15000
  ej = "1"



  
  #learnigspeed("3d", [8.510997196304478] ,1000, 5*1e-3, m, ej) #,2*v,9.073452123358896,4*v,5*v
  #learnigspeed("human", [6*v,8*v,9*v,35.86792075220431] ,1000, 5*1e-2, m, ej) #,40.90306157837341,15*v,20*v
  learnigspeed("brain", [20*v,30*v] ,1000, 5*1e-2, m, ej) #,16*v,57.57280992622677,25*v,30*v
  """
  
  learnigspeed("3d", [1*v,2*v,8.806968110818094,4*v,5*v] ,1000, 5*1e-3, m, ej)
  learnigspeed("3d", [0.25*v,0.5*v,1.972297069765665,1*v,2*v] ,1000, 5*1e-4, m, ej)
  learnigspeed("3d", [0.0625*v,0.125*v,0.7024330507416472,0.5*v,1*v] ,1000, 5*1e-5, m, ej)
  
  learnigspeed("human", [5*v,10*v,47.62657699221887,20*v,25*v] ,1000, 5*1e-2, m, ej)
  learnigspeed("human", [0.5*v,1*v,7.489913599384419,3*v,5*v] ,1000, 5*1e-3, m, ej)
  learnigspeed("human", [0.125*v,0.25*v,0.9874030101510249,1*v,2*v] ,1000, 5*1e-4, m, ej)
  
  learnigspeed("brain", [15*v,20*v,75.69107062944478,30*v,35*v] ,1000, 5*1e-2, m, ej)
  learnigspeed("brain", [1*v,3*v,15.5462334090731,7*v,9*v] ,1000, 5*1e-3, m, ej)
  learnigspeed("brain", [0.25*v,0.5*v,2.2015495125481706,1*v,2*v] ,1000, 5*1e-4, m, ej)"""

