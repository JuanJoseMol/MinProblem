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



def init_params_JJ_TGau(layers, key, sigma_W, sigma_a, R):
  trunc_gaussian = truncnorm(-R, R, loc=0, scale=sigma_W)
  Ws = []
  key, subkey = random.split(key)
  Ws.append(trunc_gaussian.rvs(size=(layers[0], layers[1])))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws



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


def init_params_DConsHuman(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfHuman(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DConsBrain(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfBrain(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DCons3d(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample3d(layers[1], lambda x,y,z: pdf3d(x,y,z,t), sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

###################################################################################


def init_params_MedoidHuman(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfHumanMediod(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_MedoidBrain(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfBrainMediod(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_Medoid3d(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample3d(layers[1], lambda x,y,z: pdf3dMediod(x,y,z,t), sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

################################################################################################

def init_params_RhoMeanHuman(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfHumanRhoMean(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_RhoMeanBrain(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfBrainRhoMean(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_RhoMean3d(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample3d(layers[1], lambda x,y,z: pdf3dRhoMean(x,y,z,t), sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

################################################################################################
def init_params_RhoMediodHuman(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfHumanRhoMediod(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_RhoMediodBrain(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], lambda x,y: pdfBrainRhoMediod(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_RhoMediod3d(layers, t, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample3d(layers[1], lambda x,y,z: pdf3dRhoMediod(x,y,z,t), sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
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
          size = im.shape
          maximun = 255#np.max(im)
          im = im/maximun
          im -= im.mean()
          layers = [3,15000,1]
          grilla = meshgrid_from_subdiv(im.shape, (-1,1))
          x_train = grilla
          y_train = im[:,:,:,None]
          print (im.shape, layers)
          sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))

          key = random.PRNGKey(0)

          LU = []


          for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
              
              
              
              params = init_params_JJ(layers, key, sigma_W, sigmaA)

              opt_init, opt_update, get_params = optimizers.sgd(lr)
              opt_state = opt_init(params)

              Y_preds = [forward_passJJ(x_train, params).reshape(size)]

              for i in range(20):
                opt_state, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])
                Y_preds.append(forward_passJJ(x_train, get_params(opt_state)).reshape(size))
              Y_preds = np.array(Y_preds)
              with open('results/tEstimation/ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt', 'w') as f:
                onp.savetxt(f, Y_preds.reshape((21,-1)))
              


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
          layers = [2,15000,1]
          grilla = meshgrid_from_subdiv(im.shape, (-1,1))
          x_train = grilla
          y_train = im[:,:,None]
          print (im.shape, layers)
          sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))

          key = random.PRNGKey(0)
          for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
              
              params = init_params_JJ(layers, key, sigma_W, sigmaA)

              opt_init, opt_update, get_params = optimizers.sgd(lr)
              opt_state = opt_init(params)

              #print (forward_passJJ(x_train, params)
              Y_preds = [forward_passJJ(x_train, params).reshape(size)]

              for i in range(20):
                opt_state, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])
                Y_preds.append(forward_passJJ(x_train, get_params(opt_state)).reshape(size))
              Y_preds = np.array(Y_preds)
              with open('results/tEstimation/ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt', 'w') as f:
                onp.savetxt(f, Y_preds.reshape((21,-1)))
             

  if tipo2 == "human":
    base_dir = "dataset/testhuman"
    for root, dirs, files in os.walk(base_dir):

      for k, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.png'):
          gz_path = os.path.join(root, file)

          img = Image.open(gz_path)
          # Convert the image to a Numpy array
          img_array = onp.array(img)
          
          
          s = (150,300)
          im1 = Image.fromarray(img_array[:,:,0])
          resized1 = im1.resize(s, Image.LANCZOS)
          im2 = Image.fromarray(img_array[:,:,1])
          resized2 = im2.resize(s, Image.LANCZOS)
          im3 = Image.fromarray(img_array[:,:,2])
          resized3 = im3.resize(s, Image.LANCZOS)
          im = onp.asarray(onp.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                        , np.array(resized3)[:,:,None]), axis=2), dtype = 'float32')
          size = im.shape
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
          layers = [2,15000,3]
          grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
          x_train = grilla
          y_train = im
          sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))      

          key = random.PRNGKey(0)


          for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
              
              params = init_params_JJ(layers, key, sigma_W, sigmaA)

              opt_init, opt_update, get_params = optimizers.sgd(lr)
              opt_state = opt_init(params)

              Y_preds = [forward_passJJ(x_train, params).reshape(size)]
              print (Y_preds[-1].shape)

              for i in range(20):
                opt_state, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])
                Y_preds.append(forward_passJJ(x_train, get_params(opt_state)).reshape(size))
                print (Y_preds[-1].shape)
              Y_preds = np.array(Y_preds)
              with open('results/tEstimation/ej-'+tipo2+'-lr'+str(lr)+'-'+str(k)+'.txt', 'w') as f:
                onp.savetxt(f, Y_preds.reshape((21,-1)))
              

                

  return None
          
  

if __name__ == "__main__":
  
  v = 0.75*np.pi
  for lr in [5*1e-4]:
    #learnigspeed("3d", [22.5*v] ,5, lr/10)
    learnigspeed("human", [75*v] ,5, lr)
    #learnigspeed("brain", [90*v] ,5, lr)
 
