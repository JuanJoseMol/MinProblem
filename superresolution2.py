import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu
from randomSampling3 import sample_from_pdf_rejection, sample2d, sample3d, pdfHuman,  pdfBrain, pdf3d
from randomSampling3 import pdf3dRhoMean, pdfBrainRhoMean, pdfHumanRhoMean, pdfLowBrainRhoMean
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
from flax import linen as nn




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
  Ws.append(sample2d(layers[1], lambda x,y: pdfLowBrainRhoMean(x,y,t), sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
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
    image1 = forward_passJJ(X, params)[None,:,:,:]
    #print (image1.shape)
    image1 = nn.avg_pool(image1, window_shape=(2, 2), strides=(1, 1), padding=((0, 1), (0, 1)))
    #print (image1.shape)
    image2 = Y[:,:,0]
    #print (image2.shape)
    image2 = np.repeat(np.repeat(image2, 2, axis=0), 2, axis=1)
    #print ("adasd")
    loss = np.mean((image1[0,:,:,0] - image2)**2)
    

    return loss

@jit
def lossh(params, X, Y):
    image1 = forward_passJJ(X, params)[None,:,:,:]
    image1 = nn.avg_pool(image1, window_shape=(2, 2), strides=(1, 1), padding=((0, 1), (0, 1)))
    image2 = np.repeat(np.repeat(Y, 2, axis=0), 2, axis=1)
    loss = np.mean((image1[0,:,:,:] - image2)**2)
    

    return loss

@jit
def loss3d(params, X, Y):
    image1 = forward_passJJ(X, params)[None,:,:,:]
    image1 = nn.avg_pool(image1, window_shape=(2, 2,2), strides=(1, 1,1), padding=((0, 1), (0, 1), (0, 1)))
    image2 = np.repeat(np.repeat(np.repeat(Y, 2, axis=0), 2, axis=1), 2, axis=2)
    loss = np.mean((image1[0,:,:,:,0] - image2)**2)
    

    return loss


@jit
def loss2(params, X, Y):
  MSE_data = np.average((np.ravel(forward_passJJ(X, params)) - np.ravel(Y))**2)
  return  MSE_data



def learnigspeed(tipo2, listw,ite, lr, lm):

  import nibabel as nib
  from skimage.transform import resize

  @partial(jit, static_argnums=(0,))
  def step(loss, i, opt_state, X_batch, Y_batch):
      params = get_params(opt_state)
      g = grad(loss)(params, X_batch, Y_batch)
      return opt_update(i, g, opt_state)



  def train(loss, loss2, X, Y, Xtotal, Ytotal, opt_state, key, nIter = 10000, batch_size = 10):
      train_loss = []
      dev_loss = []
      for it in range(nIter):
          key, subkey = random.split(key)
          #idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,))
          opt_state = step(loss, it, opt_state, X, Y)
      
          params = get_params(opt_state)
          train_loss_value = loss(params, X, Y)
          dev_loss_value = loss2(params, Xtotal, Ytotal)

          train_loss.append(train_loss_value)
          dev_loss.append(dev_loss_value)
          to_print = "it %i, train loss = %e, dev loss = %f" % (it, train_loss_value, dev_loss_value)
          print(to_print)
      return opt_state, train_loss, dev_loss



  if tipo2 == "3d":
    
    base_dir = "dataset/"

    gz_path = base_dir+ "sub-r039s002_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"

    im3 = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
    im3 = np.swapaxes(im3[7:187,16:216,5:185], 0, 1)
    im3 = im3/255.0
    im3 -= im3.mean()
    im2 = resize(im3, (50,46,46))
    im = resize(im3, (25,23,23))
    

    for m in lm:
      layers = [3,m,1]
      grilla_x = meshgrid_from_subdiv(im2.shape, (-1,1))
      grilla_y = im2
      x_train = grilla_x
      y_train = im
      print (im.shape, layers)
      sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))

      key = random.PRNGKey(0)
      for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
        
        params = init_params_JJ_uni(layers, key, sigma_W, sigmaA)

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        opt_state_normal, train_loss_normal, dev_loss_normal = train(loss3d, loss2, x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


        dicresults = {
          'train_loss': train_loss_normal,
          'dev_loss': dev_loss_normal,
          'best_param': get_params(opt_state_normal)}

        with open('super/2'+tipo2+'uniform-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'m'+str(m)+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        ######################################################################
        
        for t in [27]:
          """  
          params = init_params_DConsBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_DCons, train_loss_DCons, dev_loss_DCons = train(loss, los s2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_DCons,
            'dev_loss': dev_loss_DCons,
            'best_param': get_params(opt_state_DCons)}

          with open('super/2'+tipo2+'Mean-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)

          #####################

          params = init_params_MedoidBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_MedoidBrain, train_loss_MedoidBrain, dev_loss_MedoidBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_MedoidBrain,
            'dev_loss': dev_loss_MedoidBrain,
            'best_param': get_params(opt_state_MedoidBrain)}

          with open('super/2'+tipo2+'Medoid-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
          """
          #####################

          params = init_params_RhoMean3d(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_RhoMeanBrain, train_loss_RhoMeanBrain, dev_loss_RhoMeanBrain = train(loss3d, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_RhoMeanBrain,
            'dev_loss': dev_loss_RhoMeanBrain,
            'best_param': get_params(opt_state_RhoMeanBrain)}

          with open('super/2'+tipo2+'RhoMean-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'m'+str(m)+'.pickle', 'wb') as handle:
            pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)

          #####################
          """
          params = init_params_RhoMediodBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_RhoMediodBrain, train_loss_RhoMediodBrain, dev_loss_RhoMediodBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_RhoMediodBrain,
            'dev_loss': dev_loss_RhoMediodBrain,
            'best_param': get_params(opt_state_RhoMediodBrain)}

          with open('super/2'+tipo2+'RhoMediod-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
          """



  if tipo2 == "brain":
    base_dir = "dataset/"

    gz_path = base_dir+ "sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"

    im2 = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
    im2 = np.swapaxes(im2[7:187,16:216,120], 0, 1)
    im2 = im2/255.0
    im2 -= im2.mean()
    im = resize(im2, (100,90))

    for m in lm:
      layers = [2,m,1]
      grilla_x = meshgrid_from_subdiv(im2.shape, (-1,1))
      grilla_y = im2[:,:,None]
      x_train = grilla_x
      y_train = im[:,:,None]
      print (im.shape, layers)
      sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))

      key = random.PRNGKey(0)
      for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
        """
        params = init_params_JJ_uni(layers, key, sigma_W, sigmaA)

        opt_init, opt_update, get_params = optimizers.sgd(lr)
        opt_state = opt_init(params)

        opt_state_normal, train_loss_normal, dev_loss_normal = train(loss, loss2, x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


        dicresults = {
          'train_loss': train_loss_normal,
          'dev_loss': dev_loss_normal,
          'best_param': get_params(opt_state_normal)}

        with open('super/2'+tipo2+'uniform-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'m'+str(m)+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """
        

        ######################################################################
        
        for t in [12,27]:
          """  
          params = init_params_DConsBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_DCons, train_loss_DCons, dev_loss_DCons = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_DCons,
            'dev_loss': dev_loss_DCons,
            'best_param': get_params(opt_state_DCons)}

          with open('super/2'+tipo2+'Mean-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)

          #####################

          params = init_params_MedoidBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_MedoidBrain, train_loss_MedoidBrain, dev_loss_MedoidBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_MedoidBrain,
            'dev_loss': dev_loss_MedoidBrain,
            'best_param': get_params(opt_state_MedoidBrain)}

          with open('super/2'+tipo2+'Medoid-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
          """
          #####################

          params = init_params_RhoMeanBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.sgd(lr)
          opt_state = opt_init(params)

          opt_state_RhoMeanBrain, train_loss_RhoMeanBrain, dev_loss_RhoMeanBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_RhoMeanBrain,
            'dev_loss': dev_loss_RhoMeanBrain,
            'best_param': get_params(opt_state_RhoMeanBrain)}

          with open('super/2Low'+tipo2+'RhoMean-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'m'+str(m)+'.pickle', 'wb') as handle:
            pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)

          #####################
          """
          params = init_params_RhoMediodBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_RhoMediodBrain, train_loss_RhoMediodBrain, dev_loss_RhoMediodBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_RhoMediodBrain,
            'dev_loss': dev_loss_RhoMediodBrain,
            'best_param': get_params(opt_state_RhoMediodBrain)}

          with open('super/2'+tipo2+'RhoMediod-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
          """

  if tipo2 == "human":
    base_dir = "dataset/"

    gz_path = base_dir+ "1 (1487).jpg"

    img = Image.open(gz_path)
    # Convert the image to a Numpy array
    img_array = onp.array(img)
    size = img_array.shape
    print (size)
    s = (150,300)
    img1 = Image.fromarray(img_array[:,:,0])
    resized1 = img1.resize(s, Image.LANCZOS)
    img2 = Image.fromarray(img_array[:,:,1])
    resized2 = img2.resize(s, Image.LANCZOS)
    img3 = Image.fromarray(img_array[:,:,2])
    resized3 = img3.resize(s, Image.LANCZOS)
    im2 = onp.asarray(onp.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                  , np.array(resized3)[:,:,None]), axis=2), dtype = 'float32')
    
    im2[:,:,0] = im2[:,:,0]/255.0
    im2[:,:,1] = im2[:,:,1]/255.0
    im2[:,:,2] = im2[:,:,2]/255.0
    im2[:,:,0] -= im2[:,:,0].mean()
    im2[:,:,1] -= im2[:,:,1].mean()
    im2[:,:,2] -= im2[:,:,2].mean()
    del img_array
    del resized1
    del resized2
    del resized3
    del img1
    del img2
    del img3
    im = resize(im2, (150,75,3))
    

    

    for m in lm:
      layers = [2,m,3]
      grilla_x = meshgrid_from_subdiv(im2.shape[0:2], (-1,1))
      grilla_y = im2
      x_train = grilla_x
      y_train = im
      print (im.shape, layers)
      sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))

      key = random.PRNGKey(0)
      for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
        
        params = init_params_JJ_uni(layers, key, sigma_W, sigmaA)

        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        opt_state_normal, train_loss_normal, dev_loss_normal = train(lossh, loss2, x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


        dicresults = {
          'train_loss': train_loss_normal,
          'dev_loss': dev_loss_normal,
          'best_param': get_params(opt_state_normal)}

        with open('super/2'+tipo2+'uniform-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'m'+str(m)+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        ######################################################################
        
        for t in [24]:
          """  
          params = init_params_DConsBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_DCons, train_loss_DCons, dev_loss_DCons = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_DCons,
            'dev_loss': dev_loss_DCons,
            'best_param': get_params(opt_state_DCons)}

          with open('super/2'+tipo2+'Mean-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)

          #####################

          params = init_params_MedoidBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_MedoidBrain, train_loss_MedoidBrain, dev_loss_MedoidBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_MedoidBrain,
            'dev_loss': dev_loss_MedoidBrain,
            'best_param': get_params(opt_state_MedoidBrain)}

          with open('super/2'+tipo2+'Medoid-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
          """
          #####################

          params = init_params_RhoMeanHuman(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_RhoMeanBrain, train_loss_RhoMeanBrain, dev_loss_RhoMeanBrain = train(lossh, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_RhoMeanBrain,
            'dev_loss': dev_loss_RhoMeanBrain,
            'best_param': get_params(opt_state_RhoMeanBrain)}

          with open('super/2'+tipo2+'RhoMean-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'m'+str(m)+'.pickle', 'wb') as handle:
            pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)

          #####################
          """
          params = init_params_RhoMediodBrain(layers, t, key, sigma_W, sigmaA)

          opt_init, opt_update, get_params = optimizers.adam(lr)
          opt_state = opt_init(params)

          opt_state_RhoMediodBrain, train_loss_RhoMediodBrain, dev_loss_RhoMediodBrain = train(loss, loss2,x_train, y_train,grilla_x, grilla_y, opt_state, key, nIter = ite, batch_size = x_train.shape[0])


          dicresults = {
            'train_loss': train_loss_RhoMediodBrain,
            'dev_loss': dev_loss_RhoMediodBrain,
            'best_param': get_params(opt_state_RhoMediodBrain)}

          with open('super/2'+tipo2+'RhoMediod-lr'+str(lr)+'w-'+str(round(sigma_W,3))+'.pickle', 'wb') as handle:
          pickle.dump(dicresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
          """

  return None
          
  

if __name__ == "__main__":
  v = np.pi
  #learnigspeed("3d", [11.25*v, 22.5*v] ,1000, 1e-3, [15000])
  #learnigspeed("human", [47.5*v, 75*v] ,1000, 1e-3, [15000])
  learnigspeed("brain", [45*v, 90*v] ,1000, 1e-3, [15000])
  #learnigspeed("brain", [10*v,20*v,30*v,40*v,50*v,60*v,70*v,80*v,90*v,100*v] ,1000, 1e-2)
  #learnigspeed("brain", [45*v] ,1000, 1e-3, [30000,40000,50000])

