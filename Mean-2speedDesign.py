import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu
from randomSampling2 import sample_from_pdf_rejection, sample2d, sample3d, pdfHuman, pdfNumbers, pdfBrain, pdf3d, pdfPDE3d, pdfPDEbrain, pdfPDEhuman, pdfPDEnumbers, pdfPDE, pdfExp, pdfCons
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib
from skimage.color import rgb2gray


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

def init_params_DesignEDP(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample_from_pdf_rejection(layers[1], pdfPDE,sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DCons(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample_from_pdf_rejection(layers[1], pdfCons, sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DesignExp(layers, key, sigma_W, sigma_a):
  Ws = []
  Ws.append(sample_from_pdf_rejection(layers[1], pdfExp,sigma_a, s=0).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DesignEDPnumbers(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], pdfPDEnumbers,sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DesignEDPbrain(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], pdfPDEbrain,sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DesignEDPhuman(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], pdfPDEhuman,sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DesignEDP3d(layers, key, sigma_W, sigma_a, s=0):
  
  Ws = []
  Ws.append(sample3d(layers[1], pdfPDE3d,sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws



def init_params_DConsNumbers(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], pdfNumbers, sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DConsHuman(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], pdfHuman, sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DConsBrain(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample2d(layers[1], pdfBrain, sigma_W, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_DCons3d(layers, key, sigma_W, sigma_a, s=0):
  Ws = []
  Ws.append(sample3d(layers[1], pdf3d, sigma_a, s=s).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

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




########################################################################

def load(tipo2):
  #N_s = 151
  import os

  # List of potential file paths
  file_paths = [
      "../../../dataset/",  
      "dataset/"  
  ]
  file_path = None
  for path in file_paths:
    if os.path.exists(path):
        file_path = path
        break

  file_path = "dataset/"

  if tipo2 == "1d":
    def f(x):
      freq = 2.1
      sin = np.sin(2*np.pi*freq*x)
      return np.sign(sin)*(np.abs(sin) > 0.5)

    N_s = 240 # number of data points
    x = np.linspace(-1,1,N_s)
    im = f(x)

    x_train = np.linspace(-1,1,N_s)[:,None]
    y_train = f(x_train)
    layers = [1,2000,1]


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
    im -= im.mean()
    im = im[:,:,None]
    del ej2
    del ima
    del resized
    layers = [2,10000,1]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    #plt.imshow(im)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()


  if tipo2 == "numbers":
    from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
    (_, _), (testd, _) = mnist.load_data()
    test = testd.astype('float32') / 255.
    im = test[100,:,:]
    im -= im.mean()
    N_s = im.shape[0]
    size = (N_s, N_s)
    im = im[:,:,None]
    del test
    layers = [2,5000,1]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla
    y_train = im
    """
    plt.imshow(im)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()"""

  
  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    A = nib.load(file_path+"sub-r039s002_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    maximun = np.max(im)
    size = im.shape
    print (size)
    im = im/maximun
    im = im[:,:,None]
    del A
    layers = [2,15000,1]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla
    y_train = im
    #plt.imshow(im)
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()


  if tipo2 == "human":
    from skimage.color import rgb2gray
    #../../../
    img = Image.open(file_path+"1 (1487).jpg")
    # Convert the image to a Numpy array
    img_array = onp.array(img)[:,1:,:]
    s = (159,270)
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
    layers = [2,15000,3]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla
    y_train = im
    #plt.imshow(rgb2gray(im))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()

  if tipo2 == "3d":
      from skimage.color import rgb2gray
      #../../../
      import nibabel as nib
      from skimage.transform import resize
      A = nib.load(file_path+"sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
      #im = np.swapaxes(A[5:189,16:216,100:106], 0, 1)
      #im = resize(im, (50,46,10))
      #im =  np.swapaxes(A[5:191,14:218,98:108], 0, 1)
      #im = resize(im, (68,62,10))
      im =  np.swapaxes(A[7:187,16:216,98:108], 0, 1)
      im = resize(im, (50,45,45))
      maximun = np.max(im)
      im = im/maximun
      im -= im.mean()
      size = im.shape
      
      #im = im[:,:,None]
      del A
      layers = [3,15000,1]
      grilla = meshgrid_from_subdiv(im.shape, (-1,1))
      x_train = grilla
      y_train = im[:,:,:,None]
      print (size)
      """
      plt.imshow(im[:,:,5])
      plt.show()
      """
  return im, layers, x_train, y_train
######################################################################

@jit
def loss(params, X, Y):
  MSE_data = np.average((forward_passJJ(X, params) - Y)**2)
  return  MSE_data

def learnigspeed(tipo2, listw,ite, lr):
  #tipo = "norm"
  #tipo2 = "numbers"
  im, layers, x_train, y_train = load(tipo2)
  print (im.shape, layers)
  sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))
  #sigma_a = np.sqrt(2/(layers[-1] + layers[-2]))  #para 90, 1000: 4000 iteraciones lr -5 // para 180, 003, lr5, 4000
  #ite = 500
  #lr = 1e-4

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
        # opt_state = step(loss, it, opt_state, X, Y)
          params = get_params(opt_state)
          train_loss_value = loss(params, X, Y)
          train_loss.append(train_loss_value)
          to_print = "it %i, train loss = %e" % (it, train_loss_value)
          if it % 100 == 0:
            print(to_print)
      return opt_state, train_loss

  key = random.PRNGKey(0)

  LU = []
  LN = []
  LE = []
  LC = []
  YU = []
  YB = []
  YE = []
  YC = []

  l = [20, 40, 60, 80, 100]#[100,200,300,400]#
  for sigma_W in listw:#, (sigmaA,180), (sigmaA*1000,180)]:
      params = init_params_JJ_uni(layers, key, sigma_W, sigmaA)

      opt_init, opt_update, get_params = optimizers.sgd(lr)
      opt_state = opt_init(params)

      #for i in range(10):
      opt_state_normal, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

      LU.append(train_loss_normal)
      #YU.append(forward_passJJ(x_train, get_params(opt_state_normal)))
      
      ######################################################################

      """
      params = init_params_DesignExp(layers, key, sigma_W, sigmaA)

      opt_init, opt_update, get_params = optimizers.sgd(lr)
      opt_state = opt_init(params)

      #for i in range(10):
      opt_state_normal, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

      LN.append(train_loss_normal)
      #YU.append(forward_passJJ(x_train, get_params(opt_state_normal)))
      """
      ######################################################################
      if tipo2 == "brain":
        params = init_params_DConsBrain(layers, key, sigma_W, sigmaA)
      if tipo2 == "numbers":
        params = init_params_DConsNumbers(layers, key, sigma_W, sigmaA)
      if tipo2 == "human":
        params = init_params_DConsHuman(layers, key, sigma_W, sigmaA)
      if tipo2 == "3d":
        params = init_params_DCons3d(layers, key, sigma_W, sigmaA)

      opt_init, opt_update, get_params = optimizers.sgd(lr)
      opt_state = opt_init(params)

      #for i in range(10):
      opt_state_DCons, train_loss_DCons = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

      LC.append(train_loss_DCons)
      #YC.append(forward_passJJ(x_train, get_params(opt_state_DCons)))

      ######################################################################
      
      if tipo2 == "brain":
        params = init_params_DesignEDPbrain(layers, key, sigma_W, sigmaA)
      if tipo2 == "numbers":
        params = init_params_DesignEDPnumbers(layers, key, sigma_W, sigmaA)
      if tipo2 == "human":
        params = init_params_DesignEDPhuman(layers, key, sigma_W, sigmaA)
      if tipo2 == "3d":
        params = init_params_DesignEDP3d(layers, key, sigma_W, sigmaA)

      opt_init, opt_update, get_params = optimizers.sgd(lr)
      opt_state = opt_init(params)

      #for i in range(10):
      opt_state_designEDP, train_loss_designEDP = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

      LE.append(train_loss_designEDP)
        #YB.append(forward_passJJ(x_train, get_params(opt_state_design)))

      print (np.array(LU).shape)
      with open('speedres/MeanUni'+tipo2+'lr'+str(lr)+'-w'+str(sigma_W)+'a'+str(sigmaA.round(3))+'.txt', 'w') as f:
        onp.savetxt(f, np.array(train_loss_normal))
      with open('speedres/MeanConst'+tipo2+'lr'+str(lr)+'-w'+str(sigma_W)+'a'+str(sigmaA.round(3))+'.txt', 'w') as f:
        onp.savetxt(f, np.array(train_loss_DCons))
      with open('speedres/MeanPDE'+tipo2+'lr'+str(lr)+'-w'+str(sigma_W)+'a'+str(sigmaA.round(3))+'.txt', 'w') as f:
        onp.savetxt(f, np.array(train_loss_designEDP))


  return None
          
  """
  fig, axs = plt.subplots(1, (len(l)), figsize=(16, 4))

  ltemp = onp.array( LN + LU  + LE +LC)
  ma = np.max(ltemp)
  mi = np.min(ltemp)
  print (mi, ma)
  # Plotting the first row (pairs of lists)
  for i in range(len(l)):
      axs[i].plot(LU[i], label='Usual uniform')
      #axs[i].plot(LN[i], label='Cut normal')
      #axs[i].plot(LF[i], label='Full Pol design')
      axs[i].plot(LE[i], label='PDE design')
      axs[i].plot(LC[i], label='Cons design')
      #for k, sigma_a in enumerate([sigmaA/10,sigmaA,sigmaA*10]):
      #  axs[i].plot(L[k][i], label='PDE a='+str(sigma_a.round(3)))
      axs[i].set_title(f'R = {l[i]}')
      axs[i].set_yscale('log')
      axs[i].set_ylim(mi, ma)
  handles, labels = axs[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3)
  fig.suptitle("lr-"+str(lr)+"ite-"+str(ite))
  plt.tight_layout(rect=[0, 0.15, 1, 1])
  plt.show()
  """

if __name__ == "__main__":
  v = np.pi
  #learnigspeed("1d", [370,380,400,450] ,1000, 1e-4)
  learnigspeed("3d", [22.5*v] ,1000, 1e-2)
  learnigspeed("human", [79.5*v] ,1000, 1e-3)
  learnigspeed("brain", [90*v] ,1000, 1e-3)
  learnigspeed("numbers", [14*v] ,1000, 1e-4)
