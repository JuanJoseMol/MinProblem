import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib
from skimage.color import rgb2gray



def init_params_JJ(layers, key, sigma_W):
  Ws = []
  bs = []
  for i in range(len(layers) - 2):
    if i==0:
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[0], layers[1]))*sigma_W)
    elif i==1:
      std_glorot = np.sqrt(2/(layers[i]*2 + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i]*2, layers[i + 1]))*std_glorot)
      bs.append(np.zeros(layers[i + 1]))
    else:
      std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
      bs.append(np.zeros(layers[i + 1]))
  std_glorot = np.sqrt(2/(layers[-2] + layers[-1]))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[-2], layers[-1]))*std_glorot)

  return Ws, bs

def init_params_JJ2(layers, key, sigma_W):
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

def init_params_JJ_uni(layers, key, sigma_W):
  Ws = []
  bs = []
  for i in range(len(layers) - 2):
    if i==0:
      key, subkey = random.split(key)
      Ws.append((random.uniform(subkey, (layers[0], layers[1]), minval=-sigma_W, maxval=sigma_W)))
    elif i==1:
      std_glorot = np.sqrt(2/(layers[i]*2 + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i]*2, layers[i + 1]))*std_glorot)
      bs.append(np.zeros(layers[i + 1]))
    else:
      std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
      bs.append(np.zeros(layers[i + 1]))
  std_glorot = np.sqrt(2/(layers[-2] + layers[-1]))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[-2], layers[-1]))*std_glorot)

  return Ws, bs

def init_params_JJ2_uni(layers, key, sigma_W):
  Ws = []
  for i in range(len(layers) - 2):
    if i==0:
      key, subkey = random.split(key)
      Ws.append((random.uniform(subkey, (layers[0], layers[1]), minval=-sigma_W, maxval=sigma_W)))
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

@jit
def forward_passJJwithbias(H, params):
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  #print (H.shape, Ws[0].shape)
  H = np.matmul(H, Ws[0])
  H = np.concatenate((np.sin(H), np.cos(H)),axis = -1)
  #print (H.shape, Ws[0].shape, bs[0].shape)
  for i in range(1,N_layers-1):
      H = np.matmul(H, Ws[i]) + bs[i-1]
      H = relu(H)
  Y = np.matmul(H, Ws[-1])
  return Y

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

@jit
def forward_pass(H, params):
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  for i in range(N_layers - 1):
    H = np.matmul(H, Ws[i]) + bs[i]
    H = relu(H)
  Y = np.matmul(H, Ws[-1]) + bs[-1]
  return Y



########################################################################

def traintest(tipo2, listw, layers, lr, frame):
  #N_s = 151
  if tipo2 == "astro":

    from skimage import data
    from skimage.color import rgb2gray


    #ej = data.camera()
    ej2 = data.astronaut()
    ej2 = rgb2gray(ej2)#[20:220,120:320]
    N_s = 256
    size = (N_s, N_s)
    ima = Image.fromarray(ej2)
    resized = ima.resize(size, Image.LANCZOS)
    im = np.array(resized)
    im = im/255.0
    im -= im.mean()
    im = im[:,:,None]
    del ej2
    del ima
    del resized
    #layers = [2,m_size,256,256,1]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla#.reshape((-1,2))
    y_train = im#.reshape((-1,1))



  if tipo2 == "numbers":
    from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
    (traind, _), (_, _) = mnist.load_data()
    traind = traind.astype('float32') / 255.
    im = traind[100,:,:]
    im -= im.mean()
    N_s = im.shape[0]
    size = (N_s, N_s)
    im = im[:,:,None]
    del traind
    #layers = [2,m_size,256,256,1]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla
    y_train = im


  if tipo2 == "brain":
    #../../../
    import nibabel as nib
    A = nib.load("dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
    im = onp.swapaxes(A[7:187,16:216,120], 0, 1)
    size = im.shape
    print (size)
    im = im/255.0
    #im -= im.mean()
    im = im[:,:,None]
    del A
    #layers = [2,m_size,256,256,1]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla
    y_train = im



  if tipo2 == "human":
    from skimage.color import rgb2gray
    #../../../
    im = Image.open('dataset/1 (107).jpg')
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
    #im = im[:,:,None]
    print (im.shape)

    del resized1
    del resized2
    del resized3
    del im1
    del im2
    del im3
    #layers = [2,m_size,256,256,3]
    grilla = meshgrid_from_subdiv(im.shape[0:2], (-1,1))
    x_train = grilla
    y_train = im
  

  if tipo2 == "3d":

      import nibabel as nib
      from skimage.transform import resize
      A = nib.load("dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
      im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
      im = resize(im, (50,45,45))
      im = im/255.0
      im -= im.mean()
      size = im.shape
      
      #im = im[:,:,None]
      del A
      #layers = [3,m_size,256,256,1]
      grilla = meshgrid_from_subdiv(im.shape, (-1,1))
      x_train = grilla
      y_train = im[:,:,:,None]
      print (size)

      

  @jit
  def loss(params, X, Y):
    MSE_data = np.average((forward_passJJ(X, params) - Y)**2)
    return  MSE_data



  ######################################################################

  @partial(jit, static_argnums=(0,))
  def step(loss, i, opt_state, X_batch, Y_batch):
      params = get_params(opt_state)
      g = grad(loss)(params, X_batch, Y_batch)
      return opt_update(i, g, opt_state)


  def train(loss, X, Y, opt_state, key, nIter = 10000, batch_size = 10):
      train_loss = []
      for it in range(nIter):
          key, subkey = random.split(key)
          #idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
          opt_state = step(loss, it, opt_state, X, Y)#[idx_batch,:], Y[idx_batch,:])
          if it % 100 == 0:
              params = get_params(opt_state)
              train_loss_value = loss(params, X, Y)
              train_loss.append(train_loss_value)
              to_print = "it %i, train loss = %e" % (it, train_loss_value)
              print(to_print)
      return opt_state, train_loss


  sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))
  ite = 500
 
  tipo = "uni"



  for sigma_W in listw: 
    print ("--------------------------------------------------------")
    print ("----------------------", sigma_W, "---------------------------")
    print ("--------------------------------------------------------")
    for k in range(5):
      print ("--------------------------------------------------------")
      print ("----------------------", k, "---------------------------")
      print ("--------------------------------------------------------")
      key = random.PRNGKey(k)
      if tipo == "norm":
        params = init_params_JJ2(layers, key, sigma_W)
        print (len(params), len(params[0]), len(params[1]))
        print (params[0][0].shape, params[0][1].shape, params[0][2].shape, params[0][3].shape)
        print(params[1][0].shape, params[1][1].shape)
      else:
        params = init_params_JJ2_uni(layers, key, sigma_W)

      opt_init, opt_update, get_params = optimizers.sgd(lr)
      opt_state = opt_init(params)

      print (forward_passJJ(x_train, params).shape)
      Y_preds = [forward_passJJ(x_train, params).reshape(size)]

      for i in range(frame):
        opt_state, train_loss = train(loss,x_train, y_train, opt_state,
                                          key, nIter = ite,  batch_size = 1000)
        Y_preds.append(forward_passJJ(x_train, get_params(opt_state)).reshape(size))

      Y_preds = np.array(Y_preds)

      with open('results/NomeanML-'+tipo+'-'+tipo2+'lr'+str(lr)+'Ns'+str(size[0])+'-w'+str(round(sigma_W,3))+'a'+str(round(sigmaA,3))+'-'+str(k)+'.txt', 'w') as f:
        onp.savetxt(f, Y_preds.reshape((11,-1)) )

      




if __name__ == "__main__":
  v = np.pi#0.75*np.pi
  m_size = 10000
  #traintest("numbers",[4.5*v,9*v,14*v],1e-6,10)
  traintest("brain",[45*v, 60*v, 90*v],[2,m_size,256,256,1], 1e-4,10) #90
  #traintest("astro",[64*v, 86*v, 128*v],[2,m_size,256,256,1],1e-4,10) #128
  #traintest("human",[35*v, 46*v, 70*v],[2,m_size,256,256,3],1e-4,10) #70
  #traintest("3d",[11.25*v, 15*v, 22.5*v],[3,2*m_size,256,256,1],1e-4,10) #22.5

