import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu
from PIL import Image

######################################################

def init_params(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = random.split(key)
    Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(np.zeros(layers[i + 1]))
  return (Ws, bs)

def init_params_JJ(layers, key, sigma_W, sigma_a):
  Ws = []
  bs = []
  for i in range(len(layers) - 2):
    if i==0:
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[0], layers[1]))*sigma_W)
    else:
      std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
      bs.append(np.zeros(layers[i + 1]))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws, bs

def init_params_JJ_uni(layers, key, sigma_W, sigma_a):
  Ws = []
  bs = []
  for i in range(len(layers) - 2):
    if i==0:
      key, subkey = random.split(key)
      Ws.append((random.uniform(subkey, (layers[0], layers[1]), minval=-sigma_W, maxval=sigma_W)))
    else:
      std_glorot = np.sqrt(2/(layers[i] + layers[i + 1]))
      key, subkey = random.split(key)
      Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
      bs.append(np.zeros(layers[i + 1]))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws, bs
"""
def init_params_JJ_uni(layers, key, Wmax, sigma_a):
  Ws = []
  key, subkey = random.split(key)
  Ws.append((random.uniform(subkey, (layers[0], layers[1]), minval=-Wmax, maxval=Wmax)))
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws
"""
@jit
def forward_passJJ(H, params):
  Ws = params[0]
  bs = params[1]
  N_layers = len(Ws)
  H = np.matmul(H, Ws[0])
  H = np.concatenate((np.sin(H), np.cos(H)),axis = 1)
  for i in range(1,N_layers-1):
      H = np.matmul(H, Ws[i]) + bs[i]
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

@partial(jit, static_argnums=(0,))
def step(loss, i, opt_state, X_batch, Y_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, Y_batch)
    return opt_update(i, g, opt_state)



def train(loss, X, Y, opt_state, key, nIter = 10000, batch_size = 10):
    train_loss = []
    val_loss = []
    for it in range(nIter):
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch,:], Y[idx_batch,:])
       # opt_state = step(loss, it, opt_state, X, Y)
        if it % 100 == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, Y)
            train_loss.append(train_loss_value)
            to_print = "it %i, train loss = %e" % (it, train_loss_value)
            print(to_print)
    return opt_state, train_loss, val_loss

########################################################################



#N_s = 151


from skimage import data
from skimage.color import rgb2gray
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim

#ej = data.camera()
ej2 = data.astronaut()
ej2 = rgb2gray(ej2)#[20:220,120:320]
N_s = 200
size = (N_s, N_s)
ima = Image.fromarray(ej2)
resized = ima.resize(size, Image.LANCZOS)
im = np.array(resized)
del ej2
del ima
del resized


grilla = meshgrid_from_subdiv(im.shape, (-1,1))


x_train = flatten_all_but_lastdim(grilla)
y_train = np.ravel(im)[:,None]


print (x_train.shape)
"""
plt.imshow(im)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
"""

######################################################################

@jit
def loss(params, X, Y):
  MSE_data = np.average((forward_passJJ(X, params).reshape((N_s,N_s)) - Y.reshape((N_s,N_s)))**2)
  return  MSE_data


layers = [2,10000,1]
sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))
ite = 500
frame = 10
lr = .1
tipo = "norm"

#key = random.PRNGKey(0)
listw= [60,90,120]

for sigma_W in listw: 
  print ("--------------------------------------------------------")
  print ("----------------------", sigma_W, "---------------------------")
  print ("--------------------------------------------------------")
  for k in range(20):
    key = random.PRNGKey(k)
    if tipo == "norm":
      params = init_params_JJ(layers, key, sigma_W, sigmaA)
    else:
      params = init_params_JJ_uni(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    print (forward_passJJ(x_train, params).shape)
    Y_preds = [forward_passJJ(x_train, params).reshape((N_s,N_s))]

    for i in range(frame):
      opt_state, train_loss, val_loss = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])
      Y_preds.append(forward_passJJ(x_train, get_params(opt_state)).reshape((N_s,N_s)))

    Y_preds = np.array(Y_preds)
    #Y_preds = Y_preds[:,:,0]
    print ("asdasd", Y_preds.shape)
    with open('results/ej2dbasicNs'+str(N_s)+'-w'+str(sigma_W)+'a'+str(sigmaA)+tipo+str(k)+'.txt', 'w') as f:
      onp.savetxt(f, Y_preds.reshape((11,-1)) )


    """
    plt.figure(figsize = (8,4))
    plt.subplot(121)
    mi = np.min(im)
    ma = np.max(im)
    plt.imshow(im, cmap="gray")#, label = "$im_{real}$")
    plt.title('GT')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(Y_preds[frame,:,:], cmap="gray")#, cmap="gray")#, label = 'it %i' % frames2[k])
    #plt.title('Sig_W = '+str(sw))
    plt.axis('off')
    #cbaxes = plt.gcf().add_axes([0.9, 0.1, 0.03, 0.8]) # [left, bottom, width, height]
    #cbar = plt.colorbar(cax=cbaxes)
    #cbar.ax.tick_params(labelsize=5)
    plt.show()"""



