import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu

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

@jit
def forward_passJJ(H, params):
  Ww = params[0]
  Wa = params[1]
  H = np.matmul(H, Ww)
  H = np.concatenate((np.sin(H), np.cos(H)),axis = 1)

  Y = np.matmul(H,Wa)
  return Y

@jit
def forward_passBNClassic(H, params):
  Ww = params[0]
  Wa = params[1]
  #print (H.shape, H[0].shape)
  #print (Ww.shape, Wa.shape)
  H1 = np.matmul(H, Ww)
  #print ("1")
  #print (H1.shape, H1[0].shape)
  H2 = np.concatenate((np.sin(H1), np.cos(H1)),axis = 1)
  #print ("2")
  #print (H2.shape, H2[0].shape)
  
  Y = np.matmul(H2,Wa)
  M = np.sqrt(np.mean(Y**2))
  return Y/M
@jit
def forward_passBNMatias(H, params):
  Ww = params[0]
  Wa = params[1]
  H1 = np.matmul(H, Ww)
  H2 = np.concatenate((np.sin(H1), np.cos(H1)),axis = 1)
  #print ((H2*H2).shape, ((Wa*Wa).T).shape)
  #print (((Wa*Wa).T*(H2*H2)).shape)
  M = np.sqrt(np.mean(np.sum((Wa*Wa).T*(H2*H2), axis = 1)))
  Y = np.matmul(H2,Wa)
  return Y/M



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
    for it in range(nIter):
        params = get_params(opt_state)
        train_loss_value = loss(params, X, Y)
        train_loss.append(train_loss_value)
        to_print = "it %i, train loss = %e" % (it, train_loss_value)
        print(to_print)
        key, subkey = random.split(key)
        idx_batch = random.choice(subkey, X.shape[0], shape = (batch_size,), replace = False)
        opt_state = step(loss, it, opt_state, X[idx_batch,:], Y[idx_batch,:])
       # opt_state = step(loss, it, opt_state, X, Y)
        
        
    return opt_state, train_loss

########################################################################

def f(x):
  freq = 2.1
  sin = np.sin(2*np.pi*freq*x)
  return np.sign(sin)*(np.abs(sin) > 0.5)

N_s = 240 # number of data points
x = np.linspace(-1,1,N_s)
ft = np.fft.fft(f(x))
frequencies = np.fft.fftfreq(N_s)

x_train = np.linspace(-1,1,N_s)[:,None]
y_train = f(x_train)

######################################################################

@jit
def loss(params, X, Y):
  MSE_data = np.average((forward_passJJ(X, params) - Y)**2)
  return  MSE_data

@jit
def lossBNClassic(params, X, Y):
  MSE_data = np.average((forward_passBNClassic(X, params) - Y)**2)
  return  MSE_data

@jit
def lossBNMatias(params, X, Y):
  MSE_data = np.average((forward_passBNMatias(X, params) - Y)**2)
  return  MSE_data

tipo = "norm"
layers = [1,2000,1]
sigmaA = 0.001#np.sqrt(2/(layers[-1] + layers[-2]))  #para 90, 1000: 4000 iteraciones lr -5 // para 180, 003, lr5, 4000
ite = 1000
sigma_W = 90
"""
params = init_params_JJ(layers, random.PRNGKey(0), sigma_W, sigmaA)
r, m =forward_passBNClassic(x_train, params)
print (r.shape)
print (m)
r, m =forward_passBNMatias(x_train, params)
print (r.shape)
print (m)

"""
for k in range(1):#
    print (k)
    key = random.PRNGKey(k)
    sigma_W = 180
    l =[0.001+0.001*i for i in range(400)]
    #l = np.linspace(0.0001,0.001,100)#[0.001+0.001*i for i in range(300,400)]#np.linspace(0.001,1,1000)
    for sigma_a in l:#, (sigmaA,180), (sigmaA*1000,180)]:
    #for sigma_W in [90,180]:
    #  for sigma_a in [sigmaA*1000]:#, sigmaA*1000]:#[sigmaA*10**i for i in range(1,4)]:
        if tipo == "norm":
          params = init_params_JJ(layers, key, sigma_W, sigma_a)
        elif tipo == "uni":
          params = init_params_JJ_uni(layers, key, sigma_W, sigma_a)
        else:
          params = init_params_JJ_2(layers, key, sigma_W, sigma_a,k)
        """
        opt_init, opt_update, get_params = optimizers.adam(1e-4)
        opt_state = opt_init(params)

        opt_state, lossvalue1 = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])
        """
        print ("classic")
        params = init_params_JJ(layers, key, sigma_W, sigma_a)
        opt_init, opt_update, get_params = optimizers.adam(1e-4)
        opt_state = opt_init(params)
        opt_state, lossvalue2 = train(lossBNClassic, x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

        print ("Matias")
        params = init_params_JJ(layers, key, sigma_W, sigma_a)
        opt_init, opt_update, get_params = optimizers.adam(1e-4)
        opt_state = opt_init(params)
        opt_state, lossvalue3 = train(lossBNMatias, x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])
        """
        with open('resultdis/w'+str(sigma_W)+'a'+ format(sigma_a, ".3f") +'norm-NoBN.txt', 'w') as f:
              onp.savetxt(f,np.array(lossvalue1))"""
        with open('resultdis/w'+str(sigma_W)+'a'+ format(sigma_a, ".3f") +'norm-BNClassic.txt', 'w') as f:
              onp.savetxt(f, np.array(lossvalue2))
        with open('resultdis/w'+str(sigma_W)+'a' + format(sigma_a, ".3f") + 'norm-BNMatias.txt', 'w') as f:
              onp.savetxt(f, np.array(lossvalue3))

        """
        plt.plot(np.array(lossvalue1), label = "NoBN")
        plt.plot(np.array(lossvalue2), label = "BNclassic")
        plt.plot(np.array(lossvalue3), label = "BNMatias")
        plt.yscale('log')
        plt.legend()
        plt.show()"""


        

