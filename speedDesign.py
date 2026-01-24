import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
from jax.example_libraries import optimizers
from functools import partial
from matplotlib import pyplot as plt
from jax.nn import relu
from randomSampling import sample_from_pdf_rejection, pdfbasic, pdfbasicfull, pdfPDE, pdfExp, pdfCons
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy import integrate


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
"""
def init_params_Designbasic(layers, key, sigma_W, sigma_a):
  Ws = []
  Ws.append(sample_from_pdf_rejection(layers[1], pdfbasic,sigma_a, s=0).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws

def init_params_Designbasicfull(layers, key, sigma_W, sigma_a):
  Ws = []
  Ws.append(sample_from_pdf_rejection(layers[1], pdfbasicfull,sigma_a, s=0).reshape(layers[0], layers[1])*sigma_W)
  key, subkey = random.split(key)
  Ws.append(random.normal(subkey, (layers[1]*2, layers[-1]))*sigma_a)

  return Ws
"""

def init_params_DesignExp(layers, key, sigma_W, sigma_a):
  Ws = []
  Ws.append(sample_from_pdf_rejection(layers[1], pdfExp,sigma_a, s=0).reshape(layers[0], layers[1])*sigma_W)
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

@jit
def forward_passJJ(H, params):
  Ww = params[0]
  Wa = params[1]
  H = np.matmul(H, Ww)
  H = np.concatenate((np.sin(H), np.cos(H)),axis = 1)

  Y = np.matmul(H,Wa)
  return Y

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
        opt_state = step(loss, it, opt_state, X, Y)

        params = get_params(opt_state)
        train_loss_value = loss(params, X, Y)
        train_loss.append(train_loss_value)
        to_print = "it %i, train loss = %e" % (it, train_loss_value)
        print(to_print)
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
print (x.shape, f(x).shape)

######################################################################

@jit
def loss(params, X, Y):
  MSE_data = np.average((forward_passJJ(X, params) - Y)**2)
  return  MSE_data

tipo = "norm"
layers = [1,2000,1]
sigmaA = np.sqrt(2/(layers[-1] + layers[-2]))
sigma_a = np.sqrt(2/(layers[-1] + layers[-2]))  #para 90, 1000: 4000 iteraciones lr -5 // para 180, 003, lr5, 4000
ite = 1000
lr = 1e-4

key = random.PRNGKey(0)

LU = []
LN = []
LF = []
LE = []
LC = []
L = [[],[],[]]
YU = []
YB = []
YF = []
YE = []
YC = []

l = [350,370,380,400,450]
for sigma_W in l:#, (sigmaA,180), (sigmaA*1000,180)]:
    params = init_params_JJ_uni(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_normal, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LU.append(train_loss_normal)
    #YU.append(forward_passJJ(x_train, get_params(opt_state_normal)))
    
    ######################################################################


    params = init_params_DesignExp(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_normal, train_loss_normal = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LN.append(train_loss_normal)
    #YU.append(forward_passJJ(x_train, get_params(opt_state_normal)))
    
    ######################################################################
    """
    params = init_params_Designbasic(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_design, train_loss_design = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LB.append(train_loss_design)
    YB.append(forward_passJJ(x_train, get_params(opt_state_design)))
  
    ######################################################################

    params = init_params_Designbasicfull(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_designFull, train_loss_designFull = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LF.append(train_loss_designFull)
    YF.append(forward_passJJ(x_train, get_params(opt_state_designFull)))

    ######################################################################

    params = init_params_DesignExp(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_designExp, train_loss_designExp = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LE.append(train_loss_designExp)
    YE.append(forward_passJJ(x_train, get_params(opt_state_designExp)))

    ######################################################################
    """
    params = init_params_DCons(layers, key, sigma_W, sigmaA)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_DCons, train_loss_DCons = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LC.append(train_loss_DCons)
    YC.append(forward_passJJ(x_train, get_params(opt_state_DCons)))

    ######################################################################
    
    params = init_params_DesignEDP(layers, key, sigma_W, sigma_a, s=0)

    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    #for i in range(10):
    opt_state_designEDP, train_loss_designEDP = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

    LE.append(train_loss_designEDP)
      #YB.append(forward_passJJ(x_train, get_params(opt_state_design)))
    """
    ######################################################################
    for k, sigma_a in enumerate([sigmaA/10,sigmaA,sigmaA*10]):
      params = init_params_DesignEDP(layers, key, sigma_W, sigma_a, s=0)

      opt_init, opt_update, get_params = optimizers.adam(lr)
      opt_state = opt_init(params)

      #for i in range(10):
      opt_state_designEDP, train_loss_designEDP = train(loss,x_train, y_train, opt_state, key, nIter = ite, batch_size = x_train.shape[0])

      L[k].append(train_loss_designEDP)
      #YB.append(forward_passJJ(x_train, get_params(opt_state_design)))"""

"""
print (len(train_loss_design))
print (forward_passJJ(x_train, get_params(opt_state_design)).shape)
print (len(train_loss_normal))
print (forward_passJJ(x_train, get_params(opt_state_normal)).shape)
"""

fig, axs = plt.subplots(1, (len(l)), figsize=(16, 4))

ltemp = onp.array( LN + LU + LF + LE +LC)
ma = np.max(ltemp)
mi = np.min(ltemp)
print (mi, ma)
# Plotting the first row (pairs of lists)
for i in range(len(l)):
    axs[i].plot(LU[i], label='Usual uniform')
    axs[i].plot(LN[i], label='Cut normal')
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
fig, axs = plt.subplots(2, 5, figsize=(16, 6))

# Plotting the first row (pairs of lists)
for i in range(5):
    axs[0, i].plot(LU[i], label='D. normal')
    axs[0, i].plot(LB[i], label='D. design')
    axs[0, i].set_title(f'Sigma W = {l[i]}')
    axs[0, i].set_yscale('log')
    axs[0, i].legend()

# Plotting the second row (triplets of arrays)
for i in range(5):
    axs[1, i].plot(x, YU[i][:,0], label='D. normal')
    axs[1, i].plot(x, YB[i][:,0], label='D. design')
    axs[1, i].plot(x, f(x), label='Target f')
    axs[1, i].set_title(f'Sigma W = {l[i]}')
    axs[1, i].legend()
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 5, figsize=(16, 6))

# Plotting the first row (pairs of lists)
for i in range(5):
    axs[0, i].plot(LU[i+5], label='D. normal')
    axs[0, i].plot(LB[i+5], label='D. design')
    axs[0, i].set_title(f'Sigma W = {l[i+5]}')
    axs[0, i].set_yscale('log')
    axs[0, i].legend()

# Plotting the second row (triplets of arrays)
for i in range(5):
    axs[1, i].plot(x, YU[i+5][:,0], label='D. normal')
    axs[1, i].plot(x, YB[i+5][:,0], label='D. design')
    axs[1, i].plot(x, f(x), label='Target f')
    axs[1, i].set_title(f'Sigma W = {l[i+5]}')
    axs[1, i].legend()
plt.tight_layout()
plt.show()
"""