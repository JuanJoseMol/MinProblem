import time
import numpy as np
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
from jax.nn import relu, tanh
import scipy.ndimage as ndi
from loadRealistic import load_image, make_image_dataset
import pickle
import jax
import os
import optax
#os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
#jax.config.update("jax_default_matmul_precision", "high")

def loss_fn_true_streaming(forward, params, X_true, Y_true, *args):
    """
    Memory-safe exact MSE evaluation on large grids.
    Computes the same loss as a full forward pass, but without OOM.
    """
    N = X_true.shape[0]
    chunk_size = 4096
    def body_fun(i, acc):
        start = i * chunk_size
        end   = jnp.minimum((i + 1) * chunk_size, N)

        Xc = lax.dynamic_slice_in_dim(X_true, start, chunk_size)
        Yc = lax.dynamic_slice_in_dim(Y_true, start, chunk_size)

        pred = forward(Xc, params, *args)
        acc += jnp.sum((pred - Yc) ** 2)
        return acc

    n_chunks = (N + chunk_size - 1) // chunk_size
    total = lax.fori_loop(0, n_chunks, body_fun, 0.0)

    return total / N


def complex_gabor(z, omega0, sigma0):
    return jnp.sin(omega0 * z)* jnp.exp( - (jnp.abs(sigma0 * z)) ** 2)

def init_WIRE_params(layers, key, w0):
    Ws, bs = [], []

    # First layer
    key, subkey = random.split(key)
    Ws.append(random.uniform(subkey, (layers[0], layers[1]),
                              minval=-1/layers[0],
                              maxval= 1/layers[0]))
    bs.append(jnp.zeros(layers[1]))

    # Hidden layers
    for i in range(1, len(layers)-1):
        key, subkey = random.split(key)
        bound = jnp.sqrt(6 / (w0 * layers[i]))
        Ws.append(random.uniform(subkey, (layers[i], layers[i+1]),
                                  minval=-bound, maxval=bound))
        bs.append(jnp.zeros(layers[i+1]))

    return Ws, bs

@jax.jit
def WIRE_forward(X, params, omega0, sigma0):

    Ws, bs = params
    z = jnp.matmul(X, Ws[0])  + bs[0] 
    H = complex_gabor(z, omega0, sigma0)
    
    for i in range(1, len(Ws) - 1):
        z = jnp.matmul(H, Ws[i]) + bs[i]
        H = complex_gabor(z, omega0, sigma0)
    
    output = jnp.matmul(H, Ws[-1]) + bs[-1]
    return output 
"""
def complex_gabor(z, omega0, sigma0):
    return jnp.exp(1j * omega0 * z - (jnp.abs(sigma0 * z)) ** 2)

def init_WIRE_params(layers, key, w0):
    Ws, bs = [], []
    keys = random.split(key, len(layers) * 2)
    
    for i in range(len(layers) - 1):
        fan_in = layers[i]
        fan_out = layers[i + 1]
        
        if i == 0:
            # First layer: REAL weights
            bound = 1/ fan_in
            W = random.uniform(keys[2*i], (fan_in, fan_out), 
                              minval=-bound, maxval=bound)
            b = jnp.zeros(fan_out)
        else:
            # Complex layers - use proper complex initialization
            # The paper uses specific scaling for complex weights
            bound = jnp.sqrt(6 / (w0*layers[i])) 
            W_real = random.uniform(keys[2*i], (layers[i], layers[i+1]),
                                  minval=-bound, maxval=bound)
            W_imag = random.uniform(keys[2*i+1], (layers[i], layers[i+1]),
                                  minval=-bound, maxval=bound)
            W = (W_real + 1j * W_imag).astype(jnp.complex64)
            b = jnp.zeros(fan_out, dtype=jnp.complex64)
        
        Ws.append(W)
        bs.append(b)
    
    return Ws, bs

@jax.jit
def WIRE_forward(X, params, omega0, sigma0):
    Ws, bs = params
    
    # First layer: real to complex
    z = jnp.matmul(X, Ws[0])  + bs[0]  # matmul with complex weights
    
    # Apply activation
    H = complex_gabor(z, omega0, sigma0)
    
    # Hidden layers (all complex)
    for i in range(1, len(Ws) - 1):
        z = jnp.matmul(H, Ws[i]) + bs[i]
        H = complex_gabor(z, omega0, sigma0)
    
    # Final layer (complex to real)
    # Last weight matrix should be real or we take real part
    output = jnp.matmul(H, Ws[-1]) + bs[-1]
    
    # Return real part as final output
    return jnp.real(output)
"""


def init_FFfroze_params(layers, key, sigma):
    Ws = []
    bs = []
    N = len(layers) - 1
    keys = random.split(key, N)
    for i in range(1, N):
        if i == 1:
            std_glorot = jnp.sqrt(1 / (2 * layers[i]))
            Ws.append(random.normal(keys[i], (layers[i]*2, layers[i + 1])) * std_glorot)
            bs.append(jnp.zeros(layers[i + 1]))
        if i > 1:
            std_glorot = jnp.sqrt(1 / (layers[i]))
            Ws.append(random.normal(keys[i], (layers[i], layers[i + 1])) * std_glorot)
            bs.append(jnp.zeros(layers[i + 1]))
    return Ws, bs

@jit
def FF_frozen_forward(X, params,B):
    Ws, bs = params

    H = jnp.dot(X, B)
    H = jnp.concatenate([jnp.sin(H),
                          jnp.cos(H)], axis=-1)

    for i in range(0, len(Ws)-2):
        H = jnp.dot(H, Ws[i]) + bs[i]
        H = relu(H)  # ReLU

    return jnp.dot(H, Ws[-1])

def init_FF_params(layers, key, sigma):
    Ws = []
    bs = []
    N = len(layers) - 1
    keys = random.split(key, N)
    
    for i in range(N):
        if i == 0:
            Ws.append(random.normal(keys[i], (layers[0], layers[1])) * sigma)
        elif i == 1:
            std_glorot = jnp.sqrt(1 / (2 * layers[i]))
            Ws.append(random.normal(keys[i], (layers[i]*2, layers[i + 1])) * std_glorot)
            bs.append(jnp.zeros(layers[i + 1]))
        else:
            # Hidden layers
            std_glorot = jnp.sqrt(1 / (layers[i]))
            Ws.append(random.normal(keys[i], (layers[i], layers[i + 1])) * std_glorot)
            bs.append(jnp.zeros(layers[i + 1]))
    
    return Ws, bs

@jit
def FF_forward(X, params):
    Ws, bs = params
    H = jnp.dot(X, Ws[0])
    H = jnp.concatenate([jnp.sin(H),
                          jnp.cos(H)], axis=-1)

    for i in range(1, len(Ws)-1):
        H = jnp.dot(H, Ws[i]) + bs[i-1]
        H = relu(H)  # ReLU

    return jnp.dot(H, Ws[-1])#+ bs[-1]

def init_SIREN_params(layers, key, w0):
    Ws, bs = [], []

    # First layer
    key, subkey = random.split(key)
    Ws.append(random.uniform(subkey, (layers[0], layers[1]),
                              minval=-1/layers[0],
                              maxval= 1/layers[0]))
    bs.append(jnp.zeros(layers[1]))

    # Hidden layers
    for i in range(1, len(layers)-1):
        key, subkey = random.split(key)
        bound = jnp.sqrt(6 / layers[i]) / w0
        Ws.append(random.uniform(subkey, (layers[i], layers[i+1]),
                                  minval=-bound, maxval=bound))
        bs.append(jnp.zeros(layers[i+1]))

    return Ws, bs

@jit
def SIREN_forward(X, params, w0):
    Ws, bs = params

    H = jnp.sin(w0 * (jnp.matmul(X, Ws[0]) + bs[0]))
    for i in range(1, len(Ws)-1):
        H = jnp.sin(w0 *jnp.matmul(H, Ws[i]) + bs[i])
    return jnp.matmul(H, Ws[-1])


def init_Gaussian_params(layers, key):
    Ws, bs = [], []
    keys = random.split(key, len(layers)-1)

    for i in range(len(layers)-1):
        std = jnp.sqrt(1.0 / (layers[i]))

        #Ws.append(random.uniform(keys[i], (layers[i], layers[i+1]), minval=-0.1, maxval=0.1))
        Ws.append(random.normal(keys[i], (layers[i], layers[i + 1])) * std)
        bs.append(jnp.zeros(layers[i+1]))

    return Ws, bs

@jit
def Gaussian_forward(X, params, alpha):
    Ws, bs = params

    H = X
    for i in range(len(Ws)-1):
        Z = jnp.matmul(H, Ws[i]) + bs[i]
        H = jnp.exp(-jnp.abs(alpha* Z)**2)

    return jnp.matmul(H, Ws[-1])+ bs[-1]



def train_model_record_adam(forward, params, X, Y, X_true, Y_true, nIter, lr, ej, *args):
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return jnp.mean((forward(X, p, *args) - Y) ** 2)

    loss_history = []
    loss_true_history = [0]
    params_list = []

    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for it in range(nIter):
        params, opt_state, loss = step(params, opt_state)
        loss_history.append(loss)
        
        if it % 100 == 0:
            if (ej != "fitting" and ej != "fitting2" and ej != "fitting3"):
                loss_true = loss_fn_true_streaming(forward, params, X_true, Y_true, *args)
                loss_true_history.append(loss_true)
            print(f"Iter {it}: loss={loss:.4e}, true loss={loss_true_history[-1]:.4e}")
        if it % 10000 == 0 and (ej != "fitting" and ej != "fitting2" and ej != "fitting3"):
            params_list.append(params)

    Y_pred = forward(X, params, *args)
    params_list.append(params)

    return params_list, jnp.array(loss_history), jnp.array(loss_true_history), Y_pred


def train_model_record(forward, params, X, Y, nIter, lr, ej, *args):

    optimizer = optax.sgd(learning_rate=lr)
    opt_state = optimizer.init(params)
    
    @jit
    def loss_fn(p):
        return jnp.mean((forward(X, p, *args) - Y) ** 2)

    loss_history = []
    Y_preds = []

    @jit
    def step(params, opt_state):
        loss, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for it in range(nIter):
        params, opt_state, loss = step(params, opt_state)
        loss_history.append(loss)
        #Y_preds.append(forward(X, params, *args))

        if it % 10 == 0:
            print(f"Iter {it}: loss={loss:.4e}")

    Y_pred = forward(X, params, *args)
    #Y_preds.append(Y_pred)

    #np.save(f"realistic/short-{ej}-lr{lr}-sigmaW{sigma}.npy", np.array(Y_preds))

    return params, jnp.array(loss_history), Y_pred


def adaptive_integer_search(
    eval_loss_fn,
    low,
    high,
    coarse_step=10,
    refine_radius=10,
    criterion="test"  # "train" or "test"
):
    start = time.time()

    def select(loss_train, loss_true):
        return loss_true if criterion == "test" else loss_train

    # ---------- Stage 1: coarse scan ----------
    coarse_points = range(low, high + 1, coarse_step)
    scores = {}

    for k in coarse_points:
        loss_train, loss_true = eval_loss_fn(k)
        scores[k] = select(loss_train, loss_true)

    best_k = min(scores, key=scores.get)

    # ---------- Stage 2: local refinement ----------
    ref_low = max(low, best_k - refine_radius)
    ref_high = min(high, best_k + refine_radius)

    for k in range(ref_low, ref_high + 1):
        loss_train, loss_true = eval_loss_fn(k)
        score = select(loss_train, loss_true)
        if score < scores[best_k]:
            scores[k] = score
            best_k = k

    elapsed = time.time() - start
    return best_k, scores[best_k], elapsed

def search_sigma_FF(
    X, Y, X_true, Y_true,
    layers, nIter, lr,
    key,
    smin=1,
    smax=100,
    criterion="true"
):
    def eval_sigma(sigma):
        nonlocal key
        key, subkey = random.split(key)
        params = init_FF_params(layers, subkey, sigma)

        _, loss_hist, loss_true_hist = train_model(
            FF_forward, params,
            X, Y,
            X_true, Y_true,
            nIter, lr
        )

        return loss_hist[-1], loss_true_hist[-1]

    return adaptive_integer_search(
        eval_sigma,
        smin,
        smax,
        coarse_step=10,
        refine_radius=10,
        criterion=criterion
    )

def search_w0_SIREN(
    X, Y, X_true, Y_true,
    layers, nIter, lr,
    key,
    w0min=1,
    w0max=100,
    criterion="true"
):
    def eval_w0(w0):
        nonlocal key
        key, subkey = random.split(key)
        params = init_SIREN_params(layers, subkey, w0)

        _, loss_hist, loss_true_hist = train_model(
            SIREN_forward, params,
            X, Y,
            X_true, Y_true,
            nIter, lr,
            w0
        )

        return loss_hist[-1], loss_true_hist[-1]

    return adaptive_integer_search(
        eval_w0,
        w0min,
        w0max,
        coarse_step=10,
        refine_radius=10,
        criterion=criterion
    )


if __name__ == "__main__":
    
    import argparse   
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True) #search or savePreds or longTrain
    parser.add_argument("--ej", type=str, required=True) # "fitting", "super", "noise"
    parser.add_argument("--crit", type=str, default="train") # "train" or "test"
    parser.add_argument("--save", type=str, default="no") # "train" or "test"
    #parser.add_argument("--m", type=int) # "train" or "test"
    args = parser.parse_args()
    objective = args.obj
    ej = args.ej
    criterio = args.crit
    save = args.save
    #m = args.m

    if objective == "search":
        key = random.PRNGKey(0)

        lr = 1e-2
        layers_SIREN = [2, 300, 300, 300, 3]
        layers_FF = [2, 512, 256, 3] #[2, 300, 300, 300, 3]#

        img, imgTrue = load_image(ej)
        
        X, Y = make_image_dataset(img)
        X_true, Y_true = make_image_dataset(imgTrue)

        
        nIter_list = [5000,10000]
        latex_file = f"../data/hyperparam_search_ej{ej}_{criterio}_lr{lr}-3.tex"

            # ---------- open LaTeX file ----------
        with open(latex_file, "w") as f:
            f.write(r"""\begin{table}[h]
        \centering
        \begin{tabular}{c|ccc|ccc}
        \hline
        $n_{\text{iter}}$
        & $\sigma^*$ & Loss (FF) & Time (s)
        & $w_0^*$ & Loss (SIREN) & Time (s) \\
        \hline
        """)

        # ---------- experiment loop ----------
        for nIter in nIter_list:

            print(f"Running nIter = {nIter}")

            best_sigma, loss_ff, time_ff = search_sigma_FF(
                X, Y, X_true, Y_true, layers_FF, nIter, lr, key, smin=1, smax=100, criterion=criterio
            )

            best_w0, loss_siren, time_siren = search_w0_SIREN(
                X, Y, X_true, Y_true, layers_SIREN, nIter, lr, key, w0min=1, w0max=100, criterion=criterio
            )

            # ---------- append LaTeX row ----------
            with open(latex_file, "a") as f:
                f.write(
                    f"{nIter} & "
                    f"{best_sigma} & {loss_ff:.4f} & {time_ff:.2f} & "
                    f"{best_w0} & {loss_siren:.4f} & {time_siren:.2f} \\\\\n"
                )

        # ---------- close LaTeX table ----------
        with open(latex_file, "a") as f:
            f.write(r"""\hline
        \end{tabular}
        \caption{Hyperparameter search for fixed image and fixed criterion.}
        \end{table}
        """)
    

    
    ################################## ensayo en ejemplo individual ##################################
    if objective == "visualFF":
        start = time.time()

        img, imgTrue = load_image(ej)    
        X, Y = make_image_dataset(img)
        X_true, Y_true = make_image_dataset(imgTrue)
        
        k = 4
        key = random.PRNGKey(k)
        nIter_long = 5000

        if ej == "fitting":
            #sigma = 84 #optuna
            sigma = 112 #102 #our
        if ej == "super":
            sigma = 43.3 #optuna full #42 optuna
            #sigma = 56.7 # no full- 110 #our full
        if ej == "noise":
            sigma = 78 #
            #sigma = 121 #69 #our
        if ej == "fitting2":
            sigma = 44
            #sigma =102
        if ej == "super2":
            sigma = 46
            #sigma = 56.7
        if ej == "noise2":
            sigma = 38
            #sigma = 69
        if ej == "fitting3":
            #sigma = 84
            sigma =67
        if ej == "super3":
            #sigma = 42
            sigma = 79
        if ej == "noise3":
            sigma = 49
            #sigma = 69
        lr = 1e-1
        #sigma = 50
        print ("sigma", sigma)
        #key, subkey = random.split(key)

        layers_FF = [2, 500, 200, 3] 
        params = init_FF_params(layers_FF, key, sigma)

        params_ff, loss_hist, loss_true_hist, Y_pred = train_model(
            FF_forward, params, X, Y, X_true, Y_true,nIter_long, lr, ej)

        elapsed = time.time() - start
        print("Elapsed time (s):", elapsed)

        dic = {}
        dic["param"] = params_ff
        dic["loss"] = loss_hist
        dic["pred"] = Y_pred

        
        H, W, _ = img.shape
        img_pred = Y_pred.reshape(H, W, 3)
        img_pred = jnp.clip(img_pred, 0.0, 1.0)

        plt.figure(figsize=(15,4))
        plt.subplot(1,4,1)
        plt.semilogy(loss_hist)
        plt.semilogy(loss_true_hist, color="red")
        plt.xlabel("Iteration")
        plt.title(f"training loss, sig={sigma}")
        plt.grid(True)

        plt.subplot(1,4,2)
        plt.imshow(img)
        plt.title("GT")
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.imshow(img_pred)
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.imshow(jnp.abs(img - img_pred), cmap="Reds")
        plt. colorbar()
        plt.title("Error")
        plt.axis("off")
        if save == "yes":
            #plt.savefig(f'realistic/newourFFsgd{ej}_{nIter_long}-lr{lr}-s{sigma}.pdf', bbox_inches='tight')
            #with open(f"realistic/newourFFsgd{nIter_long}-{ej}-lr{lr}-s{sigma}-k{k}.pickle", "wb") as file:
            #    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
            plt.close()
        else:
            plt.show()


    if objective == "visualFFadam":
        start = time.time()

        img, imgTrue = load_image(ej)    
        X, Y = make_image_dataset(img)
        
        k = 4
        key = random.PRNGKey(k)
        nIter_long = 50000

        if ej == "fitting":
            #sigma = 84 #optuna sgd
            #sigma = 102 # our
            #sigma = 237.4 #optuna adam
            sigma = 111 # our full
        if ej == "super":
            #sigma = 178.8 #optuna adam
            #sigma = 56.7 #our
            #sigma = 42 #optuna
            sigma = 113 # our full
        if ej == "noise":
            #sigma = 146.4 #optuna
            #sigma = 69 #our
            #sigma = 61 #optuna adam
            sigma = 137 # our full
        if ej == "fitting2":
            #sigma = 84
            sigma =102
        if ej == "super2":
            sigma = 53.5
            #sigma = 56.7
        if ej == "noise2":
            sigma = 38.1
            #sigma = 69
        if ej == "fitting3":
            #sigma = 84
            sigma =67.4
        if ej == "super3":
            #sigma = 42
            sigma = 70.1
        if ej == "noise3":
            sigma = 39.2
            #sigma = 69
        lr = 1e-3
        #sigma = 84
        print ("sigma", sigma)

        layers_FF = [2, 2000, 200, 3] 
        params = init_FF_params(layers_FF, key, sigma)

        params_ff, loss_hist, Y_pred = train_model_record_adam(
            FF_forward, params, X, Y, nIter_long, lr, ej)

        elapsed = time.time() - start
        print("Elapsed time (s):", elapsed)

        dic = {}
        dic["param"] = params_ff
        dic["loss"] = loss_hist
        dic["pred"] = Y_pred

        
        H, W, _ = img.shape
        img_pred = Y_pred.reshape(H, W, 3)
        img_pred = jnp.clip(img_pred, 0.0, 1.0)

        plt.figure(figsize=(15,4))
        plt.subplot(1,4,1)
        plt.semilogy(loss_hist)
        plt.xlabel("Iteration")
        plt.title(f"training loss, sig={sigma}")
        plt.grid(True)

        plt.subplot(1,4,2)
        plt.imshow(img)
        plt.title("GT")
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.imshow(img_pred)
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.imshow(jnp.abs(img - img_pred), cmap="Reds")
        plt. colorbar()
        plt.title("Error")
        plt.axis("off")
        if save == "yes":
            plt.savefig(f'realistic/ourFFadam{ej}_{nIter_long}-lr{lr}-s{sigma}.pdf', bbox_inches='tight')
            with open(f"realistic/ourFFadam{nIter_long}-{ej}-lr{lr}-s{sigma}-k{k}.pickle", "wb") as file:
                pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
            plt.close()
        else:
            plt.show()

    if objective == "multiFFadam":
        start = time.time()
        
        nIter_long = 500
        lr = 1e-3
        layers_FF = [2, 2000, 200, 3] 
        k = 10

        if ej == "optuna":
            L = [("fitting", 238), ("super",179), ("noise", 146),
            ("fitting2", 90), ("super2",179), ("noise2", 176),
            ("fitting3", 219), ("super3",118), ("noise3", 193)] #opt
        if ej == "our":
            L = [("noise", 137),
            ("super2",114)] #our

        L = [("noise", 137), ("noise2", 89), ("noise3", 91)]#[("noise", 73), ("noise2", 44), ("noise3", 46)]
        for ex, sigma in L:
            k=+1


            img, imgTrue = load_image(ex)    
            X, Y = make_image_dataset(img)

            key = random.PRNGKey(k)
            params = init_FF_params(layers_FF, key, sigma)

            params_ff, loss_hist, Y_pred = train_model_record_adam(
                FF_forward, params, X, Y, nIter_long, lr, ex)

            elapsed = time.time() - start
            print("Elapsed time (s):", elapsed)

            dic = {}
            dic["param"] = params_ff
            dic["loss"] = loss_hist
            dic["pred"] = Y_pred

            
            H, W, _ = img.shape
            img_pred = Y_pred.reshape(H, W, 3)
            img_pred = jnp.clip(img_pred, 0.0, 1.0)

            plt.figure(figsize=(15,4))
            plt.subplot(1,4,1)
            plt.semilogy(loss_hist)
            plt.xlabel("Iteration")
            plt.title(f"training loss, sig={sigma}")
            plt.grid(True)

            plt.subplot(1,4,2)
            plt.imshow(img)
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.imshow(img_pred)
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.imshow(jnp.abs(img - img_pred), cmap="Reds")
            plt. colorbar()
            plt.title("Error")
            plt.axis("off")
            if save == "yes":
                plt.savefig(f'realistic/new2{ej}FFadam{ex}_{nIter_long}-lr{lr}-s{sigma}.pdf', bbox_inches='tight')
                with open(f"realistic/new2{ej}FFadam{ex}_{nIter_long}--lr{lr}-s{sigma}-k{k}.pickle", "wb") as file:
                    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
                plt.close()
            else:
                plt.show()
            

    if objective == "visualSIREN":

        
        layers_SIREN = [2, 512, 512, 512, 3]
        k = 100
        nIter_long = 50000
        
        w0 = 40
        lr = 1e-4
        #L = [("fitting", 112), ("super",110), ("noise", 121),
        #    ("fitting2", 97), ("super2",122), ("noise2", 89),
        #    ("fitting3", 119), ("super3",155), ("noise3", 91)]
        if ej == "ej1":
            L = [("fitting", 112), ("super",110), ("noise", 89)]
        if ej == "ej2":
            L = [("fitting2", 97), ("super2",122), ("noise2", 89), ("super",110), ("noise", 89)]
        if ej == "ej3":
            L = [("fitting3", 97), ("super3",122), ("noise3", 89), ("fitting", 112)]
        #L = [("noise", 121), ("noise2", 89), ("noise3", 91)]
        for ex, _ in L:
            img, imgTrue = load_image(ex)    
            X, Y = make_image_dataset(img)
            X_true, Y_true = make_image_dataset(imgTrue)
            k=+1
            key = random.PRNGKey(k)
            params = init_SIREN_params(layers_SIREN, key, w0)
            params_siren, loss_hist, loss_true_hist, Y_pred = train_model_record_adam(
                SIREN_forward, params, X, Y, X_true, Y_true, nIter_long, lr, ex, w0)

            dic = {}
            dic["param"] = params_siren
            dic["loss"] = loss_hist
            dic["loss_true"] = loss_true_hist
            dic["pred"] = Y_pred

            H, W, _ = img.shape
            img_pred = Y_pred.reshape(H, W, 3)
            img_pred = jnp.clip(img_pred, 0.0, 1.0)

            plt.figure(figsize=(15,4))
            plt.subplot(1,4,1)
            plt.semilogy(loss_hist)
            plt.xlabel("Iteration")
            plt.title(f"training loss")
            plt.grid(True)

            plt.subplot(1,4,2)
            plt.imshow(img)
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.imshow(img_pred)
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.imshow(jnp.abs(img - img_pred), cmap="Reds")
            plt. colorbar()
            plt.title("Error")
            plt.axis("off")
            if save == "yes":
                plt.savefig(f'realistic/2SIREN_adam{ex}_{nIter_long}-lr{lr}-w0{w0}.pdf', bbox_inches='tight')
                with open(f"realistic/2SIREN_adam{ex}_{nIter_long}--lr{lr}-w0{w0}-k{k}.pickle", "wb") as file:
                    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
                plt.close()
            else:
                plt.show()

    if objective == "visualGauss":

        alpha = jnp.array(30, dtype=jnp.float32)
        lr = 1e-3
        k = 200
        nIter_long = 50000
        
        if ej == "ej1":
            L = [("fitting", 112), ("super",110), ("noise", 89)]
        if ej == "ej2":
            L = [("fitting2", 97), ("super2",122), ("noise2", 89), ("super",110), ("noise", 89)]
        if ej == "ej3":
            L = [("fitting3", 97), ("super3",122), ("noise3", 89), ("fitting", 112)]

        #L = [("noise", 121), ("noise2", 89), ("noise3", 91)]

        for ex, _ in L:
            img, imgTrue = load_image(ex)    
            X, Y = make_image_dataset(img)
            X_true, Y_true = make_image_dataset(imgTrue)
            k=+1
            key = random.PRNGKey(k)
            layers_Gauss = [2, 512, 512, 512, 3]
            params = init_Gaussian_params(layers_Gauss, key)
            params_gauss, loss_hist, loss_true_hist, Y_pred = train_model_record_adam(
                Gaussian_forward, params, X, Y, X_true, Y_true, nIter_long, lr, ex, alpha)

            dic = {}
            dic["param"] = params_gauss
            dic["loss"] = loss_hist
            dic["loss_true"] = loss_true_hist
            dic["pred"] = Y_pred
            
            H, W, _ = img.shape
            img_pred = Y_pred.reshape(H, W, 3)
            img_pred = jnp.clip(img_pred, 0.0, 1.0)

            plt.figure(figsize=(15,4))
            plt.subplot(1,4,1)
            plt.semilogy(loss_hist)
            plt.xlabel("Iteration")
            plt.title(f"training loss")
            plt.grid(True)

            plt.subplot(1,4,2)
            plt.imshow(img)
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.imshow(img_pred)
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.imshow(jnp.abs(img - img_pred), cmap="Reds")
            plt. colorbar()
            plt.title("Error")
            plt.axis("off")
            if save == "yes":
                plt.savefig(f'realistic/2Gauss_adam{ex}_{nIter_long}-lr{lr}-a{alpha}.pdf', bbox_inches='tight')
                with open(f"realistic/2Gauss_adam{ex}_{nIter_long}--lr{lr}-a{alpha}-k{k}.pickle", "wb") as file:
                    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
                plt.close()
            else:
                plt.show()

    if objective == "visualWIRE":
        
        nIter_long = 50000
        omega0=20.0
        sigma0=10.0
        omega0 = jnp.array(omega0, dtype=jnp.float32)
        sigma0 = jnp.array(sigma0, dtype=jnp.float32)
        k = 300
        lr = 1e-3

        if ej == "ej1":
            L = [("fitting", 112), ("super",110), ("noise", 89)]
        if ej == "ej2":
            L = [("fitting2", 97), ("super2",122), ("noise2", 89), ("super",110), ("noise", 89)]
        if ej == "ej3":
            L = [("fitting3", 97), ("super3",122), ("noise3", 89), ("fitting", 112)]

        #L = [("noise", 121), ("noise2", 89), ("noise3", 91)]

        for ex, _ in L:
            img, imgTrue = load_image(ex)    
            X, Y = make_image_dataset(img)
            X_true, Y_true = make_image_dataset(imgTrue)
            k=+1
            key = random.PRNGKey(k)


            layers_WIRE = [2, 512, 512, 512,  3]#[2, 360, 360, 360, 3] #
            params = init_WIRE_params(layers_WIRE, key, omega0)
            params_wire, loss_hist, loss_true_hist, Y_pred = train_model_record_adam(
                WIRE_forward, params, X, Y, X_true, Y_true, nIter_long, lr, ex, omega0, sigma0)

            dic = {}
            dic["param"] = params_wire
            dic["loss"] = loss_hist
            dic["loss_true"] = loss_true_hist
            dic["pred"] = Y_pred

            
            H, W, _ = img.shape
            img_pred = Y_pred.reshape(H, W, 3)
            img_pred = jnp.clip(img_pred, 0.0, 1.0)

            plt.figure(figsize=(15,4))
            plt.subplot(1,4,1)
            plt.semilogy(loss_hist)
            plt.xlabel("Iteration")
            plt.title(f"training loss")
            plt.grid(True)

            plt.subplot(1,4,2)
            plt.imshow(img)
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.imshow(img_pred)
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.imshow(jnp.abs(img - img_pred), cmap="Reds")
            plt. colorbar()
            plt.title("Error")
            plt.axis("off")
            if save == "yes":
                plt.savefig(f'realistic/2WIRE_adam{ex}_{nIter_long}-lr{lr}-s0{sigma0}.pdf', bbox_inches='tight')
                with open(f"realistic/2WIRE_adam{ex}_{nIter_long}--lr{lr}-s0{sigma0}-k{k}.pickle", "wb") as file:
                    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
                plt.close()
            else:
                plt.show()



