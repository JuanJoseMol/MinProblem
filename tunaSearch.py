import time
import numpy as np
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
from functools import partial
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt
from PIL import Image
from jax.nn import relu
import scipy.ndimage as ndi
from loadRealistic import load_image, make_image_dataset
import optuna
import jax.lax as lax

def save_latex_table(results, ej, mode):

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Model & Hyperparam & Best value & Best loss & Total time (s) \\\\")
    lines.append("\\hline")

    for model, r in results.items():
        lines.append(
            f"{model} & {r['param']} & "
            f"{r['best_value']:.2e} & {r['best_loss']:.2e} & "
            f"{r['T_total']:.1f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Total Optuna study time for {ej}.}}")
    lines.append("\\end{table}")

    filename = f"realistic/opttunaAdam_{mode}_{ej}_trial20.tex"

    with open(filename, "w") as f:
        f.write("\n".join(lines))

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




def loss_fn_true_streaming(forward, params, X_true, Y_true, chunk_size, *args):
    """
    Memory-safe exact MSE evaluation on large grids.
    Computes the same loss as a full forward pass, but without OOM.
    """
    N = X_true.shape[0]

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

@jit
def FF_forward(X, params):
    Ws, bs = params
    H = jnp.dot(X, Ws[0])
    H = jnp.concatenate([jnp.sin(H),
                          jnp.cos(H)], axis=-1)

    for i in range(1, len(Ws)-1):
        H = jnp.dot(H, Ws[i]) + bs[i-1]
        H = relu(H)  # ReLU

    return jnp.dot(H, Ws[-1])

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
        std = jnp.sqrt(2.0 / (layers[i]))

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
        H = jnp.exp(-0.5 * Z**2 /(alpha**2)  )

    return jnp.matmul(H, Ws[-1])+ bs[-1]

def train_model(forward, params, X, Y, X_true, Y_true, nIter, lr, *args):
    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    def loss_fn(p):
        return jnp.mean((forward(X, p, *args) - Y) ** 2)

    loss_history = []

    @jit
    def step(opt_state):
        params = get_params(opt_state)
        loss, grads = value_and_grad(loss_fn)(params)
        opt_state = opt_update(0, grads, opt_state)
        return opt_state, loss

    for _ in range(nIter):
        opt_state, loss = step(opt_state)
        loss_history.append(loss)

    final_params = get_params(opt_state)

    loss_true = loss_fn_true_streaming(
        forward,
        final_params,
        X_true,
        Y_true,
        chunk_size=4096,   # increase/decrease depending on GPU/CPU memory
        *args
    )

    Y_pred = forward(X, final_params, *args)

    return final_params, jnp.array(loss_history), loss_true, Y_pred


def train_model_adam(forward, params, X, Y, X_true, Y_true, nIter, lr, *args):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    def loss_fn(p):
        return jnp.mean((forward(X, p, *args) - Y) ** 2)

    loss_history = []

    @jit
    def step(opt_state):
        params = get_params(opt_state)
        loss, grads = value_and_grad(loss_fn)(params)
        opt_state = opt_update(0, grads, opt_state)
        return opt_state, loss

    for _ in range(nIter):
        opt_state, loss = step(opt_state)
        loss_history.append(loss)

    final_params = get_params(opt_state)
    
    loss_true = loss_fn_true_streaming(
        forward,
        final_params,
        X_true,
        Y_true,
        chunk_size=4096,   # increase/decrease depending on GPU/CPU memory
        *args
    )

    Y_pred = forward(X, final_params, *args)

    return final_params, jnp.array(loss_history), loss_true, Y_pred

def train_model_adam_noise(forward, params, X, Y, X_true, Y_true, nIter, lr, *args):
    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)

    def loss_fn(p):
        return jnp.mean((forward(X, p, *args) - Y) ** 2)

    def loss_fn_true(p):
        return jnp.mean((forward(X_true, p, *args) - Y_true) ** 2)

    loss_history = []
    loss_true_history = []

    @jit
    def step(opt_state):
        params = get_params(opt_state)
        loss, grads = value_and_grad(loss_fn)(params)
        opt_state = opt_update(0, grads, opt_state)
        return opt_state, loss

    for _ in range(nIter):
        opt_state, loss = step(opt_state)
        loss_history.append(loss)
        params_now = get_params(opt_state)
        lt = loss_fn_true(params_now)
        loss_true_history.append(lt)

    final_params = get_params(opt_state)
    Y_pred = forward(X, final_params, *args)

    return final_params, jnp.array(loss_history), jnp.array(loss_true_history), Y_pred


def objective_FFadam_noise(trial):
    sigma = trial.suggest_float("sigma", 1.0, 100.0, log=True)

    lr = 1e-1
    nIter = 1000           
    layers_FF = [2, 2000, 200, 3]

    # ---- PRNG handling (DO NOT reuse keys) ----
    key = random.PRNGKey(4)
    key = random.fold_in(key, trial.number)

    params = init_FF_params(layers_FF, key, sigma)
    params_ff, loss_hist, loss_true_hist, Y_pred = train_model_adam_noise(
        FF_forward, params, X, Y, X_true, Y_true, nIter, lr)

    # ---- objective value ----
    final_loss = float(jnp.min(loss_true_hist[-1]))

    return final_loss

def objective_FF(trial):
    sigma = trial.suggest_float("sigma", 30.0, 150.0, log=True)

    lr = 1e-1
    nIter = 5000           
    layers_FF = [2, 2000, 200, 3]

    # ---- PRNG handling (DO NOT reuse keys) ----
    key = random.PRNGKey(4)
    key = random.fold_in(key, trial.number)

    params = init_FF_params(layers_FF, key, sigma)
    params_ff, loss_hist, loss_true, Y_pred = train_model(
        FF_forward, params, X, Y, X_true, Y_true, nIter, lr)

    # ---- objective value ----
    final_loss = float(loss_true)

    return final_loss

def objective_FFadam(trial):
    sigma = trial.suggest_float("sigma", 50.0, 300.0, log=True)

    lr = 1e-3
    nIter = 5000           
    layers_FF = [2, 2000, 200, 3]

    # ---- PRNG handling (DO NOT reuse keys) ----
    key = random.PRNGKey(4)
    key = random.fold_in(key, trial.number)

    params = init_FF_params(layers_FF, key, sigma)
    params_ff, loss_hist, loss_true, Y_pred = train_model_adam(
        FF_forward, params, X, Y, X_true, Y_true, nIter, lr)

    # ---- objective value ----
    final_loss = float(loss_true)

    return final_loss

def objective_Gaussian(trial):
    alpha = trial.suggest_float("alpha", 0.005, 0.5, log=True)
    lr = 1e-1
    nIter = 5000
    layers_Gaussian = [2, 512,512,512, 3]

    # ---- PRNG handling ----
    key = random.PRNGKey(0)
    key = random.fold_in(key, trial.number)

    params = init_Gaussian_params(layers_Gaussian, key)
    params_Gaussian, loss_hist, loss_true_hist, Y_pred = train_model(
        Gaussian_forward,
        params,
        X, Y,
        X_true, Y_true,
        nIter,
        lr,
        alpha
    )

    final_loss = float(loss_hist[-1])

    return final_loss

def objective_SIREN(trial):
    w0 = trial.suggest_float("w0", 300.0, 400.0, log=True)
    lr = 1e-2
    nIter = 5000
    layers_SIREN = [2, 512,512,512, 3]

    # ---- PRNG handling ----
    key = random.PRNGKey(1)
    key = random.fold_in(key, trial.number)

    params = init_SIREN_params(layers_SIREN, key, w0)
    params_siren, loss_hist, loss_true_hist, Y_pred = train_model(
        SIREN_forward,
        params,
        X, Y,
        X_true, Y_true,
        nIter,
        lr,
        w0
    )

    final_loss = float(loss_hist[-1])

    return final_loss


if __name__ == "__main__":
    
    import argparse   
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True) #FF, SIREN
    parser.add_argument("--ej", type=str, required=True) # "fitting", "super", "noise"

    args = parser.parse_args()
    mode = args.mode
    ej = args.ej


    if mode == "FFsgd":
        all_results = []
        if ej == "ej1":
            lis = ["fitting" ] #, "fitting", 
        if ej == "ej2":
            lis = ["super2", "noise2"] #, "fitting2", 
        if ej == "ej3":
            lis = ["super3","noise3"] #"fitting3",  
        for ex in lis:
            if ex == "fitting" or ex == "fitting2" or ex == "fitting3":
                k = 0
            elif ex == "super" or ex == "super2" or ex == "super3":
                k = 1
            elif ex == "noise" or ex == "noise2" or ex == "noise3":
                k = 2

            img, imgTrue = load_image(ex)
        
            X, Y = make_image_dataset(img)
            X_true, Y_true = make_image_dataset(imgTrue)

            start = time.time()

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=k)
            )

            study.optimize(objective_FF, n_trials=20)

            elapsed = time.time() - start

            best_sigma = study.best_params["sigma"]
            best_loss = study.best_value

            # ---- printed values ----
            print("Example:", ej)
            print("Best sigma:", best_sigma)
            print("Best loss:", best_loss)
            print("Elapsed time (s):", elapsed)
            print("-" * 40)

            # ---- store printed values ----
            all_results.append({
                "example": ex,
                "param": "sigma",
                "best_value": best_sigma,
                "best_loss": best_loss,
                "T_total": elapsed
            })

        # ===============================
        # Write LaTeX table (ALL examples)
        # ===============================
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\hline")
        lines.append(
            "Example & Hyperparam & Best value & Best loss & Total time (s) \\\\"
        )
        lines.append("\\hline")

        for r in all_results:
            lines.append(
                f"{r['example']} & {r['param']} & "
                f"{r['best_value']:.1f} & {r['best_loss']:.2e} & "
                f"{r['T_total']:.1f} \\\\"
            )

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{Optuna results for Fourier Features (FF).}"
        )
        lines.append("\\end{table}")

        filename = f"realistic/fittingtrue_{ej}tuna_{mode}_trial20_lr0.1.tex"
        with open(filename, "w") as f:
            f.write("\n".join(lines))

    if mode == "FFadam":
        all_results = []
        if ej == "ej1":
            lis = ["super","noise" ] #, "fitting", 
        if ej == "ej2":
            lis = ["super2", "noise2"] #, "fitting2", 
        if ej == "ej3":
            lis = ["super3","noise3"] #"fitting3",  
        #lis = ["noise", "noise2", "noise3" ]
        for ex in lis:
            if ex == "fitting" or ex == "fitting2" or ex == "fitting3":
                k = 0
            elif ex == "super" or ex == "super2" or ex == "super3":
                k = 1
            elif ex == "noise" or ex == "noise2" or ex == "noise3":
                k = 2

            img, imgTrue = load_image(ex)
        
            X, Y = make_image_dataset(img)
            X_true, Y_true = make_image_dataset(imgTrue)

            start = time.time()

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=k)
            )

            study.optimize(objective_FFadam, n_trials=20)

            elapsed = time.time() - start

            best_sigma = study.best_params["sigma"]
            best_loss = study.best_value

            # ---- printed values ----
            print("Example:", ex)
            print("Best sigma:", best_sigma)
            print("Best loss:", best_loss)
            print("Elapsed time (s):", elapsed)
            print("-" * 40)

            # ---- store printed values ----
            all_results.append({
                "example": ex,
                "param": "sigma",
                "best_value": best_sigma,
                "best_loss": best_loss,
                "T_total": elapsed
            })

        # ===============================
        # Write LaTeX table (ALL examples)
        # ===============================
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\hline")
        lines.append(
            "Example & Hyperparam & Best value & Best loss & Total time (s) \\\\"
        )
        lines.append("\\hline")

        for r in all_results:
            lines.append(
                f"{r['example']} & {r['param']} & "
                f"{r['best_value']:.1f} & {r['best_loss']:.2e} & "
                f"{r['T_total']:.1f} \\\\"
            )

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{Optuna results for Fourier Features (FF).}"
        )
        lines.append("\\end{table}")

        filename = f"realistic/true_{ej}tuna_{mode}_trial20_lr0.001.tex"
        with open(filename, "w") as f:
            f.write("\n".join(lines))

    if mode == "FFadamnoise":
        all_results = []
        lis = ["noise", "noise2", "noise3" ]
        for ex in lis:
            if ex == "fitting" or ex == "fitting2" or ex == "fitting3":
                k = 10
            elif ex == "super" or ex == "super2" or ex == "super3":
                k = 20
            elif ex == "noise" or ex == "noise2" or ex == "noise3":
                k = 30
            k+=1
            img, imgTrue = load_image(ex)
        
            X, Y = make_image_dataset(img)
            X_true, Y_true = make_image_dataset(imgTrue)

            start = time.time()

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=k)
            )

            study.optimize(objective_FFadam_noise, n_trials=20)

            elapsed = time.time() - start

            best_sigma = study.best_params["sigma"]
            best_loss = study.best_value

            # ---- printed values ----
            print("Example:", ex)
            print("Best sigma:", best_sigma)
            print("Best loss:", best_loss)
            print("Elapsed time (s):", elapsed)
            print("-" * 40)

            # ---- store printed values ----
            all_results.append({
                "example": ex,
                "param": "sigma",
                "best_value": best_sigma,
                "best_loss": best_loss,
                "T_total": elapsed
            })

        # ===============================
        # Write LaTeX table (ALL examples)
        # ===============================
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\hline")
        lines.append(
            "Example & Hyperparam & Best value & Best loss & Total time (s) \\\\"
        )
        lines.append("\\hline")

        for r in all_results:
            lines.append(
                f"{r['example']} & {r['param']} & "
                f"{r['best_value']:.1f} & {r['best_loss']:.2e} & "
                f"{r['T_total']:.1f} \\\\"
            )

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append(
            "\\caption{Optuna results for Fourier Features (FF).}"
        )
        lines.append("\\end{table}")

        filename = f"realistic/1000sgdtrue_{ej}tuna_{mode}_trial20_lr0.1.tex"
        with open(filename, "w") as f:
            f.write("\n".join(lines))

    if mode == "SIRENS":
        if ej == "fitting":
            k=0
        if ej == "super":
            k=1
        if ej == "noise":
            k=2
        start = time.time()

        study_siren = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=k)
        )
        study_siren.optimize(objective_SIREN, n_trials=20)

        elapsed = time.time() - start

        print (ej)
        print("Best sigma:", study_siren.best_params["w0"])
        print("Best loss:", study_siren.best_value)
        print("Elapsed time (s):", elapsed)

    if mode == "Gauss":

        if ej == "fitting":
            k=0
        if ej == "super":
            k=1
        if ej == "noise":
            k=2
        start = time.time()

        study_gaussian = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=k)
        )
        study_gaussian.optimize(objective_Gaussian, n_trials=20)

        elapsed = time.time() - start

        print (ej)
        print("Best sigma:", study_gaussian.best_params["alpha"])
        print("Best loss:", study_gaussian.best_value)
        print("Elapsed time (s):", elapsed)


    if mode == "all":
        results = {}
        trial = 20
        t0 = time.time()

        study_ff = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=0)
        )
        study_ff.optimize(objective_FF, n_trials=trial)

        T_total_FF = time.time() - t0

        t0 = time.time()

        study_siren = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=0)
        )
        study_siren.optimize(objective_SIREN, n_trials=trial)

        T_total_SIREN = time.time() - t0

        t0 = time.time()

        study_gaussian = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=0)
        )
        study_gaussian.optimize(objective_Gaussian, n_trials=trial)

        T_total_Gaussian = time.time() - t0

        results["FF"] = {
            "param": "sigma",
            "best_value": study_ff.best_params["sigma"],
            "best_loss": study_ff.best_value,
            "T_total": T_total_FF
        }

        results["SIREN"] = {
            "param": "w0",
            "best_value": study_siren.best_params["w0"],
            "best_loss": study_siren.best_value,
            "T_total": T_total_SIREN
        }

        results["Gaussian"] = {
            "param": "alpha",
            "best_value": study_gaussian.best_params["alpha"],
            "best_loss": study_gaussian.best_value,
            "T_total": T_total_Gaussian
        }

        save_latex_table(results, ej, mode)





